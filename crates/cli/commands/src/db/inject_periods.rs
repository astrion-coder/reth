//! `reth db inject-periods` — EIP-8188 prototype period injector. Backfills
//! [`tables::AccountLastWritten`] / [`tables::StorageLastWritten`] from an external diff
//! source, without touching consensus state (trie tables, state root). Mirrors go-ethereum's
//! `geth db inject-periods` (`cmd/geth/dbcmd_eip8188.go`), adapted to reth's sibling-table
//! design (see `crates/storage/db-api/src/tables/mod.rs`) instead of go-ethereum's in-place
//! snapshot patch.

use crate::db::{
    periods_clickhouse::{ClickHouseConfig, ClickHouseSource},
    periods_file::FileSource,
    periods_source::{AccountDiff, Source, StorageDiff, DEFAULT_BLOCKS_PER_PERIOD},
};
use alloy_primitives::BlockNumber;
use clap::Parser;
use reth_db_api::{
    cursor::DbDupCursorRO,
    database::Database,
    models::accounts::AddressStorageKey,
    tables,
    transaction::{DbTx, DbTxMut},
};
use reth_db_common::DbTool;
use reth_node_api::NodeTypesWithDB;
use reth_provider::providers::ProviderNodeTypes;
use reth_storage_api::BlockNumReader;
use std::time::{Duration, Instant};
use tracing::info;

const DEFAULT_BATCH_SIZE: usize = 10_000;
const PROGRESS_PERIOD: Duration = Duration::from_secs(5);

/// The arguments for the `reth db inject-periods` command.
#[derive(Parser, Debug)]
pub struct Command {
    /// Data source: `clickhouse` (default) or `file:///path/to/fixtures.jsonl`.
    #[arg(long, default_value = "clickhouse")]
    source: String,

    /// ClickHouse host. Required when `--source clickhouse`. No default — this is expected
    /// to be private infrastructure, not a shared/public instance.
    #[arg(long, env = "EIP8188_CLICKHOUSE_HOST")]
    clickhouse_host: Option<String>,

    /// ClickHouse HTTP port.
    #[arg(long, env = "EIP8188_CLICKHOUSE_PORT", default_value_t = 8123)]
    clickhouse_port: u16,

    /// ClickHouse username.
    #[arg(long, env = "EIP8188_CLICKHOUSE_USER", default_value = "default")]
    clickhouse_user: String,

    /// ClickHouse password.
    #[arg(long, env = "EIP8188_CLICKHOUSE_PASSWORD", default_value = "")]
    clickhouse_password: String,

    /// ClickHouse database.
    #[arg(long, env = "EIP8188_CLICKHOUSE_DATABASE", default_value = "default")]
    clickhouse_database: String,

    /// Value of the `meta_network_name` column identifying the target chain (e.g.
    /// `"sepolia"`). Required when `--source clickhouse`: some deployments host multiple
    /// networks' `canonical_execution_*` rows in one database, so this is never assumed from
    /// `--chain` or defaulted.
    #[arg(long, env = "EIP8188_CLICKHOUSE_NETWORK")]
    clickhouse_network: Option<String>,

    /// Block number at which EIP-8188 period tracking begins (also the lower bound of the
    /// diff query range).
    #[arg(long)]
    fork_block: BlockNumber,

    /// Number of blocks in each EIP-8295 period. Only used by `reth db inspect-periods` for
    /// reporting — this command stores raw block numbers, not pre-computed periods.
    #[arg(long, default_value_t = DEFAULT_BLOCKS_PER_PERIOD)]
    blocks_per_period: u64,

    /// Upper bound of the diff query range (inclusive). Defaults to the chain tip.
    #[arg(long)]
    end_block: Option<BlockNumber>,

    /// Sibling-table rows written per database transaction.
    #[arg(long, default_value_t = DEFAULT_BATCH_SIZE)]
    batch_size: usize,

    /// Read and count diffs without writing to disk.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

/// Summarizes the outcome of an `inject-periods` run.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub(crate) struct Stats {
    pub(crate) account_diffs_seen: u64,
    pub(crate) account_rows_written: u64,
    pub(crate) account_rows_missing_plain_state: u64,
    pub(crate) storage_diffs_seen: u64,
    pub(crate) storage_rows_written: u64,
    pub(crate) storage_rows_missing_plain_state: u64,
}

impl Command {
    /// Execute `db inject-periods`.
    pub async fn execute<N: ProviderNodeTypes>(&self, tool: &DbTool<N>) -> eyre::Result<()> {
        let end_block = match self.end_block {
            Some(b) => b,
            None => tool.provider_factory.best_block_number()?,
        };
        if self.fork_block > end_block {
            eyre::bail!("fork block {} is ahead of chain tip {}", self.fork_block, end_block);
        }
        if self.blocks_per_period == 0 {
            eyre::bail!("--blocks-per-period must be > 0");
        }

        let source = self.open_source()?;

        info!(
            target: "reth::cli",
            fork_block = self.fork_block,
            end_block,
            blocks_per_period = self.blocks_per_period,
            dry_run = self.dry_run,
            "EIP-8188 injector starting"
        );

        let stats = inject(
            tool,
            source.as_ref(),
            self.fork_block,
            end_block,
            self.batch_size,
            self.dry_run,
        )
        .await?;

        info!(target: "reth::cli", ?stats, "EIP-8188 injector finished");
        Ok(())
    }

    /// Constructs a [`Source`] from the `--source` flag.
    fn open_source(&self) -> eyre::Result<Box<dyn Source>> {
        match self.source.as_str() {
            "clickhouse" => {
                let host = self.clickhouse_host.clone().ok_or_else(|| {
                    eyre::eyre!("--clickhouse-host is required when --source clickhouse")
                })?;
                let network = self.clickhouse_network.clone().ok_or_else(|| {
                    eyre::eyre!("--clickhouse-network is required when --source clickhouse")
                })?;
                Ok(Box::new(ClickHouseSource::new(ClickHouseConfig {
                    host,
                    port: self.clickhouse_port,
                    user: self.clickhouse_user.clone(),
                    password: self.clickhouse_password.clone(),
                    database: self.clickhouse_database.clone(),
                    network,
                })))
            }
            src => {
                let path = src.strip_prefix("file://").ok_or_else(|| {
                    eyre::eyre!(
                        "unrecognised --source {src:?} (expected 'clickhouse' or 'file://path')"
                    )
                })?;
                Ok(Box::new(FileSource::new(path.into())))
            }
        }
    }
}

/// Consumes diffs from `source` and writes matching [`tables::AccountLastWritten`] /
/// [`tables::StorageLastWritten`] rows, for keys that exist in the corresponding plain-state
/// table. Keys not present in plain state are counted as missing and skipped — mirrors
/// go-ethereum's "account snapshot not present; skipping" guard.
pub(crate) async fn inject<N: NodeTypesWithDB>(
    tool: &DbTool<N>,
    source: &dyn Source,
    fork_block: BlockNumber,
    end_block: BlockNumber,
    batch_size: usize,
    dry_run: bool,
) -> eyre::Result<Stats> {
    let mut stats = Stats::default();
    inject_accounts(tool, source, fork_block, end_block, batch_size, dry_run, &mut stats).await?;
    inject_storage(tool, source, fork_block, end_block, batch_size, dry_run, &mut stats).await?;
    Ok(stats)
}

/// Read-only scan: which `chunk` entries are present in `PlainAccountState` and need an
/// `AccountLastWritten` write (missing row, or a different block than what's stored).
/// Shared between the dry-run (`view`) and real (`update`) transaction paths below — a `?
/// DbTxMut` bound isn't needed here since nothing is written.
fn accounts_needing_write<'a, Tx: DbTx>(
    tx: &Tx,
    chunk: &'a [AccountDiff],
) -> eyre::Result<(Vec<&'a AccountDiff>, u64)> {
    let mut needs_write = Vec::new();
    let mut missing = 0u64;
    for diff in chunk {
        if tx.get::<tables::PlainAccountState>(diff.address)?.is_none() {
            missing += 1;
            continue;
        }
        let current = tx.get::<tables::AccountLastWritten>(diff.address)?;
        if current == Some(diff.block) {
            continue; // idempotent no-op
        }
        needs_write.push(diff);
    }
    Ok((needs_write, missing))
}

async fn inject_accounts<N: NodeTypesWithDB>(
    tool: &DbTool<N>,
    source: &dyn Source,
    fork_block: BlockNumber,
    end_block: BlockNumber,
    batch_size: usize,
    dry_run: bool,
    stats: &mut Stats,
) -> eyre::Result<()> {
    let diffs = source.account_diffs(fork_block, end_block).await?;
    stats.account_diffs_seen += diffs.len() as u64;

    let mut last_progress = Instant::now();
    for chunk in diffs.chunks(batch_size) {
        // `--dry-run` opens the environment read-only (see `Command::execute`), so it must use
        // a read-only transaction here too — `update()` always requests a writable transaction
        // and errors out against a read-only-opened environment.
        let (written, missing) = if dry_run {
            tool.provider_factory.db_ref().view(|tx| -> eyre::Result<(u64, u64)> {
                let (needs_write, missing) = accounts_needing_write(tx, chunk)?;
                Ok((needs_write.len() as u64, missing))
            })??
        } else {
            tool.provider_factory.db_ref().update(|tx| -> eyre::Result<(u64, u64)> {
                let (needs_write, missing) = accounts_needing_write(tx, chunk)?;
                for diff in &needs_write {
                    tx.put::<tables::AccountLastWritten>(diff.address, diff.block)?;
                }
                Ok((needs_write.len() as u64, missing))
            })??
        };
        stats.account_rows_written += written;
        stats.account_rows_missing_plain_state += missing;

        if last_progress.elapsed() > PROGRESS_PERIOD {
            info!(target: "reth::cli", written = stats.account_rows_written, "EIP-8188 injector: accounts progress");
            last_progress = Instant::now();
        }
    }
    Ok(())
}

/// Read-only scan: which `chunk` entries are present in `PlainStorageState` and need a
/// `StorageLastWritten` write. See [`accounts_needing_write`] for why this is split out.
fn storage_needing_write<'a, Tx: DbTx>(
    tx: &Tx,
    chunk: &'a [StorageDiff],
) -> eyre::Result<(Vec<(&'a StorageDiff, AddressStorageKey)>, u64)> {
    let mut cursor = tx.cursor_dup_read::<tables::PlainStorageState>()?;
    let mut needs_write = Vec::new();
    let mut missing = 0u64;
    for diff in chunk {
        if cursor.seek_by_key_subkey(diff.address, diff.slot)?.is_none() {
            missing += 1;
            continue;
        }
        let key = AddressStorageKey((diff.address, diff.slot));
        let current = tx.get::<tables::StorageLastWritten>(key)?;
        if current == Some(diff.block) {
            continue; // idempotent no-op
        }
        needs_write.push((diff, key));
    }
    Ok((needs_write, missing))
}

async fn inject_storage<N: NodeTypesWithDB>(
    tool: &DbTool<N>,
    source: &dyn Source,
    fork_block: BlockNumber,
    end_block: BlockNumber,
    batch_size: usize,
    dry_run: bool,
    stats: &mut Stats,
) -> eyre::Result<()> {
    let diffs = source.storage_diffs(fork_block, end_block).await?;
    stats.storage_diffs_seen += diffs.len() as u64;

    let mut last_progress = Instant::now();
    for chunk in diffs.chunks(batch_size) {
        let (written, missing) = if dry_run {
            tool.provider_factory.db_ref().view(|tx| -> eyre::Result<(u64, u64)> {
                let (needs_write, missing) = storage_needing_write(tx, chunk)?;
                Ok((needs_write.len() as u64, missing))
            })??
        } else {
            tool.provider_factory.db_ref().update(|tx| -> eyre::Result<(u64, u64)> {
                let (needs_write, missing) = storage_needing_write(tx, chunk)?;
                for (diff, key) in &needs_write {
                    tx.put::<tables::StorageLastWritten>(*key, diff.block)?;
                }
                Ok((needs_write.len() as u64, missing))
            })??
        };
        stats.storage_rows_written += written;
        stats.storage_rows_missing_plain_state += missing;

        if last_progress.elapsed() > PROGRESS_PERIOD {
            info!(target: "reth::cli", written = stats.storage_rows_written, "EIP-8188 injector: storage progress");
            last_progress = Instant::now();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloy_primitives::{Address, StorageKey, U256};
    use async_trait::async_trait;
    use reth_db_common::DbTool;
    use reth_primitives_traits::{Account, StorageEntry};
    use reth_provider::test_utils::create_test_provider_factory;

    /// A [`Source`] backed by pre-arranged in-memory slices, driving `inject` deterministically.
    #[derive(Default)]
    struct FakeSource {
        accounts: Vec<AccountDiff>,
        storage: Vec<StorageDiff>,
    }

    #[async_trait]
    impl Source for FakeSource {
        async fn account_diffs(
            &self,
            _start: BlockNumber,
            _end: BlockNumber,
        ) -> eyre::Result<Vec<AccountDiff>> {
            Ok(self.accounts.clone())
        }
        async fn storage_diffs(
            &self,
            _start: BlockNumber,
            _end: BlockNumber,
        ) -> eyre::Result<Vec<StorageDiff>> {
            Ok(self.storage.clone())
        }
    }

    /// Seeds `PlainAccountState` (alice, bob) and `PlainStorageState` (alice, slot1).
    fn seed(
        tool: &DbTool<reth_provider::test_utils::MockNodeTypesWithDB>,
    ) -> (Address, Address, StorageKey) {
        let alice = Address::from([0xaa; 20]);
        let bob = Address::from([0xbb; 20]);
        let slot1 = StorageKey::from([0x01; 32]);

        tool.provider_factory
            .db_ref()
            .update(|tx| -> Result<(), reth_db_api::DatabaseError> {
                tx.put::<tables::PlainAccountState>(
                    alice,
                    Account { nonce: 1, balance: U256::from(100), bytecode_hash: None },
                )?;
                tx.put::<tables::PlainAccountState>(
                    bob,
                    Account { nonce: 0, balance: U256::from(1), bytecode_hash: None },
                )?;
                tx.put::<tables::PlainStorageState>(
                    alice,
                    StorageEntry { key: slot1, value: U256::from(0xdeadbeefu64) },
                )?;
                Ok(())
            })
            .unwrap()
            .unwrap();

        (alice, bob, slot1)
    }

    fn test_tool() -> DbTool<reth_provider::test_utils::MockNodeTypesWithDB> {
        DbTool::new(create_test_provider_factory()).unwrap()
    }

    #[tokio::test]
    async fn test_inject_end_to_end() {
        let tool = test_tool();
        let (alice, bob, slot1) = seed(&tool);

        let source = FakeSource {
            accounts: vec![
                AccountDiff { address: alice, block: 10 },
                AccountDiff { address: bob, block: 42 },
            ],
            storage: vec![StorageDiff { address: alice, slot: slot1, block: 30 }],
        };

        let stats = inject(&tool, &source, 0, 50, 10_000, false).await.unwrap();
        assert_eq!(stats.account_rows_written, 2);
        assert_eq!(stats.storage_rows_written, 1);

        let alice_block = tool
            .provider_factory
            .db_ref()
            .view(|tx| tx.get::<tables::AccountLastWritten>(alice).unwrap())
            .unwrap();
        assert_eq!(alice_block, Some(10));

        let bob_block = tool
            .provider_factory
            .db_ref()
            .view(|tx| tx.get::<tables::AccountLastWritten>(bob).unwrap())
            .unwrap();
        assert_eq!(bob_block, Some(42));

        let slot_block = tool
            .provider_factory
            .db_ref()
            .view(|tx| {
                tx.get::<tables::StorageLastWritten>(AddressStorageKey((alice, slot1))).unwrap()
            })
            .unwrap();
        assert_eq!(slot_block, Some(30));
    }

    #[tokio::test]
    async fn test_inject_missing_plain_state() {
        let tool = test_tool();
        seed(&tool);

        let ghost = Address::from([0xdd; 20]);
        let source = FakeSource {
            accounts: vec![AccountDiff { address: ghost, block: 15 }],
            ..Default::default()
        };

        let stats = inject(&tool, &source, 0, 20, 10_000, false).await.unwrap();
        assert_eq!(stats.account_rows_missing_plain_state, 1);
        assert_eq!(stats.account_rows_written, 0);
    }

    #[tokio::test]
    async fn test_inject_idempotent() {
        let tool = test_tool();
        let (alice, _bob, slot1) = seed(&tool);

        let source = FakeSource {
            accounts: vec![AccountDiff { address: alice, block: 20 }],
            storage: vec![StorageDiff { address: alice, slot: slot1, block: 25 }],
        };

        inject(&tool, &source, 0, 30, 10_000, false).await.unwrap();
        let stats = inject(&tool, &source, 0, 30, 10_000, false).await.unwrap();
        assert_eq!(stats.account_rows_written, 0, "second pass should be a no-op");
        assert_eq!(stats.storage_rows_written, 0, "second pass should be a no-op");
    }

    #[tokio::test]
    async fn test_inject_dry_run() {
        let tool = test_tool();
        let (alice, _bob, _slot1) = seed(&tool);

        let source = FakeSource {
            accounts: vec![AccountDiff { address: alice, block: 15 }],
            ..Default::default()
        };

        let stats = inject(&tool, &source, 0, 20, 10_000, true).await.unwrap();
        assert_eq!(stats.account_rows_written, 1, "dry-run should still count");

        let alice_block = tool
            .provider_factory
            .db_ref()
            .view(|tx| tx.get::<tables::AccountLastWritten>(alice).unwrap())
            .unwrap();
        assert_eq!(alice_block, None, "dry-run must not mutate disk");
    }
}
