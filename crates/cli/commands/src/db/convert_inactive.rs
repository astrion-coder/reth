//! `reth db convert-inactive` — EIP-8188/8295 prototype: moves subtrees identified by
//! `reth db identify-inactive` out of the persisted trie-node cache (`AccountsTrie`/
//! `StoragesTrie`) into an append-only cold-storage file, replacing them in the hot tables with a
//! compact stub (see [`tables::AccountTrieStubs`]/[`tables::StorageTrieStubs`]).
//!
//! # Scope and limitations — read this before relying on this command
//!
//! This command tiers only the persisted trie-node cache. Account and storage *values*
//! (`HashedAccounts`/`HashedStorages`/`PlainAccountState`/`PlainStorageState`) are never touched.
//! Freed space is real but partial: the branch-node structural overhead of a converted subtree,
//! not the leaf values themselves, which dominate total state size. This is deliberate, not an
//! oversight — moving leaf values out from under a live node without first teaching state-root
//! computation to recognize a stub and use its recorded hash directly (rather than trying to
//! rebuild it from now-missing children) would make a future full state-root recompute silently
//! produce the *wrong* root: the state root commits over the entire state, cold accounts
//! included. That "stub-aware root hashing" is real, separate, non-trivial work this command does
//! not attempt.
//!
//! Because leaf data is untouched, a live node can keep syncing/serving against a converted
//! datadir with no special-casing anywhere: it just recomputes a cold subtree from
//! `HashedAccounts`/`HashedStorages` on demand instead of hitting the (now-removed) persisted
//! cache row for it, exactly as it already must for any subtree that was never cache-promoted to
//! a persisted row in the first place (`AccountsTrie`/`StoragesTrie` only ever held a subset of
//! the trie's branch nodes — see `inactive_identifier`'s module docs for why). This is also why
//! [`capture::capture_subtree`] can safely read through the *persisted* cache (`Proof`/
//! `StorageProof`, rather than another full `StateRootBranchNodesIter` recompute) even for a
//! region whose cache rows are gone: the same graceful fallback applies.

mod capture;
mod cold_file;

use crate::db::inactive_identifier::{
    identify, IdentifyConfig, InactiveSubtree, PeriodIndex, Scope,
};
use alloy_consensus::BlockHeader as _;
use alloy_primitives::{BlockNumber, B256};
use clap::Parser;
use cold_file::ColdFileWriter;
use reth_db_api::{
    cursor::{DbCursorRO, DbCursorRW, DbDupCursorRO},
    database::Database,
    models::accounts::StorageTrieStubKey,
    tables,
    transaction::{DbTx, DbTxMut},
};
use reth_db_common::DbTool;
use reth_provider::{providers::ProviderNodeTypes, HeaderProvider};
use reth_storage_api::{BlockNumReader, StorageSettingsCache};
use reth_trie::{Nibbles, StoredNibbles, StoredNibblesSubKey};
use reth_trie_db::{
    DatabaseHashedCursorFactory, DatabaseTrieCursorFactory, StorageTrieEntryLike, TrieTableAdapter,
};
use std::path::PathBuf;
use tracing::{info, warn};

/// Metadata key `convert-inactive` uses to checkpoint the cold file's known-durable length,
/// reconciled against the file's actual size on every run — see the module's crash-safety notes.
const COLD_FILE_LEN_METADATA_KEY: &str = "eip8188.convert_inactive.cold_file_len";

/// Converts identified inactive trie subtrees to cold storage.
///
/// Moves only the persisted trie-node cache (`AccountsTrie`/`StoragesTrie`) for each converted
/// subtree into the cold file — account and storage *values* are left untouched. See this
/// module's top-level docs for the full scope/limitations.
#[derive(Parser, Debug)]
pub struct Command {
    /// Block number at which EIP-8188 period tracking begins. Used to derive the current period
    /// from the chain tip when `--current-period` isn't set.
    #[arg(long)]
    fork_block: BlockNumber,

    /// Number of blocks in each EIP-8295 period.
    #[arg(long, default_value_t = crate::db::periods_source::DEFAULT_BLOCKS_PER_PERIOD)]
    blocks_per_period: u64,

    /// Minimum age (current_period - leaf_period) for a leaf to be considered inactive.
    #[arg(long, default_value_t = 1)]
    inactive_min_age: u32,

    /// Override the current period (default: derived from the chain tip and `--fork-block`).
    #[arg(long)]
    current_period: Option<u32>,

    /// Which tries to walk: account, storage, or both.
    #[arg(long, value_enum, default_value = "both")]
    scope: Scope,

    /// Path to the append-only cold-storage file (created if absent).
    #[arg(long)]
    cold_file: PathBuf,

    /// Subtrees converted per database transaction — the crash-safety batch boundary (see the
    /// module's crash-safety notes).
    #[arg(long, default_value_t = 256)]
    batch_size: usize,

    /// Identify, capture and verify subtrees without writing to the cold file or the database.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

/// Summarizes the outcome of a `convert-inactive` run.
#[derive(Debug, Default, Clone, Copy, serde::Serialize)]
pub(crate) struct Stats {
    subtrees_identified: u64,
    subtrees_converted: u64,
    /// This exact subtree root already had a stub row — a normal outcome on a repeat run, not an
    /// error.
    subtrees_skipped_already_converted: u64,
    /// Capture succeeded but its recomputed root hash didn't match `identify()`'s — see
    /// `capture::capture_subtree`'s docs for why this can legitimately happen on a live datadir.
    subtrees_skipped_root_hash_mismatch: u64,
    account_nodes_written: u64,
    storage_nodes_written: u64,
    /// `AccountsTrie`/`StoragesTrie` rows removed (descendants plus the converted root itself).
    hot_rows_deleted: u64,
    /// Stub rows removed because a newly converted, larger subtree now subsumes them — keeps
    /// `AccountTrieStubs`/`StorageTrieStubs` free of nested/overlapping stubs. The blob(s) they
    /// pointed at become unreferenced dead space in the cold file; reclaiming that space is a
    /// "cold compaction" concern out of scope here.
    stub_rows_subsumed: u64,
    bytes_written_cold: u64,
}

impl Command {
    /// Execute `db convert-inactive`.
    pub fn execute<N: ProviderNodeTypes>(&self, tool: &DbTool<N>) -> eyre::Result<()>
    where
        <N::DB as Database>::TXMut: DbTxMut + DbTx,
    {
        if self.blocks_per_period == 0 {
            eyre::bail!("--blocks-per-period must be > 0");
        }

        let head_block = tool.provider_factory.best_block_number()?;
        let state_root = tool
            .provider_factory
            .header_by_number(head_block)?
            .ok_or_else(|| eyre::eyre!("missing canonical header at block {head_block}"))?
            .state_root();

        let current_period = match self.current_period {
            Some(period) => period,
            None => {
                if head_block < self.fork_block {
                    eyre::bail!("head block {head_block} is before fork block {}", self.fork_block);
                }
                crate::db::periods_source::compute_period(
                    head_block,
                    self.fork_block,
                    self.blocks_per_period,
                )
            }
        };

        let config = IdentifyConfig {
            fork_block: self.fork_block,
            blocks_per_period: self.blocks_per_period,
            current_period,
            inactive_min_age: self.inactive_min_age,
            scope: self.scope,
        };

        info!(target: "reth::cli", ?current_period, scope = ?self.scope, dry_run = self.dry_run, "EIP-8188 convert-inactive starting");

        let subtrees = {
            let db = tool.provider_factory.db_ref();
            let tx = db.tx()?;
            let period_index = PeriodIndex::build(&tx)?;
            let hashed_cursor_factory = DatabaseHashedCursorFactory::new(&tx);
            let (subtrees, _identify_stats) =
                identify(hashed_cursor_factory, state_root, &period_index, &config)?;
            subtrees
        };

        let mut stats = Stats { subtrees_identified: subtrees.len() as u64, ..Default::default() };
        if subtrees.is_empty() {
            info!(target: "reth::cli", "EIP-8188 convert-inactive: no inactive subtrees found");
            return Ok(());
        }

        let mut writer = ColdFileWriter::open(&self.cold_file)?;
        reconcile_cold_file(tool, &mut writer)?;

        reth_trie_db::with_adapter!(tool.provider_factory, |A| {
            for chunk in subtrees.chunks(self.batch_size.max(1)) {
                if self.dry_run {
                    dry_run_chunk::<N, A>(tool, chunk, &mut stats)?;
                } else {
                    let before = writer.valid_length();
                    convert_chunk::<N, A>(tool, chunk, &mut writer, &mut stats)?;
                    stats.bytes_written_cold += writer.valid_length() - before;
                }
            }
            Ok::<_, eyre::Report>(())
        })?;

        info!(target: "reth::cli", ?stats, "EIP-8188 convert-inactive finished");
        Ok(())
    }
}

/// Reconciles the cold file's actual length against the last checkpointed "known durable" length,
/// truncating away any orphaned trailing blob from a crash between a blob's `fsync` and its
/// hot-DB commit (see this module's top-level docs). A missing checkpoint (fresh cold file, or
/// one from before this bookkeeping existed) is left alone — there's nothing recorded to
/// reconcile against, and truncating without one would risk discarding a real, referenced blob.
fn reconcile_cold_file<N: ProviderNodeTypes>(
    tool: &DbTool<N>,
    writer: &mut ColdFileWriter,
) -> eyre::Result<()> {
    let db = tool.provider_factory.db_ref();
    let tx = db.tx()?;
    let Some(checkpoint) = read_checkpoint(&tx)? else { return Ok(()) };
    eyre::ensure!(
        writer.valid_length() >= checkpoint,
        "cold file is shorter ({} bytes) than the last recorded checkpoint ({} bytes) — it may \
         have been truncated or replaced externally",
        writer.valid_length(),
        checkpoint
    );
    if writer.valid_length() > checkpoint {
        info!(
            target: "reth::cli",
            orphaned_bytes = writer.valid_length() - checkpoint,
            "EIP-8188 convert-inactive: truncating orphaned blob from an interrupted prior run"
        );
        writer.truncate_to(checkpoint)?;
    }
    Ok(())
}

fn read_checkpoint<Tx: DbTx>(tx: &Tx) -> eyre::Result<Option<u64>> {
    let Some(bytes) = tx.get::<tables::Metadata>(COLD_FILE_LEN_METADATA_KEY.to_string())? else {
        return Ok(None);
    };
    eyre::ensure!(bytes.len() == 8, "corrupt convert-inactive checkpoint metadata");
    Ok(Some(u64::from_be_bytes(bytes.try_into().expect("checked width"))))
}

fn write_checkpoint<Tx: DbTxMut>(tx: &Tx, len: u64) -> eyre::Result<()> {
    tx.put::<tables::Metadata>(COLD_FILE_LEN_METADATA_KEY.to_string(), len.to_be_bytes().to_vec())?;
    Ok(())
}

/// Captures and verifies each subtree in `chunk` without writing anything — validates that
/// capture/verification would succeed at scale before committing to a real run.
fn dry_run_chunk<N: ProviderNodeTypes, A: TrieTableAdapter>(
    tool: &DbTool<N>,
    chunk: &[InactiveSubtree],
    stats: &mut Stats,
) -> eyre::Result<()> {
    let db = tool.provider_factory.db_ref();
    let tx = db.tx()?;

    let mut account_stub_cursor = tx.cursor_read::<tables::AccountTrieStubs>()?;
    let mut storage_stub_cursor = tx.cursor_read::<tables::StorageTrieStubs>()?;
    let hashed_cursor_factory = DatabaseHashedCursorFactory::new(&tx);
    let trie_cursor_factory = DatabaseTrieCursorFactory::<_, A>::new(&tx);

    for subtree in chunk {
        let already =
            already_converted(subtree, &mut account_stub_cursor, &mut storage_stub_cursor)?;
        capture_and_track(
            trie_cursor_factory.clone(),
            hashed_cursor_factory.clone(),
            subtree,
            already,
            stats,
        );
    }
    Ok(())
}

/// Captures, writes and commits each subtree in `chunk`, batched into one database transaction.
fn convert_chunk<N: ProviderNodeTypes, A: TrieTableAdapter>(
    tool: &DbTool<N>,
    chunk: &[InactiveSubtree],
    writer: &mut ColdFileWriter,
    stats: &mut Stats,
) -> eyre::Result<()>
where
    <N::DB as Database>::TXMut: DbTxMut + DbTx,
{
    let mut provider_rw = tool.provider_factory.provider_rw()?;

    {
        let tx = provider_rw.tx_mut();
        let mut account_trie_cursor = tx.cursor_write::<A::AccountTrieTable>()?;
        let mut storage_trie_cursor = tx.cursor_dup_write::<A::StorageTrieTable>()?;
        let mut account_stub_cursor = tx.cursor_write::<tables::AccountTrieStubs>()?;
        let mut storage_stub_cursor = tx.cursor_write::<tables::StorageTrieStubs>()?;

        let tx = provider_rw.tx_ref();
        let hashed_cursor_factory = DatabaseHashedCursorFactory::new(tx);
        let trie_cursor_factory = DatabaseTrieCursorFactory::<_, A>::new(tx);

        for subtree in chunk {
            let already =
                already_converted(subtree, &mut account_stub_cursor, &mut storage_stub_cursor)?;
            let Some(captured) = capture_and_track(
                trie_cursor_factory.clone(),
                hashed_cursor_factory.clone(),
                subtree,
                already,
                stats,
            ) else {
                continue;
            };

            let stub = writer.write_blob(&captured.nodes)?;
            let verified_hash = writer.read_and_verify(&stub)?;
            eyre::ensure!(
                verified_hash == subtree.hash,
                "cold-file write for {} subtree at {} did not round-trip: wrote for hash {}, \
                 read back {} — this indicates a bug in the cold-file encoder, not a stale cache",
                subtree.trie,
                subtree.path,
                subtree.hash,
                verified_hash,
            );

            let subtree_path = capture::parse_nibbles_hex(&subtree.path);
            match subtree.trie {
                "account" => {
                    stats.hot_rows_deleted +=
                        delete_account_trie_range::<A, _>(&mut account_trie_cursor, &subtree_path)?;
                    stats.stub_rows_subsumed +=
                        delete_account_stub_range(&mut account_stub_cursor, &subtree_path)?;
                    account_stub_cursor.upsert(StoredNibbles(subtree_path), &stub)?;
                    stats.account_nodes_written += captured.nodes.len() as u64;
                }
                "storage" => {
                    stats.hot_rows_deleted += delete_storage_trie_range::<A, _>(
                        &mut storage_trie_cursor,
                        subtree.owner,
                        &subtree_path,
                    )?;
                    stats.stub_rows_subsumed += delete_storage_stub_range(
                        &mut storage_stub_cursor,
                        subtree.owner,
                        &subtree_path,
                    )?;
                    storage_stub_cursor.upsert(
                        StorageTrieStubKey((subtree.owner, StoredNibblesSubKey(subtree_path))),
                        &stub,
                    )?;
                    stats.storage_nodes_written += captured.nodes.len() as u64;
                }
                other => return Err(eyre::eyre!("unexpected trie label {other:?}")),
            }
            stats.subtrees_converted += 1;
        }
    }

    let tx = provider_rw.tx_mut();
    write_checkpoint(tx, writer.valid_length())?;
    provider_rw.commit()?;

    Ok(())
}

/// Shared per-subtree bookkeeping: skip (and count) already-converted subtrees without doing any
/// work, otherwise capture and verify, skip (and count) a verification failure, or return the
/// captured subtree for the caller to write/commit (or, in the dry-run path, just discard).
fn capture_and_track<T, H>(
    trie_cursor_factory: T,
    hashed_cursor_factory: H,
    subtree: &InactiveSubtree,
    already_converted: bool,
    stats: &mut Stats,
) -> Option<capture::CapturedSubtree>
where
    T: reth_trie::trie_cursor::TrieCursorFactory + Clone,
    H: reth_trie::hashed_cursor::HashedCursorFactory + Clone,
{
    if already_converted {
        stats.subtrees_skipped_already_converted += 1;
        return None;
    }
    match capture::capture_subtree(trie_cursor_factory, hashed_cursor_factory, subtree) {
        Ok(captured) => Some(captured),
        Err(err) => {
            warn!(
                target: "reth::cli",
                %err,
                trie = subtree.trie,
                path = %subtree.path,
                "convert-inactive: skipping subtree, capture/verification failed"
            );
            stats.subtrees_skipped_root_hash_mismatch += 1;
            None
        }
    }
}

fn already_converted<CA, CS>(
    subtree: &InactiveSubtree,
    account_stub_cursor: &mut CA,
    storage_stub_cursor: &mut CS,
) -> eyre::Result<bool>
where
    CA: DbCursorRO<tables::AccountTrieStubs>,
    CS: DbCursorRO<tables::StorageTrieStubs>,
{
    let path = capture::parse_nibbles_hex(&subtree.path);
    Ok(match subtree.trie {
        "account" => account_stub_cursor.seek_exact(StoredNibbles(path))?.is_some(),
        "storage" => storage_stub_cursor
            .seek_exact(StorageTrieStubKey((subtree.owner, StoredNibblesSubKey(path))))?
            .is_some(),
        other => return Err(eyre::eyre!("unexpected trie label {other:?}")),
    })
}

fn delete_account_trie_range<A: TrieTableAdapter, C>(
    cursor: &mut C,
    prefix: &Nibbles,
) -> eyre::Result<u64>
where
    C: DbCursorRO<A::AccountTrieTable> + DbCursorRW<A::AccountTrieTable>,
{
    let mut deleted = 0u64;
    let mut entry = cursor.seek((*prefix).into())?;
    while let Some((key, _)) = entry {
        if !A::account_key_to_nibbles(&key).starts_with(prefix) {
            break;
        }
        cursor.delete_current()?;
        deleted += 1;
        entry = cursor.next()?;
    }
    Ok(deleted)
}

fn delete_storage_trie_range<A: TrieTableAdapter, C>(
    cursor: &mut C,
    owner: B256,
    prefix: &Nibbles,
) -> eyre::Result<u64>
where
    C: DbDupCursorRO<A::StorageTrieTable> + DbCursorRW<A::StorageTrieTable>,
{
    let Some(value) = cursor.seek_by_key_subkey(owner, (*prefix).into())? else {
        return Ok(0);
    };
    if !A::subkey_to_nibbles(value.nibbles()).starts_with(prefix) {
        return Ok(0);
    }
    cursor.delete_current()?;
    let mut deleted = 1u64;
    while let Some((next_owner, value)) = cursor.next_dup()? {
        if next_owner != owner || !A::subkey_to_nibbles(value.nibbles()).starts_with(prefix) {
            break;
        }
        cursor.delete_current()?;
        deleted += 1;
    }
    Ok(deleted)
}

fn delete_account_stub_range<C>(cursor: &mut C, prefix: &Nibbles) -> eyre::Result<u64>
where
    C: DbCursorRO<tables::AccountTrieStubs> + DbCursorRW<tables::AccountTrieStubs>,
{
    let mut deleted = 0u64;
    let mut entry = cursor.seek(StoredNibbles(*prefix))?;
    while let Some((key, _)) = entry {
        if !key.0.starts_with(prefix) {
            break;
        }
        cursor.delete_current()?;
        deleted += 1;
        entry = cursor.next()?;
    }
    Ok(deleted)
}

fn delete_storage_stub_range<C>(cursor: &mut C, owner: B256, prefix: &Nibbles) -> eyre::Result<u64>
where
    C: DbCursorRO<tables::StorageTrieStubs> + DbCursorRW<tables::StorageTrieStubs>,
{
    let mut deleted = 0u64;
    let mut entry = cursor.seek(StorageTrieStubKey((owner, StoredNibblesSubKey(*prefix))))?;
    while let Some((key, _)) = entry {
        let (key_owner, key_path) = key.0;
        if key_owner != owner || !key_path.0.starts_with(prefix) {
            break;
        }
        cursor.delete_current()?;
        deleted += 1;
        entry = cursor.next()?;
    }
    Ok(deleted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::inactive_identifier::IdentifyConfig;
    use alloy_primitives::{keccak256, Address, U256};
    use reth_db_api::database::Database;
    use reth_db_common::DbTool;
    use reth_primitives_traits::Account;
    use reth_provider::test_utils::create_test_provider_factory;
    use reth_trie_db::{DatabaseStateRoot, LegacyKeyAdapter};

    type TestStateRoot<'a, TX> = reth_trie::StateRoot<
        DatabaseTrieCursorFactory<&'a TX, LegacyKeyAdapter>,
        DatabaseHashedCursorFactory<&'a TX>,
    >;

    fn test_tool() -> DbTool<reth_provider::test_utils::MockNodeTypesWithDB> {
        DbTool::new(create_test_provider_factory()).unwrap()
    }

    /// Seeds two hashed accounts sharing nibble prefix `0x0` (so `identify()` finds them as one
    /// maximal two-leaf subtree) plus one outside that prefix, marks the first two inactive via
    /// `AccountLastWritten`, and returns the real state root computed from the seeded
    /// `HashedAccounts` (`AccountsTrie` is left empty — the walker's graceful fallback to
    /// `HashedAccounts` when the persisted cache has nothing is exactly the mechanism this whole
    /// command relies on, so exercising it with an empty cache is the more meaningful test).
    fn seed_two_leaf_subtree(
        tool: &DbTool<reth_provider::test_utils::MockNodeTypesWithDB>,
    ) -> B256 {
        // `TrieUpdates` (what `StateRootBranchNodesIter` actually yields) only records a branch
        // node once it has further nested branching of its own — a trie with only 2-3 leaves is
        // too shallow for any node to qualify, so `identify()` would see zero branch nodes and
        // report nothing, regardless of activity. Seeding a few dozen unrelated *active* accounts
        // gives the trie enough real depth for alice+bob's shared-prefix branch to actually be
        // recorded, matching the shape any real (much larger) datadir already has.
        let addr_from = |n: u32| Address::from_slice(&U256::from(n).to_be_bytes::<32>()[12..]);
        // Reserved so alice+bob's shared-prefix region contains *only* the two of them — with
        // enough filler accounts, some would otherwise land under the same prefix by chance,
        // making that region "mixed" (partly active) instead of the clean fully-inactive subtree
        // this test expects.
        let reserved_prefix = |addr: &Address| keccak256(addr).0[0] == 0x00;

        let mut alice = Address::ZERO;
        let mut bob = Address::ZERO;
        let mut n = 0u32;
        while alice == Address::ZERO || bob == Address::ZERO {
            n += 1;
            let candidate = addr_from(n);
            // Shared two-nibble prefix, so their common branch sits deep enough to actually get
            // promoted into a `TrieUpdates` entry alongside the several hundred filler accounts
            // below (empirically, a shallower shared prefix doesn't reliably get promoted at this
            // account count).
            if !reserved_prefix(&candidate) {
                continue;
            }
            if alice == Address::ZERO {
                alice = candidate;
            } else if candidate != alice {
                bob = candidate;
            }
        }
        // Filler range picked well clear of every `n` the search above may have consumed, and
        // filtered to keep alice+bob's reserved prefix exclusively theirs.
        let filler = (1_000_000u32..1_000_500).map(addr_from).filter(|a| !reserved_prefix(a));

        tool.provider_factory
            .db_ref()
            .update(|tx| -> Result<(), reth_db_api::DatabaseError> {
                for (i, addr) in [alice, bob].into_iter().chain(filler).enumerate() {
                    let account =
                        Account { nonce: i as u64, balance: U256::from(i), bytecode_hash: None };
                    tx.put::<tables::HashedAccounts>(keccak256(addr), account)?;
                }
                // Only alice and bob are marked inactive — the filler accounts are left with no
                // `AccountLastWritten` row (defaults to "active" per `scan_leaf_cluster`'s period
                // lookup miss handling), so the root stays "mixed" and alice+bob's shared-prefix
                // candidate gets emitted as its own maximal subtree instead of being subsumed into
                // one whole-trie candidate covering everything.
                tx.put::<tables::AccountLastWritten>(alice, 100)?;
                tx.put::<tables::AccountLastWritten>(bob, 100)?;
                Ok(())
            })
            .unwrap()
            .unwrap();

        let db = tool.provider_factory.db_ref();
        let tx = db.tx().unwrap();
        TestStateRoot::from_tx(&tx).root().unwrap()
    }

    fn config() -> IdentifyConfig {
        IdentifyConfig {
            fork_block: 0,
            blocks_per_period: 1000,
            current_period: 1,
            inactive_min_age: 1,
            scope: Scope::Both,
        }
    }

    #[test]
    fn test_convert_end_to_end_writes_stub_and_cold_blob() {
        let tool = test_tool();
        let state_root = seed_two_leaf_subtree(&tool);

        let subtrees = {
            let db = tool.provider_factory.db_ref();
            let tx = db.tx().unwrap();
            let period_index = PeriodIndex::build(&tx).unwrap();
            let hashed_cursor_factory = DatabaseHashedCursorFactory::new(&tx);
            let (subtrees, _stats) =
                identify(hashed_cursor_factory, state_root, &period_index, &config()).unwrap();
            subtrees
        };
        assert_eq!(subtrees.len(), 1, "expected exactly one maximal inactive account subtree");
        assert_eq!(subtrees[0].leaf_count, 2);

        let dir = tempfile::tempdir().unwrap();
        let cold_file = dir.path().join("cold.bin");
        let mut writer = ColdFileWriter::open(&cold_file).unwrap();
        let mut stats = Stats::default();

        convert_chunk::<_, LegacyKeyAdapter>(&tool, &subtrees, &mut writer, &mut stats).unwrap();

        assert_eq!(stats.subtrees_converted, 1);
        assert_eq!(stats.subtrees_skipped_already_converted, 0);
        assert_eq!(stats.subtrees_skipped_root_hash_mismatch, 0);
        assert!(stats.account_nodes_written > 0);

        // The stub row exists at the subtree's path and round-trips to the reported hash.
        let subtree_path = capture::parse_nibbles_hex(&subtrees[0].path);
        let db = tool.provider_factory.db_ref();
        let stub = db
            .view(|tx| tx.get::<tables::AccountTrieStubs>(StoredNibbles(subtree_path)).unwrap())
            .unwrap()
            .expect("stub row must exist after conversion");
        let recomputed = writer.read_and_verify(&stub).unwrap();
        assert_eq!(recomputed, subtrees[0].hash);

        // A second run is a clean no-op (idempotency), not a duplicate blob/stub.
        let mut second_stats = Stats::default();
        convert_chunk::<_, LegacyKeyAdapter>(&tool, &subtrees, &mut writer, &mut second_stats)
            .unwrap();
        assert_eq!(second_stats.subtrees_converted, 0);
        assert_eq!(second_stats.subtrees_skipped_already_converted, 1);
    }
}
