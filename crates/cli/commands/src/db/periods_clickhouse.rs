//! ClickHouse-backed [`Source`] for the EIP-8188 prototype period injector. Queries an
//! xatu-cbt-schema instance (`canonical_execution_*` tables) for the most-recent write per
//! account and per (account, slot) in a block range — see
//! <https://ethpandaops.io/data/xatu/getting-started/>. Direct port of go-ethereum's
//! `ClickHouseSource` (`cmd/geth/eip8188/clickhouse.go`), over the same HTTP transport.

use crate::db::periods_source::{AccountDiff, Source, StorageDiff};
use alloy_primitives::{Address, BlockNumber, StorageKey};
use async_trait::async_trait;
use clickhouse::{Client, Row};
use serde::Deserialize;
use std::str::FromStr;

/// Connection settings for the ClickHouse instance. No field has a hardcoded default host —
/// unlike go-ethereum's prototype, which baked in its team's private tailnet address, this
/// must be supplied via CLI flags or environment variables since the instance is private
/// infrastructure.
#[derive(Debug, Clone)]
pub(crate) struct ClickHouseConfig {
    pub(crate) host: String,
    pub(crate) port: u16,
    pub(crate) user: String,
    pub(crate) password: String,
    pub(crate) database: String,
    /// Value of the `meta_network_name` column identifying the target chain (e.g.
    /// `"sepolia"`). Required: some xatu-cbt-schema deployments host multiple networks'
    /// `canonical_execution_*` rows in a single database (no default database-per-network
    /// separation), so every query below filters on this explicitly rather than assuming the
    /// configured database is already scoped to one chain.
    pub(crate) network: String,
}

/// Queries `canonical_execution_*` diff tables, unioning them for accounts and reading
/// storage diffs directly for slots, taking the max block per key (`ARGMAX` at the SQL layer)
/// so the injector never needs to dedupe.
pub(crate) struct ClickHouseSource {
    client: Client,
    network: String,
}

impl ClickHouseSource {
    pub(crate) fn new(cfg: ClickHouseConfig) -> Self {
        let url = format!("http://{}:{}", cfg.host, cfg.port);
        let client = Client::default()
            .with_url(url)
            .with_user(cfg.user)
            .with_password(cfg.password)
            .with_database(cfg.database);
        Self { client, network: cfg.network }
    }
}

#[derive(Row, Deserialize)]
struct AccountDiffRow {
    addr: String,
    block: u64,
}

#[derive(Row, Deserialize)]
struct StorageDiffRow {
    addr: String,
    slot_key: String,
    block: u64,
}

#[async_trait]
impl Source for ClickHouseSource {
    /// "Account writes" per EIP-8188 include balance changes, nonce increments, storage
    /// mutations (indirectly tag the account), and contract creation events.
    async fn account_diffs(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<AccountDiff>> {
        const QUERY: &str = r"
            SELECT lower(address) AS addr, max(block_number) AS block
            FROM (
                SELECT address, block_number FROM canonical_execution_balance_diffs
                    WHERE block_number >= ? AND block_number <= ? AND meta_network_name = ?
                UNION ALL
                SELECT address, block_number FROM canonical_execution_nonce_diffs
                    WHERE block_number >= ? AND block_number <= ? AND meta_network_name = ?
                UNION ALL
                SELECT address, block_number FROM canonical_execution_storage_diffs
                    WHERE block_number >= ? AND block_number <= ? AND meta_network_name = ?
                UNION ALL
                SELECT contract_address AS address, block_number FROM canonical_execution_contracts
                    WHERE block_number >= ? AND block_number <= ? AND meta_network_name = ?
            )
            GROUP BY addr
        ";
        let rows: Vec<AccountDiffRow> = self
            .client
            .query(QUERY)
            .bind(start_block)
            .bind(end_block)
            .bind(&self.network)
            .bind(start_block)
            .bind(end_block)
            .bind(&self.network)
            .bind(start_block)
            .bind(end_block)
            .bind(&self.network)
            .bind(start_block)
            .bind(end_block)
            .bind(&self.network)
            .fetch_all()
            .await?;

        rows.into_iter()
            .map(|row| Ok(AccountDiff { address: Address::from_str(&row.addr)?, block: row.block }))
            .collect()
    }

    /// Filters out zero-clears at the source, so the injector only ever sees non-zero writes.
    async fn storage_diffs(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<StorageDiff>> {
        const QUERY: &str = r"
            SELECT lower(address) AS addr, lower(slot) AS slot_key, max(block_number) AS block
            FROM canonical_execution_storage_diffs
            WHERE block_number >= ? AND block_number <= ? AND meta_network_name = ?
              AND to_value != '0x0' AND to_value != '0x00' AND to_value != '0'
            GROUP BY addr, slot_key
        ";
        let rows: Vec<StorageDiffRow> = self
            .client
            .query(QUERY)
            .bind(start_block)
            .bind(end_block)
            .bind(&self.network)
            .fetch_all()
            .await?;

        rows.into_iter()
            .map(|row| {
                Ok(StorageDiff {
                    address: Address::from_str(&row.addr)?,
                    slot: StorageKey::from_str(&row.slot_key)?,
                    block: row.block,
                })
            })
            .collect()
    }
}
