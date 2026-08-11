//! `reth db inspect-periods` — reports EIP-8188 prototype period statistics. Scans
//! [`tables::AccountLastWritten`] / [`tables::StorageLastWritten`], the sibling tables
//! `reth db inject-periods` backfills. Mirrors go-ethereum's `geth db inspect-periods`
//! (`cmd/geth/eip8188/inspector.go`).

use crate::db::periods_source::{compute_period, DEFAULT_BLOCKS_PER_PERIOD};
use alloy_primitives::BlockNumber;
use clap::Parser;
use reth_db_api::{cursor::DbCursorRO, database::Database, tables, transaction::DbTx};
use reth_db_common::DbTool;
use reth_node_api::NodeTypesWithDB;
use serde::Serialize;

/// The arguments for the `reth db inspect-periods` command.
#[derive(Parser, Debug)]
pub struct Command {
    /// Emit the period report as JSON (default: pretty text).
    #[arg(long)]
    json: bool,

    /// Block number at which EIP-8188 period tracking begins. When set, the report includes
    /// the derived EIP-8295 period for the max block seen; when omitted, only raw block
    /// numbers are reported.
    #[arg(long)]
    fork_block: Option<BlockNumber>,

    /// Number of blocks in each EIP-8295 period. Only used when `--fork-block` is set.
    #[arg(long, default_value_t = DEFAULT_BLOCKS_PER_PERIOD)]
    blocks_per_period: u64,
}

/// The JSON document emitted by `db inspect-periods`. Field names are snake_case to match
/// go-ethereum's `PeriodReport` shape.
#[derive(Debug, Default, Serialize)]
pub(crate) struct PeriodReport {
    pub(crate) total_accounts_with_last_written: u64,
    pub(crate) max_account_block: BlockNumber,
    pub(crate) max_account_period: Option<u32>,
    pub(crate) total_storage_slots_with_last_written: u64,
    pub(crate) max_storage_block: BlockNumber,
    pub(crate) max_storage_period: Option<u32>,
}

impl Command {
    /// Execute `db inspect-periods`.
    pub fn execute<N: NodeTypesWithDB>(&self, tool: &DbTool<N>) -> eyre::Result<()> {
        let mut report = inspect(tool)?;
        if let Some(fork_block) = self.fork_block {
            report.max_account_period =
                Some(compute_period(report.max_account_block, fork_block, self.blocks_per_period));
            report.max_storage_period =
                Some(compute_period(report.max_storage_block, fork_block, self.blocks_per_period));
        }

        if self.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            println!("EIP-8188 period report:");
            println!(
                "  accounts:      with_last_written={}  max_block={}  max_period={}",
                report.total_accounts_with_last_written,
                report.max_account_block,
                report.max_account_period.map_or("n/a".to_string(), |p| p.to_string()),
            );
            println!(
                "  storage slots: with_last_written={}  max_block={}  max_period={}",
                report.total_storage_slots_with_last_written,
                report.max_storage_block,
                report.max_storage_period.map_or("n/a".to_string(), |p| p.to_string()),
            );
        }
        Ok(())
    }
}

/// Scans the sibling tables and tallies row counts / max block.
pub(crate) fn inspect<N: NodeTypesWithDB>(tool: &DbTool<N>) -> eyre::Result<PeriodReport> {
    Ok(tool.provider_factory.db_ref().view(|tx| {
        let mut report = PeriodReport::default();

        let mut account_cursor = tx.cursor_read::<tables::AccountLastWritten>()?;
        for row in account_cursor.walk(None)? {
            let (_, block) = row?;
            report.total_accounts_with_last_written += 1;
            report.max_account_block = report.max_account_block.max(block);
        }

        let mut storage_cursor = tx.cursor_read::<tables::StorageLastWritten>()?;
        for row in storage_cursor.walk(None)? {
            let (_, block) = row?;
            report.total_storage_slots_with_last_written += 1;
            report.max_storage_block = report.max_storage_block.max(block);
        }

        Ok::<_, reth_db_api::DatabaseError>(report)
    })??)
}
