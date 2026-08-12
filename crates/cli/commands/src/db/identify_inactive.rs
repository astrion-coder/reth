//! `reth db identify-inactive` — EIP-8295 prototype: reports maximal inactive trie subtrees as
//! JSONL. Read-only; consumes [`tables::AccountLastWritten`] / [`tables::StorageLastWritten`]
//! metadata backfilled by `reth db inject-periods` (see [`crate::db::inject_periods`]). Should be
//! run after periods have been injected — without any rows in those tables, every leaf is a
//! period-lookup miss and nothing will be reported as inactive. Mirrors go-ethereum's
//! `geth db identify-inactive` (`cmd/geth/eip8188/identifier.go`).

use crate::db::{
    inactive_identifier::{identify, IdentifyConfig, PeriodIndex, Scope},
    periods_source::{compute_period, DEFAULT_BLOCKS_PER_PERIOD},
};
use alloy_consensus::BlockHeader as _;
use alloy_primitives::BlockNumber;
use clap::Parser;
use reth_db_api::database::Database;
use reth_db_common::DbTool;
use reth_provider::{providers::ProviderNodeTypes, HeaderProvider};
use reth_storage_api::{BlockNumReader, StorageSettingsCache};
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
};
use tracing::info;

/// The arguments for the `reth db identify-inactive` command.
///
/// Known limitation: `AccountsTrie`/`StoragesTrie` are an incremental cache for fast re-hashing,
/// not a guaranteed-complete structural mirror of the trie, so this walk can only discover leaves
/// reachable through whatever happens to be cached. Reported inactive subtrees are a lower
/// bound, not exhaustive — confirmed via live testing (see `inactive_identifier` module docs).
#[derive(Parser, Debug)]
pub struct Command {
    /// Block number at which EIP-8188 period tracking begins. Used to derive the current period
    /// from the chain tip when `--current-period` isn't set.
    #[arg(long)]
    fork_block: BlockNumber,

    /// Number of blocks in each EIP-8295 period.
    #[arg(long, default_value_t = DEFAULT_BLOCKS_PER_PERIOD)]
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

    /// Path to JSONL output file (default: stdout).
    #[arg(long)]
    output: Option<PathBuf>,
}

impl Command {
    /// Execute `db identify-inactive`.
    pub fn execute<N: ProviderNodeTypes>(&self, tool: &DbTool<N>) -> eyre::Result<()> {
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
                compute_period(head_block, self.fork_block, self.blocks_per_period)
            }
        };

        let config = IdentifyConfig {
            fork_block: self.fork_block,
            blocks_per_period: self.blocks_per_period,
            current_period,
            inactive_min_age: self.inactive_min_age,
            scope: self.scope,
        };

        info!(target: "reth::cli", ?current_period, scope = ?self.scope, "EIP-8188 identify-inactive starting");

        let db = tool.provider_factory.db_ref();
        let tx = db.tx()?;
        let period_index = PeriodIndex::build(&tx)?;

        let (subtrees, stats) = reth_trie_db::with_adapter!(tool.provider_factory, |A| {
            let hashed_cursor_factory = reth_trie_db::DatabaseHashedCursorFactory::new(&tx);
            let trie_cursor_factory = reth_trie_db::DatabaseTrieCursorFactory::<_, A>::new(&tx);
            identify(
                &trie_cursor_factory,
                &hashed_cursor_factory,
                state_root,
                &period_index,
                &config,
            )?
        });

        let mut out: Box<dyn Write> = match &self.output {
            Some(path) => Box::new(BufWriter::new(File::create(path)?)),
            None => Box::new(io::stdout()),
        };
        for subtree in &subtrees {
            serde_json::to_writer(&mut out, subtree)?;
            writeln!(out)?;
        }
        out.flush()?;

        info!(target: "reth::cli", ?stats, subtrees_found = subtrees.len(), "EIP-8188 identify-inactive finished");
        Ok(())
    }
}
