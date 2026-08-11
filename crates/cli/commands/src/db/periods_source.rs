//! Shared types for the EIP-8188 prototype period injector (`reth db inject-periods` /
//! `reth db inspect-periods`). See [`crate::db::inject_periods`] and
//! [`crate::db::inspect_periods`].

use alloy_primitives::{Address, BlockNumber, StorageKey};
use async_trait::async_trait;

/// Reference period length from EIP-8295 — approximately six months at a 12 second slot time.
/// Matches go-ethereum's prototype default (`DefaultBlocksPerPeriod` in `cmd/geth/eip8188`).
pub(crate) const DEFAULT_BLOCKS_PER_PERIOD: u64 = 1_314_000;

/// A record that an account was written (balance / nonce / storage / code) at a given block.
/// Sources are expected to have already reduced their output to one row per address holding
/// the max block at which it was written (an `ARGMAX` at the source layer, e.g. in SQL).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AccountDiff {
    pub(crate) address: Address,
    pub(crate) block: BlockNumber,
}

/// A record that a specific storage slot on an account was written to a non-zero value at a
/// given block. Zero-clears must be filtered at the source; the injector trusts its input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct StorageDiff {
    pub(crate) address: Address,
    pub(crate) slot: StorageKey,
    pub(crate) block: BlockNumber,
}

/// Supplies the stream of most-recent writes for a given block range.
#[async_trait]
pub(crate) trait Source: Send + Sync {
    /// Returns one [`AccountDiff`] per address whose most-recent write fell in
    /// `[start_block, end_block]`.
    async fn account_diffs(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<AccountDiff>>;

    /// Returns one [`StorageDiff`] per `(address, slot)` whose most-recent non-zero write fell
    /// in `[start_block, end_block]`.
    async fn storage_diffs(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<StorageDiff>>;
}

/// Derives the EIP-8295 period for a given block number. Blocks before the fork and
/// zero-length periods saturate at zero. Periods that would exceed `u32` are clamped — the
/// prototype is not expected to operate on horizons that long. Mirrors go-ethereum's
/// `ComputePeriod` (`cmd/geth/eip8188/eip8188.go`).
pub(crate) fn compute_period(
    block: BlockNumber,
    fork_block: BlockNumber,
    blocks_per_period: u64,
) -> u32 {
    if block < fork_block || blocks_per_period == 0 {
        return 0;
    }
    let period = (block - fork_block) / blocks_per_period;
    period.min(u32::MAX as u64) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_period() {
        let cases = [
            ("before fork", 5, 10, 1000, 0),
            ("at fork", 10, 10, 1000, 0),
            ("one period in", 1010, 10, 1000, 1),
            ("many periods", 50_000, 0, 1000, 50),
            ("zero length saturates", 99, 0, 0, 0),
        ];
        for (name, block, fork, blocks_per_period, want) in cases {
            assert_eq!(compute_period(block, fork, blocks_per_period), want, "case: {name}");
        }
    }
}
