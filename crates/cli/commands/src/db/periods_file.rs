//! JSONL-file-backed [`Source`] for the EIP-8188 prototype period injector. Used for CI
//! fixtures/testing, or block ranges outside what a ClickHouse source covers. Mirrors
//! go-ethereum's `FileSource` (`cmd/geth/eip8188/filesource.go`) and reth's own
//! JSONL-streaming idiom in `init_from_state_dump` (`crates/storage/db-common/src/init.rs`).

use crate::db::periods_source::{AccountDiff, Source, StorageDiff};
use alloy_primitives::{Address, BlockNumber, StorageKey};
use async_trait::async_trait;
use serde::Deserialize;
use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::PathBuf,
};

/// One line of the JSONL fixture format: `kind` is `"account"` or `"storage"`; `slot` is only
/// present (and required) for `"storage"` rows.
#[derive(Debug, Deserialize)]
struct FileDiff {
    kind: String,
    block: BlockNumber,
    address: Address,
    #[serde(default)]
    slot: Option<StorageKey>,
}

/// Streams diffs from a JSONL file, one [`FileDiff`] per line. `ARGMAX` dedup (one row per
/// key) is the caller's responsibility, same contract as go-ethereum's `FileSource`.
pub(crate) struct FileSource {
    path: PathBuf,
}

impl FileSource {
    pub(crate) fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn read_lines(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<FileDiff>> {
        let file = File::open(&self.path)
            .map_err(|err| eyre::eyre!("open {}: {err}", self.path.display()))?;
        let reader = BufReader::new(file);

        let mut out = Vec::new();
        for (line_no, line) in reader.lines().enumerate() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let diff: FileDiff = serde_json::from_str(&line)
                .map_err(|err| eyre::eyre!("parse line {}: {err}", line_no + 1))?;
            if diff.block < start_block || (end_block > 0 && diff.block > end_block) {
                continue;
            }
            out.push(diff);
        }
        Ok(out)
    }
}

#[async_trait]
impl Source for FileSource {
    async fn account_diffs(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<AccountDiff>> {
        Ok(self
            .read_lines(start_block, end_block)?
            .into_iter()
            .filter(|d| d.kind == "account")
            .map(|d| AccountDiff { address: d.address, block: d.block })
            .collect())
    }

    async fn storage_diffs(
        &self,
        start_block: BlockNumber,
        end_block: BlockNumber,
    ) -> eyre::Result<Vec<StorageDiff>> {
        Ok(self
            .read_lines(start_block, end_block)?
            .into_iter()
            .filter(|d| d.kind == "storage")
            .filter_map(|d| {
                d.slot.map(|slot| StorageDiff { address: d.address, slot, block: d.block })
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_file_source_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("diffs.jsonl");
        let alice = Address::from([0xaa; 20]);
        let slot1 = StorageKey::from([0x01; 32]);
        std::fs::write(
            &path,
            format!(
                "{{\"kind\":\"account\",\"block\":12,\"address\":\"{alice}\"}}\n\
                 {{\"kind\":\"storage\",\"block\":33,\"address\":\"{alice}\",\"slot\":\"{slot1}\"}}\n"
            ),
        )
        .unwrap();

        let source = FileSource::new(path);
        let accounts = source.account_diffs(0, 50).await.unwrap();
        assert_eq!(accounts, vec![AccountDiff { address: alice, block: 12 }]);

        let storage = source.storage_diffs(0, 50).await.unwrap();
        assert_eq!(storage, vec![StorageDiff { address: alice, slot: slot1, block: 33 }]);
    }

    #[tokio::test]
    async fn test_file_source_block_range_filter() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("diffs.jsonl");
        let alice = Address::from([0xaa; 20]);
        std::fs::write(
            &path,
            format!(
                "{{\"kind\":\"account\",\"block\":5,\"address\":\"{alice}\"}}\n\
                 {{\"kind\":\"account\",\"block\":99,\"address\":\"{alice}\"}}\n"
            ),
        )
        .unwrap();

        let source = FileSource::new(path);
        let accounts = source.account_diffs(10, 20).await.unwrap();
        assert!(accounts.is_empty());
    }
}
