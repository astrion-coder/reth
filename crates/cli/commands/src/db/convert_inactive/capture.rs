//! Retrieves the full node/leaf contents of an already-identified [`InactiveSubtree`], ready for
//! [`super::cold_file::ColdFileWriter`] to serialize.
//!
//! Rather than re-walking [`reth_trie::verify::StateRootBranchNodesIter`] and trying to infer
//! Full/Short node shape from `BranchNodeCompact` masks (the same kind of mask-based guessing
//! that turned out to be unreliable for leaf-counting in `inactive_identifier`'s own history —
//! see that module's docs), this reuses [`Proof`]/[`StorageProof`]'s existing, already-correct
//! machinery: targeting a multiproof at *every* leaf key under the subtree (not a single lookup
//! key, as `eth_getProof` uses it) makes the proof retainer capture the exact RLP bytes of every
//! node at or above the subtree root, since every such node's path is an ancestor of at least one
//! included leaf. Decoding those bytes back into [`TrieNode`]s gives real, hashed,
//! consensus-correct nodes — no reconstruction guesswork.

use crate::db::inactive_identifier::{pack_seek_key, InactiveSubtree};
use alloy_primitives::{map::B256Set, B256};
use alloy_rlp::Decodable;
use reth_trie::{
    depth_first_cmp,
    hashed_cursor::{HashedCursor, HashedCursorFactory},
    proof::{Proof, StorageProof},
    trie_cursor::TrieCursorFactory,
    MultiProofTargets, Nibbles, TrieNode,
};

/// Every trie node under a subtree's root, in post-order (children before parents, root last) —
/// exactly the order [`super::cold_file::ColdFileWriter::write_blob`] requires.
pub(crate) struct CapturedSubtree {
    pub(crate) nodes: Vec<(Nibbles, TrieNode)>,
}

/// Captures `subtree`'s full node contents from the *persisted* trie cache (`T`), verifying the
/// result against `subtree.hash` (which [`crate::db::inactive_identifier::identify`] computed via
/// a from-scratch recompute). That verification is what makes it safe to use the faster
/// persisted-cache path here instead of another full recompute: if the cache is stale or missing
/// entries for this path — including, harmlessly, because this exact region was already converted
/// on an earlier run, or a previously-converted smaller subtree sits underneath it — the walker
/// that backs `Proof`/`StorageProof` falls back to reading `HashedAccounts`/`HashedStorages`
/// directly wherever the cache has nothing (the same fallback it already relies on for the large
/// fraction of any trie that was never cache-promoted to begin with), so the captured result is
/// still correct. A mismatch is still possible in one legitimate case, though: `identify()` reads
/// its own snapshot of `HashedAccounts`/`HashedStorages`, and if a concurrent `reth node` advances
/// the chain between that snapshot and this capture, the account/slot data this subtree covers may
/// have genuinely changed underneath it — the caller treats a mismatch as "skip this subtree,
/// count it, move on" rather than aborting the whole run, precisely because this is an expected
/// possibility on a live datadir, not necessarily a bug.
pub(crate) fn capture_subtree<T, H>(
    trie_cursor_factory: T,
    hashed_cursor_factory: H,
    subtree: &InactiveSubtree,
) -> eyre::Result<CapturedSubtree>
where
    T: TrieCursorFactory + Clone,
    H: HashedCursorFactory + Clone,
{
    let subtree_path = parse_nibbles_hex(&subtree.path);
    let leaf_keys = scan_leaf_keys(&subtree_path, &hashed_cursor_factory, subtree)?;
    eyre::ensure!(
        !leaf_keys.is_empty(),
        "subtree {} at {} has leaf_count {} but no matching hashed entries were found",
        subtree.trie,
        subtree.path,
        subtree.leaf_count
    );

    let proof_nodes = match subtree.trie {
        "account" => {
            Proof::new(trie_cursor_factory, hashed_cursor_factory)
                .multiproof(MultiProofTargets::accounts(leaf_keys))?
                .account_subtree
        }
        "storage" => {
            StorageProof::new_hashed(trie_cursor_factory, hashed_cursor_factory, subtree.owner)
                .storage_multiproof(leaf_keys.into_iter().collect::<B256Set>())?
                .subtree
        }
        other => return Err(eyre::eyre!("unexpected trie label {other:?}")),
    };

    let mut nodes: Vec<(Nibbles, TrieNode)> = Vec::new();
    for (path, bytes) in proof_nodes.iter() {
        if !path.starts_with(&subtree_path) {
            continue;
        }
        let mut slice: &[u8] = bytes.as_ref();
        nodes.push((*path, TrieNode::decode(&mut slice)?));
    }
    eyre::ensure!(
        !nodes.is_empty(),
        "no proof nodes retained under subtree path {} ({})",
        subtree.path,
        subtree.trie
    );

    nodes.sort_unstable_by(|(a, _), (b, _)| depth_first_cmp(a, b));
    verify_root_hash(&nodes, subtree)?;

    Ok(CapturedSubtree { nodes })
}

/// Every hashed-cursor entry whose key starts with `prefix` — the subtree's full leaf key set,
/// used as the multiproof's targets. Same seek-then-scan idiom as `inactive_identifier`'s
/// `scan_leaf_cluster`, just collecting keys instead of tallying inactivity.
fn scan_leaf_keys<H: HashedCursorFactory>(
    prefix: &Nibbles,
    hashed_cursor_factory: &H,
    subtree: &InactiveSubtree,
) -> eyre::Result<Vec<B256>> {
    fn scan(prefix: &Nibbles, mut cursor: impl HashedCursor) -> eyre::Result<Vec<B256>> {
        let mut keys = Vec::new();
        let mut entry = cursor.seek(pack_seek_key(prefix))?;
        while let Some((key, _value)) = entry {
            if !Nibbles::unpack(key).starts_with(prefix) {
                break;
            }
            keys.push(key);
            entry = cursor.next()?;
        }
        Ok(keys)
    }

    match subtree.trie {
        "account" => scan(prefix, hashed_cursor_factory.hashed_account_cursor()?),
        "storage" => scan(prefix, hashed_cursor_factory.hashed_storage_cursor(subtree.owner)?),
        other => Err(eyre::eyre!("unexpected trie label {other:?}")),
    }
}

/// `identify()`'s output only serializes a subtree's path as a hex string (matching
/// go-ethereum's JSON shape) — this recovers the [`Nibbles`] it was built from. `pub(crate)`
/// since `convert_inactive`'s orchestrator also needs it (stub-table lookups/deletes).
pub(crate) fn parse_nibbles_hex(hex: &str) -> Nibbles {
    Nibbles::from_nibbles_unchecked(
        hex.chars()
            .map(|c| c.to_digit(16).expect("identify-inactive only ever emits valid hex") as u8)
            .collect::<Vec<u8>>(),
    )
}

/// Recomputes the captured root node's own hash and checks it against `subtree.hash`. A mismatch
/// means the persisted-cache-backed capture diverged from `identify()`'s from-scratch recompute in
/// a way the walker's own fallback couldn't paper over — the caller does not proceed to write a
/// blob or mutate the hot tables for this subtree when that happens (see this module's top-level
/// docs for the legitimate, non-bug reason this can occur on a live datadir).
fn verify_root_hash(nodes: &[(Nibbles, TrieNode)], subtree: &InactiveSubtree) -> eyre::Result<()> {
    let (root_path, root_node) = nodes.last().expect("checked non-empty by the caller");
    let mut scratch = Vec::new();
    let hash = root_node.rlp(&mut scratch).as_hash().ok_or_else(|| {
        eyre::eyre!(
            "captured root for subtree {} at {} (decoded path {root_path:?}) did not resolve to \
             an independent hash",
            subtree.trie,
            subtree.path
        )
    })?;
    eyre::ensure!(
        hash == subtree.hash,
        "captured root hash mismatch for {} subtree at {}: recomputed {hash}, identify-inactive \
         reported {}",
        subtree.trie,
        subtree.path,
        subtree.hash
    );
    Ok(())
}
