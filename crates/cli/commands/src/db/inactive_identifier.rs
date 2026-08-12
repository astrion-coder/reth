//! Core algorithm for `reth db identify-inactive` — walks the persisted account trie (and, per
//! [`Scope`], per-account storage tries) via [`DepthFirstTrieIterator`] to find maximal subtrees
//! whose leaves are all "inactive" (`current_period - leaf_period >= inactive_min_age`),
//! returning them as [`InactiveSubtree`]s. Read-only; never mutates the database. Mirrors
//! go-ethereum's `identifier.go` (`cmd/geth/eip8188/identifier.go`), including the bottom-up
//! double-counting fix from `384d5dd7e`.
//!
//! Reth's persisted trie (`AccountsTrie`/`StoragesTrie`) only stores branch nodes — leaves live
//! solely in `HashedAccounts`/`HashedStorages`, discovered per branch node via the
//! `state_mask`/`tree_mask` bit difference (a nibble present in `state_mask` but absent from
//! `tree_mask` is a direct leaf child). A branch node's own hash is not self-described either:
//! it is known only from its *parent's* `hash_mask`/`hashes` (or, for the true trie root, from
//! `BranchNodeCompact::root_hash`). Both quirks are handled in [`walk_account_trie`] /
//! [`walk_storage_trie`] below.
//!
//! **Known limitation, confirmed via live testing against a real datadir**: `AccountsTrie`/
//! `StoragesTrie` are an *incremental cache* for fast re-hashing, not a guaranteed-complete
//! structural mirror of the trie — the same reason `TrieNodeIter` (deliberately not used here,
//! see below) can skip whole subtrees for incremental hash updates. A `tree_mask` bit being
//! clear does not reliably mean "exactly one leaf here"; it can also mean
//! "an entire subtree is uncached," which this walk has no way to detect or descend into (a
//! `resolve_leaf` seek still succeeds — it just silently returns the *first* hashed entry under
//! that prefix, not necessarily the *only* one). Confirmed against a freshly-initialized mainnet
//! genesis datadir (8893 accounts): only 4543 were reachable this way, with zero seek/prefix
//! mismatches reported (ruling out a bug in [`resolve_leaf`] itself) — the other ~4350 accounts
//! simply live in parts of the trie nothing here ever visits. Reported inactive subtrees are
//! therefore a **lower bound**, not exhaustive — this walk will not find every inactive subtree
//! that exists, only the ones reachable through whatever happens to be cached. A fully exhaustive
//! port would need to drive the walk from a full trie recomputation
//! (`StateRoot::root_with_progress` /`HashBuilder`, the technique
//! `crates/trie/trie/src/verify.rs`'s `StateRootBranchNodesIter` uses) built purely from
//! `HashedAccounts`/`HashedStorages`, which is structurally complete but a full trie rebuild rather
//! than a fast structural read — a materially heavier operation, deliberately out of scope for this
//! prototype.

use crate::db::periods_source::compute_period;
use alloy_primitives::{keccak256, Address, BlockNumber, B256};
use reth_db_api::{
    cursor::DbCursorRO,
    table::{Decode, Decompress},
    tables,
    transaction::DbTx,
};
use reth_etl::Collector;
use reth_trie::{
    hashed_cursor::{HashedCursor, HashedCursorFactory},
    trie_cursor::{depth_first::DepthFirstTrieIterator, TrieCursorFactory},
    Nibbles,
};
use serde::Serialize;
use std::collections::HashMap;

/// Which trie(s) `identify-inactive` walks. The account trie is always walked (it's the only
/// way to discover which accounts have storage); `Scope` controls whether account subtrees are
/// emitted and whether per-account storage tries are descended into at all. Mirrors go-ethereum's
/// `eip8188.Scope`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum Scope {
    /// Only the account trie; storage tries are never descended into.
    Account,
    /// Walk account leaves to discover storage roots, but only emit storage subtrees.
    Storage,
    /// Both account and per-contract storage subtrees are emitted.
    Both,
}

impl Scope {
    const fn emit_account(self) -> bool {
        matches!(self, Self::Account | Self::Both)
    }

    const fn descend_storage(self) -> bool {
        matches!(self, Self::Storage | Self::Both)
    }
}

/// A maximal inactive subtree root, ready to be moved to alternate storage as a unit. Field
/// names/shape match go-ethereum's `InactiveSubtree` JSON output exactly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct InactiveSubtree {
    pub(crate) trie: &'static str,
    pub(crate) owner: B256,
    pub(crate) path: String,
    pub(crate) hash: B256,
    pub(crate) leaf_count: u64,
}

/// Parameters for a single `identify` run.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IdentifyConfig {
    pub(crate) fork_block: BlockNumber,
    pub(crate) blocks_per_period: u64,
    pub(crate) current_period: u32,
    pub(crate) inactive_min_age: u32,
    pub(crate) scope: Scope,
}

impl IdentifyConfig {
    /// A leaf last written at `last_written_block` is inactive iff its age (in periods) is at
    /// least `inactive_min_age`. A leaf written in a period after `current_period` (clock skew /
    /// defensive case) is never inactive.
    fn is_inactive(&self, last_written_block: BlockNumber) -> bool {
        let leaf_period =
            compute_period(last_written_block, self.fork_block, self.blocks_per_period);
        if self.current_period < leaf_period {
            return false;
        }
        self.current_period - leaf_period >= self.inactive_min_age
    }
}

/// Aggregate counters for a single `identify` run.
#[derive(Debug, Default, Clone, Copy, Serialize)]
pub(crate) struct IdentifyStats {
    pub(crate) accounts_scanned: u64,
    pub(crate) storage_slots_scanned: u64,
    pub(crate) storage_tries_walked: u64,
    pub(crate) inactive_account_subtrees: u64,
    pub(crate) inactive_storage_subtrees: u64,
    /// A leaf discovered via a branch node's `state_mask`/`tree_mask` had no entry in
    /// `AccountLastWritten`/`StorageLastWritten`. Treated as active (never a false-positive
    /// inactive emission) — go-ethereum's analogous `SnapshotMismatches` counter, renamed since
    /// reth has no snapshot layer; this is a miss against the ephemeral period index instead.
    pub(crate) period_lookup_misses: u64,
    /// A `state_mask`/`tree_mask` nibble slot that should hold exactly one leaf (per the
    /// invariant [`resolve_leaf`] relies on) had no matching entry in the hashed cursor at all —
    /// diagnostic only, never expected to be nonzero; see [`resolve_leaf`]'s doc comment.
    pub(crate) leaf_slot_prefix_mismatches: u64,
}

/// Walks the account trie (and, per `config.scope`, per-account storage tries), returning every
/// maximal inactive subtree found plus run statistics. `state_root` is the chain header's known
/// state root for the block being walked — see [`finalize_walk`] for why an external oracle is
/// needed at all.
pub(crate) fn identify<T: TrieCursorFactory, H: HashedCursorFactory>(
    trie_cursor_factory: &T,
    hashed_cursor_factory: &H,
    state_root: B256,
    period_index: &PeriodIndex,
    config: &IdentifyConfig,
) -> eyre::Result<(Vec<InactiveSubtree>, IdentifyStats)> {
    let mut stats = IdentifyStats::default();
    let mut subtrees = Vec::new();
    walk_account_trie(
        trie_cursor_factory,
        hashed_cursor_factory,
        state_root,
        period_index,
        config,
        &mut stats,
        &mut subtrees,
    )?;
    Ok((subtrees, stats))
}

/// Ephemeral, per-run hash-keyed period index. `AccountLastWritten`/`StorageLastWritten` are
/// keyed by plain `Address`/`(Address, StorageKey)` to match `PlainAccountState`'s keyspace, but
/// a trie walk visits `HashedAccounts`/`HashedStorages` in hash order and reth keeps no
/// address-from-hash preimage table — so periods can't be probed directly mid-walk. Built once
/// per run instead of maintained as a persistent sibling table: streams the existing tables,
/// hashes each key, and sorts via [`Collector`] (the same utility `HashingAccountStage` uses to
/// build `HashedAccounts` from `PlainAccountState`), then loads the sorted output into a
/// `HashMap` for O(1) point lookups during the walk.
pub(crate) struct PeriodIndex {
    accounts: HashMap<B256, BlockNumber>,
    storage: HashMap<HashedStorageKey, BlockNumber>,
}

impl PeriodIndex {
    /// Builds the index from the current contents of `AccountLastWritten`/`StorageLastWritten`.
    pub(crate) fn build<Tx: DbTx>(tx: &Tx) -> eyre::Result<Self> {
        let mut account_collector: Collector<B256, BlockNumber> =
            Collector::new(128 * 1024 * 1024, None);
        let mut account_cursor = tx.cursor_read::<tables::AccountLastWritten>()?;
        for row in account_cursor.walk(None)? {
            let (address, block): (Address, BlockNumber) = row?;
            account_collector.insert(keccak256(address), block)?;
        }
        let mut accounts = HashMap::with_capacity(account_collector.len());
        for entry in account_collector.iter()? {
            let (key, value) = entry?;
            accounts.insert(B256::decode(&key)?, BlockNumber::decompress(&value)?);
        }

        let mut storage_collector: Collector<HashedStorageKey, BlockNumber> =
            Collector::new(128 * 1024 * 1024, None);
        let mut storage_cursor = tx.cursor_read::<tables::StorageLastWritten>()?;
        for row in storage_cursor.walk(None)? {
            let (key, block) = row?;
            let (address, slot) = key.0;
            storage_collector
                .insert(HashedStorageKey((keccak256(address), keccak256(slot))), block)?;
        }
        let mut storage = HashMap::with_capacity(storage_collector.len());
        for entry in storage_collector.iter()? {
            let (key, value) = entry?;
            storage.insert(HashedStorageKey::decode(&key)?, BlockNumber::decompress(&value)?);
        }

        Ok(Self { accounts, storage })
    }

    fn account_block(&self, hashed_address: B256) -> Option<BlockNumber> {
        self.accounts.get(&hashed_address).copied()
    }

    fn storage_block(&self, hashed_owner: B256, hashed_slot: B256) -> Option<BlockNumber> {
        self.storage.get(&HashedStorageKey((hashed_owner, hashed_slot))).copied()
    }
}

/// `keccak256(address)` concatenated with `keccak256(slot)`. Mirrors
/// [`reth_db_api::models::accounts::AddressStorageKey`]'s pattern, hash-keyed instead of
/// plain-keyed, for use as a [`Collector`] key when building [`PeriodIndex`]'s storage half.
#[derive(
    Debug,
    Default,
    Copy,
    Clone,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
struct HashedStorageKey((B256, B256));

impl reth_db_api::table::Encode for HashedStorageKey {
    type Encoded = [u8; 64];

    fn encode(self) -> Self::Encoded {
        let mut buf = [0u8; 64];
        buf[..32].copy_from_slice(self.0 .0.as_slice());
        buf[32..].copy_from_slice(self.0 .1.as_slice());
        buf
    }
}

impl Decode for HashedStorageKey {
    fn decode(value: &[u8]) -> Result<Self, reth_db_api::DatabaseError> {
        let owner = B256::from_slice(&value[..32]);
        let slot = B256::from_slice(&value[32..]);
        Ok(Self((owner, slot)))
    }
}

/// A trie node whose children (branch-node candidates and/or direct leaves) have been folded in,
/// awaiting its own parent's decision. Mirrors go-ethereum's `nodeFrame`. `hash` is resolved by
/// the caller when the frame is popped (see module docs) — `None` before that point is not
/// meaningful and is only ever read via [`fold_child`] / [`finalize_root`].
struct NodeFrame {
    path: Nibbles,
    /// Only populated for the true trie root, which self-describes its own hash. Non-root
    /// frames have their effective hash resolved from the parent's `hash_mask` in
    /// [`fold_child`], not stored here.
    root_hash: Option<B256>,
    all_inactive: bool,
    leaf_count: u64,
    candidates: Vec<InactiveSubtree>,
}

impl NodeFrame {
    fn new(path: Nibbles, root_hash: Option<B256>) -> Self {
        Self { path, root_hash, all_inactive: true, leaf_count: 0, candidates: Vec::new() }
    }
}

fn walk_account_trie<T: TrieCursorFactory, H: HashedCursorFactory>(
    trie_cursor_factory: &T,
    hashed_cursor_factory: &H,
    state_root: B256,
    period_index: &PeriodIndex,
    config: &IdentifyConfig,
    stats: &mut IdentifyStats,
    subtrees: &mut Vec<InactiveSubtree>,
) -> eyre::Result<()> {
    let cursor = trie_cursor_factory.account_trie_cursor()?;
    let iter = DepthFirstTrieIterator::new(cursor);
    let mut hashed_cursor = hashed_cursor_factory.hashed_account_cursor()?;
    let emit_output = config.scope.emit_account();

    let mut stack: Vec<NodeFrame> = Vec::new();
    for item in iter {
        let (path, branch) = item?;
        let mut frame = NodeFrame::new(path, branch.root_hash);

        while stack.last().is_some_and(|f| f.path.len() > path.len() && f.path.starts_with(&path)) {
            let child = stack.pop().expect("checked above");
            let nibble = child.path.get(path.len()).expect("child path longer than parent");
            let hash = branch.hash_mask.is_bit_set(nibble).then(|| branch.hash_for_nibble(nibble));
            let mut ctx =
                EmitCtx { trie_label: "account", owner: B256::ZERO, emit_output, subtrees, stats };
            fold_child(child, hash, &mut frame, &mut ctx);
        }

        for nibble in 0u8..16 {
            if !branch.state_mask.is_bit_set(nibble) || branch.tree_mask.is_bit_set(nibble) {
                continue;
            }
            let mut leaf_prefix = path;
            leaf_prefix.push(nibble);
            let Some(hashed_key) = resolve_leaf(&leaf_prefix, &mut hashed_cursor)? else {
                stats.leaf_slot_prefix_mismatches += 1;
                continue
            };

            frame.leaf_count += 1;
            stats.accounts_scanned += 1;
            let inactive = match period_index.account_block(hashed_key) {
                Some(block) => config.is_inactive(block),
                None => {
                    stats.period_lookup_misses += 1;
                    false
                }
            };
            if !inactive {
                frame.all_inactive = false;
            }

            if config.scope.descend_storage() {
                stats.storage_tries_walked += 1;
                walk_storage_trie(
                    trie_cursor_factory,
                    hashed_cursor_factory,
                    hashed_key,
                    period_index,
                    config,
                    stats,
                    subtrees,
                )?;
            }
        }

        stack.push(frame);
    }

    let mut ctx =
        EmitCtx { trie_label: "account", owner: B256::ZERO, emit_output, subtrees, stats };
    finalize_walk(stack, Some(state_root), &mut ctx);

    Ok(())
}

fn walk_storage_trie<T: TrieCursorFactory, H: HashedCursorFactory>(
    trie_cursor_factory: &T,
    hashed_cursor_factory: &H,
    owner: B256,
    period_index: &PeriodIndex,
    config: &IdentifyConfig,
    stats: &mut IdentifyStats,
    subtrees: &mut Vec<InactiveSubtree>,
) -> eyre::Result<()> {
    let cursor = trie_cursor_factory.storage_trie_cursor(owner)?;
    let iter = DepthFirstTrieIterator::new(cursor);
    let mut hashed_cursor = hashed_cursor_factory.hashed_storage_cursor(owner)?;

    let mut stack: Vec<NodeFrame> = Vec::new();
    for item in iter {
        let (path, branch) = item?;
        let mut frame = NodeFrame::new(path, branch.root_hash);

        while stack.last().is_some_and(|f| f.path.len() > path.len() && f.path.starts_with(&path)) {
            let child = stack.pop().expect("checked above");
            let nibble = child.path.get(path.len()).expect("child path longer than parent");
            let hash = branch.hash_mask.is_bit_set(nibble).then(|| branch.hash_for_nibble(nibble));
            let mut ctx =
                EmitCtx { trie_label: "storage", owner, emit_output: true, subtrees, stats };
            fold_child(child, hash, &mut frame, &mut ctx);
        }

        for nibble in 0u8..16 {
            if !branch.state_mask.is_bit_set(nibble) || branch.tree_mask.is_bit_set(nibble) {
                continue;
            }
            let mut leaf_prefix = path;
            leaf_prefix.push(nibble);
            let Some(hashed_slot) = resolve_leaf(&leaf_prefix, &mut hashed_cursor)? else {
                stats.leaf_slot_prefix_mismatches += 1;
                continue;
            };

            frame.leaf_count += 1;
            stats.storage_slots_scanned += 1;
            let inactive = match period_index.storage_block(owner, hashed_slot) {
                Some(block) => config.is_inactive(block),
                None => {
                    stats.period_lookup_misses += 1;
                    false
                }
            };
            if !inactive {
                frame.all_inactive = false;
            }
        }

        stack.push(frame);
    }

    // No external oracle exists for a storage trie's own root hash (unlike the account trie,
    // where the chain header provides one) — `Account` carries no `storage_root` field, it's
    // computed only at hash-time. `finalize_walk` handles that gracefully: a `None` root hash
    // just means the whole-storage-trie candidate can't be formed, falling back to flushing
    // whatever inner candidates it collected instead.
    let mut ctx = EmitCtx { trie_label: "storage", owner, emit_output: true, subtrees, stats };
    finalize_walk(stack, None, &mut ctx);

    Ok(())
}

/// Seeks `cursor` to the first hashed entry at or after `leaf_prefix` (zero-padded to a full
/// 32-byte key) and returns its key, or `None` if no entry actually starts with `leaf_prefix`
/// (defensive — `state_mask`/`tree_mask` are *expected* to guarantee exactly one such entry, but
/// a mismatch is treated as a miss rather than trusted blindly). Note this expectation can be
/// wrong in a different way that this check can't catch: if more than one hashed entry shares
/// `leaf_prefix` (an uncached subtree — see this module's top-level docs), only the first is
/// returned and the rest are silently invisible to the walk, with no mismatch to report.
fn resolve_leaf<C: HashedCursor>(
    leaf_prefix: &Nibbles,
    cursor: &mut C,
) -> eyre::Result<Option<B256>> {
    let mut buf = [0u8; 32];
    let packed = leaf_prefix.pack();
    buf[..packed.len()].copy_from_slice(&packed);
    let seek_key = B256::from(buf);

    let Some((key, _value)) = cursor.seek(seek_key)? else { return Ok(None) };
    if !Nibbles::unpack(key).starts_with(leaf_prefix) {
        return Ok(None);
    }
    Ok(Some(key))
}

/// Bundles the per-trie-walk constants (which trie, whose storage, whether output is enabled for
/// this trie under the active [`Scope`]) with the run-wide output sinks, so [`fold_child`] and
/// [`finalize_root`] don't need to thread them as separate parameters.
struct EmitCtx<'a> {
    trie_label: &'static str,
    owner: B256,
    emit_output: bool,
    subtrees: &'a mut Vec<InactiveSubtree>,
    stats: &'a mut IdentifyStats,
}

/// Folds a just-popped child frame into its parent, per go-ethereum's bug-fixed `finalize`
/// (`384d5dd7e`): a fully-inactive child with its own hash becomes a single candidate on the
/// parent, and its own inner candidates are dropped (subsumed — this is the double-counting fix).
/// A fully-inactive child with no hash of its own (embedded) has no standalone identity to
/// candidate, so its inner candidates bubble up to the parent instead. Any child that isn't
/// fully inactive marks the parent as such too, and — if this trie's output is enabled — flushes
/// the child's own candidates as final emissions (they'll never be subsumed by anything, since
/// this parent no longer qualifies).
fn fold_child(
    popped: NodeFrame,
    popped_hash: Option<B256>,
    parent: &mut NodeFrame,
    ctx: &mut EmitCtx<'_>,
) {
    parent.leaf_count += popped.leaf_count;
    if popped.all_inactive {
        if let Some(hash) = popped_hash {
            parent.candidates.push(InactiveSubtree {
                trie: ctx.trie_label,
                owner: ctx.owner,
                path: nibbles_hex(&popped.path),
                hash,
                leaf_count: popped.leaf_count,
            });
        } else {
            parent.candidates.extend(popped.candidates);
        }
    } else {
        parent.all_inactive = false;
        if ctx.emit_output {
            for candidate in popped.candidates {
                emit(candidate, ctx.subtrees, ctx.stats);
            }
        }
    }
}

/// Resolves whatever frame(s) are left on the stack once a trie walk's iterator is exhausted.
///
/// Unlike go-ethereum's raw trie, reth's persisted trie tables don't necessarily contain a row
/// for the absolute trie root: when every one of the 16 top-level nibbles is itself a further
/// branch, reth stores 16 independent single-nibble rows and no combining row at the empty path
/// at all (its hash is redundant with the block header's `state_root`, so nothing re-persists
/// it). So after the walk, `stack` may hold anywhere from zero to sixteen unresolved top-level
/// frames instead of the single finished root [`fold_child`]/[`finalize_root`] alone assume.
///
/// - Zero frames: an empty trie, nothing to report.
/// - One frame whose path is already empty: a real persisted root row exists — finalize it
///   directly, preferring its own `root_hash` but falling back to `known_root_hash` if unset.
/// - Otherwise: no single row represents the root. Synthesize one at the empty path, using
///   `known_root_hash` (the chain header's state root for the account trie; `None` for storage
///   tries, which have no such external oracle — `Account` carries no `storage_root` field, it's
///   only computed at hash-time). Each remaining top-level frame folds in "embedded"-style (its own
///   hash isn't known — there's no parent row to read a `hash_mask` from), so its inner candidates
///   bubble up rather than becoming one top-level candidate on their own.
fn finalize_walk(mut stack: Vec<NodeFrame>, known_root_hash: Option<B256>, ctx: &mut EmitCtx<'_>) {
    if stack.is_empty() {
        return;
    }
    if stack.len() == 1 && stack[0].path.is_empty() {
        let mut root = stack.pop().expect("checked len == 1");
        if root.root_hash.is_none() {
            root.root_hash = known_root_hash;
        }
        finalize_root(root, ctx);
        return;
    }
    let mut root = NodeFrame::new(Nibbles::default(), known_root_hash);
    for frame in stack {
        fold_child(frame, None, &mut root, ctx);
    }
    finalize_root(root, ctx);
}

/// Finalizes the frame with no parent (the trie root itself), per go-ethereum's `finalize` when
/// `len(stack) == 0`.
fn finalize_root(popped: NodeFrame, ctx: &mut EmitCtx<'_>) {
    if !ctx.emit_output {
        return;
    }
    if popped.all_inactive &&
        let Some(hash) = popped.root_hash
    {
        emit(
            InactiveSubtree {
                trie: ctx.trie_label,
                owner: ctx.owner,
                path: nibbles_hex(&popped.path),
                hash,
                leaf_count: popped.leaf_count,
            },
            ctx.subtrees,
            ctx.stats,
        );
        return;
    }
    for candidate in popped.candidates {
        emit(candidate, ctx.subtrees, ctx.stats);
    }
}

fn emit(subtree: InactiveSubtree, subtrees: &mut Vec<InactiveSubtree>, stats: &mut IdentifyStats) {
    match subtree.trie {
        "account" => stats.inactive_account_subtrees += 1,
        "storage" => stats.inactive_storage_subtrees += 1,
        other => unreachable!("unexpected trie label {other:?}"),
    }
    subtrees.push(subtree);
}

fn nibbles_hex(nibbles: &Nibbles) -> String {
    nibbles.iter().map(|nibble| format!("{nibble:x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(current_period: u32, inactive_min_age: u32) -> IdentifyConfig {
        IdentifyConfig {
            fork_block: 0,
            blocks_per_period: 1000,
            current_period,
            inactive_min_age,
            scope: Scope::Both,
        }
    }

    #[test]
    fn test_is_inactive_boundary() {
        let cfg = config(10, 2);
        assert!(!cfg.is_inactive(10_000)); // period 10, age 0 < 2
        assert!(!cfg.is_inactive(9_000)); // period 9, age 1 < 2
        assert!(cfg.is_inactive(8_000)); // period 8, age 2 >= 2
        assert!(cfg.is_inactive(0)); // period 0, age 10 >= 2
    }

    #[test]
    fn test_is_inactive_future_period_defensive() {
        // A leaf whose period is ahead of `current_period` (clock skew) is never inactive.
        let cfg = config(1, 0);
        assert!(!cfg.is_inactive(5_000)); // period 5 > current_period 1
    }

    #[test]
    fn test_is_inactive_no_fork_yet() {
        let cfg = config(0, 0);
        assert!(cfg.is_inactive(0)); // period 0, age 0 >= inactive_min_age 0
    }

    fn leaf_frame(path: &[u8], inactive: bool) -> NodeFrame {
        NodeFrame {
            path: Nibbles::from_nibbles(path),
            root_hash: None,
            all_inactive: inactive,
            leaf_count: 1,
            candidates: Vec::new(),
        }
    }

    #[test]
    fn test_fold_child_all_inactive_with_hash_becomes_single_candidate() {
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();
        let mut parent = NodeFrame::new(Nibbles::from_nibbles([0x1]), None);
        let child = leaf_frame(&[0x1, 0x2], true);

        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: true,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        fold_child(child, Some(B256::repeat_byte(0xaa)), &mut parent, &mut ctx);

        assert_eq!(parent.candidates.len(), 1);
        assert!(parent.all_inactive);
        assert_eq!(parent.leaf_count, 1);
        assert!(subtrees.is_empty(), "nothing emitted yet, parent hasn't been finalized");
    }

    #[test]
    fn test_fold_child_double_counting_fix_drops_inner_candidates() {
        // A grandchild candidate should be subsumed by its (fully-inactive, hashed) parent when
        // that parent is itself folded into the grandparent — the grandchild must NOT also
        // survive as a separate candidate. This is the go-ethereum `384d5dd7e` regression.
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();

        // `child` already has one candidate (as if a grandchild was folded into it earlier) but
        // is itself fully inactive and has its own hash, so it subsumes that candidate.
        let mut child = leaf_frame(&[0x1, 0x2], true);
        child.candidates.push(InactiveSubtree {
            trie: "account",
            owner: B256::ZERO,
            path: "0102ff".to_string(),
            hash: B256::repeat_byte(0xcc),
            leaf_count: 1,
        });

        let mut grandparent = NodeFrame::new(Nibbles::from_nibbles([0x1]), None);
        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: true,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        fold_child(child, Some(B256::repeat_byte(0xbb)), &mut grandparent, &mut ctx);

        assert_eq!(
            grandparent.candidates.len(),
            1,
            "only the child itself should be a candidate, not also its inner grandchild"
        );
        assert_eq!(grandparent.candidates[0].hash, B256::repeat_byte(0xbb));
    }

    #[test]
    fn test_fold_child_embedded_bubbles_up_candidates() {
        // A fully-inactive child with no hash of its own (embedded) can't be a standalone
        // candidate, so its inner candidates propagate to the parent unchanged.
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();

        let mut child = leaf_frame(&[0x1, 0x2], true);
        let inner = InactiveSubtree {
            trie: "account",
            owner: B256::ZERO,
            path: "0102ff".to_string(),
            hash: B256::repeat_byte(0xcc),
            leaf_count: 1,
        };
        child.candidates.push(inner.clone());

        let mut parent = NodeFrame::new(Nibbles::from_nibbles([0x1]), None);
        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: true,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        fold_child(child, None /* embedded: no hash */, &mut parent, &mut ctx);

        assert_eq!(parent.candidates, vec![inner]);
    }

    #[test]
    fn test_fold_child_active_child_flushes_candidates_and_marks_parent_active() {
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();

        let mut child = leaf_frame(&[0x1, 0x2], false);
        child.candidates.push(InactiveSubtree {
            trie: "account",
            owner: B256::ZERO,
            path: "0102ff".to_string(),
            hash: B256::repeat_byte(0xcc),
            leaf_count: 1,
        });

        let mut parent = NodeFrame::new(Nibbles::from_nibbles([0x1]), None);
        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: true,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        fold_child(child, Some(B256::repeat_byte(0xdd)), &mut parent, &mut ctx);

        assert!(!parent.all_inactive);
        assert!(parent.candidates.is_empty(), "child's own candidates were flushed, not carried");
        assert_eq!(subtrees.len(), 1);
        assert_eq!(subtrees[0].hash, B256::repeat_byte(0xcc));
        assert_eq!(stats.inactive_account_subtrees, 1);
    }

    #[test]
    fn test_finalize_root_all_inactive_emits_single_subtree() {
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();
        let root = NodeFrame {
            path: Nibbles::new(),
            root_hash: Some(B256::repeat_byte(0xee)),
            all_inactive: true,
            leaf_count: 3,
            candidates: vec![InactiveSubtree {
                trie: "account",
                owner: B256::ZERO,
                path: "01".to_string(),
                hash: B256::repeat_byte(0x11),
                leaf_count: 1,
            }],
        };

        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: true,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        finalize_root(root, &mut ctx);

        assert_eq!(subtrees.len(), 1);
        assert_eq!(subtrees[0].hash, B256::repeat_byte(0xee));
        assert_eq!(subtrees[0].leaf_count, 3);
    }

    #[test]
    fn test_finalize_root_mixed_emits_candidates_not_root() {
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();
        let candidate = InactiveSubtree {
            trie: "account",
            owner: B256::ZERO,
            path: "01".to_string(),
            hash: B256::repeat_byte(0x11),
            leaf_count: 1,
        };
        let root = NodeFrame {
            path: Nibbles::new(),
            root_hash: Some(B256::repeat_byte(0xee)),
            all_inactive: false,
            leaf_count: 3,
            candidates: vec![candidate.clone()],
        };

        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: true,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        finalize_root(root, &mut ctx);

        assert_eq!(subtrees, vec![candidate]);
    }

    #[test]
    fn test_finalize_root_suppressed_when_emit_output_false() {
        let mut stats = IdentifyStats::default();
        let mut subtrees = Vec::new();
        let root = NodeFrame {
            path: Nibbles::new(),
            root_hash: Some(B256::repeat_byte(0xee)),
            all_inactive: true,
            leaf_count: 1,
            candidates: Vec::new(),
        };

        let mut ctx = EmitCtx {
            trie_label: "account",
            owner: B256::ZERO,
            emit_output: false,
            subtrees: &mut subtrees,
            stats: &mut stats,
        };
        finalize_root(root, &mut ctx);

        assert!(subtrees.is_empty(), "Scope::Storage must suppress account-trie emissions");
    }

    #[test]
    fn test_resolve_leaf_prefix_mismatch_is_treated_as_miss() {
        use reth_trie::hashed_cursor::mock::MockHashedCursorFactory;
        use std::collections::BTreeMap;

        // No accounts at all under prefix `0xf`.
        let mut accounts = BTreeMap::new();
        accounts.insert(B256::repeat_byte(0x00), reth_primitives_traits::Account::default());
        let factory = MockHashedCursorFactory::new(accounts, Default::default());
        let mut cursor = factory.hashed_account_cursor().unwrap();

        let prefix = Nibbles::from_nibbles([0xf]);
        assert_eq!(resolve_leaf(&prefix, &mut cursor).unwrap(), None);
    }
}
