//! Core algorithm for `reth db identify-inactive` — walks the complete account trie (and, per
//! [`Scope`], per-account storage tries) to find maximal subtrees whose leaves are all "inactive"
//! (`current_period - leaf_period >= inactive_min_age`), returning them as [`InactiveSubtree`]s.
//! Read-only; never mutates the database. Mirrors go-ethereum's `identifier.go`
//! (`cmd/geth/eip8188/identifier.go`), including the bottom-up double-counting fix from
//! `384d5dd7e`.
//!
//! The walk is driven by [`StateRootBranchNodesIter`], which recomputes the entire trie from
//! `HashedAccounts`/`HashedStorages` via `StateRoot::root_with_progress`/`HashBuilder` rather than
//! reading the persisted `AccountsTrie`/`StoragesTrie` tables — those tables are an incremental
//! cache for fast re-hashing, not a guaranteed-complete structural mirror of the trie in general,
//! so recomputing from the hashed tables is the more robust source in cases where the two diverge
//! (on a freshly-initialized datadir they don't diverge at all — `reth db repair-trie` confirms
//! the persisted cache already matches a fresh recompute exactly — but a synced node's cache can).
//!
//! `StateRootBranchNodesIter` only yields branch nodes, not leaves. Per branch node, each of the
//! 16 nibbles falls into one of two cases, using [`alloy_trie::BranchNodeCompact`]'s masks:
//! * `state_mask` clear: no child at all.
//! * `state_mask` set, `tree_mask` set: the child is itself a branch node that will arrive later as
//!   its own [`BranchNode`] stream item — handled by [`pop_closed_children`]/[`fold_child`].
//! * `state_mask` set, `tree_mask` clear: everything else — genuinely could be a single leaf, or a
//!   real multi-leaf branch. Two earlier versions of this code got this case wrong in different
//!   ways: the first assumed `tree_mask` clear always meant exactly one leaf and undercounted by
//!   nearly half; the second tried checking `hash_mask` instead (set whenever alloy-trie's
//!   `store_branch_node` pushes a branch node's hash directly) — an improvement, but alloy-trie's
//!   `HashBuilder::update_masks` *clears* `hash_mask` again whenever an extension node (a
//!   shared-nibble-prefix run) sits between this position and the branch, so a real multi-leaf
//!   branch reached via an extension still looks mask-identical to a single leaf. There is no mask
//!   combination at this level that reliably tells the two apart — [`scan_leaf_cluster`] resolves
//!   it by always scanning forward from the seek point rather than trusting masks, using
//!   `hash_mask`/`hash_for_nibble` only to decide whether a found multi-leaf cluster has a real
//!   hash of its own (for [`fold_child`] to use) or must be inherited into a larger ancestor
//!   candidate instead. See [`walk`]'s nibble-scan loops.

use crate::db::periods_source::compute_period;
use alloy_primitives::{keccak256, Address, BlockNumber, B256};
use alloy_trie::BranchNodeCompact;
use reth_db_api::{
    cursor::DbCursorRO,
    table::{Decode, Decompress},
    tables,
    transaction::DbTx,
};
use reth_etl::Collector;
use reth_trie::{
    hashed_cursor::{HashedCursor, HashedCursorFactory},
    verify::{BranchNode, StateRootBranchNodesIter},
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
    /// A `state_mask`-set, `tree_mask`-clear nibble slot ([`scan_leaf_cluster`]'s target) had no
    /// matching entry in the hashed cursor at all. Since the walk recomputes the complete trie
    /// from `HashedAccounts`/`HashedStorages` rather than reading a possibly-incomplete persisted
    /// cache, this is a genuine invariant violation if it's ever nonzero — a real bug in
    /// [`scan_leaf_cluster`] or the trie recomputation itself, not expected cache-gap noise.
    pub(crate) leaf_slot_prefix_mismatches: u64,
}

/// Walks the complete account trie (and, per `config.scope`, per-account storage tries),
/// returning every maximal inactive subtree found plus run statistics. `state_root` is the chain
/// header's known state root, used as the account trie's root hash — the freshly recomputed root
/// node's own `root_hash` field is not reliably populated by `HashBuilder` in practice, so the
/// header is the authoritative source instead (see [`finalize_walk`]).
pub(crate) fn identify<H: HashedCursorFactory + Clone>(
    hashed_cursor_factory: H,
    state_root: B256,
    period_index: &PeriodIndex,
    config: &IdentifyConfig,
) -> eyre::Result<(Vec<InactiveSubtree>, IdentifyStats)> {
    let mut stats = IdentifyStats::default();
    let mut subtrees = Vec::new();
    walk(hashed_cursor_factory, state_root, period_index, config, &mut stats, &mut subtrees)?;
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

/// Pops and folds every child frame on `stack` that `(path, branch)` closes (i.e. whose path is a
/// proper descendant of `path`), returning the new, not-yet-pushed frame for `path`. Shared
/// between the account-trie and per-owner storage-trie handling in [`walk`] — this part of the
/// algorithm doesn't differ between the two.
fn pop_closed_children(
    stack: &mut Vec<NodeFrame>,
    path: Nibbles,
    branch: &BranchNodeCompact,
    ctx: &mut EmitCtx<'_>,
) -> NodeFrame {
    let mut frame = NodeFrame::new(path, branch.root_hash);
    while stack.last().is_some_and(|f| f.path.len() > path.len() && f.path.starts_with(&path)) {
        let child = stack.pop().expect("checked above");
        let nibble = child.path.get(path.len()).expect("child path longer than parent");
        let hash = branch.hash_mask.is_bit_set(nibble).then(|| branch.hash_for_nibble(nibble));
        fold_child(child, hash, &mut frame, ctx);
    }
    frame
}

/// Drives a single [`StateRootBranchNodesIter`], dispatching each incoming branch node to either
/// the account-trie stack or whichever storage-trie stack is currently open. Storage nodes for one
/// account always finish before the next account's storage nodes begin (the iterator's own
/// ordering guarantee), which is the trigger for finalizing one owner's storage stack and starting
/// the next — no recursion needed, unlike the old per-account `walk_storage_trie` call: storage
/// descent now happens automatically, since `StateRoot::root_with_progress` always computes every
/// account's storage root regardless of `config.scope` (it's needed for the account leaf's own RLP
/// value). Under `Scope::Account`, incoming storage nodes are simply skipped rather than folded.
fn walk<H: HashedCursorFactory + Clone>(
    hashed_cursor_factory: H,
    state_root: B256,
    period_index: &PeriodIndex,
    config: &IdentifyConfig,
    stats: &mut IdentifyStats,
    subtrees: &mut Vec<InactiveSubtree>,
) -> eyre::Result<()> {
    let emit_account = config.scope.emit_account();
    let descend_storage = config.scope.descend_storage();

    let branch_iter = StateRootBranchNodesIter::new(hashed_cursor_factory.clone());
    let mut account_hashed_cursor = hashed_cursor_factory.hashed_account_cursor()?;

    let mut account_stack: Vec<NodeFrame> = Vec::new();
    let mut storage_owner: Option<B256> = None;
    let mut storage_stack: Vec<NodeFrame> = Vec::new();
    let mut storage_hashed_cursor = None;

    for item in branch_iter {
        match item? {
            BranchNode::Account(path, branch) => {
                let mut frame = {
                    let mut ctx = EmitCtx {
                        trie_label: "account",
                        owner: B256::ZERO,
                        emit_output: emit_account,
                        subtrees,
                        stats,
                    };
                    pop_closed_children(&mut account_stack, path, &branch, &mut ctx)
                };

                for nibble in 0u8..16 {
                    if !branch.state_mask.is_bit_set(nibble) {
                        continue;
                    }
                    if branch.tree_mask.is_bit_set(nibble) {
                        // Arrives later as its own stream item; `pop_closed_children` handles it.
                        continue;
                    }

                    let mut leaf_prefix = path;
                    leaf_prefix.push(nibble);

                    let cluster = scan_leaf_cluster(
                        &leaf_prefix,
                        &mut account_hashed_cursor,
                        config,
                        |key| period_index.account_block(key),
                        &mut stats.accounts_scanned,
                        &mut stats.period_lookup_misses,
                    )?;
                    if cluster.leaf_count == 0 {
                        stats.leaf_slot_prefix_mismatches += 1;
                        continue;
                    }

                    let hash =
                        branch.hash_mask.is_bit_set(nibble).then(|| branch.hash_for_nibble(nibble));
                    let mut ctx = EmitCtx {
                        trie_label: "account",
                        owner: B256::ZERO,
                        emit_output: emit_account,
                        subtrees,
                        stats,
                    };
                    fold_child(cluster, hash, &mut frame, &mut ctx);
                }

                account_stack.push(frame);
            }
            BranchNode::Storage(owner, path, branch) => {
                if !descend_storage {
                    continue;
                }
                if storage_owner != Some(owner) {
                    if let Some(prev_owner) = storage_owner.take() {
                        let mut ctx = EmitCtx {
                            trie_label: "storage",
                            owner: prev_owner,
                            emit_output: true,
                            subtrees,
                            stats,
                        };
                        finalize_walk(std::mem::take(&mut storage_stack), None, &mut ctx);
                    }
                    storage_owner = Some(owner);
                    storage_hashed_cursor =
                        Some(hashed_cursor_factory.hashed_storage_cursor(owner)?);
                    stats.storage_tries_walked += 1;
                }

                let mut frame = {
                    let mut ctx = EmitCtx {
                        trie_label: "storage",
                        owner,
                        emit_output: true,
                        subtrees,
                        stats,
                    };
                    pop_closed_children(&mut storage_stack, path, &branch, &mut ctx)
                };

                let cursor = storage_hashed_cursor.as_mut().expect("just set above");
                for nibble in 0u8..16 {
                    if !branch.state_mask.is_bit_set(nibble) {
                        continue;
                    }
                    if branch.tree_mask.is_bit_set(nibble) {
                        continue;
                    }

                    let mut leaf_prefix = path;
                    leaf_prefix.push(nibble);

                    let cluster = scan_leaf_cluster(
                        &leaf_prefix,
                        cursor,
                        config,
                        |key| period_index.storage_block(owner, key),
                        &mut stats.storage_slots_scanned,
                        &mut stats.period_lookup_misses,
                    )?;
                    if cluster.leaf_count == 0 {
                        stats.leaf_slot_prefix_mismatches += 1;
                        continue;
                    }

                    let hash =
                        branch.hash_mask.is_bit_set(nibble).then(|| branch.hash_for_nibble(nibble));
                    let mut ctx = EmitCtx {
                        trie_label: "storage",
                        owner,
                        emit_output: true,
                        subtrees,
                        stats,
                    };
                    fold_child(cluster, hash, &mut frame, &mut ctx);
                }

                storage_stack.push(frame);
            }
        }
    }

    if let Some(prev_owner) = storage_owner {
        let mut ctx = EmitCtx {
            trie_label: "storage",
            owner: prev_owner,
            emit_output: true,
            subtrees,
            stats,
        };
        // No external oracle exists for a storage trie's own root hash (unlike the account trie,
        // where the chain header provides one) — `Account` carries no `storage_root` field, it's
        // computed only at hash-time. `finalize_walk` handles that gracefully: a `None` root hash
        // just means the whole-storage-trie candidate can't be formed, falling back to flushing
        // whatever inner candidates it collected instead.
        finalize_walk(storage_stack, None, &mut ctx);
    }

    let mut ctx = EmitCtx {
        trie_label: "account",
        owner: B256::ZERO,
        emit_output: emit_account,
        subtrees,
        stats,
    };
    finalize_walk(account_stack, Some(state_root), &mut ctx);

    Ok(())
}

/// Zero-pads `leaf_prefix` (nibbles) out to a full 32-byte key suitable for
/// [`HashedCursor::seek`]. `pub(crate)` (rather than private) so `convert_inactive::capture` can
/// reuse it for its own prefix scans instead of duplicating the same packing logic.
pub(crate) fn pack_seek_key(leaf_prefix: &Nibbles) -> B256 {
    let mut buf = [0u8; 32];
    let packed = leaf_prefix.pack();
    buf[..packed.len()].copy_from_slice(&packed);
    B256::from(buf)
}

/// Scans every hashed-cursor entry whose key starts with `leaf_prefix`, folding each in as an
/// individual leaf via `period_lookup`, and returns the resulting cluster as a standalone
/// [`NodeFrame`] (not yet folded into anything — the caller does that via [`fold_child`], passing
/// the cluster's hash from the parent branch node's `hash_for_nibble` when `hash_mask` has it,
/// or `None` otherwise — `fold_child` already handles a hashless fully-inactive child correctly
/// by bubbling its (empty, for a single leaf) candidate list up instead of trying to stand it up
/// as its own candidate).
///
/// This always scans rather than trusting a single seek, because neither `tree_mask` nor
/// `hash_mask` reliably distinguishes "exactly one leaf" from "a real multi-leaf branch" for a
/// `state_mask`-set, `tree_mask`-clear nibble: `hash_mask` is set when a branch node's hash is
/// pushed directly, but alloy-trie's `HashBuilder::update_masks` *clears* it again whenever an
/// extension node (a shared-nibble-prefix run) sits between this position and that branch — see
/// this module's top-level docs. So a real multi-leaf branch reached via an extension looks
/// identical, mask-wise, to a single leaf; only actually scanning the hashed entries tells them
/// apart.
fn scan_leaf_cluster<C: HashedCursor>(
    leaf_prefix: &Nibbles,
    cursor: &mut C,
    config: &IdentifyConfig,
    mut period_lookup: impl FnMut(B256) -> Option<BlockNumber>,
    scanned: &mut u64,
    period_lookup_misses: &mut u64,
) -> eyre::Result<NodeFrame> {
    let mut cluster = NodeFrame::new(*leaf_prefix, None);

    let mut entry = cursor.seek(pack_seek_key(leaf_prefix))?;
    while let Some((key, _value)) = entry {
        if !Nibbles::unpack(key).starts_with(leaf_prefix) {
            break;
        }

        cluster.leaf_count += 1;
        *scanned += 1;
        let inactive = match period_lookup(key) {
            Some(block) => config.is_inactive(block),
            None => {
                *period_lookup_misses += 1;
                false
            }
        };
        if !inactive {
            cluster.all_inactive = false;
        }

        entry = cursor.next()?;
    }

    Ok(cluster)
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
/// Because the walk now recomputes the complete trie via [`StateRootBranchNodesIter`] rather than
/// reading persisted rows, `stack` should always hold exactly one frame whose path is empty — the
/// true root, guaranteed to be the last item `HashBuilder` folds. The general handling below is
/// kept as a defensive fallback rather than assumed away entirely (e.g. it also protects against a
/// bug in this module's own owner-boundary tracking leaving stray frames behind), not because
/// reth's trie tables are known to omit rows the way the old persisted-cache-based walk had to
/// account for.
///
/// - Zero frames: an empty trie, nothing to report.
/// - One frame whose path is already empty (the expected case): finalize it directly, preferring
///   its own `root_hash` but falling back to `known_root_hash` if unset — in practice `root_hash`
///   is essentially never populated by `HashBuilder`, so `known_root_hash` (the chain header's
///   state root for the account trie; `None` for storage tries, which have no such external oracle
///   — `Account` carries no `storage_root` field, it's only computed at hash-time) is the one that
///   actually matters.
/// - Otherwise (should not normally happen): synthesize a root at the empty path from
///   `known_root_hash`. Each remaining top-level frame folds in "embedded"-style (its own hash
///   isn't known — there's no parent row to read a `hash_mask` from), so its inner candidates
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
    fn test_scan_leaf_cluster_no_match_returns_empty() {
        use reth_trie::hashed_cursor::mock::MockHashedCursorFactory;
        use std::collections::BTreeMap;

        // No accounts at all under prefix `0xf`.
        let mut accounts = BTreeMap::new();
        accounts.insert(B256::repeat_byte(0x00), reth_primitives_traits::Account::default());
        let factory = MockHashedCursorFactory::new(accounts, Default::default());
        let mut cursor = factory.hashed_account_cursor().unwrap();

        let prefix = Nibbles::from_nibbles([0xf]);
        let cfg = config(0, 0);
        let mut scanned = 0u64;
        let mut misses = 0u64;
        let cluster =
            scan_leaf_cluster(&prefix, &mut cursor, &cfg, |_| None, &mut scanned, &mut misses)
                .unwrap();

        assert_eq!(cluster.leaf_count, 0);
        assert_eq!(scanned, 0);
    }

    #[test]
    fn test_scan_leaf_cluster_finds_every_leaf_under_a_shared_prefix() {
        // The actual regression this module's bug fix is about: a `tree_mask`-clear nibble can
        // hide more than one leaf (a shallow branch, or one reached via an extension node), and
        // trusting a single seek silently drops every leaf after the first. `scan_leaf_cluster`
        // must find all of them.
        use reth_trie::hashed_cursor::mock::MockHashedCursorFactory;
        use std::collections::BTreeMap;

        let mut accounts = BTreeMap::new();
        let mut key_a = [0u8; 32];
        key_a[0] = 0x0a; // nibble path: 0, a, ...
        let mut key_b = [0u8; 32];
        key_b[0] = 0x0b; // nibble path: 0, b, ... — shares the `0` prefix with key_a
        let mut key_c = [0u8; 32];
        key_c[0] = 0x10; // nibble path: 1, ... — outside the `0` prefix
        accounts.insert(B256::from(key_a), reth_primitives_traits::Account::default());
        accounts.insert(B256::from(key_b), reth_primitives_traits::Account::default());
        accounts.insert(B256::from(key_c), reth_primitives_traits::Account::default());

        let factory = MockHashedCursorFactory::new(accounts, Default::default());
        let mut cursor = factory.hashed_account_cursor().unwrap();

        let prefix = Nibbles::from_nibbles([0x0]);
        let cfg = config(0, 0);
        let mut scanned = 0u64;
        let mut misses = 0u64;
        let cluster =
            scan_leaf_cluster(&prefix, &mut cursor, &cfg, |_| None, &mut scanned, &mut misses)
                .unwrap();

        assert_eq!(cluster.leaf_count, 2, "both key_a and key_b share the `0` prefix");
        assert_eq!(scanned, 2);
        assert_eq!(misses, 2, "period_lookup returns None for both in this test");
    }
}
