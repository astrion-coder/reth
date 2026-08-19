//! The append-only cold-storage file format: blobs of serialized trie nodes, one per converted
//! subtree, and the [`ColdFileWriter`] that appends and reads them back.
//!
//! Each blob is a 16-byte [`BlobHeader`] followed by a body of [`ColdNode`]s in post-order
//! (children before parents, root last), mirroring the design doc's "Cold State Format" section.
//! Every entry is self-describing (an explicit kind tag plus explicit length fields for
//! variable-length parts), so the body can be re-parsed sequentially with no separate directory —
//! [`recompute_root_hash`] relies on exactly that to verify a write round-trips correctly.
//!
//! Two ambiguities in the design doc are resolved here by convention (documented since the
//! upstream EIPs leave them unspecified): the header's 16 bytes split as `version(u16) +
//! reserved(6 bytes) + root_offset(u32) + root_size(u32)`, matching the stub's own `u32`
//! offset/size widths; and both `root_offset` fields (header and [`ColdStub::root_in_blob`]) are
//! relative to the start of the blob's *body*, not the blob as a whole (i.e. they don't include
//! the 16-byte header).

use alloy_primitives::B256;
use reth_trie::{BranchNode, ColdStub, ExtensionNode, LeafNode, Nibbles, RlpNode, TrieNode};
use std::{
    collections::HashMap,
    fs::{File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::Path,
};

/// Appends blobs to a single cold-storage file and reads them back for verification.
pub(crate) struct ColdFileWriter {
    file: File,
    next_offset: u64,
}

impl ColdFileWriter {
    /// Opens (creating if absent) in append mode. `valid_length()` starts at the file's current
    /// size — callers resuming after a crash should [`Self::truncate_to`] a known-good
    /// checkpoint length first (see `convert_inactive.rs`'s crash-safety handling).
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).read(true).open(path)?;
        let next_offset = file.metadata()?.len();
        Ok(Self { file, next_offset })
    }

    /// Current file length — every byte before this is either a complete, referenced blob or
    /// dead space safe to truncate away.
    pub(crate) const fn valid_length(&self) -> u64 {
        self.next_offset
    }

    /// Drops any trailing bytes past `len` — used to clean up a blob that was fsynced but whose
    /// hot-DB commit never happened (a crash between those two steps).
    pub(crate) fn truncate_to(&mut self, len: u64) -> io::Result<()> {
        self.file.set_len(len)?;
        self.next_offset = len;
        Ok(())
    }

    /// Serializes `nodes` (already in post-order — children before parents; the last entry is
    /// the subtree root) into one self-contained blob, appends and fsyncs it, and returns the
    /// stub pointing at it. The `fsync` here, not the caller's later hot-DB commit, is the point
    /// past which this blob must be treated as durable-but-possibly-orphaned — see
    /// `convert_inactive.rs`'s crash-safety notes.
    pub(crate) fn write_blob(&mut self, nodes: &[(Nibbles, TrieNode)]) -> eyre::Result<ColdStub> {
        let root_path = &nodes.last().ok_or_else(|| eyre::eyre!("cannot write an empty blob"))?.0;

        let mut body = Vec::new();
        let mut placed: HashMap<Nibbles, (u32, u32)> = HashMap::with_capacity(nodes.len());
        for (path, node) in nodes {
            let cold_node = to_cold_node(path, node, &placed)?;
            let start = body.len() as u32;
            cold_node.encode(&mut body);
            placed.insert(*path, (start, body.len() as u32 - start));
        }
        let (root_offset, root_size) =
            *placed.get(root_path).expect("root path was just inserted above");

        let mut blob = Vec::with_capacity(BlobHeader::ENCODED_LEN + body.len());
        BlobHeader { root_offset, root_size }.encode(&mut blob);
        blob.extend_from_slice(&body);

        let blob_offset = self.next_offset;
        self.file.write_all(&blob)?;
        self.file.sync_data()?;
        self.next_offset += blob.len() as u64;

        Ok(ColdStub { blob_offset, root_in_blob: root_offset, root_size })
    }

    /// Reads a previously written blob back and recomputes its root hash from scratch. Used both
    /// as a post-write self-check in `convert_inactive.rs` (never trust a write without reading
    /// it back) and directly by tests.
    pub(crate) fn read_and_verify(&mut self, stub: &ColdStub) -> eyre::Result<B256> {
        self.file.seek(SeekFrom::Start(stub.blob_offset))?;

        let mut header_buf = vec![0u8; BlobHeader::ENCODED_LEN];
        self.file.read_exact(&mut header_buf)?;
        let header = BlobHeader::decode(&header_buf)?;

        let body_len = header.root_offset as usize + header.root_size as usize;
        let mut body = vec![0u8; body_len];
        self.file.read_exact(&mut body)?;

        recompute_root_hash(&body, &header)
    }
}

/// The 16-byte blob header: `version(u16) + reserved(6 bytes) + root_offset(u32) + root_size(u32)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlobHeader {
    root_offset: u32,
    root_size: u32,
}

impl BlobHeader {
    const ENCODED_LEN: usize = 16;
    const VERSION: u16 = 1;

    fn encode(&self, buf: &mut Vec<u8>) {
        buf.extend_from_slice(&Self::VERSION.to_be_bytes());
        buf.extend_from_slice(&[0u8; 6]);
        buf.extend_from_slice(&self.root_offset.to_be_bytes());
        buf.extend_from_slice(&self.root_size.to_be_bytes());
    }

    fn decode(buf: &[u8]) -> eyre::Result<Self> {
        eyre::ensure!(buf.len() >= Self::ENCODED_LEN, "blob header truncated");
        let version = u16::from_be_bytes(buf[0..2].try_into().expect("checked width"));
        eyre::ensure!(version == Self::VERSION, "unsupported cold-file blob version {version}");
        let root_offset = u32::from_be_bytes(buf[8..12].try_into().expect("checked width"));
        let root_size = u32::from_be_bytes(buf[12..16].try_into().expect("checked width"));
        Ok(Self { root_offset, root_size })
    }
}

/// One serialized MPT node inside a blob body. `Full` mirrors a branch node's 17 slots (16
/// nibbles + a value slot that's always [`ChildSlot::Empty`] in practice — account/storage tries
/// use fixed 32-byte keys, so values only ever terminate at a leaf, never a branch — kept only
/// for structural fidelity to the spec). `Short` covers both extension nodes (child points at
/// another node) and leaves (child is the inline value) per go-ethereum's `shortNode` convention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ColdNode {
    Full(Vec<ChildSlot>),
    Short { key: Nibbles, child: ChildSlot },
}

impl ColdNode {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Full(slots) => {
                buf.push(0);
                for slot in slots {
                    slot.encode(buf);
                }
            }
            Self::Short { key, child } => {
                buf.push(1);
                let key_bytes: Vec<u8> = key.iter().collect();
                buf.push(key_bytes.len() as u8);
                buf.extend_from_slice(&key_bytes);
                child.encode(buf);
            }
        }
    }

    /// Decodes one node starting at `buf[0]`; returns it and the number of bytes consumed.
    fn decode(buf: &[u8]) -> eyre::Result<(Self, usize)> {
        eyre::ensure!(!buf.is_empty(), "cold node buffer empty");
        match buf[0] {
            0 => {
                let mut offset = 1;
                let mut slots = Vec::with_capacity(17);
                for _ in 0..17 {
                    let (slot, consumed) = ChildSlot::decode(&buf[offset..])?;
                    slots.push(slot);
                    offset += consumed;
                }
                Ok((Self::Full(slots), offset))
            }
            1 => {
                eyre::ensure!(buf.len() >= 2, "truncated short node key length");
                let key_len = buf[1] as usize;
                eyre::ensure!(buf.len() >= 2 + key_len, "truncated short node key");
                let key = Nibbles::from_nibbles_unchecked(&buf[2..2 + key_len]);
                let (child, consumed) = ChildSlot::decode(&buf[2 + key_len..])?;
                Ok((Self::Short { key, child }, 2 + key_len + consumed))
            }
            other => Err(eyre::eyre!("unknown cold node kind {other}")),
        }
    }
}

/// A branch node's per-nibble child reference (or an extension/leaf's single child), tagged with
/// how to resolve it. Mirrors the design doc's four child-slot kinds exactly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ChildSlot {
    Empty,
    /// The child's own Keccak hash plus its location in this blob.
    Hashed {
        hash: B256,
        offset: u32,
        size: u32,
    },
    /// No independent hash (the child's RLP encoding is under 32 bytes) — only its location.
    Embedded {
        offset: u32,
        size: u32,
    },
    /// A leaf's value, stored directly (the serialized RLP account/storage value).
    InlineValue(Vec<u8>),
}

impl ChildSlot {
    fn encode(&self, buf: &mut Vec<u8>) {
        match self {
            Self::Empty => buf.push(0),
            Self::Hashed { hash, offset, size } => {
                buf.push(1);
                buf.extend_from_slice(hash.as_slice());
                buf.extend_from_slice(&offset.to_be_bytes());
                buf.extend_from_slice(&size.to_be_bytes());
            }
            Self::Embedded { offset, size } => {
                buf.push(2);
                buf.extend_from_slice(&offset.to_be_bytes());
                buf.extend_from_slice(&size.to_be_bytes());
            }
            Self::InlineValue(bytes) => {
                buf.push(3);
                buf.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
                buf.extend_from_slice(bytes);
            }
        }
    }

    fn decode(buf: &[u8]) -> eyre::Result<(Self, usize)> {
        eyre::ensure!(!buf.is_empty(), "child slot buffer empty");
        match buf[0] {
            0 => Ok((Self::Empty, 1)),
            1 => {
                eyre::ensure!(buf.len() >= 41, "truncated hashed child slot");
                let hash = B256::from_slice(&buf[1..33]);
                let offset = u32::from_be_bytes(buf[33..37].try_into().expect("checked width"));
                let size = u32::from_be_bytes(buf[37..41].try_into().expect("checked width"));
                Ok((Self::Hashed { hash, offset, size }, 41))
            }
            2 => {
                eyre::ensure!(buf.len() >= 9, "truncated embedded child slot");
                let offset = u32::from_be_bytes(buf[1..5].try_into().expect("checked width"));
                let size = u32::from_be_bytes(buf[5..9].try_into().expect("checked width"));
                Ok((Self::Embedded { offset, size }, 9))
            }
            3 => {
                eyre::ensure!(buf.len() >= 5, "truncated inline-value child slot");
                let len = u32::from_be_bytes(buf[1..5].try_into().expect("checked width")) as usize;
                eyre::ensure!(buf.len() >= 5 + len, "truncated inline-value payload");
                Ok((Self::InlineValue(buf[5..5 + len].to_vec()), 5 + len))
            }
            other => Err(eyre::eyre!("unknown child slot kind {other}")),
        }
    }
}

/// Converts one decoded `TrieNode` at `path` into its [`ColdNode`] form. `placed` must already
/// contain every child this node references — true as long as `nodes` is supplied in post-order,
/// which is what [`ColdFileWriter::write_blob`] requires of its caller.
fn to_cold_node(
    path: &Nibbles,
    node: &TrieNode,
    placed: &HashMap<Nibbles, (u32, u32)>,
) -> eyre::Result<ColdNode> {
    match node {
        TrieNode::Branch(branch) => {
            let mut slots = Vec::with_capacity(17);
            let mut stack_iter = branch.stack.iter();
            for nibble in 0u8..16 {
                if branch.state_mask.is_bit_set(nibble) {
                    let rlp = stack_iter
                        .next()
                        .ok_or_else(|| eyre::eyre!("branch stack shorter than state_mask"))?;
                    let mut child_path = *path;
                    child_path.push(nibble);
                    slots.push(child_slot_for(rlp, &child_path, placed)?);
                } else {
                    slots.push(ChildSlot::Empty);
                }
            }
            slots.push(ChildSlot::Empty); // value slot: always empty, see module docs
            Ok(ColdNode::Full(slots))
        }
        TrieNode::Extension(ext) => {
            let mut child_path = *path;
            child_path.extend(&ext.key);
            let child = child_slot_for(&ext.child, &child_path, placed)?;
            Ok(ColdNode::Short { key: ext.key, child })
        }
        TrieNode::Leaf(leaf) => {
            Ok(ColdNode::Short { key: leaf.key, child: ChildSlot::InlineValue(leaf.value.clone()) })
        }
        TrieNode::EmptyRoot => {
            Err(eyre::eyre!("unexpected empty root node inside a non-empty inactive subtree"))
        }
    }
}

fn child_slot_for(
    rlp: &RlpNode,
    child_path: &Nibbles,
    placed: &HashMap<Nibbles, (u32, u32)>,
) -> eyre::Result<ChildSlot> {
    let (offset, size) = *placed.get(child_path).ok_or_else(|| {
        eyre::eyre!("child at {child_path:?} not yet placed — nodes must be supplied in post-order")
    })?;
    Ok(match rlp.as_hash() {
        Some(hash) => ChildSlot::Hashed { hash, offset, size },
        None => ChildSlot::Embedded { offset, size },
    })
}

/// Sequentially decodes every [`ColdNode`] in a blob's body — self-describing framing means no
/// separate directory is needed — replaying them in the same post-order they were written to
/// resolve each node's [`RlpNode`] bottom-up (an [`ChildSlot::Embedded`] child is looked up in
/// `resolved` by its offset, always already present since it was written, and therefore decoded,
/// before its parent), and returns the resolved root's hash.
fn recompute_root_hash(body: &[u8], header: &BlobHeader) -> eyre::Result<B256> {
    let limit = header.root_offset as usize + header.root_size as usize;
    eyre::ensure!(limit <= body.len(), "blob body shorter than header's recorded root range");

    let mut resolved: HashMap<u32, RlpNode> = HashMap::new();
    let mut offset = 0usize;
    let mut last: Option<(usize, RlpNode)> = None;
    while offset < limit {
        let (node, consumed) = ColdNode::decode(&body[offset..])?;
        let rlp = cold_node_to_rlp(&node, &resolved)?;
        resolved.insert(offset as u32, rlp.clone());
        last = Some((offset, rlp));
        offset += consumed;
    }

    let (last_offset, last_rlp) = last.ok_or_else(|| eyre::eyre!("blob body is empty"))?;
    eyre::ensure!(
        offset == limit && last_offset == header.root_offset as usize,
        "blob body entries did not align exactly with the header's recorded root range"
    );

    last_rlp
        .as_hash()
        .ok_or_else(|| eyre::eyre!("blob root did not resolve to an independent hash"))
}

fn cold_node_to_rlp(node: &ColdNode, resolved: &HashMap<u32, RlpNode>) -> eyre::Result<RlpNode> {
    let resolve = |slot: &ChildSlot| -> eyre::Result<RlpNode> {
        match slot {
            ChildSlot::Empty | ChildSlot::InlineValue(_) => {
                Err(eyre::eyre!("child slot has no independent RLP node"))
            }
            ChildSlot::Hashed { hash, .. } => Ok(RlpNode::word_rlp(hash)),
            ChildSlot::Embedded { offset, .. } => resolved
                .get(offset)
                .cloned()
                .ok_or_else(|| eyre::eyre!("embedded child at offset {offset} not yet resolved")),
        }
    };

    let trie_node = match node {
        ColdNode::Full(slots) => {
            let mut branch = BranchNode::default();
            for (nibble, slot) in slots.iter().take(16).enumerate() {
                if !matches!(slot, ChildSlot::Empty) {
                    branch.state_mask.set_bit(nibble as u8);
                    branch.stack.push(resolve(slot)?);
                }
            }
            TrieNode::Branch(branch)
        }
        ColdNode::Short { key, child: ChildSlot::InlineValue(value) } => {
            TrieNode::Leaf(LeafNode::new(*key, value.clone()))
        }
        ColdNode::Short { key, child } => {
            TrieNode::Extension(ExtensionNode::new(*key, resolve(child)?))
        }
    };

    let mut scratch = Vec::new();
    Ok(trie_node.rlp(&mut scratch))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    /// Builds a tiny two-leaf subtree: a branch at the empty path with two leaf children under
    /// nibbles `0x1` and `0x2`, already in post-order (leaves first, branch last).
    fn two_leaf_subtree() -> Vec<(Nibbles, TrieNode)> {
        // Values sized so each leaf stays small enough to embed in the branch (exercising the
        // `ChildSlot::Embedded` path) while the branch's own encoding still exceeds 32 bytes and
        // therefore hashes (exercising the "genuine subtree root" path every real InactiveSubtree
        // takes — see `capture::verify_root_hash`'s doc comment).
        let leaf_a = TrieNode::Leaf(LeafNode::new(Nibbles::from_nibbles([0xa]), vec![0xaa; 10]));
        let leaf_b = TrieNode::Leaf(LeafNode::new(Nibbles::from_nibbles([0xb]), vec![0xbb; 10]));

        // Each `.rlp()` call appends to its buffer rather than clearing it first, so each needs
        // its own fresh scratch buffer — reusing one across calls would hash the leftover bytes
        // from the previous call too.
        let rlp_a = leaf_a.rlp(&mut Vec::new());
        let rlp_b = leaf_b.rlp(&mut Vec::new());

        let mut branch = BranchNode::default();
        branch.state_mask.set_bit(0x1);
        branch.state_mask.set_bit(0x2);
        branch.stack.push(rlp_a);
        branch.stack.push(rlp_b);

        vec![
            (Nibbles::from_nibbles([0x1]), leaf_a),
            (Nibbles::from_nibbles([0x2]), leaf_b),
            (Nibbles::default(), TrieNode::Branch(branch)),
        ]
    }

    #[test]
    fn test_write_blob_and_read_and_verify_round_trips() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cold.bin");
        let nodes = two_leaf_subtree();

        let mut scratch = Vec::new();
        let expected_hash = nodes[2]
            .1
            .rlp(&mut scratch)
            .as_hash()
            .expect("branch with two hashed-size leaves must hash, not embed");

        let mut writer = ColdFileWriter::open(&path).unwrap();
        let stub = writer.write_blob(&nodes).unwrap();
        assert_eq!(stub.blob_offset, 0);
        assert!(stub.root_in_blob > 0, "leaves are written before the branch root");

        let recomputed = writer.read_and_verify(&stub).unwrap();
        assert_eq!(recomputed, expected_hash);
    }

    #[test]
    fn test_write_blob_rejects_empty_input() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cold.bin");
        let mut writer = ColdFileWriter::open(&path).unwrap();
        assert!(writer.write_blob(&[]).is_err());
    }

    #[test]
    fn test_second_blob_appends_after_first() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cold.bin");
        let mut writer = ColdFileWriter::open(&path).unwrap();

        let stub_a = writer.write_blob(&two_leaf_subtree()).unwrap();
        let stub_b = writer.write_blob(&two_leaf_subtree()).unwrap();

        assert_eq!(stub_a.blob_offset, 0);
        assert!(stub_b.blob_offset > stub_a.blob_offset);
        assert_eq!(
            writer.valid_length(),
            stub_b.blob_offset + 16 + u64::from(stub_b.root_in_blob + stub_b.root_size)
        );
    }

    #[test]
    fn test_truncate_to_drops_orphaned_tail() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("cold.bin");
        let mut writer = ColdFileWriter::open(&path).unwrap();

        writer.write_blob(&two_leaf_subtree()).unwrap();
        let checkpoint = writer.valid_length();
        writer.write_blob(&two_leaf_subtree()).unwrap();
        assert!(writer.valid_length() > checkpoint);

        writer.truncate_to(checkpoint).unwrap();
        assert_eq!(writer.valid_length(), checkpoint);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), checkpoint);
    }
}
