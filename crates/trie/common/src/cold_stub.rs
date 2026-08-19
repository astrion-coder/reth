/// Value type for the `AccountTrieStubs`/`StorageTrieStubs` tables (EIP-8188/8295 prototype).
///
/// Points at the cold-file blob a converted subtree's nodes were serialized into by
/// `reth db convert-inactive`. Unlike the design doc's 17-byte "primary stub" (which embeds a
/// marker byte directly into `AccountsTrie`/`StoragesTrie`'s own value encoding), this fork
/// stores stubs in a separate sibling table instead of overloading `BranchNodeCompact`'s
/// encoding — the row's existence in the stub table *is* the marker, so only the remaining
/// 16 bytes (blob offset + root-in-blob offset + root size) need to be stored here. See
/// `crates/cli/commands/src/db/convert_inactive.rs` for the writer/reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(any(test, feature = "serde"), derive(serde::Serialize, serde::Deserialize))]
pub struct ColdStub {
    /// Absolute byte offset of the blob's start in the cold file.
    pub blob_offset: u64,
    /// Offset of the subtree's root node, relative to the start of the blob's body.
    pub root_in_blob: u32,
    /// Byte size of the root node's serialized encoding.
    pub root_size: u32,
}

#[cfg(any(test, feature = "reth-codec"))]
impl reth_codecs::Compact for ColdStub {
    fn to_compact<B>(&self, buf: &mut B) -> usize
    where
        B: bytes::BufMut + AsMut<[u8]>,
    {
        buf.put_u64(self.blob_offset);
        buf.put_u32(self.root_in_blob);
        buf.put_u32(self.root_size);
        16
    }

    fn from_compact(buf: &[u8], _len: usize) -> (Self, &[u8]) {
        let blob_offset = u64::from_be_bytes(buf[0..8].try_into().expect("checked width"));
        let root_in_blob = u32::from_be_bytes(buf[8..12].try_into().expect("checked width"));
        let root_size = u32::from_be_bytes(buf[12..16].try_into().expect("checked width"));
        (Self { blob_offset, root_in_blob, root_size }, &buf[16..])
    }
}

#[cfg(any(test, feature = "reth-codec"))]
reth_codecs::impl_compression_for_compact!(ColdStub);

#[cfg(test)]
mod tests {
    use super::*;
    use reth_codecs::Compact;

    #[test]
    fn test_cold_stub_compact_round_trip() {
        let stub = ColdStub {
            blob_offset: 0x1122_3344_5566_7788,
            root_in_blob: 0xaabbccdd,
            root_size: 0x1234,
        };
        let mut buf = Vec::new();
        let written = stub.to_compact(&mut buf);
        assert_eq!(written, 16);
        assert_eq!(buf.len(), 16);
        let (decoded, rest) = ColdStub::from_compact(&buf, 16);
        assert_eq!(decoded, stub);
        assert!(rest.is_empty());
    }
}
