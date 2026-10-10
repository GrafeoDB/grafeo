//! The `OverlayDeletions` section of a 0.5.x database file, read for
//! migration: the base nodes and edges deleted after `compact()`, which the
//! fold leaves out (see [`fold`](super::fold)).
//!
//! 0.5.x wrote it as one raw chunk: the magic `GODL`, version 1, three
//! reserved bytes, the node ids and then the edge ids (each a `u64` count
//! and the ids, `u64` LE), and a CRC32 of all of it. Nothing writes it any
//! more. Removed in 0.7.0 with the other 0.5.x readers.

use grafeo_common::storage::section::SectionType;
use grafeo_common::storage::{SectionSource, legacy_bytes};
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::error::{Error, Result};

use super::section::in_section;

/// Magic bytes identifying an OverlayDeletions section ("Grafeo Overlay
/// Deletion Log").
const MAGIC: [u8; 4] = *b"GODL";

/// The layout 0.5.x wrote.
const FORMAT_VERSION: u8 = 1;

/// The base nodes and edges a 0.5.x `OverlayDeletions` section lists.
///
/// # Errors
///
/// Returns [`Error::Corruption`] naming the section when its bytes do not
/// decode, and [`Error::Serialization`] when the section is not one raw
/// chunk (only a 0.6.0 development build wrote it otherwise) or of another
/// layout version.
pub(super) fn read_deletions(source: &dyn SectionSource) -> Result<(Vec<NodeId>, Vec<EdgeId>)> {
    let Some(bytes) =
        legacy_bytes(source).map_err(|e| in_section(SectionType::OverlayDeletions, e))?
    else {
        return Err(Error::Serialization(
            "section OverlayDeletions: not the 0.5.x layout (a 0.6.0 development build wrote \
             it); recreate the database with this build"
                .to_string(),
        ));
    };
    decode(&bytes)
}

fn decode(data: &[u8]) -> Result<(Vec<NodeId>, Vec<EdgeId>)> {
    if data.len() < 8 + 8 + 8 + 4 {
        return Err(Error::corruption("OverlayDeletions section too short"));
    }
    if data[..4] != MAGIC {
        return Err(Error::corruption(format!(
            "OverlayDeletions magic mismatch: expected {MAGIC:?}, got {:?}",
            &data[..4],
        )));
    }
    let version = data[4];
    if version != FORMAT_VERSION {
        return Err(Error::Serialization(format!(
            "unsupported OverlayDeletions section version {version}, expected {FORMAT_VERSION}",
        )));
    }

    // CRC verifies the entire prefix up to the trailing 4 bytes.
    let payload = &data[..data.len() - 4];
    let stored_crc = u32::from_le_bytes(data[data.len() - 4..].try_into().unwrap());
    let actual_crc = crc32fast::hash(payload);
    if stored_crc != actual_crc {
        return Err(Error::corruption(format!(
            "OverlayDeletions CRC mismatch: stored {stored_crc:#010X}, computed {actual_crc:#010X}",
        )));
    }

    let mut pos = 8usize;
    let read_u64 = |buf: &[u8], pos: &mut usize| -> Result<u64> {
        if *pos + 8 > buf.len() {
            return Err(Error::corruption("OverlayDeletions truncated mid-entry"));
        }
        let v = u64::from_le_bytes(buf[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(v)
    };

    let node_count_u64 = read_u64(data, &mut pos)?;
    let node_count = usize::try_from(node_count_u64).map_err(|_| {
        Error::Serialization(format!(
            "OverlayDeletions node_count {node_count_u64} exceeds usize on this target",
        ))
    })?;
    // Sanity bound: each node id is 8 bytes, plus 8 bytes for edge_count
    // and 4 trailing CRC bytes. Reject obvious garbage early so we don't
    // pre-allocate huge vecs from a corrupt header.
    if node_count
        .checked_mul(8)
        .map_or(true, |n| pos + n + 8 + 4 > data.len())
    {
        return Err(Error::corruption(format!(
            "OverlayDeletions node_count {node_count} exceeds section size",
        )));
    }
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(NodeId(read_u64(data, &mut pos)?));
    }

    let edge_count_u64 = read_u64(data, &mut pos)?;
    let edge_count = usize::try_from(edge_count_u64).map_err(|_| {
        Error::Serialization(format!(
            "OverlayDeletions edge_count {edge_count_u64} exceeds usize on this target",
        ))
    })?;
    if edge_count
        .checked_mul(8)
        .map_or(true, |n| pos + n + 4 > data.len())
    {
        return Err(Error::corruption(format!(
            "OverlayDeletions edge_count {edge_count} exceeds section size",
        )));
    }
    let mut edges = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        edges.push(EdgeId(read_u64(data, &mut pos)?));
    }

    Ok((nodes, edges))
}

#[cfg(test)]
mod tests {
    use grafeo_common::storage::{ChunkMeta, ImageSource, MemoryImage, SectionSink};

    use super::*;

    /// The `OverlayDeletions` section of the database `compact()` wrote with
    /// 0.5.44: Vincent, detach-deleted after `compact()`, who had no edges.
    const DELETIONS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.44/overlay_deletions.bin"
    ));

    fn read(bytes: &[u8]) -> Result<(Vec<NodeId>, Vec<EdgeId>)> {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::OverlayDeletions, 1)
            .unwrap();
        image.write_chunk(ChunkMeta::raw(), bytes).unwrap();
        read_deletions(&*image.section_source(SectionType::OverlayDeletions).unwrap())
    }

    /// The log of `nodes` and `edges` as 0.5.x wrote it.
    fn encode(nodes: &[u64], edges: &[u64]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&[FORMAT_VERSION, 0, 0, 0]);
        for ids in [nodes, edges] {
            bytes.extend_from_slice(&(ids.len() as u64).to_le_bytes());
            for id in ids {
                bytes.extend_from_slice(&id.to_le_bytes());
            }
        }
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes
    }

    #[test]
    fn the_0_5_44_log_lists_the_deleted_node() {
        let (nodes, edges) = read(DELETIONS).unwrap();
        assert_eq!(nodes.len(), 1, "Vincent");
        assert!(edges.is_empty(), "Vincent had no edges");
        assert_eq!(
            encode(&[nodes[0].as_u64()], &[]),
            DELETIONS,
            "the layout this reader expects is the one 0.5.44 wrote"
        );
    }

    #[test]
    fn a_log_of_nodes_and_edges_reads_back() {
        let (nodes, edges) = read(&encode(&[3, 19, 88], &[7])).unwrap();
        assert_eq!(nodes, [NodeId::new(3), NodeId::new(19), NodeId::new(88)]);
        assert_eq!(edges, [EdgeId::new(7)]);
    }

    #[test]
    fn a_damaged_log_is_refused() {
        let mut flipped = DELETIONS.to_vec();
        flipped[16] ^= 0xFF;
        let error = read(&flipped).unwrap_err().to_string();
        assert!(error.contains("CRC mismatch"), "{error}");

        let error = read(&DELETIONS[..20]).unwrap_err().to_string();
        assert!(error.contains("too short"), "{error}");

        let mut version = encode(&[1], &[]);
        version[4] = 2;
        let error = read(&version).unwrap_err().to_string();
        assert!(error.contains("version 2"), "{error}");
    }

    /// A count larger than the bytes can hold is refused before the ids are
    /// allocated, also when the CRC32 matches.
    #[test]
    fn a_count_no_log_could_hold_is_refused() {
        let mut bytes = encode(&[1], &[]);
        bytes[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        let len = bytes.len();
        let crc = crc32fast::hash(&bytes[..len - 4]);
        bytes[len - 4..].copy_from_slice(&crc.to_le_bytes());
        let error = read(&bytes).unwrap_err().to_string();
        assert!(error.contains("node_count"), "{error}");
    }

    #[test]
    fn a_chunked_log_is_refused() {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::OverlayDeletions, 2)
            .unwrap();
        image.write_chunk(ChunkMeta::meta(), &[1, 0]).unwrap();
        let error = read_deletions(&*image.section_source(SectionType::OverlayDeletions).unwrap())
            .unwrap_err()
            .to_string();
        assert!(error.contains("development build"), "{error}");
    }
}
