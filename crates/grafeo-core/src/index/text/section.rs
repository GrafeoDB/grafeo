//! Text Index section serializer for the `.grafeo` container format.
//!
//! Persists the BM25 inverted indexes (posting lists, document lengths) of
//! all text indexes, so an open does not rebuild them from the LPG
//! properties.
//!
//! ## Section version 2: streams (written by checkpoints)
//!
//! [`Section::write_to`] writes a metadata chunk, then one byte stream per
//! index ([`ChunkKind::Stream`] pieces of graph 0, cut at the byte cap). Each
//! stream is written as [`InvertedIndex::visit`] hands the index over:
//!
//! - The metadata chunk is the bincode of `TextMeta`: the layout byte (1),
//!   the byte cap the pieces were cut with, and the index keys
//!   ("label:property") in strictly increasing order.
//! - Stream `i` holds the index `keys[i]`:
//!   `[k1 f64][b f64][total_length u64][doc_count u64][term_count u64]`,
//!   then `doc_count` records `[node u64][length u32]`, then `term_count`
//!   records `[term_len u32][term][count u64]` each followed by `count`
//!   postings `[node u64][term_frequency u32]`. All little-endian.
//!
//! The layout is canonical: the same postings give the same bytes, whatever
//! order their documents were inserted in, and a restored index writes the
//! bytes it was read from. A reader checks it as it goes and refuses a
//! stream that breaks any of these:
//!
//! - k1 and b are finite;
//! - node ids strictly increase among the document lengths and within each
//!   posting list, and terms (UTF-8) strictly increase;
//! - every length, posting count and term frequency is at least 1, and
//!   every posting's node has a document length (read before the lists);
//! - the document lengths and the term frequencies each add up to
//!   `total_length`;
//! - the stream ends right after its last posting list.
//!
//! [`Section::read_from`] restores each index it was given from the stream
//! with its key, one posting list at a time, and skips the streams of indexes
//! it was not given. A refused stream names the index, which is then left
//! empty with the configuration it had: the engine builds it from the data.
//!
//! ## Memory and locks while writing
//!
//! A checkpoint holds one piece and a few KiB of gathered bytes per stream,
//! never a whole posting list's or index's bytes. The text section's
//! exception to bounded memory: to write in a defined order it gathers
//! references to the index's terms (16 bytes per term) and a copy of its
//! document lengths (16 bytes per document), plus a sorted copy of one
//! posting list at a time when the index does not hold that list in node
//! order (16 bytes per posting).
//!
//! The checkpoint holds the index's read lock while it writes the index's
//! stream to the sink, so a writer of that index (and, as `parking_lot` is
//! fair, every reader queued behind that writer) waits until the stream is
//! written.
//!
//! ## 0.5.x format (one raw chunk)
//!
//! A 0.5.x file holds the section as one raw chunk, the bincode of
//! `TextIndexSnapshot` (snapshot version 1), which `read_from` hands to
//! [`Section::deserialize`]. [`Section::serialize`] still writes it, for the
//! spill path. The next checkpoint after reading it writes version 2.

use std::io::{self, Read, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{
    ChunkKind, ChunkMeta, Section, SectionSink, SectionSource, SectionType, check_version,
    legacy_bytes,
};
use grafeo_common::storage::{ChunkCaps, ChunkStreamReader, ChunkStreamWriter, stream_error};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use super::{BM25Config, InvertedIndex, PostingsVisitor};

/// The section version this build writes: 1 was the bincode snapshot of
/// 0.5.x, 2 the streams. A source of version 2 is read as streams, a single
/// raw chunk of any version as 0.5.x bytes.
pub(crate) const TEXT_SECTION_VERSION: u8 = 2;

/// The version byte inside the 0.5.x snapshot, which [`Section::serialize`]
/// writes ([`Section::deserialize`] does not check it).
const SNAPSHOT_VERSION: u8 = 1;

/// The layout byte of the metadata chunk this build writes and reads.
const META_LAYOUT: u8 = 1;

/// The most bytes a metadata chunk's decoding may claim, so a corrupt length
/// prefix fails the decode instead of requesting a huge allocation. Far
/// above the metadata of any real set of indexes (a few bytes plus the key
/// per index).
const META_DECODE_LIMIT: usize = 1 << 24;

/// The most postings a read allocates for up front: a count read from a
/// stream is not trusted with an allocation, a list grows as its postings
/// arrive.
const PREALLOCATE_AT_MOST: usize = 64;

/// The bytes of a term a read takes at a time, so a term length read from a
/// stream cannot request a huge allocation before the bytes are there.
const TERM_READ_STEP: usize = 4096;

/// The bytes of postings or document lengths the writer gathers before it
/// hands them to the stream: a long posting list goes out in a few writes,
/// without a copy of the list.
const WRITE_BATCH_BYTES: usize = 4096;

// ── Snapshot types (0.5.x) ──────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct TextIndexSnapshot {
    version: u8,
    indexes: Vec<SingleIndexSnapshot>,
}

#[derive(Serialize, Deserialize)]
struct SingleIndexSnapshot {
    /// Index key: "label:property"
    key: String,
    /// BM25 parameters
    k1: f64,
    b: f64,
    /// Postings: term -> vec of (node_id, term_freq)
    postings: Vec<(String, Vec<(NodeId, u32)>)>,
    /// Document lengths: node_id -> token count
    doc_lengths: Vec<(NodeId, u32)>,
    /// Sum of all document lengths
    total_length: u64,
}

// ── Version 2 metadata ──────────────────────────────────────────────

/// What the metadata chunk of version 2 holds.
#[derive(Serialize, Deserialize)]
struct TextMeta {
    /// [`META_LAYOUT`].
    layout: u8,
    /// The byte cap the streams were cut with (readers accept any).
    max_bytes: u32,
    /// The index keys in strictly increasing order; index `keys[i]` is
    /// stream `i`.
    keys: Vec<String>,
}

// ── Section implementation ──────────────────────────────────────────

/// Text Index section for the `.grafeo` container.
pub struct TextIndexSection {
    indexes: Vec<(String, Arc<RwLock<InvertedIndex>>)>,
    dirty: AtomicBool,
    /// The caps [`Section::write_to`] cuts the streams with, taken when the
    /// section is built.
    caps: ChunkCaps,
}

impl TextIndexSection {
    /// Create a new Text Index section from the current indexes, with this
    /// thread's chunk caps ([`ChunkCaps::current`]).
    pub fn new(indexes: Vec<(String, Arc<RwLock<InvertedIndex>>)>) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
            caps: ChunkCaps::current(),
        }
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

/// Names the text index `key` in `error`, keeping its kind.
fn in_index(error: Error, key: &str) -> Error {
    let place = format!("in the stream of text index '{key}'");
    match error {
        Error::Serialization(message) => Error::Serialization(format!("{message}, {place}")),
        Error::Internal(message) => Error::Internal(format!("{message}, {place}")),
        Error::InvalidValue(message) => Error::InvalidValue(format!("{message}, {place}")),
        Error::Io(inner) => Error::Io(io::Error::new(inner.kind(), format!("{inner}, {place}"))),
        other => other,
    }
}

/// `count` as the u32 the stream stores, or an error naming `what`.
fn count_u32(count: usize, what: &str) -> Result<u32> {
    u32::try_from(count).map_err(|_| {
        Error::Serialization(format!(
            "section TextIndex: {what} {count} does not fit the 32 bits the stream stores"
        ))
    })
}

/// The metadata chunk of `keys`, which come in strictly increasing order.
fn encode_meta(keys: Vec<String>, caps: ChunkCaps) -> Result<Vec<u8>> {
    let meta = TextMeta {
        layout: META_LAYOUT,
        max_bytes: caps.max_bytes,
        keys,
    };
    bincode::serde::encode_to_vec(&meta, bincode::config::standard()).map_err(|error| {
        Error::Serialization(format!(
            "section TextIndex: the metadata chunk cannot be encoded: {error}"
        ))
    })
}

/// Writes an index into its stream as [`InvertedIndex::visit`] hands it
/// over.
struct PostingsWriter<'s> {
    stream: ChunkStreamWriter<'s>,
    /// Bytes gathered for the stream: the header, terms, postings and
    /// document lengths, written out once they reach [`WRITE_BATCH_BYTES`]
    /// and when the stream finishes.
    scratch: Vec<u8>,
    /// Whether a write into `stream` failed: the stream then keeps the sink's
    /// own error, which `finish` returns.
    write_failed: bool,
}

impl<'s> PostingsWriter<'s> {
    /// A writer of stream `stream` of graph 0 into `sink`.
    fn new(sink: &'s mut dyn SectionSink, stream: u32, caps: ChunkCaps) -> Self {
        Self {
            stream: ChunkStreamWriter::new(sink, 0, stream, caps),
            scratch: Vec::new(),
            write_failed: false,
        }
    }

    /// Writes the gathered bytes into the stream.
    fn write_scratch(&mut self) -> Result<()> {
        let written = self.stream.write_all(&self.scratch);
        self.scratch.clear();
        written.map_err(|error| {
            self.write_failed = true;
            stream_error(SectionType::TextIndex, error)
        })
    }

    /// Writes the gathered bytes once they reach [`WRITE_BATCH_BYTES`].
    fn write_batch(&mut self) -> Result<()> {
        if self.scratch.len() >= WRITE_BATCH_BYTES {
            self.write_scratch()
        } else {
            Ok(())
        }
    }

    /// Writes what is still gathered and the stream's last piece.
    fn finish(mut self) -> Result<()> {
        match self.write_scratch() {
            Ok(()) => self.stream.finish().map(drop),
            // The stream keeps the sink's error: return it, not its text.
            Err(error) => Err(self.stream.finish().err().unwrap_or(error)),
        }
    }
}

impl PostingsVisitor for PostingsWriter<'_> {
    fn header(
        &mut self,
        config: &BM25Config,
        total_length: u64,
        term_count: usize,
        doc_count: usize,
    ) -> Result<()> {
        self.scratch.extend_from_slice(&config.k1.to_le_bytes());
        self.scratch.extend_from_slice(&config.b.to_le_bytes());
        self.scratch.extend_from_slice(&total_length.to_le_bytes());
        self.scratch
            .extend_from_slice(&(doc_count as u64).to_le_bytes());
        self.scratch
            .extend_from_slice(&(term_count as u64).to_le_bytes());
        self.write_batch()
    }

    fn posting_list(
        &mut self,
        term: &str,
        count: usize,
        postings: &mut dyn Iterator<Item = (NodeId, u32)>,
    ) -> Result<()> {
        let term_length = count_u32(term.len(), "a term of length")?;
        self.scratch.extend_from_slice(&term_length.to_le_bytes());
        self.scratch.extend_from_slice(term.as_bytes());
        self.scratch
            .extend_from_slice(&(count as u64).to_le_bytes());
        let mut written = 0usize;
        for (node, frequency) in postings {
            self.scratch.extend_from_slice(&node.as_u64().to_le_bytes());
            self.scratch.extend_from_slice(&frequency.to_le_bytes());
            written += 1;
            self.write_batch()?;
        }
        if written != count {
            return Err(Error::Internal(format!(
                "section TextIndex: the posting list of term '{term}' was announced with {count} \
                 postings and holds {written}"
            )));
        }
        self.write_batch()
    }

    fn doc_length(&mut self, node: NodeId, length: u32) -> Result<()> {
        self.scratch.extend_from_slice(&node.as_u64().to_le_bytes());
        self.scratch.extend_from_slice(&length.to_le_bytes());
        self.write_batch()
    }
}

/// Writes `index` as stream `stream` of graph 0.
fn write_index(
    sink: &mut dyn SectionSink,
    stream: u32,
    caps: ChunkCaps,
    index: &InvertedIndex,
) -> Result<()> {
    let mut writer = PostingsWriter::new(sink, stream, caps);
    match index.visit(&mut writer) {
        Ok(()) => writer.finish(),
        // The stream keeps the sink's error: return it, not its text.
        Err(error) if writer.write_failed => Err(writer.stream.finish().err().unwrap_or(error)),
        Err(error) => Err(error),
    }
}

/// The metadata chunk of a version 2 source, after checking that it comes
/// first, that its keys increase, and that every other chunk is a piece of a
/// stream it lists.
fn read_meta(source: &dyn SectionSource) -> Result<TextMeta> {
    let chunks = source.chunks();
    match chunks.first() {
        Some(first) if *first == ChunkMeta::meta() => {}
        Some(first) if first.kind == ChunkKind::Meta => {
            return Err(Error::Serialization(format!(
                "section TextIndex: the metadata chunk has codec {}, graph {}, column {}, first \
                 row {} and rows {}, where all are 0",
                first.codec, first.graph_id, first.column_id, first.row_start, first.row_count
            )));
        }
        Some(first) => {
            return Err(Error::Serialization(format!(
                "section TextIndex: the first chunk is of kind {:?}, where the metadata chunk \
                 comes first",
                first.kind
            )));
        }
        None => {
            return Err(Error::Serialization(
                "section TextIndex: the section holds no chunk, not even its metadata chunk"
                    .to_string(),
            ));
        }
    }
    let bytes = source.fetch(0)?;
    let (meta, read): (TextMeta, usize) = bincode::serde::decode_from_slice(
        &bytes,
        bincode::config::standard().with_limit::<META_DECODE_LIMIT>(),
    )
    .map_err(|error| {
        Error::Serialization(format!(
            "section TextIndex: the metadata chunk does not decode: {error}"
        ))
    })?;
    if read != bytes.len() {
        return Err(Error::Serialization(format!(
            "section TextIndex: the metadata chunk holds {} bytes after its {read} bytes of \
             metadata",
            bytes.len() - read
        )));
    }
    if meta.layout != META_LAYOUT {
        return Err(Error::Serialization(format!(
            "section TextIndex: the metadata chunk has layout {}, this build reads layout \
             {META_LAYOUT}",
            meta.layout
        )));
    }
    for pair in meta.keys.windows(2) {
        if pair[0] >= pair[1] {
            return Err(Error::Serialization(format!(
                "section TextIndex: the metadata lists text index '{}' after '{}', but keys are \
                 strictly increasing",
                pair[1], pair[0]
            )));
        }
    }
    for chunk in &chunks[1..] {
        let listed = usize::try_from(chunk.column_id).is_ok_and(|stream| stream < meta.keys.len());
        if chunk.kind != ChunkKind::Stream || chunk.graph_id != 0 || !listed {
            return Err(Error::Serialization(format!(
                "section TextIndex: a chunk of kind {:?} for graph {}, stream {}, where only \
                 pieces of the {} streams of graph 0 (one per index) follow the metadata chunk",
                chunk.kind,
                chunk.graph_id,
                chunk.column_id,
                meta.keys.len()
            )));
        }
    }
    Ok(meta)
}

/// Reads one index stream, fixed-size fields at a time.
struct PostingsReader<'s> {
    stream: ChunkStreamReader<'s>,
}

impl PostingsReader<'_> {
    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut buffer = [0u8; N];
        self.stream
            .read_exact(&mut buffer)
            .map_err(|error| stream_error(SectionType::TextIndex, error))?;
        Ok(buffer)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }

    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes()?))
    }

    /// `count` (read from the stream) as a `usize`.
    fn usize(count: u64, what: &str) -> Result<usize> {
        usize::try_from(count).map_err(|_| {
            Error::Serialization(format!(
                "section TextIndex: {what} {count} does not fit this platform"
            ))
        })
    }

    /// A term: its length, then its UTF-8 bytes, read [`TERM_READ_STEP`]
    /// bytes at a time.
    fn term(&mut self) -> Result<String> {
        let length = Self::usize(u64::from(self.u32()?), "a term length of")?;
        let mut bytes = Vec::with_capacity(length.min(TERM_READ_STEP));
        while bytes.len() < length {
            let start = bytes.len();
            bytes.resize(start + (length - start).min(TERM_READ_STEP), 0);
            self.stream
                .read_exact(&mut bytes[start..])
                .map_err(|error| stream_error(SectionType::TextIndex, error))?;
        }
        String::from_utf8(bytes).map_err(|error| {
            Error::Serialization(format!("section TextIndex: a term is not UTF-8 ({error})"))
        })
    }

    /// Refuses a stream that holds bytes after its last posting list.
    fn expect_end(&mut self) -> Result<()> {
        let mut probe = [0u8; 1];
        match self.stream.read(&mut probe) {
            Ok(0) => Ok(()),
            Ok(_) => Err(Error::Serialization(
                "section TextIndex: the stream holds bytes after its last posting list".to_string(),
            )),
            Err(error) => Err(stream_error(SectionType::TextIndex, error)),
        }
    }

    /// Reads the index into `index`: the header, the document lengths, then
    /// one posting list at a time, checking each as it comes (see the module
    /// documentation).
    fn restore(&mut self, index: &mut InvertedIndex) -> Result<()> {
        let k1 = self.f64()?;
        let b = self.f64()?;
        if !k1.is_finite() || !b.is_finite() {
            return Err(Error::Serialization(format!(
                "section TextIndex: the BM25 parameters k1 {k1} and b {b} are not both finite"
            )));
        }
        let total_length = self.u64()?;
        let doc_count = self.u64()?;
        let term_count = self.u64()?;
        index.begin_restore(BM25Config { k1, b }, total_length);
        self.restore_doc_lengths(index, doc_count, total_length)?;
        self.restore_posting_lists(index, term_count, total_length)?;
        self.expect_end()
    }

    /// Reads `doc_count` document lengths into `index`.
    fn restore_doc_lengths(
        &mut self,
        index: &mut InvertedIndex,
        doc_count: u64,
        total_length: u64,
    ) -> Result<()> {
        let mut previous: Option<NodeId> = None;
        // Saturates instead of overflowing; it never reaches the total then.
        let mut sum: u128 = 0;
        for _ in 0..doc_count {
            let node = NodeId::new(self.u64()?);
            if let Some(previous) = previous
                && node <= previous
            {
                return Err(Error::Serialization(format!(
                    "section TextIndex: document {} follows document {}, but node ids are \
                     strictly increasing",
                    node.as_u64(),
                    previous.as_u64()
                )));
            }
            let length = self.u32()?;
            if length == 0 {
                return Err(Error::Serialization(format!(
                    "section TextIndex: document {} has length 0, but an indexed document has \
                     at least one token",
                    node.as_u64()
                )));
            }
            sum = sum.saturating_add(u128::from(length));
            index.restore_doc_length(node, length);
            previous = Some(node);
        }
        if sum != u128::from(total_length) {
            return Err(Error::Serialization(format!(
                "section TextIndex: the document lengths add up to {sum}, the stream gives a \
                 total length {total_length}"
            )));
        }
        Ok(())
    }

    /// Reads `term_count` posting lists into `index`, whose document lengths
    /// are restored already.
    fn restore_posting_lists(
        &mut self,
        index: &mut InvertedIndex,
        term_count: u64,
        total_length: u64,
    ) -> Result<()> {
        let mut previous: Option<String> = None;
        // Saturates instead of overflowing; it never reaches the total then.
        let mut sum: u128 = 0;
        for _ in 0..term_count {
            let term = self.term()?;
            if let Some(previous) = &previous
                && term <= *previous
            {
                return Err(Error::Serialization(format!(
                    "section TextIndex: the term '{term}' follows '{previous}', but terms are \
                     strictly increasing"
                )));
            }
            let count = Self::usize(self.u64()?, "a posting count of")?;
            if count == 0 {
                return Err(Error::Serialization(format!(
                    "section TextIndex: the posting list of term '{term}' is empty"
                )));
            }
            let mut postings = Vec::with_capacity(count.min(PREALLOCATE_AT_MOST));
            let mut previous_node: Option<NodeId> = None;
            for _ in 0..count {
                let node = NodeId::new(self.u64()?);
                let frequency = self.u32()?;
                if let Some(previous_node) = previous_node
                    && node <= previous_node
                {
                    return Err(Error::Serialization(format!(
                        "section TextIndex: the posting list of term '{term}' holds node {} \
                         after node {}, but its nodes are strictly increasing",
                        node.as_u64(),
                        previous_node.as_u64()
                    )));
                }
                if !index.contains(node) {
                    return Err(Error::Serialization(format!(
                        "section TextIndex: the posting list of term '{term}' holds node {}, \
                         which has no document length",
                        node.as_u64()
                    )));
                }
                if frequency == 0 {
                    return Err(Error::Serialization(format!(
                        "section TextIndex: the posting list of term '{term}' gives node {} a \
                         term frequency of 0",
                        node.as_u64()
                    )));
                }
                sum = sum.saturating_add(u128::from(frequency));
                postings.push((node, frequency));
                previous_node = Some(node);
            }
            index.restore_posting_list(term.clone(), postings);
            previous = Some(term);
        }
        if sum != u128::from(total_length) {
            return Err(Error::Serialization(format!(
                "section TextIndex: the term frequencies add up to {sum}, the stream gives a \
                 total length {total_length}"
            )));
        }
        Ok(())
    }
}

/// Restores `index` from stream `stream`; on an error, leaves it empty with
/// the configuration it had.
fn read_index(source: &dyn SectionSource, stream: u32, index: &mut InvertedIndex) -> Result<()> {
    let config = index.config().clone();
    let mut reader = PostingsReader {
        stream: ChunkStreamReader::new(source, 0, stream),
    };
    let restored = reader.restore(index);
    if restored.is_err() {
        // No half index stays behind for a search to use.
        index.begin_restore(config, 0);
    }
    restored
}

impl Section for TextIndexSection {
    fn section_type(&self) -> SectionType {
        SectionType::TextIndex
    }

    fn version(&self) -> u8 {
        TEXT_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let indexes: Vec<SingleIndexSnapshot> = self
            .indexes
            .iter()
            .map(|(key, index_lock)| {
                let index = index_lock.read();
                let config = index.config();
                let (postings, doc_lengths, total_length) = index.snapshot();

                SingleIndexSnapshot {
                    key: key.clone(),
                    k1: config.k1,
                    b: config.b,
                    postings,
                    doc_lengths,
                    total_length,
                }
            })
            .collect();

        let snapshot = TextIndexSnapshot {
            version: SNAPSHOT_VERSION,
            indexes,
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("Text Index section serialization failed: {e}")))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let config = bincode::config::standard();
        let (snapshot, _): (TextIndexSnapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| {
                Error::Serialization(format!("Text Index section deserialization failed: {e}"))
            })?;

        for idx_snap in snapshot.indexes {
            if let Some((_, index_lock)) = self.indexes.iter().find(|(k, _)| *k == idx_snap.key) {
                let mut index = index_lock.write();
                index.set_config(BM25Config {
                    k1: idx_snap.k1,
                    b: idx_snap.b,
                });
                index.restore(
                    idx_snap.postings,
                    idx_snap.doc_lengths,
                    idx_snap.total_length,
                );
            }
        }

        Ok(())
    }

    /// Writes the metadata chunk, then each index as its own stream, in key
    /// order.
    ///
    /// Each index's read lock is held while its stream is written to the
    /// sink: a writer of that index (and, as `parking_lot` is fair, every
    /// reader queued behind that writer) waits until the stream is written.
    ///
    /// # Errors
    ///
    /// Returns the sink's error; [`Error::Serialization`] when a count does
    /// not fit the 32 bits the stream stores it in; [`Error::Internal`] when
    /// the section was given two indexes with one key.
    fn write_to(&self, sink: &mut dyn SectionSink) -> Result<()> {
        let mut indexes: Vec<&(String, Arc<RwLock<InvertedIndex>>)> = self.indexes.iter().collect();
        indexes.sort_by(|(left, _), (right, _)| left.cmp(right));
        if let Some(pair) = indexes.windows(2).find(|pair| pair[0].0 == pair[1].0) {
            return Err(Error::Internal(format!(
                "section TextIndex: two text indexes with the key '{}'",
                pair[0].0
            )));
        }
        let keys = indexes.iter().map(|(key, _)| key.clone()).collect();
        sink.write_chunk(ChunkMeta::meta(), &encode_meta(keys, self.caps)?)?;
        for (position, (key, index)) in indexes.into_iter().enumerate() {
            let stream = u32::try_from(position).map_err(|_| {
                Error::Serialization(format!(
                    "section TextIndex: {position} text indexes do not fit the 32-bit stream \
                     numbers"
                ))
            })?;
            write_index(sink, stream, self.caps, &index.read())
                .map_err(|error| in_index(error, key))?;
        }
        Ok(())
    }

    /// Restores each index this section was given from the stream with its
    /// key; a single raw chunk (0.5.x bytes) goes to
    /// [`deserialize`](Section::deserialize).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] for a section of another version, a
    /// metadata chunk or a chunk sequence this writer does not produce, or a
    /// stream that does not hold its index exactly (the index named, and left
    /// empty); any error from fetching a chunk.
    fn read_from(&mut self, source: &dyn SectionSource) -> Result<()> {
        let legacy = legacy_bytes(source).map_err(|error| match error {
            Error::Serialization(message) => {
                Error::Serialization(format!("section TextIndex: {message}"))
            }
            other => other,
        })?;
        if let Some(bytes) = legacy {
            return self.deserialize(&bytes);
        }
        check_version(SectionType::TextIndex, source, TEXT_SECTION_VERSION)?;
        let meta = read_meta(source)?;
        for (position, key) in meta.keys.iter().enumerate() {
            let Some((_, index)) = self.indexes.iter().find(|(given, _)| given == key) else {
                continue;
            };
            let stream = u32::try_from(position).map_err(|_| {
                Error::Serialization(format!(
                    "section TextIndex: the metadata lists {position} indexes, more than 32-bit \
                     stream numbers reach"
                ))
            })?;
            read_index(source, stream, &mut index.write()).map_err(|error| in_index(error, key))?;
        }
        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.indexes
            .iter()
            .map(|(_, idx)| idx.read().heap_memory_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::storage::{ImageSource, MemoryImage};
    use grafeo_common::testing::chunk_caps::with_chunk_caps;

    /// Caps that cut every stream of the tests into many pieces.
    const TINY: ChunkCaps = ChunkCaps {
        max_rows: 3,
        max_bytes: 512,
    };

    /// The words the test documents are made of.
    const WORDS: [&str; 12] = [
        "alix",
        "gus",
        "vincent",
        "mia",
        "jules",
        "amsterdam",
        "berlin",
        "paris",
        "prague",
        "canal",
        "bridge",
        "museum",
    ];

    type Indexes = Vec<(String, Arc<RwLock<InvertedIndex>>)>;

    /// Document `seed`: one to seven words picked by the seed, so terms
    /// appear in many documents, some more than once, and documents differ
    /// in length.
    fn document(seed: u64) -> String {
        let words = WORDS.len() as u64;
        (0..=seed % 7)
            .map(|position| {
                let word = (seed * 3 + position * 19 + position * position * 88) % words;
                WORDS[usize::try_from(word).unwrap()]
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// An index of `documents`, each a node and its text.
    fn index_of(
        config: BM25Config,
        documents: impl IntoIterator<Item = (u64, String)>,
    ) -> Arc<RwLock<InvertedIndex>> {
        let mut index = InvertedIndex::new(config);
        for (node, text) in documents {
            index.insert(NodeId::new(node), &text);
        }
        Arc::new(RwLock::new(index))
    }

    /// "Doc:body" with documents 1 to `count` and the default parameters,
    /// and "Note:text" with `count` documents of other words and node ids, k1
    /// 1.9 and b 0.3.
    fn two_indexes(count: u64) -> [(String, Arc<RwLock<InvertedIndex>>); 2] {
        [
            (
                "Doc:body".to_string(),
                index_of(
                    BM25Config::default(),
                    (1..=count).map(|seed| (seed, document(seed))),
                ),
            ),
            (
                "Note:text".to_string(),
                index_of(
                    BM25Config { k1: 1.9, b: 0.3 },
                    (1..=count).map(|seed| (seed * 3 + 88, document(seed + 19))),
                ),
            ),
        ]
    }

    /// Empty indexes for `keys`, as the engine hands a section to read into.
    fn shells(keys: &[&str]) -> Indexes {
        keys.iter()
            .map(|key| {
                let index = InvertedIndex::new(BM25Config::default());
                (key.to_string(), Arc::new(RwLock::new(index)))
            })
            .collect()
    }

    /// The image `indexes` write, with `caps` taken when the section is
    /// built.
    fn written(indexes: Indexes, caps: ChunkCaps) -> MemoryImage {
        let section = with_chunk_caps(caps, || TextIndexSection::new(indexes));
        MemoryImage::from_sections(&[&section]).unwrap()
    }

    /// Reads the text section of `image` into shells for `keys`.
    fn read_back(image: &MemoryImage, keys: &[&str]) -> (Result<()>, Indexes) {
        let shells = shells(keys);
        let source = image
            .section_source(SectionType::TextIndex)
            .expect("the image holds a text section");
        let read = TextIndexSection::new(shells.clone()).read_from(&*source);
        (read, shells)
    }

    /// The chunks of the text section of `image`, with their bytes.
    fn chunks(image: &MemoryImage) -> Vec<(ChunkMeta, Vec<u8>)> {
        let source = image.section_source(SectionType::TextIndex).unwrap();
        source
            .chunks()
            .iter()
            .enumerate()
            .map(|(index, meta)| (*meta, source.fetch(index).unwrap().to_vec()))
            .collect()
    }

    /// An image of a text section of `version` holding `chunks`.
    fn image_of(version: u8, chunks: &[(ChunkMeta, Vec<u8>)]) -> MemoryImage {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::TextIndex, version)
            .unwrap();
        for (meta, bytes) in chunks {
            image.write_chunk(*meta, bytes).unwrap();
        }
        image
    }

    /// `image` with its chunks changed by `edit`.
    fn edited(
        image: &MemoryImage,
        edit: impl FnOnce(&mut Vec<(ChunkMeta, Vec<u8>)>),
    ) -> MemoryImage {
        let mut chunks = chunks(image);
        edit(&mut chunks);
        image_of(TEXT_SECTION_VERSION, &chunks)
    }

    /// The position in `chunks` of the last piece of stream `stream`.
    fn last_piece(chunks: &[(ChunkMeta, Vec<u8>)], stream: u32) -> usize {
        chunks
            .iter()
            .rposition(|(meta, _)| meta.kind == ChunkKind::Stream && meta.column_id == stream)
            .expect("the stream has a piece")
    }

    /// The bytes of the pieces of stream `stream`, in order.
    fn pieces(chunks: &[(ChunkMeta, Vec<u8>)], stream: u32) -> Vec<&[u8]> {
        chunks
            .iter()
            .filter(|(meta, _)| meta.kind == ChunkKind::Stream && meta.column_id == stream)
            .map(|(_, bytes)| bytes.as_slice())
            .collect()
    }

    fn decode_meta(bytes: &[u8]) -> TextMeta {
        bincode::serde::decode_from_slice(bytes, bincode::config::standard())
            .unwrap()
            .0
    }

    fn encode_meta_of(meta: &TextMeta) -> Vec<u8> {
        bincode::serde::encode_to_vec(meta, bincode::config::standard()).unwrap()
    }

    type Snapshot = (Vec<(String, Vec<(NodeId, u32)>)>, Vec<(NodeId, u32)>, u64);

    /// `index.snapshot()` with every posting list in node order: what an
    /// index restored from a stream of `index` holds.
    fn in_node_order(index: &InvertedIndex) -> Snapshot {
        let (mut postings, lengths, total) = index.snapshot();
        for (_, list) in &mut postings {
            list.sort_by_key(|(node, _)| *node);
        }
        (postings, lengths, total)
    }

    /// Search results by descending score, ties by node id (`search` leaves
    /// ties in hash order), each score as its bits.
    fn ranking(index: &InvertedIndex, query: &str) -> Vec<(NodeId, u64)> {
        let mut results = index.search(query, usize::MAX);
        results.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
        results
            .into_iter()
            .map(|(node, score)| (node, score.to_bits()))
            .collect()
    }

    /// A stream spelled out as the version 2 layout reads.
    struct Spelled(Vec<u8>);

    impl Spelled {
        /// The header: BM25 parameters, total length, document and term
        /// counts.
        fn header(parameters: (f64, f64), total_length: u64, documents: u64, terms: u64) -> Self {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&parameters.0.to_le_bytes());
            bytes.extend_from_slice(&parameters.1.to_le_bytes());
            bytes.extend_from_slice(&total_length.to_le_bytes());
            bytes.extend_from_slice(&documents.to_le_bytes());
            bytes.extend_from_slice(&terms.to_le_bytes());
            Self(bytes)
        }

        /// The posting list of `term`, its count as a u64.
        fn term(mut self, term: &[u8], postings: &[(u64, u32)]) -> Self {
            self.0
                .extend_from_slice(&u32::try_from(term.len()).unwrap().to_le_bytes());
            self.0.extend_from_slice(term);
            self.0
                .extend_from_slice(&(postings.len() as u64).to_le_bytes());
            for (node, frequency) in postings {
                self.0.extend_from_slice(&node.to_le_bytes());
                self.0.extend_from_slice(&frequency.to_le_bytes());
            }
            self
        }

        /// The length of the document of `node`.
        fn document(mut self, node: u64, length: u32) -> Self {
            self.0.extend_from_slice(&node.to_le_bytes());
            self.0.extend_from_slice(&length.to_le_bytes());
            self
        }

        /// `bytes` as they are.
        fn raw(mut self, bytes: &[u8]) -> Self {
            self.0.extend_from_slice(bytes);
            self
        }
    }

    /// A version 2 image listing `keys`, stream `i` holding `streams[i]` as
    /// one piece.
    fn crafted(keys: &[&str], streams: Vec<Spelled>) -> MemoryImage {
        let meta = TextMeta {
            layout: 1,
            max_bytes: 512,
            keys: keys.iter().map(ToString::to_string).collect(),
        };
        let mut chunks = vec![(ChunkMeta::meta(), encode_meta_of(&meta))];
        for (stream, spelled) in streams.into_iter().enumerate() {
            let stream = u32::try_from(stream).unwrap();
            chunks.push((ChunkMeta::stream_piece(0, stream, 0), spelled.0));
        }
        image_of(TEXT_SECTION_VERSION, &chunks)
    }

    #[test]
    fn text_indexes_round_trip_through_streams() {
        let [doc, note] = two_indexes(300);
        // Handed over out of key order, as the store's hash map may.
        let image = written(vec![note.clone(), doc.clone()], TINY);
        let source = image.section_source(SectionType::TextIndex).unwrap();
        assert_eq!(source.section_version(), 2, "the section version");
        drop(source);
        let chunks = chunks(&image);
        assert_eq!(
            chunks[0].0,
            ChunkMeta::meta(),
            "the metadata chunk comes first"
        );
        let meta = decode_meta(&chunks[0].1);
        assert_eq!(
            (meta.layout, meta.max_bytes, meta.keys),
            (
                1,
                512,
                vec!["Doc:body".to_string(), "Note:text".to_string()]
            ),
            "layout 1, the cap the pieces were cut with, the keys in order"
        );
        for stream in 0..2 {
            let pieces = pieces(&chunks, stream);
            assert!(
                pieces.len() > 3,
                "stream {stream} is cut into {} pieces",
                pieces.len()
            );
            assert!(
                pieces.iter().all(|piece| piece.len() <= 512),
                "stream {stream}: no piece is larger than the cap"
            );
        }

        let (read, restored) = read_back(&image, &["Doc:body", "Note:text"]);
        read.unwrap();
        for ((key, restored), (_, original)) in restored.iter().zip([&doc, &note]) {
            let (restored, original) = (restored.read(), original.read());
            assert_eq!(original.len(), 300, "{key}: every document is indexed");
            assert_eq!(restored.snapshot(), in_node_order(&original), "{key}");
            assert_eq!(
                (
                    restored.config().k1.to_bits(),
                    restored.config().b.to_bits()
                ),
                (
                    original.config().k1.to_bits(),
                    original.config().b.to_bits()
                ),
                "{key}: the BM25 parameters"
            );
            for query in ["alix amsterdam", "museum", "gus prague bridge canal"] {
                let expected = ranking(&original, query);
                assert!(!expected.is_empty(), "{key}: '{query}' finds documents");
                assert_eq!(ranking(&restored, query), expected, "{key}: '{query}'");
            }
        }
    }

    #[test]
    fn a_posting_list_larger_than_the_cap_streams_across_pieces() {
        // Amsterdam in 5,000 documents, twice in every third.
        let city = index_of(
            BM25Config::default(),
            (1..=5_000).map(|node| {
                let text = if node % 3 == 0 {
                    "Amsterdam Amsterdam"
                } else {
                    "Amsterdam"
                };
                (node, text.to_string())
            }),
        );
        assert_eq!(city.read().term_count(), 1, "one term");
        let image = written(vec![("City:name".to_string(), Arc::clone(&city))], TINY);
        let chunks = chunks(&image);
        let pieces = pieces(&chunks, 0);
        let length: usize = pieces.iter().map(|piece| piece.len()).sum();
        assert_eq!(
            length,
            40 + 5_000 * 12 + (4 + 9 + 8 + 5_000 * 12),
            "the header, 5,000 document lengths, one list of 5,000 postings"
        );
        assert_eq!(pieces.len(), length.div_ceil(512));
        assert!(
            pieces[..pieces.len() - 1]
                .iter()
                .all(|piece| piece.len() == 512),
            "every piece but the last is full"
        );

        let (read, restored) = read_back(&image, &["City:name"]);
        read.unwrap();
        let restored = restored[0].1.read();
        assert_eq!(restored.snapshot(), in_node_order(&city.read()));
        let ranked = ranking(&restored, "amsterdam");
        assert_eq!(ranked.len(), 5_000, "every document is found");
        assert_eq!(ranked, ranking(&city.read(), "amsterdam"));
    }

    #[test]
    fn a_truncated_stream_is_refused() {
        let [doc, note] = two_indexes(40);
        let image = written(vec![doc, note], TINY);
        let keys = ["Doc:body", "Note:text"];
        let cases = [
            (
                "the last piece of Doc:body dropped",
                edited(&image, |chunks| {
                    let at = last_piece(chunks, 0);
                    chunks.remove(at);
                }),
                0,
                "ends before",
            ),
            (
                "Doc:body without its last byte",
                edited(&image, |chunks| {
                    let at = last_piece(chunks, 0);
                    chunks[at].1.pop();
                }),
                0,
                "ends before",
            ),
            (
                "a byte after the end of Doc:body",
                edited(&image, |chunks| {
                    let at = last_piece(chunks, 0);
                    chunks[at].1.push(3);
                }),
                0,
                "after its last",
            ),
            (
                "Note:text without a piece",
                edited(&image, |chunks| {
                    chunks
                        .retain(|(meta, _)| meta.kind != ChunkKind::Stream || meta.column_id != 1);
                }),
                1,
                "ends before",
            ),
        ];
        for (case, image, failed, reason) in cases {
            let (read, restored) = read_back(&image, &keys);
            let error = read.expect_err(case);
            let message = error.to_string();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
            assert!(
                message.contains("TextIndex")
                    && message.contains(keys[failed])
                    && message.contains(reason),
                "{case}: {message}"
            );
            let failed = restored[failed].1.read();
            assert!(
                failed.is_empty() && failed.term_count() == 0,
                "{case}: no half index stays behind"
            );
        }
    }

    #[test]
    fn a_0_5_text_section_still_loads() {
        let [doc, note] = two_indexes(40);
        let bytes = TextIndexSection::new(vec![doc.clone(), note.clone()])
            .serialize()
            .unwrap();
        assert_eq!(bytes[0], 1, "the 0.5.x snapshot starts with its version, 1");
        let image = MemoryImage::from_raw(vec![(SectionType::TextIndex, bytes)]).unwrap();

        let (read, restored) = read_back(&image, &["Doc:body", "Note:text"]);
        read.unwrap();
        for ((key, restored), (_, original)) in restored.iter().zip([&doc, &note]) {
            let (restored, original) = (restored.read(), original.read());
            assert_eq!(restored.snapshot(), original.snapshot(), "{key}");
            assert_eq!(
                (restored.config().k1, restored.config().b),
                (original.config().k1, original.config().b),
                "{key}: the BM25 parameters"
            );
        }
    }

    #[test]
    fn the_same_indexes_write_the_same_bytes() {
        // Built twice, so every map has its own hash seed, and handed over in
        // both orders.
        let [doc, note] = two_indexes(60);
        let [doc_again, note_again] = two_indexes(60);
        let first = chunks(&written(vec![doc.clone(), note.clone()], TINY));
        assert_eq!(
            chunks(&written(vec![note_again, doc_again], TINY)),
            first,
            "the same indexes built again"
        );
        assert_eq!(
            chunks(&written(vec![doc, note], TINY)),
            first,
            "the same indexes written again"
        );
    }

    #[test]
    fn the_metadata_and_a_stream_hold_the_documented_layout() {
        let mut index = InvertedIndex::new(BM25Config { k1: 1.9, b: 0.3 });
        index.insert(NodeId::new(88), "Paris Mia Paris");
        index.insert(NodeId::new(3), "Berlin Paris");
        index.insert(NodeId::new(19), "Amsterdam");
        let image = written(
            vec![("City:name".to_string(), Arc::new(RwLock::new(index)))],
            ChunkCaps::DEFAULT,
        );
        let chunks = chunks(&image);
        // Bincode: layout 1, the cap 1 MiB as a u32 behind its marker 0xFC,
        // one key of nine bytes.
        let mut meta = vec![1, 0xFC, 0x00, 0x00, 0x10, 0x00, 1, 9];
        meta.extend_from_slice(b"City:name");
        assert_eq!(chunks[0], (ChunkMeta::meta(), meta), "the metadata chunk");
        let stream = Spelled::header((1.9, 0.3), 6, 3, 4)
            .document(3, 2)
            .document(19, 1)
            .document(88, 3)
            .term(b"amsterdam", &[(19, 1)])
            .term(b"berlin", &[(3, 1)])
            .term(b"mia", &[(88, 1)])
            // In node order, although node 88 was inserted first.
            .term(b"paris", &[(3, 1), (88, 2)]);
        assert_eq!(
            chunks[1..],
            [(ChunkMeta::stream_piece(0, 0, 0), stream.0)],
            "one piece: the documents in node order, then the terms in order"
        );
    }

    #[test]
    fn indexes_the_section_was_not_given_are_skipped() {
        let [doc, note] = two_indexes(40);
        let image = written(vec![doc, note.clone()], TINY);
        // Doc:body's stream damaged: nothing reads it.
        let image = edited(&image, |chunks| {
            let at = last_piece(chunks, 0);
            chunks.remove(at);
        });
        let (read, restored) = read_back(&image, &["Note:text", "City:name"]);
        read.unwrap();
        assert_eq!(
            restored[0].1.read().snapshot(),
            in_node_order(&note.1.read()),
            "Note:text"
        );
        assert!(
            restored[1].1.read().is_empty(),
            "City:name, which the section does not hold, stays empty"
        );
    }

    #[test]
    fn an_empty_index_and_a_section_without_indexes_round_trip() {
        let image = written(Vec::new(), TINY);
        let only = chunks(&image);
        assert_eq!(only.len(), 1, "only the metadata chunk");
        assert_eq!(
            decode_meta(&only[0].1).keys,
            Vec::<String>::new(),
            "no keys"
        );
        let (read, restored) = read_back(&image, &["Doc:body"]);
        read.unwrap();
        assert!(restored[0].1.read().is_empty());

        let empty = index_of(BM25Config { k1: 0.3, b: 0.88 }, std::iter::empty());
        let image = written(vec![("Doc:body".to_string(), empty)], TINY);
        assert_eq!(
            chunks(&image)[1..],
            [(
                ChunkMeta::stream_piece(0, 0, 0),
                Spelled::header((0.3, 0.88), 0, 0, 0).0
            )],
            "a header only"
        );
        let (read, restored) = read_back(&image, &["Doc:body"]);
        read.unwrap();
        let restored = restored[0].1.read();
        assert!(restored.is_empty() && restored.term_count() == 0);
        assert_eq!((restored.config().k1, restored.config().b), (0.3, 0.88));
    }

    #[test]
    fn crafted_text_sections_are_refused() {
        let [doc, note] = two_indexes(20);
        let good = chunks(&written(vec![doc, note], TINY));
        let changed = |edit: &dyn Fn(&mut Vec<(ChunkMeta, Vec<u8>)>)| {
            let mut chunks = good.clone();
            edit(&mut chunks);
            image_of(TEXT_SECTION_VERSION, &chunks)
        };
        let with_meta = |change: &dyn Fn(&mut TextMeta)| {
            changed(&|chunks| {
                let mut meta = decode_meta(&chunks[0].1);
                change(&mut meta);
                chunks[0].1 = encode_meta_of(&meta);
            })
        };
        let parameters = (1.2, 0.75);
        let cases: Vec<(&str, MemoryImage, &str)> = vec![
            ("another version", image_of(3, &good), "version 3"),
            (
                "a raw chunk among the streams",
                changed(&|chunks| chunks.push((ChunkMeta::raw(), b"Vincent".to_vec()))),
                "raw",
            ),
            (
                "the metadata chunk after a piece",
                changed(&|chunks| chunks.swap(0, 1)),
                "first",
            ),
            (
                "a metadata chunk of a graph",
                changed(&|chunks| chunks[0].0.graph_id = 3),
                "graph 3",
            ),
            (
                "another layout",
                with_meta(&|meta| meta.layout = 2),
                "layout 2",
            ),
            (
                "bytes after the metadata",
                changed(&|chunks| chunks[0].1.push(19)),
                "after",
            ),
            (
                "keys out of order",
                with_meta(&|meta| meta.keys.reverse()),
                "strictly increasing",
            ),
            (
                "a key listed twice",
                with_meta(&|meta| meta.keys[1] = meta.keys[0].clone()),
                "strictly increasing",
            ),
            (
                "a chunk of another kind",
                changed(&|chunks| chunks.push((ChunkMeta::column(0, 0, 0, 3, 0), vec![88]))),
                "Column",
            ),
            (
                "a stream the metadata does not list",
                changed(&|chunks| chunks.push((ChunkMeta::stream_piece(0, 2, 0), vec![88]))),
                "stream 2",
            ),
            (
                "a stream of another graph",
                changed(&|chunks| chunks.push((ChunkMeta::stream_piece(3, 0, 0), vec![88]))),
                "graph 3",
            ),
            (
                "terms out of order",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 2, 2, 2)
                            .document(3, 1)
                            .document(19, 1)
                            .term(b"paris", &[(3, 1)])
                            .term(b"berlin", &[(19, 1)]),
                    ],
                ),
                "strictly increasing",
            ),
            (
                "a term listed twice",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 2, 2, 2)
                            .document(3, 1)
                            .document(19, 1)
                            .term(b"paris", &[(3, 1)])
                            .term(b"paris", &[(19, 1)]),
                    ],
                ),
                "strictly increasing",
            ),
            (
                "documents out of node order",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 2, 2, 1)
                            .document(19, 1)
                            .document(3, 1)
                            .term(b"paris", &[(3, 1), (19, 1)]),
                    ],
                ),
                "strictly increasing",
            ),
            (
                "a document listed twice",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 2, 2, 1)
                            .document(3, 1)
                            .document(3, 1)
                            .term(b"paris", &[(3, 2)]),
                    ],
                ),
                "strictly increasing",
            ),
            (
                "lengths that do not add up to the total",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 88, 1, 1)
                            .document(3, 1)
                            .term(b"paris", &[(3, 1)]),
                    ],
                ),
                "total length 88",
            ),
            (
                "a term that is not UTF-8",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 1, 1, 1)
                            .document(3, 1)
                            .term(&[0xFF, 0xFE], &[(3, 1)]),
                    ],
                ),
                "UTF-8",
            ),
            (
                "a posting count above 32 bits past the end of the stream",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 1, 1, 1)
                            .document(3, 1)
                            .raw(&5u32.to_le_bytes())
                            .raw(b"paris")
                            .raw(&(u64::from(u32::MAX) + 1).to_le_bytes())
                            .raw(&3u64.to_le_bytes())
                            .raw(&1u32.to_le_bytes()),
                    ],
                ),
                "ends before",
            ),
            (
                "a term length past the end of the stream",
                crafted(
                    &["Doc:body"],
                    vec![
                        Spelled::header(parameters, 1, 1, 1)
                            .document(3, 1)
                            .raw(&u32::MAX.to_le_bytes())
                            .raw(b"paris"),
                    ],
                ),
                "ends before",
            ),
        ];
        for (case, image, reason) in cases {
            let (read, restored) = read_back(&image, &["Doc:body", "Note:text"]);
            let error = read.expect_err(case);
            let message = error.to_string();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
            assert!(
                message.contains("TextIndex") && message.contains(reason),
                "{case}: {message}"
            );
            assert!(
                restored
                    .iter()
                    .all(|(_, index)| index.read().is_empty() && index.read().term_count() == 0),
                "{case}: nothing is restored"
            );
        }
    }

    #[test]
    fn inconsistent_streams_are_refused() {
        let parameters = (1.2, 0.75);
        let cases: Vec<(&str, Spelled, &str)> = vec![
            (
                "a posting for a node without a document length",
                Spelled::header(parameters, 2, 1, 2)
                    .document(3, 2)
                    .term(b"berlin", &[(19, 1)])
                    .term(b"paris", &[(3, 1)]),
                "node 19, which has no document length",
            ),
            (
                "one node twice in a list",
                Spelled::header(parameters, 2, 1, 1)
                    .document(3, 2)
                    .term(b"paris", &[(3, 1), (3, 1)]),
                "strictly increasing",
            ),
            (
                "the nodes of a list out of order",
                Spelled::header(parameters, 2, 2, 1)
                    .document(3, 1)
                    .document(19, 1)
                    .term(b"paris", &[(19, 1), (3, 1)]),
                "strictly increasing",
            ),
            (
                "a term frequency of 0",
                Spelled::header(parameters, 1, 1, 2)
                    .document(3, 1)
                    .term(b"berlin", &[(3, 0)])
                    .term(b"paris", &[(3, 1)]),
                "term frequency of 0",
            ),
            (
                "an empty posting list",
                Spelled::header(parameters, 1, 1, 2)
                    .document(3, 1)
                    .term(b"berlin", &[])
                    .term(b"paris", &[(3, 1)]),
                "is empty",
            ),
            (
                "a document length of 0",
                Spelled::header(parameters, 1, 2, 1)
                    .document(3, 0)
                    .document(19, 1)
                    .term(b"paris", &[(19, 1)]),
                "length 0",
            ),
            (
                "term frequencies that do not add up to the total length",
                Spelled::header(parameters, 88, 1, 1)
                    .document(3, 88)
                    .term(b"paris", &[(3, 1)]),
                "term frequencies add up to 1",
            ),
            (
                "k1 NaN",
                Spelled::header((f64::NAN, 0.75), 1, 1, 1)
                    .document(3, 1)
                    .term(b"paris", &[(3, 1)]),
                "not both finite",
            ),
            (
                "b infinite",
                Spelled::header((1.2, f64::INFINITY), 1, 1, 1)
                    .document(3, 1)
                    .term(b"paris", &[(3, 1)]),
                "not both finite",
            ),
        ];
        for (case, spelled, reason) in cases {
            let image = crafted(&["Doc:body"], vec![spelled]);
            let (read, restored) = read_back(&image, &["Doc:body"]);
            let error = read.expect_err(case);
            let message = error.to_string();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
            assert!(
                message.contains("TextIndex")
                    && message.contains("Doc:body")
                    && message.contains(reason),
                "{case}: {message}"
            );
            let index = restored[0].1.read();
            assert!(
                index.is_empty() && index.term_count() == 0,
                "{case}: nothing is restored"
            );
        }
    }

    #[test]
    fn the_same_postings_write_the_same_bytes_whatever_the_insert_history() {
        let documents: Vec<(u64, String)> = (1..=60).map(|seed| (seed, document(seed))).collect();
        let forward = index_of(BM25Config::default(), documents.clone());
        // Inserted backwards, then every third document indexed again, which
        // moves it to the end of its lists.
        let mut backward = InvertedIndex::new(BM25Config::default());
        for (node, text) in documents.iter().rev() {
            backward.insert(NodeId::new(*node), text);
        }
        for (node, text) in documents.iter().filter(|(node, _)| node % 3 == 0) {
            backward.insert(NodeId::new(*node), text);
        }
        assert_ne!(
            backward.snapshot(),
            forward.read().snapshot(),
            "the two indexes hold their lists in different orders"
        );
        let first = chunks(&written(vec![("Doc:body".to_string(), forward)], TINY));
        assert_eq!(
            chunks(&written(
                vec![("Doc:body".to_string(), Arc::new(RwLock::new(backward)))],
                TINY
            )),
            first,
            "the same postings, inserted in another order"
        );

        let (read, restored) = read_back(&image_of(TEXT_SECTION_VERSION, &first), &["Doc:body"]);
        read.unwrap();
        assert_eq!(
            chunks(&written(restored, TINY)),
            first,
            "a restored index writes the bytes it was read from"
        );
    }

    #[test]
    fn a_failed_read_keeps_the_configuration_the_index_had() {
        // The stream's parameters and its document come before the posting
        // list it lacks.
        let image = crafted(
            &["Doc:body"],
            vec![Spelled::header((1.9, 0.3), 1, 1, 1).document(3, 1)],
        );
        let shell = Arc::new(RwLock::new(InvertedIndex::new(BM25Config {
            k1: 0.3,
            b: 0.88,
        })));
        let source = image.section_source(SectionType::TextIndex).unwrap();
        let read = TextIndexSection::new(vec![("Doc:body".to_string(), Arc::clone(&shell))])
            .read_from(&*source);
        assert!(read.is_err(), "the stream ends before its posting list");
        let shell = shell.read();
        assert!(
            shell.is_empty() && shell.term_count() == 0,
            "nothing is restored"
        );
        assert_eq!(
            (shell.config().k1, shell.config().b),
            (0.3, 0.88),
            "the configuration the index had, not the stream's"
        );
    }

    /// A sink that refuses its third chunk.
    struct RefusesThird {
        image: MemoryImage,
        written: usize,
    }

    impl SectionSink for RefusesThird {
        fn write_chunk(&mut self, meta: ChunkMeta, bytes: &[u8]) -> Result<()> {
            self.written += 1;
            if self.written == 3 {
                return Err(Error::Io(io::Error::other("the disk of Jules is full")));
            }
            self.image.write_chunk(meta, bytes)
        }
    }

    #[test]
    fn a_sink_error_fails_the_write_with_its_own_error() {
        // The sink refuses the third chunk: with 300 documents a piece of a
        // batch Doc:body writes during its visit; with 40 a piece of the bytes
        // Doc:body gathered until its stream finishes; with 3 the last (and
        // only) piece of Note:text (Doc:body's stream is one piece then).
        for (documents, refused) in [(300, "Doc:body"), (40, "Doc:body"), (3, "Note:text")] {
            let [doc, note] = two_indexes(documents);
            let section = with_chunk_caps(TINY, || TextIndexSection::new(vec![doc, note]));
            let mut sink = RefusesThird {
                image: MemoryImage::new(),
                written: 0,
            };
            sink.image
                .begin_section(SectionType::TextIndex, TEXT_SECTION_VERSION)
                .unwrap();
            let error = section.write_to(&mut sink).unwrap_err();
            let message = error.to_string();
            assert!(matches!(error, Error::Io(_)), "{documents}: {error:?}");
            assert!(
                message.contains("the disk of Jules is full") && message.contains(refused),
                "{documents}: {message}"
            );
            assert_eq!(
                sink.written, 3,
                "{documents}: nothing is written after the refusal"
            );
        }
    }

    #[test]
    fn a_metadata_length_past_the_limit_is_refused_before_it_is_allocated() {
        // Layout 1 and cap 0, then: one key claiming 4 GiB of bytes behind
        // the u32 marker 0xFC, which the decode limit refuses before reading;
        // a key list claiming 2^60 keys behind the u64 marker 0xFD, which
        // grows only as keys arrive and ends with the bytes.
        let huge_key = vec![1, 0, 1, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut huge_list = vec![1, 0, 0xFD];
        huge_list.extend_from_slice(&(1u64 << 60).to_le_bytes());
        for (case, meta, reason) in [
            ("a key", huge_key, "LimitExceeded"),
            ("the key list", huge_list, "UnexpectedEnd"),
        ] {
            let image = image_of(TEXT_SECTION_VERSION, &[(ChunkMeta::meta(), meta)]);
            let error = read_back(&image, &["Doc:body"]).0.unwrap_err();
            let message = error.to_string();
            assert!(
                matches!(error, Error::Serialization(_))
                    && message.contains("does not decode")
                    && message.contains(reason),
                "{case}: {message}"
            );
        }
    }

    #[test]
    fn two_indexes_with_one_key_are_not_written() {
        let [doc, _] = two_indexes(3);
        let twice = TextIndexSection::new(vec![doc.clone(), doc]);
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::TextIndex, TEXT_SECTION_VERSION)
            .unwrap();
        let error = twice.write_to(&mut image).unwrap_err();
        assert!(
            matches!(error, Error::Internal(_)) && error.to_string().contains("Doc:body"),
            "{error:?}"
        );
        assert_eq!(image.chunk_count(), 0, "not even the metadata chunk");
    }

    #[test]
    fn a_posting_list_that_does_not_hold_its_count_is_not_written() {
        for (announced, held) in [(3, 1), (1, 3)] {
            let mut image = MemoryImage::new();
            image
                .begin_section(SectionType::TextIndex, TEXT_SECTION_VERSION)
                .unwrap();
            let mut writer = PostingsWriter::new(&mut image, 0, ChunkCaps::DEFAULT);
            let postings: Vec<(NodeId, u32)> =
                (0..held).map(|node| (NodeId::new(node), 1)).collect();
            let error = writer
                .posting_list("mia", announced, &mut postings.into_iter())
                .unwrap_err();
            assert!(
                matches!(error, Error::Internal(_)) && error.to_string().contains("mia"),
                "announced {announced}, held {held}: {error:?}"
            );
        }
    }

    #[test]
    fn text_section_round_trip() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "rust graph database");
        index.insert(NodeId::new(2), "python web framework");
        index.insert(NodeId::new(3), "rust systems programming");

        let index_arc = Arc::new(RwLock::new(index));
        let section = TextIndexSection::new(vec![(
            "Item:description".to_string(),
            Arc::clone(&index_arc),
        )]);

        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty(), "bytes is empty");

        // Restore into a fresh index
        let fresh = InvertedIndex::new(BM25Config::default());
        let fresh_arc = Arc::new(RwLock::new(fresh));
        let mut section2 =
            TextIndexSection::new(vec![("Item:description".to_string(), fresh_arc.clone())]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(fresh_arc.read().len(), 3);
        // 8 unique terms: rust, graph, database, python, web, framework, systems, programming
        assert!(fresh_arc.read().term_count() > 0);
    }

    #[test]
    fn text_section_empty() {
        let section = TextIndexSection::new(vec![]);
        let bytes = section.serialize().expect("serialize should succeed");

        let mut section2 = TextIndexSection::new(vec![]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");
    }

    #[test]
    fn text_section_type() {
        let section = TextIndexSection::new(vec![]);
        assert_eq!(section.section_type(), SectionType::TextIndex);
        assert_eq!(section.version(), TEXT_SECTION_VERSION);
    }

    #[test]
    fn text_section_dirty_tracking() {
        let section = TextIndexSection::new(vec![]);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }
}
