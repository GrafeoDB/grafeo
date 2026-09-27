//! RDF section serializer for the `.grafeo` container format.
//!
//! Implements the [`Section`] trait for RDF triple data (triples, named graphs).
//! Uses a block-based binary format (v2) with a shared string table for
//! efficient serialization and CRC integrity checking.
//!
//! # Layout
//!
//! ```text
//! [Header 32B: magic "RDFB", version, triple_count, graph_count]
//! [StringTable: deduplicated N-Triples strings]
//! [TripleData: packed (subject_idx, predicate_idx, object_idx) * triple_count]
//! [NamedGraph 0: name_idx + embedded sub-section]
//! [NamedGraph 1: ...]
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::codec::limits::checked_u32;
use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::utils::error::{Error, Result};

use crate::graph::rdf::{RdfStore, Term, Triple};

/// Current RDF section format version (v2 = block-based).
const RDF_SECTION_VERSION: u8 = 2;

/// Magic bytes for the RDF block format.
const RDF_BLOCK_MAGIC: [u8; 4] = *b"RDFB";
/// Header: magic(4) + version(1) + flags(1) + triple_count(4) + graph_count(4) + pad(18) = 32
const HEADER_SIZE: usize = 32;

// ── String table (shared with LPG block module pattern) ────────────

struct StringTableBuilder {
    strings: Vec<String>,
    index: hashbrown::HashMap<String, u32>,
}

impl StringTableBuilder {
    fn new() -> Self {
        Self {
            strings: Vec::new(),
            index: hashbrown::HashMap::new(),
        }
    }

    fn intern(&mut self, s: &str) -> u32 {
        if let Some(&idx) = self.index.get(s) {
            return idx;
        }
        // More than u32::MAX strings cannot be held in memory in practice; if it
        // ever happens the index saturates and `serialize` rejects the table.
        let idx = u32::try_from(self.strings.len()).unwrap_or(u32::MAX);
        self.strings.push(s.to_owned());
        self.index.insert(s.to_owned(), idx);
        idx
    }

    /// # Errors
    ///
    /// Fails if a count, length or offset does not fit the format's `u32` fields.
    fn serialize(&self) -> Result<Vec<u8>> {
        let count = checked_u32(self.strings.len(), "RDF string table count")?;
        let mut packed = Vec::new();
        let mut offsets = Vec::with_capacity(self.strings.len());
        for s in &self.strings {
            offsets.push(checked_u32(packed.len(), "RDF string table size")?);
            let bytes = s.as_bytes();
            packed.extend_from_slice(&checked_u32(bytes.len(), "RDF string length")?.to_le_bytes());
            packed.extend_from_slice(bytes);
        }

        let mut buf = Vec::with_capacity(4 + offsets.len() * 4 + packed.len());
        buf.extend_from_slice(&count.to_le_bytes());
        for off in &offsets {
            buf.extend_from_slice(&off.to_le_bytes());
        }
        buf.extend_from_slice(&packed);
        Ok(buf)
    }
}

struct StringTableReader<'a> {
    data: &'a [u8],
    count: u32,
    offsets_start: usize,
    packed_start: usize,
}

impl<'a> StringTableReader<'a> {
    fn new(data: &'a [u8]) -> Option<Self> {
        if data.len() < 4 {
            return None;
        }
        let count = u32::from_le_bytes(data[0..4].try_into().ok()?);
        let offsets_start = 4;
        let packed_start = offsets_start + (count as usize) * 4;
        if data.len() < packed_start {
            return None;
        }
        Some(Self {
            data,
            count,
            offsets_start,
            packed_start,
        })
    }

    fn get(&self, index: u32) -> Option<&'a str> {
        if index >= self.count {
            return None;
        }
        let off_pos = self.offsets_start + (index as usize) * 4;
        let rel_offset =
            u32::from_le_bytes(self.data[off_pos..off_pos + 4].try_into().ok()?) as usize;
        let abs_offset = self.packed_start + rel_offset;
        if abs_offset + 4 > self.data.len() {
            return None;
        }
        let len =
            u32::from_le_bytes(self.data[abs_offset..abs_offset + 4].try_into().ok()?) as usize;
        let str_start = abs_offset + 4;
        if str_start + len > self.data.len() {
            return None;
        }
        std::str::from_utf8(&self.data[str_start..str_start + len]).ok()
    }
}

// ── Serialization ──────────────────────────────────────────────────

fn write_rdf_blocks(store: &RdfStore, named_graphs: &[(String, Arc<RdfStore>)]) -> Result<Vec<u8>> {
    let mut strings = StringTableBuilder::new();

    // Phase 1: intern strings for this level's triples and graph names.
    // Named graph triple strings are NOT interned here: each named graph
    // serializes its own string table via the recursive write_rdf_blocks
    // call, so interning them at the top level would only waste space.
    let triples: Vec<_> = store.triples().into_iter().collect();
    for t in &triples {
        strings.intern(&t.subject().to_string());
        strings.intern(&t.predicate().to_string());
        strings.intern(&t.object().to_string());
    }
    for (name, _graph) in named_graphs {
        strings.intern(name);
    }

    // Serialize triple data: [subject_idx:u32][predicate_idx:u32][object_idx:u32] per triple
    let mut triple_data = Vec::with_capacity(triples.len() * 12);
    for t in &triples {
        let s_idx = strings.intern(&t.subject().to_string());
        let p_idx = strings.intern(&t.predicate().to_string());
        let o_idx = strings.intern(&t.object().to_string());
        triple_data.extend_from_slice(&s_idx.to_le_bytes());
        triple_data.extend_from_slice(&p_idx.to_le_bytes());
        triple_data.extend_from_slice(&o_idx.to_le_bytes());
    }
    let triple_crc = crc32fast::hash(&triple_data);

    // Serialize named graphs (each as a nested section)
    let mut graph_blocks: Vec<(u32, Vec<u8>)> = Vec::new(); // (name_idx, data)
    for (name, graph) in named_graphs {
        let name_idx = strings.intern(name);
        let nested = write_rdf_blocks(graph, &[])?;
        graph_blocks.push((name_idx, nested));
    }

    // Re-serialize string table (may have grown during triple interning)
    let st_data = strings.serialize()?;
    let st_crc = crc32fast::hash(&st_data);

    // Calculate total size
    // Header: 32 bytes
    // String table: 4 (len) + st_data.len() + 4 (crc)
    // Triple data: 4 (len) + triple_data.len() + 4 (crc)
    // Named graphs: for each, 4 (name_idx) + 4 (len) + data + 4 (crc)
    let mut total = HEADER_SIZE;
    total += 4 + st_data.len() + 4; // string table block
    total += 4 + triple_data.len() + 4; // triple data block
    for (_, data) in &graph_blocks {
        total += 4 + 4 + data.len() + 4; // name_idx + len + data + crc
    }

    let mut buf = Vec::with_capacity(total);

    // Write header
    buf.extend_from_slice(&RDF_BLOCK_MAGIC);
    buf.push(RDF_SECTION_VERSION);
    buf.push(0); // flags
    buf.extend_from_slice(&checked_u32(triples.len(), "RDF triple count")?.to_le_bytes());
    buf.extend_from_slice(&checked_u32(named_graphs.len(), "RDF named graph count")?.to_le_bytes());
    // Pad to 32 bytes
    buf.extend_from_slice(&[0u8; 18]);
    debug_assert_eq!(buf.len(), HEADER_SIZE);

    // Write string table block: [length:u32][data][crc:u32]
    buf.extend_from_slice(
        &checked_u32(st_data.len(), "RDF string table block length")?.to_le_bytes(),
    );
    buf.extend_from_slice(&st_data);
    buf.extend_from_slice(&st_crc.to_le_bytes());

    // Write triple data block: [length:u32][data][crc:u32]
    buf.extend_from_slice(
        &checked_u32(triple_data.len(), "RDF triple block length")?.to_le_bytes(),
    );
    buf.extend_from_slice(&triple_data);
    buf.extend_from_slice(&triple_crc.to_le_bytes());

    // Write named graph blocks: [name_idx:u32][length:u32][data][crc:u32]
    for (name_idx, data) in &graph_blocks {
        let crc = crc32fast::hash(data);
        buf.extend_from_slice(&name_idx.to_le_bytes());
        buf.extend_from_slice(
            &checked_u32(data.len(), "RDF named graph block length")?.to_le_bytes(),
        );
        buf.extend_from_slice(data);
        buf.extend_from_slice(&crc.to_le_bytes());
    }

    Ok(buf)
}

// ── Deserialization ────────────────────────────────────────────────

fn read_rdf_blocks(data: &[u8], store: &RdfStore) -> Result<()> {
    if data.len() < HEADER_SIZE {
        return Err(Error::Serialization(
            "RDF block section too short for header".to_string(),
        ));
    }
    if data[0..4] != RDF_BLOCK_MAGIC {
        return Err(Error::Serialization(
            "invalid RDF block magic bytes".to_string(),
        ));
    }
    // data[4] = version, data[5] = flags (reserved for future use)
    let triple_count = u32::from_le_bytes(data[6..10].try_into().unwrap()) as usize;
    let graph_count = u32::from_le_bytes(data[10..14].try_into().unwrap()) as usize;

    let mut pos = HEADER_SIZE;

    // Read string table block
    if pos + 4 > data.len() {
        return Err(Error::Serialization(
            "RDF section truncated at string table length".to_string(),
        ));
    }
    let st_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + st_len + 4 > data.len() {
        return Err(Error::Serialization(
            "RDF section truncated at string table data".to_string(),
        ));
    }
    let st_data = &data[pos..pos + st_len];
    pos += st_len;
    let expected_crc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let actual_crc = crc32fast::hash(st_data);
    if expected_crc != actual_crc {
        return Err(Error::Serialization(format!(
            "RDF string table CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"
        )));
    }
    pos += 4;

    let strings = StringTableReader::new(st_data)
        .ok_or_else(|| Error::Serialization("invalid RDF string table".to_string()))?;

    // Read triple data block
    if pos + 4 > data.len() {
        return Err(Error::Serialization(
            "RDF section truncated at triple data length".to_string(),
        ));
    }
    let td_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
    pos += 4;
    if pos + td_len + 4 > data.len() {
        return Err(Error::Serialization(
            "RDF section truncated at triple data".to_string(),
        ));
    }
    let triple_data = &data[pos..pos + td_len];
    pos += td_len;
    let expected_crc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let actual_crc = crc32fast::hash(triple_data);
    if expected_crc != actual_crc {
        return Err(Error::Serialization(format!(
            "RDF triple data CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"
        )));
    }
    pos += 4;

    // Parse triples
    let mut tp = 0;
    for _ in 0..triple_count {
        if tp + 12 > triple_data.len() {
            return Err(Error::Serialization(
                "RDF triple data truncated".to_string(),
            ));
        }
        let s_idx = u32::from_le_bytes(triple_data[tp..tp + 4].try_into().unwrap());
        tp += 4;
        let p_idx = u32::from_le_bytes(triple_data[tp..tp + 4].try_into().unwrap());
        tp += 4;
        let o_idx = u32::from_le_bytes(triple_data[tp..tp + 4].try_into().unwrap());
        tp += 4;

        let s_str = strings
            .get(s_idx)
            .ok_or_else(|| Error::Serialization(format!("invalid subject string index {s_idx}")))?;
        let p_str = strings.get(p_idx).ok_or_else(|| {
            Error::Serialization(format!("invalid predicate string index {p_idx}"))
        })?;
        let o_str = strings
            .get(o_idx)
            .ok_or_else(|| Error::Serialization(format!("invalid object string index {o_idx}")))?;

        if let (Some(s), Some(p), Some(o)) = (
            Term::from_ntriples(s_str),
            Term::from_ntriples(p_str),
            Term::from_ntriples(o_str),
        ) {
            store.insert(Triple::new(s, p, o));
        }
    }

    // Read named graphs
    for _ in 0..graph_count {
        if pos + 8 > data.len() {
            return Err(Error::Serialization(
                "RDF section truncated at named graph header".to_string(),
            ));
        }
        let name_idx = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let graph_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + graph_len + 4 > data.len() {
            return Err(Error::Serialization(
                "RDF section truncated at named graph data".to_string(),
            ));
        }
        let graph_data = &data[pos..pos + graph_len];
        pos += graph_len;
        let expected_crc = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        let actual_crc = crc32fast::hash(graph_data);
        if expected_crc != actual_crc {
            return Err(Error::Serialization(format!(
                "RDF named graph CRC mismatch: expected {expected_crc:08x}, got {actual_crc:08x}"
            )));
        }
        pos += 4;

        let graph_name = strings.get(name_idx).ok_or_else(|| {
            Error::Serialization(format!("invalid graph name string index {name_idx}"))
        })?;

        store.create_graph(graph_name);
        if let Some(graph_store) = store.graph(graph_name) {
            read_rdf_blocks(graph_data, &graph_store)?;
        }
    }

    Ok(())
}

// ── Section implementation ──────────────────────────────────────────

/// RDF store section for the `.grafeo` container.
pub struct RdfStoreSection {
    store: Arc<RdfStore>,
    dirty: AtomicBool,
}

impl RdfStoreSection {
    /// Create a new RDF section wrapping the given store.
    pub fn new(store: Arc<RdfStore>) -> Self {
        Self {
            store,
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }

    /// Access the underlying store.
    #[must_use]
    pub fn store(&self) -> &Arc<RdfStore> {
        &self.store
    }
}

impl Section for RdfStoreSection {
    fn section_type(&self) -> SectionType {
        SectionType::RdfStore
    }

    fn version(&self) -> u8 {
        RDF_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let named_graphs: Vec<(String, Arc<RdfStore>)> = self
            .store
            .graph_names()
            .into_iter()
            .filter_map(|name| {
                self.store
                    .graph(&name)
                    .map(|graph| (name, Arc::clone(&graph)))
            })
            .collect();

        write_rdf_blocks(&self.store, &named_graphs)
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        read_rdf_blocks(data, &self.store)
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.store.len() * 200
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rdf_section_round_trip() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/alix"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            Term::iri("http://example.org/gus"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Gus"),
        ));

        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty());
        assert_eq!(&bytes[0..4], b"RDFB");

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(section2.store().len(), 2);
    }

    #[test]
    fn rdf_section_type() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(store);
        assert_eq!(section.section_type(), SectionType::RdfStore);
    }

    #[test]
    fn rdf_section_version() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(store);
        assert_eq!(section.version(), RDF_SECTION_VERSION);
    }

    #[test]
    fn rdf_section_dirty_tracking() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(store);

        assert!(!section.is_dirty(), "new section should be clean");

        section.mark_dirty();
        assert!(
            section.is_dirty(),
            "section should be dirty after mark_dirty"
        );

        section.mark_clean();
        assert!(
            !section.is_dirty(),
            "section should be clean after mark_clean"
        );
    }

    #[test]
    fn rdf_section_memory_usage() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/vincent"),
            Term::iri("http://xmlns.com/foaf/0.1/knows"),
            Term::iri("http://example.org/jules"),
        ));
        let section = RdfStoreSection::new(store);
        let usage = section.memory_usage();
        assert_eq!(usage, 200);
    }

    #[test]
    fn rdf_section_named_graph_round_trip() {
        let store = Arc::new(RdfStore::new());

        store.insert(Triple::new(
            Term::iri("http://example.org/mia"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Mia"),
        ));

        store.create_graph("http://example.org/graph/butch");
        if let Some(named) = store.graph("http://example.org/graph/butch") {
            named.insert(Triple::new(
                Term::iri("http://example.org/butch"),
                Term::iri("http://xmlns.com/foaf/0.1/name"),
                Term::literal("Butch"),
            ));
            named.insert(Triple::new(
                Term::iri("http://example.org/butch"),
                Term::iri("http://xmlns.com/foaf/0.1/knows"),
                Term::iri("http://example.org/mia"),
            ));
        }

        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize named graphs");

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize named graphs");

        assert_eq!(section2.store().len(), 1);

        let names = section2.store().graph_names();
        assert_eq!(names.len(), 1);
        assert_eq!(names[0], "http://example.org/graph/butch");

        let named = section2
            .store()
            .graph("http://example.org/graph/butch")
            .expect("named graph should exist");
        assert_eq!(named.len(), 2);
    }

    #[test]
    fn rdf_section_deserialize_invalid_data() {
        let store = Arc::new(RdfStore::new());
        let mut section = RdfStoreSection::new(store);
        let bad_bytes = &[0xFF, 0xFE, 0xFD, 0x00, 0x01];
        let result = section.deserialize(bad_bytes);
        assert!(
            result.is_err(),
            "corrupted data should fail deserialization"
        );
    }

    #[test]
    fn rdf_section_empty_store_round_trip() {
        let store = Arc::new(RdfStore::new());
        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().expect("serialize empty store");

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2
            .deserialize(&bytes)
            .expect("deserialize empty store");
        assert_eq!(section2.store().len(), 0);
        assert_eq!(section2.memory_usage(), 0);
    }

    #[test]
    fn rdf_section_crc_corruption_detected() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(
            Term::iri("http://example.org/test"),
            Term::iri("http://example.org/pred"),
            Term::literal("value"),
        ));

        let section = RdfStoreSection::new(Arc::clone(&store));
        let mut bytes = section.serialize().unwrap();

        // Corrupt a byte near the end (triple data area)
        let last = bytes.len() - 5;
        bytes[last] ^= 0xFF;

        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        assert!(section2.deserialize(&bytes).is_err());
    }

    #[test]
    fn rdf_section_string_deduplication() {
        let store = Arc::new(RdfStore::new());
        let pred = Term::iri("http://xmlns.com/foaf/0.1/name");
        // Same predicate used in multiple triples
        for i in 0..100 {
            store.insert(Triple::new(
                Term::iri(format!("http://example.org/node{i}")),
                pred.clone(),
                Term::literal(format!("Name{i}")),
            ));
        }

        let section = RdfStoreSection::new(Arc::clone(&store));
        let bytes = section.serialize().unwrap();

        // Verify round-trip
        let store2 = Arc::new(RdfStore::new());
        let mut section2 = RdfStoreSection::new(store2);
        section2.deserialize(&bytes).unwrap();
        assert_eq!(section2.store().len(), 100);
    }
}
