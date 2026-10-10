//! The RDF section in chunks (section version 3).
//!
//! The section starts with its metadata chunk ([`ChunkMeta::meta`], laid out
//! by [`encode_rdf_meta`]): the caps the section was written with and its
//! graphs, each with its name and its number of triples. Graph 0 is the
//! default graph, with an empty name, then come the named graphs in name
//! order; a graph's id is its position, so ids are this checkpoint's (and a
//! named graph with an empty name, which a store can hold, stays apart from
//! the default graph).
//!
//! Then, per graph in id order, its triple table. A row is a triple; rows
//! count from 0 in the order [`RdfStore::for_each_triple`] visits them (by
//! subject, predicate and object, so the same triples give the same bytes),
//! in row groups of `max_rows` rows. One [`RowsChunker`] per group writes
//! three columns, [`COLUMN_SUBJECT`], [`COLUMN_PREDICATE`] and
//! [`COLUMN_OBJECT`], each value the term's N-Triples string (its
//! `Display`); a chunk of strings is a `Dict` chunk, the dictionary of its
//! own terms, so the section has no term table of its own. The three columns
//! share their chunk boundaries: their chunks come as three in a row with
//! one range, every row with a value, at most `max_rows` rows and
//! `max_bytes` bytes each (a larger value in a chunk of its own). A graph
//! without triples writes no chunk; the metadata lists it, so a load creates
//! it.
//!
//! Memory: the writer holds the open chunk of each of the three columns and
//! one reference per triple of every graph: it takes them (each graph's
//! under its triple set's read lock, released before writing) and sorts them
//! before the metadata chunk, whose triple counts they give. The reader
//! holds the metadata, the three chunks of one range and their terms.
//!
//! The reader refuses, naming the graph and rows: a first chunk other than
//! the metadata chunk, or a second one; metadata of another layout, with
//! caps of zero or of more rows than the format's row cap (65,536), a named
//! default graph, named graphs out of name order, or a count or name length
//! past the bytes left; a chunk of another kind than `Column`; a graph the
//! metadata does not list, or whose chunks come after those of a later
//! graph; a range that does not start where the graph's
//! rows before it end (from 0), holds more than `max_rows` rows, crosses a
//! row group or reaches past the graph's triple count; a subject chunk not
//! followed by the predicate and object chunks of its range; a chunk with
//! epochs or a row without a value; a value that is not a string or not an
//! N-Triples term ([`Term::from_ntriples`]); a graph whose rows stop short
//! of its triple count.

use std::collections::HashMap;
use std::sync::Arc;

use grafeo_common::storage::section::SectionSource;
use grafeo_common::storage::{ChunkCaps, ChunkKind, ChunkMeta, ChunkNamespace, SectionSink};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, Result};

use super::{RdfStore, Term, Triple};
use crate::codec::column_chunk::decode_column_chunk_bytes;
use crate::codec::{ChunkColumn, RowsChunker};

/// The RDF section's version: chunks, as this module writes them.
pub(crate) const RDF_SECTION_VERSION: u8 = 3;
/// The layout byte of the metadata chunk.
pub(crate) const RDF_META_LAYOUT: u8 = 1;
/// The column of the triples' subjects.
pub(crate) const COLUMN_SUBJECT: u32 = 0;
/// The column of the triples' predicates.
pub(crate) const COLUMN_PREDICATE: u32 = 1;
/// The column of the triples' objects.
pub(crate) const COLUMN_OBJECT: u32 = 2;
/// The three columns with their names, in the order their chunks come.
const COLUMNS: [(u32, &str); 3] = [
    (COLUMN_SUBJECT, "subject"),
    (COLUMN_PREDICATE, "predicate"),
    (COLUMN_OBJECT, "object"),
];
/// The fewest bytes a graph takes in the metadata chunk: its name length
/// and its triple count.
const GRAPH_LEAST_BYTES: usize = 4 + 8;

/// The metadata chunk of an RDF section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RdfMeta {
    /// [`RDF_META_LAYOUT`].
    pub layout: u8,
    /// The rows per row group and chunk the section was written with.
    pub max_rows: u32,
    /// The byte cap the section was written with.
    pub max_bytes: u32,
    /// Graph `i` has graph id `i`: graph 0 is the default graph, with an
    /// empty name, then the named graphs in name order.
    pub graphs: Vec<GraphMeta>,
}

/// One graph of an RDF section.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GraphMeta {
    /// The graph's name; empty for the default graph.
    pub name: String,
    /// The graph's number of triples: its rows are `[0, triples)`.
    pub triples: u64,
}

/// Writes the metadata chunk, then per graph (in id order) its triples.
///
/// # Errors
///
/// Returns [`Error::InvalidValue`] for caps [`ChunkCaps::validate`]
/// refuses; [`Error::Serialization`] for more graphs or a longer name than
/// the metadata chunk holds; the row chunker's and the sink's errors.
pub(crate) fn write_rdf_chunks(
    store: &RdfStore,
    caps: ChunkCaps,
    sink: &mut dyn SectionSink,
) -> Result<()> {
    caps.validate()?;
    let mut names = store.graph_names();
    names.sort_unstable();
    let mut graphs = vec![(String::new(), store.sorted_triples())];
    for name in names {
        // A graph dropped since the names were read is left out.
        if let Some(graph) = store.graph(&name) {
            graphs.push((name, graph.sorted_triples()));
        }
    }
    let meta = RdfMeta {
        layout: RDF_META_LAYOUT,
        max_rows: caps.max_rows,
        max_bytes: caps.max_bytes,
        graphs: graphs
            .iter()
            .map(|(name, triples)| GraphMeta {
                name: name.clone(),
                // A `usize` count fits a u64 on every supported target.
                triples: u64::try_from(triples.len()).unwrap_or(u64::MAX),
            })
            .collect(),
    };
    sink.write_chunk(ChunkMeta::meta(), &encode_rdf_meta(&meta)?)?;
    // `encode_rdf_meta` refused more graphs than a u32 counts.
    for (graph_id, (_, triples)) in (0u32..).zip(&graphs) {
        write_triple_table(triples, graph_id, caps, sink)?;
    }
    Ok(())
}

/// Writes `triples` as the rows of graph `graph_id`, one row chunker per row
/// group.
fn write_triple_table(
    triples: &[Arc<Triple>],
    graph_id: u32,
    caps: ChunkCaps,
    sink: &mut dyn SectionSink,
) -> Result<()> {
    let group_rows = u64::from(caps.max_rows);
    let mut chunker: Option<RowsChunker> = None;
    for (row, triple) in (0u64..).zip(triples) {
        if row.is_multiple_of(group_rows)
            && let Some(full) = chunker.take()
        {
            full.finish(sink)?;
        }
        let open = chunker.get_or_insert_with(|| {
            let columns = COLUMNS
                .iter()
                .map(|&(column_id, _)| ChunkColumn {
                    kind: ChunkKind::Column,
                    namespace: ChunkNamespace::Section,
                    column_id,
                })
                .collect();
            RowsChunker::new(graph_id, columns, row, caps)
        });
        let cells = [triple.subject(), triple.predicate(), triple.object()]
            .into_iter()
            .map(|term| Some((Value::from(term.to_string()), 0)))
            .collect();
        open.push(sink, row, cells)?;
    }
    match chunker {
        Some(last) => last.finish(sink),
        None => Ok(()),
    }
}

/// Encodes a metadata chunk. Little-endian:
///
/// | Field | Encoding |
/// | --- | --- |
/// | layout | u8, [`RDF_META_LAYOUT`] |
/// | max_rows, max_bytes | u32 each |
/// | graphs | a count u32, then per graph its name (a length u32 and UTF-8) and its triple count u64 |
///
/// Nothing in the layout limits its size: a reader checks every count and
/// length against the bytes left, so the chunk lists as many graphs as the
/// store has, with names as long as they are.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when there are more than `u32::MAX`
/// graphs or a name is longer than `u32::MAX` bytes.
pub(crate) fn encode_rdf_meta(meta: &RdfMeta) -> Result<Vec<u8>> {
    let mut out = vec![meta.layout];
    out.extend_from_slice(&meta.max_rows.to_le_bytes());
    out.extend_from_slice(&meta.max_bytes.to_le_bytes());
    let count = u32::try_from(meta.graphs.len()).map_err(|_| {
        Error::Serialization(format!(
            "RDF metadata chunk: {} graphs, more than the {} a section holds",
            meta.graphs.len(),
            u32::MAX
        ))
    })?;
    out.extend_from_slice(&count.to_le_bytes());
    for graph in &meta.graphs {
        let length = u32::try_from(graph.name.len()).map_err(|_| {
            Error::Serialization(format!(
                "RDF metadata chunk: a graph name of {} bytes, longer than the {} a section \
                 holds",
                graph.name.len(),
                u32::MAX
            ))
        })?;
        out.extend_from_slice(&length.to_le_bytes());
        out.extend_from_slice(graph.name.as_bytes());
        out.extend_from_slice(&graph.triples.to_le_bytes());
    }
    Ok(out)
}

/// Decodes a metadata chunk of [`encode_rdf_meta`]'s layout. Every count and
/// length is checked against the bytes left before anything is allocated.
///
/// # Errors
///
/// Returns [`Error::Corruption`] for another layout than
/// [`RDF_META_LAYOUT`], a count or name length past the bytes left (naming
/// the byte offset), a name that is not UTF-8, bytes after the metadata,
/// caps of zero or of more rows than the format's row cap, no graph, a named
/// graph 0, or named graphs that are not in strictly increasing name order.
pub(crate) fn decode_rdf_meta(bytes: &[u8]) -> Result<RdfMeta> {
    let mut reader = MetaReader { bytes, pos: 0 };
    let layout = reader.u8("layout")?;
    if layout != RDF_META_LAYOUT {
        return Err(reader.refuse(
            0,
            format!("layout {layout}, this build reads layout {RDF_META_LAYOUT}"),
        ));
    }
    let max_rows = reader.u32("max_rows")?;
    let max_bytes = reader.u32("max_bytes")?;
    let count = reader.count()?;
    let mut graphs = Vec::with_capacity(count);
    for _ in 0..count {
        graphs.push(GraphMeta {
            name: reader.name()?,
            triples: reader.u64("triple count")?,
        });
    }
    if reader.pos != bytes.len() {
        return Err(reader.refuse(
            reader.pos,
            format!("{} bytes after the metadata", bytes.len() - reader.pos),
        ));
    }

    let corrupt = |what: String| Error::corruption(format!("RDF metadata chunk: {what}"));
    let caps = ChunkCaps {
        max_rows,
        max_bytes,
    };
    caps.validate().map_err(|error| {
        corrupt(format!(
            "caps of {max_rows} rows and {max_bytes} bytes: {error}"
        ))
    })?;
    match graphs.first() {
        None => {
            return Err(corrupt(
                "lists no graph, where graph 0 is the default graph".to_string(),
            ));
        }
        Some(graph) if !graph.name.is_empty() => {
            return Err(corrupt(format!(
                "graph 0 is named {:?}, but the default graph has no name",
                graph.name
            )));
        }
        Some(_) => {}
    }
    for (graph_id, pair) in (2u32..).zip(graphs[1..].windows(2)) {
        if pair[0].name >= pair[1].name {
            return Err(corrupt(format!(
                "{} follows {}: named graphs come in name order, each once",
                describe_graph(graph_id, &pair[1].name),
                describe_graph(graph_id - 1, &pair[0].name)
            )));
        }
    }
    Ok(RdfMeta {
        layout,
        max_rows,
        max_bytes,
        graphs,
    })
}

/// Reads a metadata chunk from its first byte.
struct MetaReader<'b> {
    bytes: &'b [u8],
    pos: usize,
}

impl MetaReader<'_> {
    /// The error of what is wrong at byte `at`.
    fn refuse(&self, at: usize, what: String) -> Error {
        Error::corruption(format!("RDF metadata chunk, byte {at}: {what}"))
    }

    /// The bytes after the read position.
    fn left(&self) -> usize {
        self.bytes.len() - self.pos
    }

    /// The next `N` bytes.
    fn take<const N: usize>(&mut self, what: &str) -> Result<[u8; N]> {
        let taken: [u8; N] = self
            .bytes
            .get(self.pos..)
            .and_then(|rest| rest.get(..N))
            .and_then(|bytes| bytes.try_into().ok())
            .ok_or_else(|| self.refuse(self.pos, format!("the chunk ends inside its {what}")))?;
        self.pos += N;
        Ok(taken)
    }

    fn u8(&mut self, what: &str) -> Result<u8> {
        Ok(self.take::<1>(what)?[0])
    }

    fn u32(&mut self, what: &str) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take::<4>(what)?))
    }

    fn u64(&mut self, what: &str) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take::<8>(what)?))
    }

    /// The graph count, refused when the bytes left cannot hold that many
    /// graphs.
    fn count(&mut self) -> Result<usize> {
        let at = self.pos;
        let count = usize::try_from(self.u32("graph count")?).unwrap_or(usize::MAX);
        if count.saturating_mul(GRAPH_LEAST_BYTES) > self.left() {
            return Err(self.refuse(
                at,
                format!(
                    "{count} graphs, but only {} bytes are left for them",
                    self.left()
                ),
            ));
        }
        Ok(count)
    }

    /// A graph name: its length (checked against the bytes left) and UTF-8.
    fn name(&mut self) -> Result<String> {
        let at = self.pos;
        let length = usize::try_from(self.u32("graph name length")?).unwrap_or(usize::MAX);
        if length > self.left() {
            return Err(self.refuse(
                at,
                format!(
                    "a graph name of {length} bytes, but only {} bytes are left",
                    self.left()
                ),
            ));
        }
        let text = &self.bytes[self.pos..self.pos + length];
        let name = std::str::from_utf8(text)
            .map_err(|error| {
                self.refuse(self.pos, format!("a graph name that is not UTF-8: {error}"))
            })?
            .to_string();
        self.pos += length;
        Ok(name)
    }
}

/// How a graph is named in errors.
fn describe_graph(graph_id: u32, name: &str) -> String {
    if graph_id == 0 {
        "the default graph".to_string()
    } else {
        format!("graph {graph_id} {name:?}")
    }
}

/// Applies the chunks of an RDF section of version 3 to `store`, the three
/// chunks of one range at a time, with one `batch_insert` per range.
///
/// The graphs the metadata lists are created, also when they hold no
/// triple. A triple is stored as it was written, whatever kinds its terms
/// have.
///
/// # Errors
///
/// Returns [`Error::Corruption`] naming the graph and rows of what the
/// module documentation lists as refused, and any error from fetching a
/// chunk.
pub(crate) fn read_rdf_chunks(store: &RdfStore, source: &dyn SectionSource) -> Result<()> {
    let chunks = source.chunks();
    match chunks.first() {
        Some(first) if *first == ChunkMeta::meta() => {}
        Some(first) => {
            return Err(Error::corruption(format!(
                "RDF section: the first chunk is a {:?} chunk of graph {}, column {}, not the \
                 metadata chunk",
                first.kind, first.graph_id, first.column_id
            )));
        }
        None => {
            return Err(Error::corruption(
                "RDF section: no chunk, where the metadata chunk comes first",
            ));
        }
    }
    let meta = decode_rdf_meta(&source.fetch(0)?)?;
    let named: Vec<Arc<RdfStore>> = meta.graphs[1..]
        .iter()
        .map(|graph| store.graph_or_create(&graph.name))
        .collect();

    // The graph whose chunks come now, and the rows read of each graph.
    let mut graph_id = 0u32;
    let mut rows_read = vec![0u64; meta.graphs.len()];
    let mut index = 1;
    while index < chunks.len() {
        let head = chunks[index];
        let place = Place::of(&meta, &head);
        if head.kind == ChunkKind::Meta {
            return Err(Error::corruption(format!(
                "RDF section: chunk {index} is a second metadata chunk"
            )));
        }
        if head.kind != ChunkKind::Column {
            return Err(place.error(format!(
                "a {:?} chunk, where an RDF section holds only Column chunks after its metadata",
                head.kind
            )));
        }
        let graph_count = meta.graphs.len();
        let position = usize::try_from(head.graph_id)
            .ok()
            .filter(|&position| position < graph_count)
            .ok_or_else(|| {
                place.error(format!(
                    "the metadata chunk lists {graph_count} graphs, not graph {}",
                    head.graph_id
                ))
            })?;
        let target = match position {
            0 => store,
            _ => &*named[position - 1],
        };
        if head.graph_id < graph_id {
            return Err(place.error(format!(
                "after the chunks of graph {graph_id}: the chunks of a graph come together, \
                 graphs in id order"
            )));
        }
        graph_id = head.graph_id;
        place.check_range(
            &head,
            &meta,
            rows_read[position],
            meta.graphs[position].triples,
        )?;
        let triples = read_range(source, chunks, index, &place)?;
        target.batch_insert(triples);
        rows_read[position] = place.row_end;
        index += COLUMNS.len();
    }
    for ((graph_id, graph), read) in (0u32..).zip(&meta.graphs).zip(rows_read) {
        if read != graph.triples {
            return Err(Error::corruption(format!(
                "RDF section, {}: its chunks end at row {read}, where the metadata counts {} \
                 triples",
                describe_graph(graph_id, &graph.name),
                graph.triples
            )));
        }
    }
    Ok(())
}

/// Where a range of rows is, for errors.
struct Place {
    /// The graph, as [`describe_graph`] names it (by its id alone when the
    /// metadata does not list it).
    graph: String,
    /// The first row of the range.
    row_start: u64,
    /// The row after the range.
    row_end: u64,
}

impl Place {
    /// The range of `chunk`.
    fn of(meta: &RdfMeta, chunk: &ChunkMeta) -> Self {
        let graph = usize::try_from(chunk.graph_id)
            .ok()
            .and_then(|position| meta.graphs.get(position))
            .map_or_else(
                || format!("graph {}", chunk.graph_id),
                |graph| describe_graph(chunk.graph_id, &graph.name),
            );
        Self {
            graph,
            row_start: chunk.row_start,
            row_end: chunk.row_start.saturating_add(u64::from(chunk.row_count)),
        }
    }

    /// An error about the whole range.
    fn error(&self, what: impl std::fmt::Display) -> Error {
        Error::corruption(format!(
            "RDF section, {}, rows [{}, {}): {what}",
            self.graph, self.row_start, self.row_end
        ))
    }

    /// An error about row `row` of the range, in column `column`.
    fn row_error(&self, row: u64, column: &str, what: impl std::fmt::Display) -> Error {
        Error::corruption(format!(
            "RDF section, {}, row {row}, {column}: {what}",
            self.graph
        ))
    }

    /// Refuses a subject chunk, `head`, whose rows are not the next rows of
    /// its graph (`next_row` on), do not fit one row group or reach past the
    /// graph's `triples`.
    fn check_range(
        &self,
        head: &ChunkMeta,
        meta: &RdfMeta,
        next_row: u64,
        triples: u64,
    ) -> Result<()> {
        if head.column_id != COLUMN_SUBJECT {
            return Err(self.error(format!(
                "column {} where the subject column {COLUMN_SUBJECT} comes first",
                head.column_id
            )));
        }
        if head.row_count == 0 || head.row_count > meta.max_rows {
            return Err(self.error(format!(
                "{} rows, where a chunk holds 1 to {} rows (one row group)",
                head.row_count, meta.max_rows
            )));
        }
        let group_rows = u64::from(meta.max_rows);
        if self.row_start / group_rows != (self.row_end - 1) / group_rows {
            return Err(self.error(format!(
                "the rows cross the boundary of two row groups of {group_rows} rows"
            )));
        }
        if self.row_start != next_row {
            return Err(self.error(format!(
                "the rows do not start at row {next_row}, where the graph's rows before them end"
            )));
        }
        if self.row_end > triples {
            return Err(self.error(format!(
                "rows past the graph's {triples} triples, the metadata's count"
            )));
        }
        Ok(())
    }
}

/// Reads the subject, predicate and object chunks from chunk `index`, whose
/// range is `place`, as triples. A term that repeats in a chunk is read
/// once, and its rows share it.
fn read_range(
    source: &dyn SectionSource,
    chunks: &[ChunkMeta],
    index: usize,
    place: &Place,
) -> Result<Vec<Triple>> {
    let head = chunks[index];
    let mut columns: Vec<Vec<Term>> = Vec::with_capacity(COLUMNS.len());
    for (offset, &(column_id, column)) in COLUMNS.iter().enumerate() {
        let Some(chunk) = chunks.get(index + offset) else {
            let missing: Vec<&str> = COLUMNS[offset..].iter().map(|&(_, name)| name).collect();
            return Err(place.error(format!(
                "the subject chunk without its {} chunks",
                missing.join(" and ")
            )));
        };
        if chunk.kind != ChunkKind::Column || chunk.column_id != column_id {
            return Err(place.error(format!(
                "a {:?} chunk of column {} where the {column} column {column_id} comes",
                chunk.kind, chunk.column_id
            )));
        }
        if (chunk.graph_id, chunk.row_start, chunk.row_count)
            != (head.graph_id, head.row_start, head.row_count)
        {
            return Err(place.error(format!(
                "the {column} chunk holds graph {}, rows [{}, {})",
                chunk.graph_id,
                chunk.row_start,
                chunk.row_start.saturating_add(u64::from(chunk.row_count))
            )));
        }
        let bytes = source.fetch(index + offset)?;
        let decoded = decode_column_chunk_bytes(&bytes, chunk.codec, chunk.row_count)
            .map_err(|error| place.error(format!("the {column} chunk: {error}")))?;
        if decoded.epochs.is_some() {
            return Err(place.error(format!(
                "the {column} chunk carries epochs, which RDF chunks do not"
            )));
        }
        let no_value = |row: u32| {
            place.error(format!(
                "the {column} chunk has no value for row {}",
                place.row_start + u64::from(row)
            ))
        };
        let mut terms = Vec::with_capacity(decoded.values.len());
        // The terms read so far, by the address of their string: a `Dict`
        // chunk hands every row of one entry the same string.
        let mut read: HashMap<usize, Term> = HashMap::new();
        // The next row that must have a value: every row has one.
        let mut next = 0u32;
        for (row, value) in &decoded.values {
            if *row != next {
                return Err(no_value(next));
            }
            let absolute = place.row_start + u64::from(*row);
            let Value::String(text) = value else {
                return Err(place.row_error(
                    absolute,
                    column,
                    format!("{value:?}, where an N-Triples string belongs"),
                ));
            };
            let term = match read.get(&text.as_ptr().addr()) {
                Some(term) => term.clone(),
                None => {
                    let term = Term::from_ntriples(text)
                        .map_err(|error| place.row_error(absolute, column, error))?;
                    read.insert(text.as_ptr().addr(), term.clone());
                    term
                }
            };
            terms.push(term);
            next += 1;
        }
        if next != chunk.row_count {
            return Err(no_value(next));
        }
        columns.push(terms);
    }
    let [subjects, predicates, objects]: [Vec<Term>; 3] = columns
        .try_into()
        .map_err(|_| place.error("the three columns were read but not kept"))?;
    // Unchecked: the store holds what it was given, whatever the terms'
    // kinds, and a section gives it back as it was written.
    Ok(subjects
        .into_iter()
        .zip(predicates)
        .zip(objects)
        .map(|((subject, predicate), object)| Triple::new_unchecked(subject, predicate, object))
        .collect())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use bytes::Bytes;
    use grafeo_common::storage::section::{Section, SectionSource};
    use grafeo_common::storage::{
        ChunkCaps, ChunkKind, ChunkMeta, ImageSource, MemoryImage, SectionSink, SectionType,
    };
    use grafeo_common::testing::chunk_caps::with_chunk_caps;
    use grafeo_common::types::Value;
    use grafeo_common::utils::error::Result;

    use super::{
        GraphMeta, RDF_SECTION_VERSION, RdfMeta, decode_rdf_meta, encode_rdf_meta, write_rdf_chunks,
    };
    use crate::codec::column_chunk::{ChunkCodec, decode_column_chunk, encode_column_chunk};
    use crate::graph::rdf::term::unusual_terms;
    use crate::graph::rdf::{Literal, RdfStore, RdfStoreSection, Term, Triple};

    /// The chunks `write_rdf_chunks` writes for `store` with `caps`.
    fn chunks(store: &RdfStore, caps: ChunkCaps) -> Vec<(ChunkMeta, Bytes)> {
        let mut image = MemoryImage::new();
        image
            .begin_section(SectionType::RdfStore, RDF_SECTION_VERSION)
            .unwrap();
        write_rdf_chunks(store, caps, &mut image).unwrap();
        let source = image.section_source(SectionType::RdfStore).unwrap();
        (0..source.chunks().len())
            .map(|index| (source.chunks()[index], source.fetch(index).unwrap()))
            .collect()
    }

    /// The metadata chunk of `chunks`, decoded.
    fn meta(chunks: &[(ChunkMeta, Bytes)]) -> RdfMeta {
        assert_eq!(chunks[0].0, ChunkMeta::meta());
        decode_rdf_meta(&chunks[0].1).unwrap()
    }

    /// A graph of a metadata chunk.
    fn graph(name: &str, triples: u64) -> GraphMeta {
        GraphMeta {
            name: name.to_string(),
            triples,
        }
    }

    /// Writes `store` with `RdfStoreSection::with_caps` into a `MemoryImage`
    /// and reads it with `RdfStoreSection::read_from` into a new store.
    fn round_trip(store: &Arc<RdfStore>, caps: ChunkCaps) -> Arc<RdfStore> {
        let section = RdfStoreSection::with_caps(Arc::clone(store), caps);
        let image = MemoryImage::from_sections(&[&section]).unwrap();
        let source = image.section_source(SectionType::RdfStore).unwrap();
        assert_eq!(source.section_version(), 3);
        let back = Arc::new(RdfStore::new());
        RdfStoreSection::new(Arc::clone(&back))
            .read_from(&*source)
            .unwrap();
        back
    }

    /// Every graph's triples as sorted N-Triples lines, the graph name first;
    /// a named graph also as a line of its own, so empty graphs count.
    fn sorted(store: &RdfStore) -> Vec<String> {
        let mut lines: Vec<String> = store
            .triples()
            .iter()
            .map(|triple| format!("\"\" {triple}"))
            .collect();
        for name in store.graph_names() {
            lines.push(format!("{name:?}"));
            let graph = store.graph(&name).unwrap();
            lines.extend(
                graph
                    .triples()
                    .iter()
                    .map(|triple| format!("{name:?} {triple}")),
            );
        }
        lines.sort();
        lines
    }

    /// Whether every graph of `a` and `b` holds the same triples, compared as
    /// terms.
    fn same_triples(a: &RdfStore, b: &RdfStore) -> bool {
        let set = |store: &RdfStore| {
            store
                .triples()
                .iter()
                .map(|triple| (**triple).clone())
                .collect::<std::collections::HashSet<Triple>>()
        };
        let mut names_a = a.graph_names();
        let mut names_b = b.graph_names();
        names_a.sort();
        names_b.sort();
        set(a) == set(b)
            && names_a == names_b
            && names_a
                .iter()
                .all(|name| set(&a.graph(name).unwrap()) == set(&b.graph(name).unwrap()))
    }

    fn iri(local: &str) -> Term {
        Term::iri(format!("http://example.org/{local}"))
    }

    #[test]
    fn triples_of_every_term_kind_round_trip_in_named_graphs() {
        let store = Arc::new(RdfStore::new());
        let alix = Term::iri("http://example.org/alix");
        store.insert(Triple::new(
            alix.clone(),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix"),
        ));
        store.insert(Triple::new(
            alix.clone(),
            Term::iri("http://example.org/motto"),
            Term::lang_literal("Gus \"de bus\"\nAmsterdam", "nl"),
        ));
        store.insert(Triple::new(
            Term::blank("b0"),
            Term::iri("http://example.org/km"),
            Term::typed_literal("1030", "http://www.w3.org/2001/XMLSchema#integer"),
        ));
        store.create_graph("http://example.org/trips");
        store
            .graph("http://example.org/trips")
            .unwrap()
            .insert(Triple::new(
                alix,
                Term::iri("http://example.org/visited"),
                Term::iri("http://example.org/paris"),
            ));
        store.create_graph("http://example.org/empty");
        let back = round_trip(
            &store,
            ChunkCaps {
                max_rows: 2,
                max_bytes: 1 << 20,
            },
        );
        assert_eq!(sorted(&back), sorted(&store));
        assert!(same_triples(&back, &store));
        assert_eq!(back.graph_names().len(), 2, "the empty graph is kept");
    }

    #[test]
    fn non_ascii_text_escapes_and_whitespace_round_trip() {
        let store = Arc::new(RdfStore::new());
        let trips = store.graph_or_create("http://example.org/Kraków trips ");
        for (index, term) in unusual_terms().into_iter().enumerate() {
            let subject = if term.is_literal() {
                iri(&format!("subject/{index}"))
            } else {
                term.clone()
            };
            let triple = Triple::new_unchecked(subject, iri("holds"), term.clone());
            store.insert(triple.clone());
            trips.insert(Triple::new_unchecked(
                Term::blank(format!("b{index} ")),
                Term::iri("http://example.org/Kraków"),
                term,
            ));
        }
        for caps in [
            ChunkCaps::DEFAULT,
            ChunkCaps {
                max_rows: 1,
                max_bytes: 1,
            },
            ChunkCaps {
                max_rows: 4,
                max_bytes: 300,
            },
        ] {
            let back = round_trip(&store, caps);
            assert!(same_triples(&back, &store), "{caps:?}");
            assert_eq!(sorted(&back), sorted(&store), "{caps:?}");
        }
    }

    /// What a release build stores from `INSERT { ?s ?p ?o } WHERE { .. }`
    /// with a literal bound to the subject comes back as it was, without the
    /// debug check of `Triple::new`.
    #[test]
    fn a_triple_the_store_holds_comes_back_whatever_its_term_kinds() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new_unchecked(
            Term::literal("Alix"),
            Term::blank("b0"),
            iri("gus"),
        ));
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        assert!(same_triples(&back, &store));
    }

    #[test]
    fn the_three_columns_share_their_chunks() {
        let store = RdfStore::new();
        let people = ["alix", "gus", "vincent", "mia", "jules"];
        for person in people {
            store.insert(Triple::new(iri(person), iri("lives_in"), iri("amsterdam")));
        }
        store.insert(Triple::new(iri("amsterdam"), iri("near"), iri("berlin")));
        store.insert(Triple::new(
            iri("berlin"),
            iri("name"),
            Term::lang_literal("Berlin", "de"),
        ));
        let written = chunks(
            &store,
            ChunkCaps {
                max_rows: 3,
                max_bytes: 1 << 20,
            },
        );
        let layout: Vec<(ChunkKind, u32, u32, u64, u32)> = written
            .iter()
            .map(|(meta, _)| {
                (
                    meta.kind,
                    meta.graph_id,
                    meta.column_id,
                    meta.row_start,
                    meta.row_count,
                )
            })
            .collect();
        let mut expected = vec![(ChunkKind::Meta, 0, 0, 0, 0)];
        for (row_start, row_count) in [(0, 3), (3, 3), (6, 1)] {
            for column in 0..3 {
                expected.push((ChunkKind::Column, 0, column, row_start, row_count));
            }
        }
        assert_eq!(layout, expected);
        for (chunk, bytes) in &written[1..] {
            assert_eq!(chunk.codec, ChunkCodec::Dict.to_byte(), "{chunk:?}");
            let decoded = decode_column_chunk(bytes, chunk.codec, chunk.row_count).unwrap();
            assert_eq!(decoded.values.len(), chunk.row_count as usize, "every row");
            assert_eq!(decoded.epochs, None);
        }
    }

    #[test]
    fn a_long_literal_cuts_the_chunks_of_all_three_columns() {
        let store = Arc::new(RdfStore::new());
        for (index, city) in ["Amsterdam", "Berlin", "Paris", "Prague"]
            .iter()
            .enumerate()
        {
            store.insert(Triple::new(
                iri(city),
                iri("description"),
                Term::literal(city.repeat(index * 19 + 3)),
            ));
        }
        let caps = ChunkCaps {
            max_rows: 64,
            max_bytes: 320,
        };
        let written = chunks(&store, caps);
        let ranges: Vec<Vec<(u64, u32)>> = (0..3)
            .map(|column| {
                written
                    .iter()
                    .filter(|(meta, _)| meta.kind == ChunkKind::Column && meta.column_id == column)
                    .map(|(meta, _)| (meta.row_start, meta.row_count))
                    .collect()
            })
            .collect();
        assert!(ranges[0].len() > 1, "the cap cut the group: {ranges:?}");
        assert_eq!(ranges[0], ranges[1]);
        assert_eq!(ranges[0], ranges[2]);
        for (chunk, bytes) in &written[1..] {
            let values = decode_column_chunk(bytes, chunk.codec, chunk.row_count)
                .unwrap()
                .values
                .len();
            assert!(
                bytes.len() <= 320 || values == 1,
                "{} bytes, {values} values",
                bytes.len()
            );
        }
        assert!(same_triples(&round_trip(&store, caps), &store));
    }

    #[test]
    fn an_empty_store_writes_only_its_metadata_chunk() {
        let written = chunks(&RdfStore::new(), ChunkCaps::DEFAULT);
        assert_eq!(written.len(), 1);
        assert_eq!(
            meta(&written),
            RdfMeta {
                layout: 1,
                max_rows: 65_536,
                max_bytes: 1 << 20,
                graphs: vec![graph("", 0)],
            }
        );
        // Named graphs are listed in name order, empty or not.
        let store = RdfStore::new();
        for name in ["http://example.org/prague", "", "http://example.org/berlin"] {
            store.create_graph(name);
        }
        store
            .graph("http://example.org/prague")
            .unwrap()
            .insert(Triple::new(iri("mia"), iri("visits"), iri("prague")));
        let written = chunks(&store, ChunkCaps::DEFAULT);
        assert_eq!(
            meta(&written).graphs,
            [
                graph("", 0),
                graph("", 0),
                graph("http://example.org/berlin", 0),
                graph("http://example.org/prague", 1)
            ],
            "every graph with its triple count"
        );
        let graphs: Vec<u32> = written[1..].iter().map(|(meta, _)| meta.graph_id).collect();
        assert_eq!(graphs, [3, 3, 3]);
        let back = round_trip(&Arc::new(store), ChunkCaps::DEFAULT);
        assert_eq!(
            back.graph_count(),
            3,
            "a named graph with an empty name too"
        );
        assert!(back.graph("").unwrap().is_empty());
    }

    /// Two stores with the same triples write the same bytes, whatever order
    /// the triples and graphs came in (the triple sets are hash sets with
    /// their own random seeds).
    #[test]
    fn the_same_triples_give_the_same_bytes() {
        let triples: Vec<Triple> = (0..40)
            .map(|index| {
                Triple::new(
                    iri(["alix", "gus", "vincent", "mia", "jules"][index % 5]),
                    iri(["knows", "visits", "likes"][index % 3]),
                    Term::literal(format!("{} {index}", ["Amsterdam", "Berlin"][index % 2])),
                )
            })
            .collect();
        // Two triples that differ only in a language tag.
        let triples: Vec<Triple> =
            triples
                .into_iter()
                .chain(["cs", "sk"].map(|tag| {
                    Triple::new(iri("mia"), iri("name"), Term::lang_literal("Praha", tag))
                }))
                .collect();
        let build = |reverse: bool| {
            let store = RdfStore::new();
            let mut order: Vec<&Triple> = triples.iter().collect();
            let mut names = vec!["http://example.org/trips", "http://example.org/empty"];
            if reverse {
                order.reverse();
                names.reverse();
            }
            for name in names {
                store.create_graph(name);
            }
            let trips = store.graph("http://example.org/trips").unwrap();
            for triple in order {
                store.insert(triple.clone());
                trips.insert(triple.clone());
            }
            store
        };
        let caps = ChunkCaps {
            max_rows: 8,
            max_bytes: 512,
        };
        let first = chunks(&build(false), caps);
        assert_eq!(first, chunks(&build(true), caps));
        assert_eq!(first, chunks(&build(false), caps));
    }

    /// The metadata chunk lists every graph however long the names are: a
    /// fixed limit on its size would fail the checkpoint of a store with many
    /// graphs of long names (here 2,000 names of 9,000 bytes, 18 MB).
    #[test]
    fn many_long_graph_names_round_trip() {
        let store = Arc::new(RdfStore::new());
        let long = "Amsterdam/".repeat(900);
        for index in 0..2_000 {
            store.create_graph(&format!("http://example.org/{long}{index:04}"));
        }
        store
            .graph(&format!("http://example.org/{long}1988"))
            .unwrap()
            .insert(Triple::new(iri("mia"), iri("visits"), iri("berlin")));
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        assert_eq!(back.graph_count(), 2_000);
        assert!(same_triples(&back, &store));
    }

    #[test]
    fn a_section_writes_chunks_with_the_caps_it_was_built_with() {
        let store = Arc::new(RdfStore::new());
        for person in ["alix", "gus", "vincent", "mia", "jules"] {
            store.insert(Triple::new(iri(person), iri("lives_in"), iri("paris")));
        }
        let tiny = ChunkCaps {
            max_rows: 2,
            max_bytes: 1 << 20,
        };
        let subject_chunks = |section: &RdfStoreSection| {
            let image = MemoryImage::from_sections(&[section as &dyn Section]).unwrap();
            let source = image.section_source(SectionType::RdfStore).unwrap();
            assert_eq!(source.section_version(), 3);
            source
                .chunks()
                .iter()
                .filter(|chunk| chunk.kind == ChunkKind::Column && chunk.column_id == 0)
                .count()
        };
        let built_with = RdfStoreSection::with_caps(Arc::clone(&store), tiny);
        assert_eq!(subject_chunks(&built_with), 3, "5 triples in groups of 2");
        let built_under = with_chunk_caps(tiny, || RdfStoreSection::new(Arc::clone(&store)));
        assert_eq!(
            subject_chunks(&built_under),
            3,
            "the caps of the thread that built it"
        );
        let default = RdfStoreSection::new(Arc::clone(&store));
        assert_eq!(subject_chunks(&default), 1, "the default caps");
    }

    #[test]
    fn a_0_5_rdf_section_still_loads() {
        let store = Arc::new(RdfStore::new());
        let trips = store.graph_or_create("http://example.org/trips");
        store.create_graph("http://example.org/empty");
        for (index, term) in unusual_terms().into_iter().enumerate() {
            store.insert(Triple::new_unchecked(
                iri(&format!("subject/{index}")),
                iri("holds"),
                term.clone(),
            ));
            trips.insert(Triple::new_unchecked(
                Term::blank(format!("b{index} ")),
                iri("Kraków"),
                term,
            ));
        }
        let bytes = RdfStoreSection::new(Arc::clone(&store))
            .serialize()
            .unwrap();
        let image = MemoryImage::from_raw(vec![(SectionType::RdfStore, bytes)]).unwrap();
        let back = Arc::new(RdfStore::new());
        RdfStoreSection::new(Arc::clone(&back))
            .read_from(&*image.section_source(SectionType::RdfStore).unwrap())
            .unwrap();
        assert!(same_triples(&back, &store));
        assert_eq!(sorted(&back), sorted(&store));
    }

    /// Serves crafted chunks as they are, repeats and all.
    struct Crafted(Vec<ChunkMeta>, Vec<Bytes>, u8);

    impl SectionSource for Crafted {
        fn chunks(&self) -> &[ChunkMeta] {
            &self.0
        }

        fn fetch(&self, index: usize) -> Result<Bytes> {
            Ok(self.1[index].clone())
        }

        fn stored_length(&self, index: usize) -> Result<u64> {
            Ok(self.1[index].len() as u64)
        }

        fn section_version(&self) -> u8 {
            self.2
        }
    }

    /// One chunk of a crafted section.
    enum Chunk {
        /// The metadata of a section written with caps of 3 rows: the default
        /// graph and graph 1, `http://example.org/trips`, each of 6 triples.
        Meta,
        /// The same metadata with these triple counts of the default graph
        /// and graph 1.
        MetaCounts(u64, u64),
        /// A metadata chunk of these bytes.
        MetaBytes(Vec<u8>),
        /// The three column chunks of rows `[row_start, row_start + count)`
        /// of `graph`, each row a valid triple.
        Triples {
            graph: u32,
            row_start: u64,
            count: u32,
        },
        /// One column chunk with these values.
        Column {
            graph: u32,
            column: u32,
            row_start: u64,
            row_count: u32,
            values: Vec<(u32, Value)>,
        },
        /// Any chunk.
        Raw(ChunkMeta, Vec<u8>),
    }

    fn crafted_meta() -> RdfMeta {
        crafted_meta_counts(6, 6)
    }

    fn crafted_meta_counts(default: u64, trips: u64) -> RdfMeta {
        RdfMeta {
            layout: 1,
            max_rows: 3,
            max_bytes: 1 << 20,
            graphs: vec![graph("", default), graph("http://example.org/trips", trips)],
        }
    }

    fn encode_meta(meta: &RdfMeta) -> Vec<u8> {
        encode_rdf_meta(meta).unwrap()
    }

    /// The N-Triples string of `column` in a valid triple at `row`.
    fn term_text(column: u32, row: u64) -> String {
        match column {
            0 => format!("<http://example.org/person/{row}>"),
            1 => "<http://xmlns.com/foaf/0.1/name>".to_string(),
            _ => format!("\"Alix {row}\""),
        }
    }

    fn column_values(column: u32, row_start: u64, count: u32) -> Vec<(u32, Value)> {
        (0..count)
            .map(|offset| {
                (
                    offset,
                    Value::from(term_text(column, row_start + u64::from(offset))),
                )
            })
            .collect()
    }

    fn column_chunk(
        graph: u32,
        column: u32,
        row_start: u64,
        row_count: u32,
        values: &[(u32, Value)],
        epochs: Option<&[u64]>,
    ) -> (ChunkMeta, Bytes) {
        let (codec, bytes) = encode_column_chunk(row_count, values, epochs).unwrap();
        (
            ChunkMeta::column(graph, column, row_start, row_count, codec.to_byte()),
            Bytes::from(bytes),
        )
    }

    /// Loads `chunks`, of a section of `version`, into a new store.
    fn load_crafted_version(chunks: Vec<Chunk>, version: u8) -> Result<RdfStore> {
        let mut metas = Vec::new();
        let mut bytes = Vec::new();
        let mut push = |(meta, data): (ChunkMeta, Bytes)| {
            metas.push(meta);
            bytes.push(data);
        };
        for chunk in chunks {
            match chunk {
                Chunk::Meta => push((ChunkMeta::meta(), Bytes::from(encode_meta(&crafted_meta())))),
                Chunk::MetaCounts(default, trips) => push((
                    ChunkMeta::meta(),
                    Bytes::from(encode_meta(&crafted_meta_counts(default, trips))),
                )),
                Chunk::MetaBytes(data) => push((ChunkMeta::meta(), Bytes::from(data))),
                Chunk::Triples {
                    graph,
                    row_start,
                    count,
                } => {
                    for column in 0..3 {
                        let values = column_values(column, row_start, count);
                        push(column_chunk(graph, column, row_start, count, &values, None));
                    }
                }
                Chunk::Column {
                    graph,
                    column,
                    row_start,
                    row_count,
                    values,
                } => push(column_chunk(
                    graph, column, row_start, row_count, &values, None,
                )),
                Chunk::Raw(meta, data) => push((meta, Bytes::from(data))),
            }
        }
        let store = Arc::new(RdfStore::new());
        RdfStoreSection::new(Arc::clone(&store)).read_from(&Crafted(metas, bytes, version))?;
        Ok(Arc::try_unwrap(store).unwrap_or_else(|_| unreachable!("the section is gone")))
    }

    fn load_crafted(chunks: Vec<Chunk>) -> Result<RdfStore> {
        load_crafted_version(chunks, RDF_SECTION_VERSION)
    }

    #[test]
    fn crafted_chunks_of_valid_triples_load() {
        let store = load_crafted(vec![
            Chunk::MetaCounts(4, 2),
            Chunk::Triples {
                graph: 0,
                row_start: 0,
                count: 3,
            },
            Chunk::Triples {
                graph: 0,
                row_start: 3,
                count: 1,
            },
            Chunk::Triples {
                graph: 1,
                row_start: 0,
                count: 2,
            },
        ])
        .unwrap();
        assert_eq!(store.len(), 4);
        assert_eq!(store.graph("http://example.org/trips").unwrap().len(), 2);
        let alix = Triple::new(
            Term::iri("http://example.org/person/3"),
            Term::iri("http://xmlns.com/foaf/0.1/name"),
            Term::literal("Alix 3"),
        );
        assert!(store.contains(&alix));
    }

    #[test]
    fn crafted_rdf_chunks_are_refused() {
        use Chunk::{Column, Meta, MetaBytes, MetaCounts, Raw, Triples};
        let triples = |graph, row_start, count| Triples {
            graph,
            row_start,
            count,
        };
        let only = |column: u32, row_start: u64, count: u32| Column {
            graph: 0,
            column,
            row_start,
            row_count: count,
            values: column_values(column, row_start, count),
        };
        let with_meta = |meta: RdfMeta| MetaBytes(encode_meta(&meta));
        let (_, strings) =
            encode_column_chunk(1, &[(0, Value::from("<http://example.org/a>"))], None).unwrap();
        let (_, epochs) = column_chunk(
            0,
            0,
            0,
            1,
            &[(0, Value::from("<http://example.org/a>"))],
            Some(&[3]),
        );
        let mut trailing = encode_meta(&crafted_meta());
        trailing.push(0);
        let cases: Vec<(&str, Vec<Chunk>, &[&str])> = vec![
            (
                "a subject column without the others",
                vec![Meta, only(0, 0, 3)],
                &["the default graph, rows [0, 3)", "predicate"],
            ),
            (
                "a gap between two ranges",
                vec![Meta, triples(0, 0, 3), triples(0, 4, 2)],
                &["the default graph, rows [4, 6)", "row 3"],
            ),
            (
                "a missing row",
                vec![
                    Meta,
                    only(0, 0, 3),
                    only(1, 0, 3),
                    Column {
                        graph: 0,
                        column: 2,
                        row_start: 0,
                        row_count: 3,
                        values: vec![(0, Value::from("\"Alix\"")), (2, Value::from("\"Gus\""))],
                    },
                ],
                &["the default graph, rows [0, 3)", "object", "row 1"],
            ),
            (
                "an object that is no term",
                vec![
                    Meta,
                    triples(1, 0, 3),
                    Column {
                        graph: 1,
                        column: 0,
                        row_start: 3,
                        row_count: 1,
                        values: column_values(0, 3, 1),
                    },
                    Column {
                        graph: 1,
                        column: 1,
                        row_start: 3,
                        row_count: 1,
                        values: column_values(1, 3, 1),
                    },
                    Column {
                        graph: 1,
                        column: 2,
                        row_start: 3,
                        row_count: 1,
                        values: vec![(0, Value::from("<<not a term"))],
                    },
                ],
                &[
                    "graph 1 \"http://example.org/trips\", row 3, object",
                    "an IRI without its closing '>'",
                    "<<not a term",
                ],
            ),
            (
                "a chunk before the metadata",
                vec![triples(0, 0, 1), Meta],
                &["metadata chunk"],
            ),
            (
                "no metadata at all",
                vec![triples(0, 0, 1)],
                &["metadata chunk"],
            ),
            (
                "a second metadata chunk",
                vec![Meta, triples(0, 0, 1), Meta],
                &["a second metadata chunk"],
            ),
            (
                "a graph the metadata does not list",
                vec![Meta, triples(2, 0, 1)],
                &["graph 2", "the metadata chunk lists 2 graphs"],
            ),
            (
                "graphs out of order",
                vec![Meta, triples(1, 0, 1), triples(0, 0, 1)],
                &[
                    "the default graph, rows [0, 1)",
                    "after the chunks of graph 1",
                ],
            ),
            (
                "a graph that comes back",
                vec![Meta, triples(0, 0, 1), triples(1, 0, 1), triples(0, 1, 1)],
                &[
                    "the default graph, rows [1, 2)",
                    "after the chunks of graph 1",
                ],
            ),
            (
                "rows that do not start at 0",
                vec![Meta, triples(1, 1, 2)],
                &["graph 1 \"http://example.org/trips\", rows [1, 3)", "row 0"],
            ),
            (
                "rows read twice",
                vec![Meta, triples(0, 0, 2), triples(0, 1, 2)],
                &["the default graph, rows [1, 3)", "row 2"],
            ),
            (
                "more rows than a row group",
                vec![Meta, triples(0, 0, 4)],
                &["the default graph, rows [0, 4)", "row group"],
            ),
            (
                "a chunk across two row groups",
                vec![Meta, triples(0, 0, 2), triples(0, 2, 2)],
                &["the default graph, rows [2, 4)", "row group"],
            ),
            (
                "columns out of order",
                vec![Meta, only(1, 0, 1), only(0, 0, 1), only(2, 0, 1)],
                &["the default graph, rows [0, 1)", "column 1", "subject"],
            ),
            (
                "an unknown column",
                vec![Meta, only(0, 0, 1), only(1, 0, 1), only(3, 0, 1)],
                &["the default graph, rows [0, 1)", "column 3", "object"],
            ),
            (
                "columns of other rows",
                vec![Meta, only(0, 0, 2), only(1, 0, 1), only(2, 0, 2)],
                &["the default graph, rows [0, 2)", "rows [0, 1)"],
            ),
            (
                "a history chunk",
                vec![
                    Meta,
                    Raw(
                        ChunkMeta::history(0, 0, 0, 1, ChunkCodec::Dict.to_byte()),
                        strings.clone(),
                    ),
                ],
                &["the default graph, rows [0, 1)", "History"],
            ),
            (
                "a stream piece",
                vec![Meta, Raw(ChunkMeta::stream_piece(0, 0, 0), vec![3])],
                &["Stream"],
            ),
            (
                "epochs on a chunk",
                vec![
                    Meta,
                    Raw(
                        ChunkMeta::column(0, 0, 0, 1, ChunkCodec::Dict.to_byte()),
                        epochs.to_vec(),
                    ),
                    only(1, 0, 1),
                    only(2, 0, 1),
                ],
                &["the default graph, rows [0, 1)", "epochs"],
            ),
            (
                "a value that is not a string",
                vec![
                    Meta,
                    only(0, 0, 1),
                    only(1, 0, 1),
                    Column {
                        graph: 0,
                        column: 2,
                        row_start: 0,
                        row_count: 1,
                        values: vec![(0, Value::Int64(88))],
                    },
                ],
                &["the default graph, row 0, object", "Int64"],
            ),
            (
                "another layout",
                vec![with_meta(RdfMeta {
                    layout: 2,
                    ..crafted_meta()
                })],
                &["layout 2"],
            ),
            (
                "bytes after the metadata",
                vec![MetaBytes(trailing)],
                &["1 bytes after the metadata"],
            ),
            (
                "caps of zero",
                vec![with_meta(RdfMeta {
                    max_rows: 0,
                    ..crafted_meta()
                })],
                &["caps"],
            ),
            (
                // Without triples, so nothing but the caps is wrong.
                "caps above the format's row cap",
                vec![with_meta(RdfMeta {
                    max_rows: ChunkCaps::DEFAULT.max_rows + 1,
                    ..crafted_meta_counts(0, 0)
                })],
                &["caps of 65537 rows", "65536"],
            ),
            (
                "a named default graph",
                vec![with_meta(RdfMeta {
                    graphs: vec![graph("http://example.org/alix", 0)],
                    ..crafted_meta()
                })],
                &["graph 0", "the default graph has no name"],
            ),
            (
                "no graph at all",
                vec![with_meta(RdfMeta {
                    graphs: Vec::new(),
                    ..crafted_meta()
                })],
                &["no graph"],
            ),
            (
                "a named graph listed twice",
                vec![with_meta(RdfMeta {
                    graphs: vec![
                        graph("", 0),
                        graph("http://example.org/b", 0),
                        graph("http://example.org/b", 0),
                    ],
                    ..crafted_meta()
                })],
                &["graph 2 \"http://example.org/b\"", "name order"],
            ),
            (
                "named graphs out of name order",
                vec![with_meta(RdfMeta {
                    graphs: vec![
                        graph("", 0),
                        graph("http://example.org/b", 0),
                        graph("http://example.org/a", 0),
                    ],
                    ..crafted_meta()
                })],
                &["graph 2 \"http://example.org/a\"", "name order"],
            ),
            (
                "a missing last row",
                vec![
                    Meta,
                    only(0, 0, 3),
                    only(1, 0, 3),
                    Column {
                        graph: 0,
                        column: 2,
                        row_start: 0,
                        row_count: 3,
                        values: vec![(0, Value::from("\"Alix\"")), (1, Value::from("\"Gus\""))],
                    },
                ],
                &[
                    "the default graph, rows [0, 3)",
                    "the object chunk has no value for row 2",
                ],
            ),
            (
                "a missing trailing range",
                vec![MetaCounts(4, 0), triples(0, 0, 3)],
                &[
                    "the default graph: its chunks end at row 3, where the metadata counts 4 triples",
                ],
            ),
            (
                "rows past the graph's count",
                vec![MetaCounts(2, 0), triples(0, 0, 3)],
                &[
                    "the default graph, rows [0, 3)",
                    "rows past the graph's 2 triples",
                ],
            ),
            (
                "a named graph without its chunks",
                vec![MetaCounts(0, 2)],
                &[
                    "graph 1 \"http://example.org/trips\": its chunks end at row 0, where the \
                     metadata counts 2 triples",
                ],
            ),
        ];
        for (name, chunks, words) in cases {
            let error = load_crafted(chunks)
                .err()
                .unwrap_or_else(|| panic!("{name}: loaded"))
                .to_string();
            for word in words {
                assert!(error.contains(word), "{name}: {error:?} lacks {word:?}");
            }
        }
    }

    /// A metadata chunk that claims more than its bytes hold is refused
    /// before anything is allocated for it.
    #[test]
    fn a_metadata_chunk_claiming_more_than_it_holds_is_refused() {
        let head = |count: u32| {
            let mut bytes = vec![1];
            bytes.extend_from_slice(&3u32.to_le_bytes());
            bytes.extend_from_slice(&(1u32 << 20).to_le_bytes());
            bytes.extend_from_slice(&count.to_le_bytes());
            bytes
        };
        let refused = |bytes: &[u8]| match decode_rdf_meta(bytes) {
            Err(grafeo_common::utils::error::Error::Corruption(corruption)) => corruption.what,
            other => panic!("{other:?}"),
        };

        let mut many = head(u32::MAX);
        many.extend_from_slice(&[0; 12]);
        assert_eq!(
            refused(&many),
            "RDF metadata chunk, byte 9: 4294967295 graphs, but only 12 bytes are left for them"
        );

        let mut long = head(1);
        long.extend_from_slice(&u32::MAX.to_le_bytes());
        long.extend_from_slice(b"Alix");
        long.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(
            refused(&long),
            "RDF metadata chunk, byte 13: a graph name of 4294967295 bytes, but only 12 bytes \
             are left"
        );

        let mut not_utf8 = head(1);
        not_utf8.extend_from_slice(&2u32.to_le_bytes());
        not_utf8.extend_from_slice(&[0xC3, 0x28]);
        not_utf8.extend_from_slice(&0u64.to_le_bytes());
        assert!(refused(&not_utf8).contains("a graph name that is not UTF-8"));

        let valid = encode_meta(&crafted_meta());
        assert_eq!(decode_rdf_meta(&valid).unwrap(), crafted_meta());
        for cut in 0..valid.len() {
            assert!(decode_rdf_meta(&valid[..cut]).is_err(), "cut at {cut}");
        }
    }

    /// The named graphs a metadata chunk lists cost nothing until triples
    /// arrive: a small chunk naming 2,000 graphs allocates no index.
    #[test]
    fn a_metadata_chunk_listing_many_graphs_allocates_no_indexes() {
        let mut graphs = vec![graph("", 0)];
        graphs.extend((0..2_000).map(|index| graph(&format!("g{index:04}"), 0)));
        let meta = RdfMeta {
            graphs,
            ..crafted_meta()
        };
        let store = load_crafted(vec![Chunk::MetaBytes(encode_meta(&meta))]).unwrap();
        assert_eq!(store.graph_count(), 2_000);
        let allocated: usize = store
            .graph_names()
            .iter()
            .map(|name| store.graph(name).unwrap().index_allocation_bytes())
            .sum();
        assert_eq!(allocated, 0, "2,000 empty named graphs");
    }

    #[test]
    fn chunks_of_another_section_version_are_refused() {
        let error = load_crafted_version(vec![Chunk::Meta], 2)
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("RdfStore")
                && error.contains("version 2")
                && error.contains("version 3"),
            "{error}"
        );
    }

    /// A term that repeats in a chunk is read once: the triples read back
    /// share its string instead of holding a copy each.
    #[test]
    fn a_term_repeated_in_a_chunk_is_read_once() {
        let store = Arc::new(RdfStore::new());
        for person in ["alix", "gus", "vincent", "mia", "jules"] {
            store.insert(Triple::new(iri(person), iri("lives_in"), iri("amsterdam")));
        }
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        let address = |term: &Term| term.as_iri().unwrap().as_str().as_ptr();
        let triples = back.triples();
        for triple in &triples {
            assert_eq!(address(triple.predicate()), address(triples[0].predicate()));
            assert_eq!(address(triple.object()), address(triples[0].object()));
        }
        assert_ne!(
            address(triples[0].subject()),
            address(triples[1].subject()),
            "distinct terms keep their own strings"
        );
    }

    #[test]
    fn a_literal_type_survives_next_to_a_plain_one() {
        let store = Arc::new(RdfStore::new());
        store.insert(Triple::new(iri("mia"), iri("age"), Term::literal("19")));
        store.insert(Triple::new(
            iri("mia"),
            iri("age"),
            Term::typed_literal("19", Literal::XSD_INTEGER),
        ));
        let back = round_trip(&store, ChunkCaps::DEFAULT);
        assert_eq!(back.len(), 2);
        assert!(same_triples(&back, &store));
    }

    /// A sink that refuses its third chunk.
    struct RefusesThird(usize);

    impl SectionSink for RefusesThird {
        fn write_chunk(&mut self, _meta: ChunkMeta, _bytes: &[u8]) -> Result<()> {
            self.0 += 1;
            if self.0 == 3 {
                return Err(grafeo_common::utils::error::Error::Internal(
                    "Jules refuses the chunk".to_string(),
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn a_sink_error_ends_the_write() {
        let store = RdfStore::new();
        store.insert(Triple::new(iri("jules"), iri("knows"), iri("vincent")));
        let error = write_rdf_chunks(&store, ChunkCaps::DEFAULT, &mut RefusesThird(0))
            .unwrap_err()
            .to_string();
        assert!(error.contains("Jules refuses"), "{error}");
        let zero = ChunkCaps {
            max_rows: 0,
            max_bytes: 1,
        };
        assert!(write_rdf_chunks(&store, zero, &mut RefusesThird(0)).is_err());
    }
}
