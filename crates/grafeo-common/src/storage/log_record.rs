//! Log records: the typed, framed records of the write-ahead log (WAL v2).
//!
//! A committed transaction is one group of records in the WAL: a
//! [`GroupBegin`], then one record per logical write, in the order the writes
//! were applied. The frames that carry a group (and mark its end) belong to
//! the WAL's container in grafeo-storage, which treats a frame's records as
//! opaque bytes; this module defines the records and their bytes. Records hold
//! no engine types, so the same records serve the live write path and replay.
//!
//! A record is the encoding of a change ([`change`](crate::change)): a data
//! record ([`LogRecord::Data`]) holds the op of one change set entry, a
//! [`DataOp`], and the storage key of its graph; a standalone record
//! ([`LogRecord::Standalone`]) holds a [`StandaloneOp`]. A change set encodes
//! its entries without copying them ([`ChangeSet::log_records`] and
//! [`LogRecordRef`]); entries' before-images are never logged.
//!
//! Every record is framed as a catalog record is
//! ([`catalog_record`](super::catalog_record)), with its own kinds and
//! maximum:
//!
//! | Bytes | Field |
//! | --- | --- |
//! | 1 | kind |
//! | 1 | flags: bit 0 is [`RECORD_REQUIRED`]; a reader refuses another of bits 0 to 3 and ignores bits 4 to 7 |
//! | 4 | payload length, u32 little endian; the framed record takes at most [`MAX_RECORD_BYTES`] |
//! | length | payload: bincode (standard configuration) of the kind's fields |
//!
//! | Kind | Record | Payload |
//! | --- | --- | --- |
//! | 1 | [`GroupBegin`](LogRecord::GroupBegin) | epoch, timestamp in milliseconds, [`Origin`] |
//! | 16 | [`CreateNode`](DataOp::CreateNode) | graph, id, labels, properties |
//! | 17 | [`DeleteNode`](DataOp::DeleteNode) | graph, id |
//! | 18 | [`CreateEdge`](DataOp::CreateEdge) | graph, id, source node, target node, edge type, properties |
//! | 19 | [`DeleteEdge`](DataOp::DeleteEdge) | graph, id |
//! | 20 | [`SetNodeProperty`](DataOp::SetNodeProperty) | graph, id, key, value |
//! | 21 | [`RemoveNodeProperty`](DataOp::RemoveNodeProperty) | graph, id, key |
//! | 22 | [`SetEdgeProperty`](DataOp::SetEdgeProperty) | graph, id, key, value |
//! | 23 | [`RemoveEdgeProperty`](DataOp::RemoveEdgeProperty) | graph, id, key |
//! | 24 | [`AddNodeLabel`](DataOp::AddNodeLabel) | graph, id, label |
//! | 25 | [`RemoveNodeLabel`](DataOp::RemoveNodeLabel) | graph, id, label |
//! | 32 | [`CreateGraph`](StandaloneOp::CreateGraph) | name |
//! | 33 | [`DropGraph`](StandaloneOp::DropGraph) | name |
//! | 34 | reserved for copying a graph (`CREATE GRAPH ... AS COPY OF`) | |
//! | 40 | [`PutCatalog`](StandaloneOp::PutCatalog) | the catalog record's kind (u8), then its payload |
//! | 41 | [`DropCatalog`](StandaloneOp::DropCatalog) | the dropped catalog record's kind (u8), then its [`CatalogKey`] |
//! | 64 | [`InsertTriple`](DataOp::InsertTriple) | graph, subject, predicate, object |
//! | 65 | [`DeleteTriple`](DataOp::DeleteTriple) | graph, subject, predicate, object |
//! | 66 | [`Create`](RdfGraphOp::Create) an RDF graph | name |
//! | 67 | [`Drop`](RdfGraphOp::Drop) RDF graphs | [`RdfGraphTarget`] |
//! | 68 | [`Clear`](RdfGraphOp::Clear) RDF graphs | [`RdfGraphTarget`] |
//! | 69 | [`Copy`](RdfGraphOp::Copy) an RDF graph | source graph, target graph |
//! | 70 | [`Move`](RdfGraphOp::Move) an RDF graph | source graph, target graph |
//! | 71 | [`Add`](RdfGraphOp::Add) an RDF graph to another | source graph, target graph |
//!
//! In bincode's standard configuration an integer is a variable-length
//! number (one byte below 251), a string its length and its UTF-8 bytes, an
//! option 0 for none or 1 and the value, a list its count and its elements,
//! and an enum its variant number (from 0, in declaration order) and its
//! fields. So:
//!
//! - an epoch, an id and a timestamp are u64 numbers;
//! - a graph is an option of the graph's storage key, none for the default
//!   graph (of the LPG store or the RDF store, by the record's kind); a source
//!   or target graph of kinds 69 to 71 is the same;
//! - labels are a list of strings, properties a list of (key, value) pairs;
//! - a value is a byte string holding the value's encoding by the lossless
//!   value codec ([`value_codec`](super::value_codec)), so every value comes
//!   back with its kind and bits: NaN payloads, sub-millisecond timestamps,
//!   offsets and counters included;
//! - a term is a [`TermRecord`]: 0 an IRI, 1 a blank node (each a string),
//!   2 a literal (value, datatype IRI, optional language tag);
//! - a catalog record's payload is the one the catalog section stores, with
//!   the catalog's own maximum ([`MAX_CATALOG_RECORD_PAYLOAD`]); a
//!   [`CatalogKey`] is the name of what it drops, the graph of a binding, or
//!   the graph and the [`IndexKeyRecord`] of an index.
//!
//! Every record names its graph (no record depends on another one before
//! it), its table by its kind (nodes, edges, triples, graphs, catalog) and,
//! for a property, its key.
//!
//! Kind 0 is never written. A reader skips a record of a kind it does not
//! know when the record's required flag is clear, and refuses it naming the
//! kind when the flag is set. The same holds for a catalog record of a kind
//! it does not know inside kind 40 or 41: the log record's flag decides, and
//! a writer gives the log record the catalog record's flag. Every kind of
//! this release is written required. A kind's payload never changes: a
//! release that needs other fields adds a kind, and the enums of a payload
//! only get variants appended.
//!
//! # What a damaged record can allocate
//!
//! The reader checks a payload's length against the maximum, and reads the
//! payload into a buffer that grows with the bytes present. It then decodes
//! the payload in place: a string or value is read only once its length is
//! found within the payload's bytes, lists grow as their elements decode,
//! and the value codec checks every count against the bytes left. So the
//! memory a record decodes into is proportional to the bytes it holds, and
//! no length or count it claims is allocated before the bytes are there.
//! A list of labels or properties takes at most 8 bytes of slots per byte
//! of its elements plus 1 MiB (twice that while it grows): a list past that
//! is refused when read, and when written, so every record written decodes.
//! The payload of a catalog record inside kind 40 is the exception: it is
//! decoded as the catalog section decodes it, whose damaged string lengths
//! may claim up to the catalog's decode limit (16 MiB, see
//! [`catalog_record`](super::catalog_record)).
//!
//! [`ChangeSet::log_records`]: crate::change::ChangeSet::log_records

use std::fmt;
use std::io;
use std::marker::PhantomData;

use serde::de::{DeserializeOwned, SeqAccess, Visitor};
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use smallvec::SmallVec;

use super::catalog_record::{
    CatalogKey, CatalogRecord, IndexKeyRecord, MAX_CATALOG_RECORD_PAYLOAD, PayloadWriter,
    RECORD_HEADER_BYTES, RECORD_REQUIRED, RecordFraming, Text, encode_bincode_payload,
    encode_framed_record, optional_text, read_framed_records, text,
};
use super::value_codec::{decode_value, encode_value, encoded_len, nests_too_deep};
use crate::change::{DataOp, GraphRef, Labels, Properties, RdfGraphOp, StandaloneOp};
use crate::types::{ArcStr, EdgeId, EpochId, NodeId, PropertyKey, Value};
use crate::utils::error::{Error, Result};

/// The most bytes one framed log record may take, its header included:
/// 1 GiB less 64 bytes.
///
/// The writer refuses a larger record before it writes a byte of it, and
/// the reader refuses a header that claims one before it reads the payload.
/// The 64 bytes leave room for a WAL frame to carry a record of this size on
/// its own: a frame's payload holds at most 1 GiB, of which a FIRST frame's
/// prologue takes 8 bytes and an encrypted frame's nonce and tag 28.
pub const MAX_RECORD_BYTES: usize = (1 << 30) - 64;

/// The most bytes a log record's payload may hold: [`MAX_RECORD_BYTES`]
/// less the record header.
const MAX_RECORD_PAYLOAD: u32 = (1 << 30) - 64 - 6;

// A record at the maximum fits a WAL frame of its own: a frame's payload
// holds at most 1 GiB, less a FIRST frame's 8-byte prologue and an encrypted
// frame's 12-byte nonce and 16-byte tag.
const _: () = assert!(
    MAX_RECORD_BYTES + 8 + 12 + 16 <= 1 << 30,
    "a record at the maximum must fit a WAL frame of its own"
);

/// The framing of log records.
const LOG_FRAMING: RecordFraming = RecordFraming {
    what: "log record",
    max_payload: MAX_RECORD_PAYLOAD,
};

const KIND_GROUP_BEGIN: u8 = 1;
const KIND_CREATE_NODE: u8 = 16;
const KIND_DELETE_NODE: u8 = 17;
const KIND_CREATE_EDGE: u8 = 18;
const KIND_DELETE_EDGE: u8 = 19;
const KIND_SET_NODE_PROPERTY: u8 = 20;
const KIND_REMOVE_NODE_PROPERTY: u8 = 21;
const KIND_SET_EDGE_PROPERTY: u8 = 22;
const KIND_REMOVE_EDGE_PROPERTY: u8 = 23;
const KIND_ADD_NODE_LABEL: u8 = 24;
const KIND_REMOVE_NODE_LABEL: u8 = 25;
const KIND_CREATE_GRAPH: u8 = 32;
const KIND_DROP_GRAPH: u8 = 33;
const KIND_PUT_CATALOG: u8 = 40;
const KIND_DROP_CATALOG: u8 = 41;
const KIND_INSERT_TRIPLE: u8 = 64;
const KIND_DELETE_TRIPLE: u8 = 65;
const KIND_CREATE_RDF_GRAPH: u8 = 66;
const KIND_DROP_RDF_GRAPH: u8 = 67;
const KIND_CLEAR_RDF_GRAPH: u8 = 68;
const KIND_COPY_RDF_GRAPH: u8 = 69;
const KIND_MOVE_RDF_GRAPH: u8 = 70;
const KIND_ADD_RDF_GRAPH: u8 = 71;

// ── Record types ────────────────────────────────────────────────────

/// What wrote a group: the first record of every group says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Origin {
    /// An explicit or auto-commit transaction of a session.
    Transaction,
    /// A write statement run as its own transaction.
    Statement,
    /// A direct call on the database or a graph handle.
    Direct,
    /// A schema change, an index or graph command, or an RDF graph operation.
    Schema,
    /// A bulk import.
    Bulk,
}

/// The first record of a group: the epoch its changes take and when it was
/// written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupBegin {
    /// The epoch the group's changes commit at; replay applies them there.
    pub epoch: EpochId,
    /// When the group was written, in milliseconds since the Unix epoch.
    pub timestamp_ms: u64,
    /// What wrote the group.
    pub origin: Origin,
}

/// An RDF term, as a log record stores it: typed, so replay parses nothing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TermRecord {
    /// An IRI.
    Iri(#[serde(deserialize_with = "text")] String),
    /// A blank node, by its label.
    Blank(#[serde(deserialize_with = "text")] String),
    /// A literal.
    Literal {
        /// The lexical form.
        #[serde(deserialize_with = "text")]
        value: String,
        /// The datatype IRI.
        #[serde(deserialize_with = "text")]
        datatype: String,
        /// The language tag, if it has one.
        #[serde(deserialize_with = "optional_text")]
        language: Option<String>,
    },
}

/// An RDF triple, as a log record stores it.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TripleRecord {
    /// The subject.
    pub subject: TermRecord,
    /// The predicate.
    pub predicate: TermRecord,
    /// The object.
    pub object: TermRecord,
}

/// The graphs an RDF `DROP` or `CLEAR` acts on.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RdfGraphTarget {
    /// The default graph.
    Default,
    /// One named graph.
    Named(#[serde(deserialize_with = "text")] String),
    /// Every named graph (`NAMED`).
    AllNamed,
    /// The default graph and every named graph (`ALL`).
    All,
}

/// One record of the WAL. Matches over it are exhaustive on purpose: a kind
/// added later must be handled by every reader of this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum LogRecord {
    /// The first record of a group (kind 1).
    GroupBegin(GroupBegin),
    /// One change to an entity or a triple of a graph: the op of a change
    /// set entry (kinds 16 to 25, 64 and 65).
    Data {
        /// The graph's storage key; `None` for the default graph of the
        /// op's model (the LPG store's or the RDF store's).
        graph: Option<String>,
        /// What was applied.
        op: DataOp,
    },
    /// A change applied on its own: a graph, catalog or RDF graph operation
    /// (kinds 32, 33, 40, 41 and 66 to 71).
    Standalone(StandaloneOp),
}

/// A log record that borrows its op, so a change set encodes its entries
/// without copying them ([`ChangeSet::log_records`](crate::change::ChangeSet::log_records)).
/// It encodes to the bytes of the [`LogRecord`] it stands for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LogRecordRef<'a> {
    /// The first record of a group (kind 1).
    GroupBegin(GroupBegin),
    /// One change to an entity or a triple of a graph.
    Data {
        /// The graph's storage key; `None` for the default graph of the
        /// op's model.
        graph: Option<&'a str>,
        /// What was applied.
        op: &'a DataOp,
    },
    /// A change applied on its own.
    Standalone(&'a StandaloneOp),
}

impl DataOp {
    /// The kind of the op's log record (see the module documentation's
    /// table): 16 to 25 for nodes, edges, properties and labels, 64 and 65
    /// for triples.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::CreateNode { .. } => KIND_CREATE_NODE,
            Self::DeleteNode { .. } => KIND_DELETE_NODE,
            Self::CreateEdge { .. } => KIND_CREATE_EDGE,
            Self::DeleteEdge { .. } => KIND_DELETE_EDGE,
            Self::SetNodeProperty { .. } => KIND_SET_NODE_PROPERTY,
            Self::RemoveNodeProperty { .. } => KIND_REMOVE_NODE_PROPERTY,
            Self::SetEdgeProperty { .. } => KIND_SET_EDGE_PROPERTY,
            Self::RemoveEdgeProperty { .. } => KIND_REMOVE_EDGE_PROPERTY,
            Self::AddNodeLabel { .. } => KIND_ADD_NODE_LABEL,
            Self::RemoveNodeLabel { .. } => KIND_REMOVE_NODE_LABEL,
            Self::InsertTriple { .. } => KIND_INSERT_TRIPLE,
            Self::DeleteTriple { .. } => KIND_DELETE_TRIPLE,
        }
    }
}

impl StandaloneOp {
    /// The kind of the op's log record (see the module documentation's
    /// table): 32 and 33 for graphs, 40 and 41 for the catalog, 66 to 71
    /// for RDF graphs.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::CreateGraph { .. } => KIND_CREATE_GRAPH,
            Self::DropGraph { .. } => KIND_DROP_GRAPH,
            Self::PutCatalog(_) => KIND_PUT_CATALOG,
            Self::DropCatalog(_) => KIND_DROP_CATALOG,
            Self::RdfGraph(op) => op.kind(),
        }
    }
}

impl RdfGraphOp {
    /// The kind of the op's log record: 66 to 71.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::Create { .. } => KIND_CREATE_RDF_GRAPH,
            Self::Drop { .. } => KIND_DROP_RDF_GRAPH,
            Self::Clear { .. } => KIND_CLEAR_RDF_GRAPH,
            Self::Copy { .. } => KIND_COPY_RDF_GRAPH,
            Self::Move { .. } => KIND_MOVE_RDF_GRAPH,
            Self::Add { .. } => KIND_ADD_RDF_GRAPH,
        }
    }
}

impl LogRecord {
    /// The record's kind, the first byte of its frame (see the module
    /// documentation's table).
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::GroupBegin(_) => KIND_GROUP_BEGIN,
            Self::Data { op, .. } => op.kind(),
            Self::Standalone(op) => op.kind(),
        }
    }

    /// The graph a data record writes: its storage key, and the model of
    /// its op. `None` for the other records.
    #[must_use]
    pub fn graph(&self) -> Option<GraphRef> {
        match self {
            Self::Data { graph, op } => Some(GraphRef {
                model: op.model(),
                key: graph.as_deref().map(ArcStr::from),
            }),
            Self::GroupBegin(_) | Self::Standalone(_) => None,
        }
    }

    /// The record, borrowed.
    #[must_use]
    pub fn borrowed(&self) -> LogRecordRef<'_> {
        match self {
            Self::GroupBegin(group) => LogRecordRef::GroupBegin(*group),
            Self::Data { graph, op } => LogRecordRef::Data {
                graph: graph.as_deref(),
                op,
            },
            Self::Standalone(op) => LogRecordRef::Standalone(op),
        }
    }

    /// Appends the record framed: `[kind u8][flags u8][length u32 LE][payload]`.
    /// Every kind of this release is written with [`RECORD_REQUIRED`].
    ///
    /// # Errors
    ///
    /// As [`LogRecordRef::encode_framed`].
    pub fn encode_framed(&self, out: &mut Vec<u8>) -> Result<()> {
        self.borrowed().encode_framed(out)
    }

    /// The record of `kind` whose payload is `payload`, or `None` for a
    /// record this release does not know and may skip (see
    /// [`read_framed_records`]).
    fn decode_payload(kind: u8, required: bool, payload: &[u8]) -> Result<Option<Self>> {
        let data = |graph: Option<String>, op: DataOp| Self::Data { graph, op };
        let rdf_graph = |op: RdfGraphOp| Self::Standalone(StandaloneOp::RdfGraph(op));
        let record = match kind {
            KIND_GROUP_BEGIN => {
                let (epoch, timestamp_ms, origin): (u64, u64, Origin) = decode(payload)?;
                Self::GroupBegin(GroupBegin {
                    epoch: EpochId::new(epoch),
                    timestamp_ms,
                    origin,
                })
            }
            KIND_CREATE_NODE => {
                let (Graph(graph), id, OwnedLabels(labels), OwnedProperties(properties)) =
                    decode(payload)?;
                data(
                    graph,
                    DataOp::CreateNode {
                        id: NodeId::new(id),
                        labels,
                        properties,
                    },
                )
            }
            KIND_DELETE_NODE => {
                let (Graph(graph), id) = decode(payload)?;
                data(
                    graph,
                    DataOp::DeleteNode {
                        id: NodeId::new(id),
                    },
                )
            }
            KIND_CREATE_EDGE => {
                let (Graph(graph), id, src, dst, Name(edge_type), OwnedProperties(properties)) =
                    decode(payload)?;
                data(
                    graph,
                    DataOp::CreateEdge {
                        id: EdgeId::new(id),
                        src: NodeId::new(src),
                        dst: NodeId::new(dst),
                        edge_type,
                        properties,
                    },
                )
            }
            KIND_DELETE_EDGE => {
                let (Graph(graph), id) = decode(payload)?;
                data(
                    graph,
                    DataOp::DeleteEdge {
                        id: EdgeId::new(id),
                    },
                )
            }
            KIND_SET_NODE_PROPERTY => {
                let (Graph(graph), id, Name(key), OwnedValue(value)) = decode(payload)?;
                data(
                    graph,
                    DataOp::SetNodeProperty {
                        id: NodeId::new(id),
                        key: PropertyKey::new(key),
                        value,
                    },
                )
            }
            KIND_REMOVE_NODE_PROPERTY => {
                let (Graph(graph), id, Name(key)) = decode(payload)?;
                data(
                    graph,
                    DataOp::RemoveNodeProperty {
                        id: NodeId::new(id),
                        key: PropertyKey::new(key),
                    },
                )
            }
            KIND_SET_EDGE_PROPERTY => {
                let (Graph(graph), id, Name(key), OwnedValue(value)) = decode(payload)?;
                data(
                    graph,
                    DataOp::SetEdgeProperty {
                        id: EdgeId::new(id),
                        key: PropertyKey::new(key),
                        value,
                    },
                )
            }
            KIND_REMOVE_EDGE_PROPERTY => {
                let (Graph(graph), id, Name(key)) = decode(payload)?;
                data(
                    graph,
                    DataOp::RemoveEdgeProperty {
                        id: EdgeId::new(id),
                        key: PropertyKey::new(key),
                    },
                )
            }
            KIND_ADD_NODE_LABEL => {
                let (Graph(graph), id, Name(label)) = decode(payload)?;
                data(
                    graph,
                    DataOp::AddNodeLabel {
                        id: NodeId::new(id),
                        label,
                    },
                )
            }
            KIND_REMOVE_NODE_LABEL => {
                let (Graph(graph), id, Name(label)) = decode(payload)?;
                data(
                    graph,
                    DataOp::RemoveNodeLabel {
                        id: NodeId::new(id),
                        label,
                    },
                )
            }
            KIND_CREATE_GRAPH => {
                let Text(name) = decode(payload)?;
                Self::Standalone(StandaloneOp::CreateGraph { name })
            }
            KIND_DROP_GRAPH => {
                let Text(name) = decode(payload)?;
                Self::Standalone(StandaloneOp::DropGraph { name })
            }
            KIND_PUT_CATALOG => match decode_catalog_record(required, payload)? {
                Some(record) => Self::Standalone(StandaloneOp::PutCatalog(record)),
                None => return Ok(None),
            },
            KIND_DROP_CATALOG => match decode_catalog_key(required, payload)? {
                Some(key) => Self::Standalone(StandaloneOp::DropCatalog(key)),
                None => return Ok(None),
            },
            KIND_INSERT_TRIPLE => {
                let (Graph(graph), triple) = decode(payload)?;
                data(
                    graph,
                    DataOp::InsertTriple {
                        triple: Box::new(triple),
                    },
                )
            }
            KIND_DELETE_TRIPLE => {
                let (Graph(graph), triple) = decode(payload)?;
                data(
                    graph,
                    DataOp::DeleteTriple {
                        triple: Box::new(triple),
                    },
                )
            }
            KIND_CREATE_RDF_GRAPH => {
                let Text(name) = decode(payload)?;
                rdf_graph(RdfGraphOp::Create { name })
            }
            KIND_DROP_RDF_GRAPH => rdf_graph(RdfGraphOp::Drop {
                target: decode(payload)?,
            }),
            KIND_CLEAR_RDF_GRAPH => rdf_graph(RdfGraphOp::Clear {
                target: decode(payload)?,
            }),
            KIND_COPY_RDF_GRAPH => {
                let (Graph(source), Graph(target)) = decode(payload)?;
                rdf_graph(RdfGraphOp::Copy { source, target })
            }
            KIND_MOVE_RDF_GRAPH => {
                let (Graph(source), Graph(target)) = decode(payload)?;
                rdf_graph(RdfGraphOp::Move { source, target })
            }
            KIND_ADD_RDF_GRAPH => {
                let (Graph(source), Graph(target)) = decode(payload)?;
                rdf_graph(RdfGraphOp::Add { source, target })
            }
            _ => return Ok(None),
        };
        Ok(Some(record))
    }
}

impl LogRecordRef<'_> {
    /// The record's kind, the first byte of its frame.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::GroupBegin(_) => KIND_GROUP_BEGIN,
            Self::Data { op, .. } => op.kind(),
            Self::Standalone(op) => op.kind(),
        }
    }

    /// Appends the record framed: `[kind u8][flags u8][length u32 LE][payload]`,
    /// the bytes of the [`LogRecord`] it stands for. Every kind of this
    /// release is written with [`RECORD_REQUIRED`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when the framed record would take
    /// more than [`MAX_RECORD_BYTES`] (for values, counted before one is
    /// encoded), a list of labels or properties is past what a reader decodes
    /// (see the module documentation), the value codec refuses a value, or
    /// the catalog refuses the catalog record of a
    /// [`PutCatalog`](StandaloneOp::PutCatalog) (as the catalog section
    /// would, its maximum included). `out` is then left as it was.
    pub fn encode_framed(&self, out: &mut Vec<u8>) -> Result<()> {
        encode_framed_record(&LOG_FRAMING, self.kind(), RECORD_REQUIRED, out, |out| {
            // The value codec encodes each value whole before the payload
            // takes it: refuse them from their sizes first.
            out.check_room(self.value_bytes())?;
            match *self {
                Self::GroupBegin(group) => encode_bincode_payload(
                    &(group.epoch.as_u64(), group.timestamp_ms, group.origin),
                    out,
                ),
                Self::Data { graph, op } => {
                    check_list_slots(op)?;
                    encode_data(graph, op, out)
                }
                Self::Standalone(op) => encode_standalone(op, out),
            }
        })
    }

    /// The bytes the value codec writes for the record's values, counted
    /// without encoding them. A value nested deeper than a property value may
    /// be ([`nests_too_deep`]) counts nothing, so the count recurses no
    /// deeper than that: the value codec refuses it, as it reaches the depth
    /// it refuses, or it is written through the payload's maximum as before.
    fn value_bytes(&self) -> usize {
        let counted = |value: &Value| {
            if nests_too_deep(value) {
                0
            } else {
                encoded_len(value)
            }
        };
        let Self::Data { op, .. } = self else {
            return 0;
        };
        match op {
            DataOp::CreateNode { properties, .. } | DataOp::CreateEdge { properties, .. } => {
                properties
                    .iter()
                    .map(|(_, value)| counted(value))
                    .fold(0, usize::saturating_add)
            }
            DataOp::SetNodeProperty { value, .. } | DataOp::SetEdgeProperty { value, .. } => {
                counted(value)
            }
            DataOp::DeleteNode { .. }
            | DataOp::DeleteEdge { .. }
            | DataOp::RemoveNodeProperty { .. }
            | DataOp::RemoveEdgeProperty { .. }
            | DataOp::AddNodeLabel { .. }
            | DataOp::RemoveNodeLabel { .. }
            | DataOp::InsertTriple { .. }
            | DataOp::DeleteTriple { .. } => 0,
        }
    }
}

/// Writes the payload of a data record: its graph, then the op's fields.
fn encode_data(graph: Option<&str>, op: &DataOp, out: &mut PayloadWriter<'_>) -> Result<()> {
    match op {
        DataOp::CreateNode {
            id,
            labels,
            properties,
        } => encode_bincode_payload(
            &(
                graph,
                id.as_u64(),
                NamesRef(labels),
                PropertiesRef(properties),
            ),
            out,
        ),
        DataOp::DeleteNode { id } => encode_bincode_payload(&(graph, id.as_u64()), out),
        DataOp::CreateEdge {
            id,
            src,
            dst,
            edge_type,
            properties,
        } => encode_bincode_payload(
            &(
                graph,
                id.as_u64(),
                src.as_u64(),
                dst.as_u64(),
                edge_type.as_str(),
                PropertiesRef(properties),
            ),
            out,
        ),
        DataOp::DeleteEdge { id } => encode_bincode_payload(&(graph, id.as_u64()), out),
        DataOp::SetNodeProperty { id, key, value } => {
            encode_bincode_payload(&(graph, id.as_u64(), key.as_str(), ValueRef(value)), out)
        }
        DataOp::RemoveNodeProperty { id, key } => {
            encode_bincode_payload(&(graph, id.as_u64(), key.as_str()), out)
        }
        DataOp::SetEdgeProperty { id, key, value } => {
            encode_bincode_payload(&(graph, id.as_u64(), key.as_str(), ValueRef(value)), out)
        }
        DataOp::RemoveEdgeProperty { id, key } => {
            encode_bincode_payload(&(graph, id.as_u64(), key.as_str()), out)
        }
        DataOp::AddNodeLabel { id, label } | DataOp::RemoveNodeLabel { id, label } => {
            encode_bincode_payload(&(graph, id.as_u64(), label.as_str()), out)
        }
        DataOp::InsertTriple { triple } | DataOp::DeleteTriple { triple } => {
            encode_bincode_payload(&(graph, &**triple), out)
        }
    }
}

/// Writes the payload of a standalone record.
fn encode_standalone(op: &StandaloneOp, out: &mut PayloadWriter<'_>) -> Result<()> {
    match op {
        StandaloneOp::CreateGraph { name } | StandaloneOp::DropGraph { name } => {
            encode_bincode_payload(name, out)
        }
        StandaloneOp::PutCatalog(record) => encode_catalog_record(record, out),
        StandaloneOp::DropCatalog(key) => encode_catalog_key(key, out),
        StandaloneOp::RdfGraph(op) => match op {
            RdfGraphOp::Create { name } => encode_bincode_payload(name, out),
            RdfGraphOp::Drop { target } | RdfGraphOp::Clear { target } => {
                encode_bincode_payload(target, out)
            }
            RdfGraphOp::Copy { source, target }
            | RdfGraphOp::Move { source, target }
            | RdfGraphOp::Add { source, target } => encode_bincode_payload(&(source, target), out),
        },
    }
}

/// Writes a catalog record's kind and the payload the catalog section
/// stores for it, refused as the catalog section refuses it.
fn encode_catalog_record(record: &CatalogRecord, out: &mut PayloadWriter<'_>) -> Result<()> {
    let mut framed = Vec::new();
    record.encode_framed(&mut framed)?;
    io::Write::write_all(out, &[record.kind()])?;
    io::Write::write_all(out, &framed[RECORD_HEADER_BYTES..])?;
    Ok(())
}

/// Writes a catalog key's kind and the key.
fn encode_catalog_key(key: &CatalogKey, out: &mut PayloadWriter<'_>) -> Result<()> {
    io::Write::write_all(out, &[key.kind()])?;
    match key {
        CatalogKey::Schema(name)
        | CatalogKey::NodeType(name)
        | CatalogKey::EdgeType(name)
        | CatalogKey::GraphType(name)
        | CatalogKey::GraphBinding(name)
        | CatalogKey::Constraint(name)
        | CatalogKey::IndexName(name)
        | CatalogKey::Procedure(name) => encode_bincode_payload(name, out),
        CatalogKey::Index { graph, index } => encode_bincode_payload(&(graph, index), out),
    }
}

/// Splits a catalog record's or key's kind from what follows it.
fn catalog_kind<'a>(what: &str, payload: &'a [u8]) -> Result<(u8, &'a [u8])> {
    match payload.split_first() {
        Some((&0, _)) => Err(Error::Serialization(format!(
            "a {what} of kind 0, which is never written"
        ))),
        Some((&kind, rest)) => Ok((kind, rest)),
        None => Err(Error::Serialization(format!(
            "the payload is empty: it holds no {what} kind"
        ))),
    }
}

/// The error for a required catalog record or key of a kind this release
/// does not know.
fn unknown_catalog_kind(what: &str, kind: u8) -> Error {
    Error::Serialization(format!(
        "a {what} of kind {kind}, which this release does not know (a newer release wrote it), \
         in a required record"
    ))
}

/// Reads the catalog record of a [`PutCatalog`](StandaloneOp::PutCatalog),
/// as the catalog section reads it, or `None` for a kind this release does
/// not know in a record it may skip.
fn decode_catalog_record(required: bool, payload: &[u8]) -> Result<Option<CatalogRecord>> {
    const WHAT: &str = "catalog record";
    let (kind, catalog_payload) = catalog_kind(WHAT, payload)?;
    if !u32::try_from(catalog_payload.len())
        .is_ok_and(|length| length <= MAX_CATALOG_RECORD_PAYLOAD)
    {
        return Err(Error::Serialization(format!(
            "a {WHAT} of {} bytes, more than the {MAX_CATALOG_RECORD_PAYLOAD} a {WHAT} may hold",
            catalog_payload.len()
        )));
    }
    match CatalogRecord::decode_payload(kind, catalog_payload)? {
        Some(record) => Ok(Some(record)),
        None if required => Err(unknown_catalog_kind(WHAT, kind)),
        None => Ok(None),
    }
}

/// Reads the key of a [`DropCatalog`](StandaloneOp::DropCatalog), or `None`
/// for a kind this release does not know in a record it may skip.
fn decode_catalog_key(required: bool, payload: &[u8]) -> Result<Option<CatalogKey>> {
    const WHAT: &str = "catalog key";
    let (kind, key) = catalog_kind(WHAT, payload)?;
    let name = || decode(key).map(|Text(name)| name);
    let key = match kind {
        1 => CatalogKey::Schema(name()?),
        2 => CatalogKey::NodeType(name()?),
        3 => CatalogKey::EdgeType(name()?),
        4 => CatalogKey::GraphType(name()?),
        5 => CatalogKey::GraphBinding(name()?),
        6 => CatalogKey::Constraint(name()?),
        7 => {
            let (Graph(graph), index): (Graph, IndexKeyRecord) = decode(key)?;
            CatalogKey::Index { graph, index }
        }
        8 => CatalogKey::IndexName(name()?),
        9 => CatalogKey::Procedure(name()?),
        _ if required => return Err(unknown_catalog_kind(WHAT, kind)),
        _ => return Ok(None),
    };
    Ok(Some(key))
}

/// Reads the framed log records of `payload` (the records of a WAL frame)
/// and calls `apply` with each record of a known kind, in order.
///
/// A record of an unknown kind, or a [`PutCatalog`](StandaloneOp::PutCatalog)
/// or [`DropCatalog`](StandaloneOp::DropCatalog) of an unknown catalog kind,
/// is skipped when its required flag is clear. The records must fill
/// `payload` exactly.
///
/// # Errors
///
/// Returns [`Error::Serialization`] naming the record (its index, from 0,
/// and the byte of `payload` it starts at) for:
///
/// - kind 0, or a flag among bits 0 to 3 other than [`RECORD_REQUIRED`]
///   (bits 4 to 7 are ignored);
/// - an unknown kind, or a catalog record of an unknown kind, with the
///   required flag set;
/// - a framed record longer than [`MAX_RECORD_BYTES`], or a catalog record
///   longer than [`MAX_CATALOG_RECORD_PAYLOAD`];
/// - a payload that ends inside a record's header or payload;
/// - a record payload that does not decode as its kind's fields, or has
///   bytes left over.
///
/// An error of `apply` comes back as it is and stops the read.
pub fn read_log_records(
    payload: &[u8],
    apply: &mut dyn FnMut(LogRecord) -> Result<()>,
) -> Result<()> {
    let mut reader = payload;
    read_framed_records(
        &LOG_FRAMING,
        &mut reader,
        &mut LogRecord::decode_payload,
        apply,
    )
}

// ── Payload encoding ────────────────────────────────────────────────

/// Decodes a whole payload as bincode (standard configuration) of `T`.
///
/// No bincode limit: `T` reads its strings, byte strings and lists in place
/// (see the module documentation), so a length the payload claims is never
/// allocated, and a limit would bound nothing.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when the payload does not decode or has
/// bytes left over.
fn decode<T: DeserializeOwned>(payload: &[u8]) -> Result<T> {
    match bincode::serde::decode_from_slice::<T, _>(payload, bincode::config::standard()) {
        Ok((record, used)) if used == payload.len() => Ok(record),
        Ok((_, used)) => Err(Error::Serialization(format!(
            "the payload decodes from {used} of its {} bytes; the rest is left over",
            payload.len()
        ))),
        Err(error) => Err(Error::Serialization(format!(
            "the payload does not decode: {error}"
        ))),
    }
}

/// A value, written as the bytes of the lossless value codec.
struct ValueRef<'a>(&'a Value);

impl Serialize for ValueRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let mut bytes = Vec::new();
        encode_value(self.0, &mut bytes).map_err(S::Error::custom)?;
        serializer.serialize_bytes(&bytes)
    }
}

/// Properties, written as a list of (key, value) pairs.
struct PropertiesRef<'a>(&'a [(PropertyKey, Value)]);

impl Serialize for PropertiesRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_seq(
            self.0
                .iter()
                .map(|(key, value)| (key.as_str(), ValueRef(value))),
        )
    }
}

/// Names (labels), written as a list of strings.
struct NamesRef<'a>(&'a [ArcStr]);

impl Serialize for NamesRef<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(ArcStr::as_str))
    }
}

/// A value read as [`ValueRef`] writes it, in place: the value codec reads
/// it from the payload's bytes and checks every count against the bytes
/// left.
struct OwnedValue(Value);

impl<'de> Deserialize<'de> for OwnedValue {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_bytes(ValueVisitor).map(OwnedValue)
    }
}

struct ValueVisitor;

impl Visitor<'_> for ValueVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the bytes of one encoded value")
    }

    fn visit_bytes<E: serde::de::Error>(self, bytes: &[u8]) -> std::result::Result<Value, E> {
        let mut pos = 0;
        let value = decode_value(bytes, &mut pos).map_err(E::custom)?;
        if pos != bytes.len() {
            return Err(E::custom(format_args!(
                "value codec: {} trailing bytes after a value of {pos} bytes",
                bytes.len() - pos
            )));
        }
        Ok(value)
    }
}

// ── The list rule ───────────────────────────────────────────────────
//
// A label decodes into a name: 8 bytes in its list and, unless it is
// empty, a shared string that keeps a header of 16 bytes (its length and
// its count) before its bytes. A property decodes into a (key, value) pair
// of 48 bytes in its list and its key's header. That is up to 24 and 64
// bytes from as little as one and three bytes of payload: unchecked, a
// record of empty labels decoded into 13 times its size while its list
// grew, and one of properties of an empty key and a null into 25 times. So
// a list of labels or properties is held to a rule, as it decodes and as
// it is written: its slots (an element's place in the list and its name's
// header) take at most `SLOT_BYTES_PER_BYTE` bytes per byte its elements
// are encoded in, plus `LIST_ALLOWANCE_BYTES`, at every point of the list.
// The reader refuses the first element that breaks it, before that element
// takes a slot; the writer refuses such a record before it writes a byte,
// so every record written decodes. No record the engine writes comes near
// it: labels and property keys are distinct names, and the allowance alone
// holds 43,690 labels or 16,384 properties.
//
// What a list holds, decoded: at most 8 bytes of slots per byte of its
// elements plus 1 MiB, and while it grows (the old buffer and the new one
// during a copy) at most twice that. Its names' bytes take what they take
// in the payload. Values are decoded by the value codec, as documented
// there.

/// The bytes of slots a list of labels or properties may take per byte its
/// elements are encoded in (see "The list rule" above).
const SLOT_BYTES_PER_BYTE: usize = 8;

/// The bytes of slots a list of labels or properties may take beyond
/// [`SLOT_BYTES_PER_BYTE`] per byte of its elements: 1 MiB.
const LIST_ALLOWANCE_BYTES: usize = 1 << 20;

/// The slot of a label, as the rule counts it on every target: its name's
/// place in [`Labels`] and its name's header ([`NAME_HEADER_BYTES`]), 24
/// bytes on a 64-bit target and fewer on a 32-bit one.
const LABEL_SLOT_BYTES: usize = 24;

/// The slot of a property, as the rule counts it on every target: its
/// (key, value) pair's place in [`Properties`] and its key's header
/// ([`NAME_HEADER_BYTES`]), 64 bytes on a 64-bit target.
const PROPERTY_SLOT_BYTES: usize = 64;

/// The header a name's shared string ([`ArcStr`]) keeps before its bytes,
/// in the same allocation: its length and its count. An empty name is a
/// static string and takes none.
const NAME_HEADER_BYTES: usize = 2 * size_of::<usize>();

const _: () = assert!(
    size_of::<ArcStr>() + NAME_HEADER_BYTES <= LABEL_SLOT_BYTES
        && size_of::<(PropertyKey, Value)>() + NAME_HEADER_BYTES <= PROPERTY_SLOT_BYTES,
    "the rule counts at least the slots a list takes"
);

/// The bytes bincode's standard configuration takes for a length: one below
/// 251, then a marker and 2, 4 or 8 bytes.
const fn varint_bytes(length: usize) -> usize {
    if length < 251 {
        1
    } else if length <= 0xFFFF {
        3
    } else if length <= 0xFFFF_FFFF {
        5
    } else {
        9
    }
}

/// The bytes a string of `length` bytes takes in a payload.
const fn string_bytes(length: usize) -> usize {
    varint_bytes(length).saturating_add(length)
}

/// The bytes a property takes in a payload: its key, then its value as a
/// byte string of the value codec's encoding.
fn property_bytes(key: &str, value: &Value) -> usize {
    // A value nested deeper than a write accepts is refused by the codec;
    // counting it as empty keeps the count from recursing that deep.
    let value_length = if nests_too_deep(value) {
        0
    } else {
        encoded_len(value)
    };
    string_bytes(key.len()).saturating_add(string_bytes(value_length))
}

/// The list rule, followed as a list's elements are added.
struct ListSlots {
    /// What the list holds, for an error.
    what: &'static str,
    /// The slot of one element.
    slot_bytes: usize,
    elements: usize,
    /// The bytes the elements so far are encoded in.
    encoded: usize,
}

impl ListSlots {
    const fn new(what: &'static str, slot_bytes: usize) -> Self {
        Self {
            what,
            slot_bytes,
            elements: 0,
            encoded: 0,
        }
    }

    /// Adds an element encoded in `encoded` bytes, refusing it when the
    /// list's slots would pass the rule with it.
    fn add(&mut self, encoded: usize) -> std::result::Result<(), String> {
        self.elements = self.elements.saturating_add(1);
        self.encoded = self.encoded.saturating_add(encoded);
        let slots = self.elements.saturating_mul(self.slot_bytes);
        if slots > self.allowance() {
            return Err(format!(
                "a list of {} {} would take {slots} bytes of slots from {} bytes, more than \
                 {SLOT_BYTES_PER_BYTE} per byte plus {LIST_ALLOWANCE_BYTES}",
                self.elements, self.what, self.encoded
            ));
        }
        Ok(())
    }

    /// The bytes of slots the list may take now.
    fn allowance(&self) -> usize {
        self.encoded
            .saturating_mul(SLOT_BYTES_PER_BYTE)
            .saturating_add(LIST_ALLOWANCE_BYTES)
    }

    /// How many elements the list may hold now.
    fn allowed_elements(&self) -> usize {
        self.allowance() / self.slot_bytes
    }
}

/// Refuses an op whose labels or properties break the list rule, from the
/// sizes they will be encoded in, as the reader would refuse them.
fn check_list_slots(op: &DataOp) -> Result<()> {
    let refused = |reason: String| Error::Serialization(format!("refused to write {reason}"));
    // A list that fits the allowance cannot break the rule at any point.
    let fits = |count: usize, slot: usize| count.saturating_mul(slot) <= LIST_ALLOWANCE_BYTES;
    let properties_of = |properties: &[(PropertyKey, Value)]| -> Result<()> {
        if fits(properties.len(), PROPERTY_SLOT_BYTES) {
            return Ok(());
        }
        let mut slots = ListSlots::new(<(Name, OwnedValue)>::WHAT, PROPERTY_SLOT_BYTES);
        for (key, value) in properties {
            slots
                .add(property_bytes(key.as_str(), value))
                .map_err(refused)?;
        }
        Ok(())
    };
    match op {
        DataOp::CreateNode {
            labels, properties, ..
        } => {
            if !fits(labels.len(), LABEL_SLOT_BYTES) {
                let mut slots = ListSlots::new(Name::WHAT, LABEL_SLOT_BYTES);
                for label in labels {
                    slots.add(string_bytes(label.len())).map_err(refused)?;
                }
            }
            properties_of(properties)
        }
        DataOp::CreateEdge { properties, .. } => properties_of(properties),
        DataOp::DeleteNode { .. }
        | DataOp::DeleteEdge { .. }
        | DataOp::SetNodeProperty { .. }
        | DataOp::RemoveNodeProperty { .. }
        | DataOp::SetEdgeProperty { .. }
        | DataOp::RemoveEdgeProperty { .. }
        | DataOp::AddNodeLabel { .. }
        | DataOp::RemoveNodeLabel { .. }
        | DataOp::InsertTriple { .. }
        | DataOp::DeleteTriple { .. } => Ok(()),
    }
}

/// An element of a list read with [`Grown`]: its slot and the bytes it
/// takes in the payload, for the list rule.
trait ListElement {
    /// What a list of such elements holds, for an error.
    const WHAT: &'static str;
    /// The slot of one element, as the rule counts it.
    const SLOT_BYTES: usize;
    /// The bytes the element took in the payload.
    fn encoded_bytes(&self) -> usize;
}

/// A name in a list is a label: only labels come as a list of names.
impl ListElement for Name {
    const WHAT: &'static str = "labels";
    const SLOT_BYTES: usize = LABEL_SLOT_BYTES;

    fn encoded_bytes(&self) -> usize {
        string_bytes(self.0.len())
    }
}

impl ListElement for (Name, OwnedValue) {
    const WHAT: &'static str = "properties";
    const SLOT_BYTES: usize = PROPERTY_SLOT_BYTES;

    fn encoded_bytes(&self) -> usize {
        property_bytes(&self.0.0, &self.1.0)
    }
}

/// The list a [`Grown`] decodes into, as the change set holds it: it takes
/// each element as it decodes, and grows only as far as it is told.
trait GrownList<T>: Default {
    /// The elements it holds.
    fn len(&self) -> usize;
    /// The elements it has room for.
    fn capacity(&self) -> usize;
    /// Makes room for exactly `additional` more elements.
    fn reserve_exact(&mut self, additional: usize);
    /// Takes an element, into room made for it.
    fn push(&mut self, element: T);
}

impl GrownList<Name> for Labels {
    fn len(&self) -> usize {
        SmallVec::len(self)
    }

    fn capacity(&self) -> usize {
        SmallVec::capacity(self)
    }

    fn reserve_exact(&mut self, additional: usize) {
        SmallVec::reserve_exact(self, additional);
    }

    fn push(&mut self, Name(label): Name) {
        SmallVec::push(self, label);
    }
}

impl GrownList<(Name, OwnedValue)> for Properties {
    fn len(&self) -> usize {
        Vec::len(self)
    }

    fn capacity(&self) -> usize {
        Vec::capacity(self)
    }

    fn reserve_exact(&mut self, additional: usize) {
        Vec::reserve_exact(self, additional);
    }

    fn push(&mut self, (Name(key), OwnedValue(value)): (Name, OwnedValue)) {
        Vec::push(self, (PropertyKey::new(key), value));
    }
}

/// A list read element by element into `C` under the list rule: it grows
/// as its elements decode, never ahead of what the rule allows for the
/// elements read so far, so the count the payload claims is never
/// allocated ahead of them (serde's own lists allocate up to 1 MiB ahead),
/// and a list that breaks the rule is refused at the element that breaks
/// it.
struct Grown<T, C>(C, PhantomData<T>);

impl<'de, T: Deserialize<'de> + ListElement, C: GrownList<T>> Deserialize<'de> for Grown<T, C> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_seq(GrownVisitor(PhantomData))
    }
}

struct GrownVisitor<T, C>(PhantomData<(T, C)>);

impl<'de, T: Deserialize<'de> + ListElement, C: GrownList<T>> Visitor<'de> for GrownVisitor<T, C> {
    type Value = Grown<T, C>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a list")
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Grown<T, C>, A::Error> {
        // The count the payload claims caps the growth, so a list ends with
        // no spare capacity; it is never reserved ahead of the elements.
        let claimed = seq.size_hint().unwrap_or(usize::MAX);
        // A list whose count fits the allowance cannot break the rule, so
        // its elements' sizes are not counted (a value's would be walked).
        let ruled = claimed.saturating_mul(T::SLOT_BYTES) > LIST_ALLOWANCE_BYTES;
        let mut slots = ListSlots::new(T::WHAT, T::SLOT_BYTES);
        let mut items = C::default();
        while let Some(item) = seq.next_element::<T>()? {
            let encoded = if ruled { item.encoded_bytes() } else { 0 };
            slots.add(encoded).map_err(serde::de::Error::custom)?;
            if items.len() == items.capacity() {
                let length = items.len();
                let target = length
                    .saturating_mul(2)
                    .max(16)
                    .min(claimed)
                    .min(slots.allowed_elements())
                    .max(length + 1);
                items.reserve_exact(target - length);
            }
            items.push(item);
        }
        Ok(Grown(items, PhantomData))
    }
}

/// Properties read as [`PropertiesRef`] writes them, in place.
struct OwnedProperties(Properties);

impl<'de> Deserialize<'de> for OwnedProperties {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let Grown(properties, _) =
            Grown::<(Name, OwnedValue), Properties>::deserialize(deserializer)?;
        Ok(OwnedProperties(properties))
    }
}

/// Labels read as [`NamesRef`] writes them, in place.
struct OwnedLabels(Labels);

impl<'de> Deserialize<'de> for OwnedLabels {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let Grown(labels, _) = Grown::<Name, Labels>::deserialize(deserializer)?;
        Ok(OwnedLabels(labels))
    }
}

/// A graph's storage key, read in place; `None` for the default graph.
struct Graph(Option<String>);

impl<'de> Deserialize<'de> for Graph {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        optional_text(deserializer).map(Graph)
    }
}

/// A name (a label, an edge type, a property key), read in place into the
/// shared string the change set holds (see [`text`]).
struct Name(ArcStr);

impl<'de> Deserialize<'de> for Name {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        deserializer.deserialize_str(NameVisitor).map(Name)
    }
}

struct NameVisitor;

impl Visitor<'_> for NameVisitor {
    type Value = ArcStr;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a name")
    }

    fn visit_str<E: serde::de::Error>(self, name: &str) -> std::result::Result<ArcStr, E> {
        Ok(ArcStr::from(name))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use super::*;
    use crate::storage::catalog_record::{
        ConstraintRecord, DistanceMetricRecord, EdgeTypeRecord, EndpointPair, GraphBindingRecord,
        GraphTypeRecord, IndexKindRecord, IndexNameKindRecord, IndexNameRecord, IndexRecord,
        NamedConstraintKindRecord, NodeTypeRecord, ProcedureRecord, PropertyRecord,
        PropertyTypeRecord, QuantizationRecord, SchemaRecord, read_catalog_records,
    };
    use crate::types::{Date, Duration, Time, Timestamp, ZonedDatetime};

    const XSD_STRING: &str = "http://www.w3.org/2001/XMLSchema#string";

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    fn labels(items: &[&str]) -> Labels {
        items.iter().map(|&item| ArcStr::from(item)).collect()
    }

    fn iri(value: &str) -> TermRecord {
        TermRecord::Iri(value.to_string())
    }

    fn triple(subject: TermRecord, predicate: TermRecord, object: TermRecord) -> Box<TripleRecord> {
        Box::new(TripleRecord {
            subject,
            predicate,
            object,
        })
    }

    /// A data record of `op` in the graph of storage key `graph`.
    fn data(graph: Option<&str>, op: DataOp) -> LogRecord {
        LogRecord::Data {
            graph: graph.map(str::to_string),
            op,
        }
    }

    fn standalone(op: StandaloneOp) -> LogRecord {
        LogRecord::Standalone(op)
    }

    fn rdf_graph(op: RdfGraphOp) -> LogRecord {
        LogRecord::Standalone(StandaloneOp::RdfGraph(op))
    }

    fn framed(record: &LogRecord) -> Vec<u8> {
        let mut bytes = Vec::new();
        record.encode_framed(&mut bytes).unwrap();
        bytes
    }

    fn read_all(bytes: &[u8]) -> Result<Vec<LogRecord>> {
        let mut records = Vec::new();
        read_log_records(bytes, &mut |record| {
            records.push(record);
            Ok(())
        })?;
        Ok(records)
    }

    fn read_error(bytes: &[u8]) -> String {
        match read_all(bytes) {
            Ok(records) => panic!("{} records read, expected an error", records.len()),
            Err(error) => error.to_string(),
        }
    }

    fn encoded_value(value: &Value) -> Vec<u8> {
        let mut bytes = Vec::new();
        encode_value(value, &mut bytes).unwrap();
        bytes
    }

    /// Whether two values are the same bit for bit: the value codec gives
    /// every value exactly one encoding, floats by their bits and times with
    /// their offsets, where `Value`'s own equality says NaN is not NaN and
    /// compares times by their instant.
    fn same(left: &Value, right: &Value) -> bool {
        encoded_value(left) == encoded_value(right)
    }

    fn schema(name: &str) -> CatalogRecord {
        CatalogRecord::Schema(SchemaRecord {
            name: name.to_string(),
        })
    }

    /// One record of every kind, in kind order.
    fn one_of_each() -> Vec<LogRecord> {
        vec![
            LogRecord::GroupBegin(GroupBegin {
                epoch: EpochId::new(88),
                timestamp_ms: 1_696_500_000_123,
                origin: Origin::Transaction,
            }),
            data(
                None,
                DataOp::CreateNode {
                    id: NodeId::new(3),
                    labels: labels(&["Person"]),
                    properties: vec![("name".into(), Value::from("Alix"))],
                },
            ),
            data(
                Some("trips"),
                DataOp::DeleteNode {
                    id: NodeId::new(19),
                },
            ),
            data(
                None,
                DataOp::CreateEdge {
                    id: EdgeId::new(88),
                    src: NodeId::new(3),
                    dst: NodeId::new(19),
                    edge_type: "KNOWS".into(),
                    properties: vec![("since".into(), Value::Int64(1988))],
                },
            ),
            data(
                None,
                DataOp::DeleteEdge {
                    id: EdgeId::new(88),
                },
            ),
            data(
                None,
                DataOp::SetNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                    value: Value::from("Paris"),
                },
            ),
            data(
                None,
                DataOp::RemoveNodeProperty {
                    id: NodeId::new(3),
                    key: "city".into(),
                },
            ),
            data(
                Some("trips"),
                DataOp::SetEdgeProperty {
                    id: EdgeId::new(88),
                    key: "km".into(),
                    value: Value::Int64(319),
                },
            ),
            data(
                Some("trips"),
                DataOp::RemoveEdgeProperty {
                    id: EdgeId::new(88),
                    key: "km".into(),
                },
            ),
            data(
                None,
                DataOp::AddNodeLabel {
                    id: NodeId::new(19),
                    label: "City".into(),
                },
            ),
            data(
                None,
                DataOp::RemoveNodeLabel {
                    id: NodeId::new(19),
                    label: "City".into(),
                },
            ),
            standalone(StandaloneOp::CreateGraph {
                name: "trips".into(),
            }),
            standalone(StandaloneOp::DropGraph {
                name: "trips".into(),
            }),
            standalone(StandaloneOp::PutCatalog(schema("travel"))),
            standalone(StandaloneOp::DropCatalog(CatalogKey::Schema(
                "travel".into(),
            ))),
            data(
                None,
                DataOp::InsertTriple {
                    triple: triple(
                        iri("ex:Gus"),
                        iri("ex:knows"),
                        TermRecord::Blank("b3".into()),
                    ),
                },
            ),
            data(
                Some("ex:g"),
                DataOp::DeleteTriple {
                    triple: triple(
                        TermRecord::Blank("b3".into()),
                        iri("ex:name"),
                        TermRecord::Literal {
                            value: "Mia".into(),
                            datatype: "xsd:string".into(),
                            language: Some("nl".into()),
                        },
                    ),
                },
            ),
            rdf_graph(RdfGraphOp::Create {
                name: "ex:g".into(),
            }),
            rdf_graph(RdfGraphOp::Drop {
                target: RdfGraphTarget::Named("ex:g".into()),
            }),
            rdf_graph(RdfGraphOp::Clear {
                target: RdfGraphTarget::All,
            }),
            rdf_graph(RdfGraphOp::Copy {
                source: None,
                target: Some("ex:b".into()),
            }),
            rdf_graph(RdfGraphOp::Move {
                source: Some("ex:b".into()),
                target: Some("ex:g".into()),
            }),
            rdf_graph(RdfGraphOp::Add {
                source: Some("ex:g".into()),
                target: None,
            }),
        ]
    }

    /// Records holding every variant of every enum the records use, and a
    /// catalog record and key of every catalog kind.
    fn every_variant() -> Vec<LogRecord> {
        let mut records: Vec<LogRecord> = [
            Origin::Transaction,
            Origin::Statement,
            Origin::Direct,
            Origin::Schema,
            Origin::Bulk,
        ]
        .into_iter()
        .map(|origin| {
            LogRecord::GroupBegin(GroupBegin {
                epoch: EpochId::new(u64::MAX),
                timestamp_ms: 3,
                origin,
            })
        })
        .collect();
        let terms = [
            iri("http://example.org/Vincent"),
            TermRecord::Blank("b19".into()),
            TermRecord::Literal {
                value: "Jules".into(),
                datatype: XSD_STRING.into(),
                language: None,
            },
            TermRecord::Literal {
                value: "Amsterdam".into(),
                datatype: "http://www.w3.org/1999/02/22-rdf-syntax-ns#langString".into(),
                language: Some("nl".into()),
            },
        ];
        for object in terms {
            records.push(data(
                Some("http://example.org/people"),
                DataOp::InsertTriple {
                    triple: triple(
                        TermRecord::Blank("b3".into()),
                        iri("http://example.org/name"),
                        object,
                    ),
                },
            ));
        }
        for target in [
            RdfGraphTarget::Default,
            RdfGraphTarget::Named("http://example.org/people".into()),
            RdfGraphTarget::AllNamed,
            RdfGraphTarget::All,
        ] {
            records.push(rdf_graph(RdfGraphOp::Drop {
                target: target.clone(),
            }));
            records.push(rdf_graph(RdfGraphOp::Clear { target }));
        }
        for record in catalog_records() {
            records.push(standalone(StandaloneOp::PutCatalog(record)));
        }
        for key in catalog_keys() {
            records.push(standalone(StandaloneOp::DropCatalog(key)));
        }
        records.push(data(
            Some(""),
            DataOp::CreateNode {
                id: NodeId::new(u64::MAX),
                labels: Labels::new(),
                properties: Vec::new(),
            },
        ));
        records.push(data(
            Some("Prague"),
            DataOp::CreateNode {
                id: NodeId::new(0),
                labels: labels(&["Person", "Employee", ""]),
                properties: vec![
                    ("name".into(), Value::from("Butch")),
                    ("".into(), Value::Null),
                    ("age".into(), Value::Int64(-19)),
                ],
            },
        ));
        records
    }

    /// A catalog record of every catalog kind.
    fn catalog_records() -> Vec<CatalogRecord> {
        vec![
            schema("travel"),
            CatalogRecord::NodeType(NodeTypeRecord {
                name: "City".into(),
                properties: vec![PropertyRecord {
                    name: "founded".into(),
                    data_type: PropertyTypeRecord::Timestamp,
                    nullable: true,
                    default_value: Some(Value::Timestamp(Timestamp::from_micros(
                        1_696_500_000_123_457,
                    ))),
                }],
                constraints: Vec::new(),
                parent_types: names(&["Place"]),
                key_labels: Vec::new(),
            }),
            CatalogRecord::EdgeType(EdgeTypeRecord {
                name: "ROUTE".into(),
                properties: Vec::new(),
                constraints: Vec::new(),
                endpoints: vec![EndpointPair {
                    source: Some("City".into()),
                    target: None,
                }],
                key_labels: Vec::new(),
            }),
            CatalogRecord::GraphType(GraphTypeRecord {
                name: "travel".into(),
                node_types: names(&["City"]),
                edge_types: names(&["ROUTE"]),
                open: true,
            }),
            CatalogRecord::GraphBinding(GraphBindingRecord {
                graph: "trips".into(),
                graph_type: "travel".into(),
            }),
            CatalogRecord::Constraint(ConstraintRecord {
                name: "person_name".into(),
                label: "Person".into(),
                properties: names(&["name"]),
                kind: NamedConstraintKindRecord::Unique,
            }),
            CatalogRecord::Index(IndexRecord {
                graph: Some("trips".into()),
                index: IndexKindRecord::Vector {
                    label: "Doc".into(),
                    property: "emb".into(),
                    dimensions: 88,
                    metric: DistanceMetricRecord::Euclidean,
                    m: 19,
                    ef_construction: 88,
                    quantization: QuantizationRecord::Product { num_subvectors: 3 },
                },
            }),
            CatalogRecord::IndexName(IndexNameRecord {
                name: "person_city".into(),
                label: "Person".into(),
                property: "city".into(),
                kind: IndexNameKindRecord::Hash,
            }),
            CatalogRecord::Procedure(ProcedureRecord {
                name: "visitors".into(),
                params: vec![("city".into(), "STRING".into())],
                returns: vec![("name".into(), "STRING".into())],
                body: "MATCH (p:Person {city: $city}) RETURN p.name AS name".into(),
            }),
        ]
    }

    /// A catalog key of every catalog kind and index kind.
    fn catalog_keys() -> Vec<CatalogKey> {
        vec![
            CatalogKey::Schema("travel".into()),
            CatalogKey::NodeType("City".into()),
            CatalogKey::EdgeType("ROUTE".into()),
            CatalogKey::GraphType("travel".into()),
            CatalogKey::GraphBinding("trips".into()),
            CatalogKey::Constraint("person_name".into()),
            CatalogKey::Index {
                graph: None,
                index: IndexKeyRecord::Property { key: "id".into() },
            },
            CatalogKey::Index {
                graph: Some("trips".into()),
                index: IndexKeyRecord::Vector {
                    label: "Doc".into(),
                    property: "emb".into(),
                },
            },
            CatalogKey::Index {
                graph: Some("Berlin".into()),
                index: IndexKeyRecord::Text {
                    label: "Doc".into(),
                    property: "body".into(),
                },
            },
            CatalogKey::IndexName("person_city".into()),
            CatalogKey::Procedure("visitors".into()),
        ]
    }

    /// One value of every kind, with the edge cases a lossy encoding would
    /// change: a NaN payload, -0.0, sub-millisecond timestamps, offsets,
    /// counters.
    fn every_value() -> Vec<Value> {
        let counter = |entries: &[(&str, u64)]| {
            Arc::new(
                entries
                    .iter()
                    .map(|(replica, count)| ((*replica).to_string(), *count))
                    .collect::<HashMap<_, _>>(),
            )
        };
        vec![
            Value::Null,
            Value::Bool(true),
            Value::Int64(i64::MIN),
            Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058)),
            Value::Float64(f64::from_bits(0xFFF0_0000_0000_0019)),
            Value::Float64(-0.0),
            Value::from("Barcelona"),
            Value::Bytes(Arc::from(vec![3u8, 19, 88])),
            Value::Date(Date::from_days(-3)),
            Value::Time(
                Time::from_nanos(3_600_000_000_019)
                    .unwrap()
                    .with_offset(3600),
            ),
            Value::Time(Time::from_nanos(88).unwrap()),
            Value::Timestamp(Timestamp::from_micros(1_696_500_000_123_457)),
            Value::Timestamp(Timestamp::from_micros(-19)),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_000_001),
                7200,
            )),
            Value::Duration(Duration::new(3, 19, 88)),
            Value::List(Arc::from(vec![
                Value::Float64(f64::NAN),
                Value::List(Arc::from(vec![Value::from("Berlin")])),
            ])),
            Value::Map(Arc::new(
                [
                    (PropertyKey::new("city"), Value::from("Prague")),
                    (PropertyKey::new("stops"), Value::Int64(19)),
                ]
                .into_iter()
                .collect(),
            )),
            Value::Vector(Arc::from(vec![3.0f32, -19.5, f32::from_bits(0x7FC0_0058)])),
            Value::Path {
                nodes: Arc::from(vec![Value::Int64(3), Value::Int64(19)]),
                edges: Arc::from(vec![Value::Int64(88)]),
            },
            Value::GCounter(counter(&[("Alix", 3), ("Gus", 19), ("Vincent", 88)])),
            Value::OnCounter {
                pos: counter(&[("Mia", 88)]),
                neg: counter(&[("Jules", 3), ("Butch", 19)]),
            },
        ]
    }

    #[test]
    fn every_record_kind_round_trips() {
        let mut bytes = Vec::new();
        for record in one_of_each() {
            record.encode_framed(&mut bytes).unwrap();
        }
        let back = read_all(&bytes).unwrap();
        assert_eq!(back, one_of_each());
        let kinds: Vec<u8> = back.iter().map(LogRecord::kind).collect();
        assert_eq!(
            kinds,
            [
                1, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 32, 33, 40, 41, 64, 65, 66, 67, 68, 69,
                70, 71
            ],
            "one kind per record, numbered as the module documents"
        );
    }

    #[test]
    fn every_variant_round_trips_and_encodes_to_the_same_bytes_again() {
        for record in every_variant() {
            let bytes = framed(&record);
            let back = read_all(&bytes).unwrap();
            assert_eq!(back, std::slice::from_ref(&record), "{record:?}");
            assert_eq!(framed(&back[0]), bytes, "{record:?}");
        }
    }

    /// A catalog key identifies records of its own catalog kind.
    #[test]
    fn every_catalog_kind_has_a_record_and_a_key() {
        let record_kinds: Vec<u8> = catalog_records().iter().map(CatalogRecord::kind).collect();
        assert_eq!(record_kinds, [1, 2, 3, 4, 5, 6, 7, 8, 9]);
        let key_kinds: Vec<u8> = catalog_keys().iter().map(CatalogKey::kind).collect();
        assert_eq!(key_kinds, [1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9]);
    }

    /// Characterization: pins the bytes of one record of every kind.
    #[test]
    fn the_record_layouts_are_pinned() {
        #[rustfmt::skip]
        const EXPECTED: &[&[u8]] = &[
            &[
                1, 1, 11, 0, 0, 0, // kind 1 (GroupBegin), required, 11 bytes
                88, // epoch 88
                253, 123, 165, 71, 255, 138, 1, 0, 0, // timestamp 1,696,500,000,123 ms (a u64)
                0, // origin Transaction
            ],
            &[
                16, 1, 26, 0, 0, 0, // kind 16 (CreateNode), required, 26 bytes
                0, // the default graph
                3, // id 3
                1, 6, 80, 101, 114, 115, 111, 110, // labels ["Person"]
                1, // one property
                4, 110, 97, 109, 101, // "name"
                9, 4, 4, 0, 0, 0, 65, 108, 105, 120, // the value codec's "Alix", 9 bytes
            ],
            &[
                17, 1, 8, 0, 0, 0, // kind 17 (DeleteNode), required, 8 bytes
                1, 5, 116, 114, 105, 112, 115, // graph "trips"
                19, // id 19
            ],
            &[
                18, 1, 27, 0, 0, 0, // kind 18 (CreateEdge), required, 27 bytes
                0, 88, 3, 19, // the default graph, id 88, from 3 to 19
                5, 75, 78, 79, 87, 83, // type "KNOWS"
                1, // one property
                5, 115, 105, 110, 99, 101, // "since"
                9, 2, 196, 7, 0, 0, 0, 0, 0, 0, // the value codec's 1988, 9 bytes
            ],
            &[
                19, 1, 2, 0, 0, 0, // kind 19 (DeleteEdge), required, 2 bytes
                0, 88, // the default graph, id 88
            ],
            &[
                20, 1, 18, 0, 0, 0, // kind 20 (SetNodeProperty), required, 18 bytes
                0, 3, // the default graph, id 3
                4, 99, 105, 116, 121, // "city"
                10, 4, 5, 0, 0, 0, 80, 97, 114, 105, 115, // the value codec's "Paris", 10 bytes
            ],
            &[
                21, 1, 7, 0, 0, 0, // kind 21 (RemoveNodeProperty), required, 7 bytes
                0, 3, 4, 99, 105, 116, 121, // the default graph, id 3, "city"
            ],
            &[
                22, 1, 21, 0, 0, 0, // kind 22 (SetEdgeProperty), required, 21 bytes
                1, 5, 116, 114, 105, 112, 115, // graph "trips"
                88, 2, 107, 109, // id 88, "km"
                9, 2, 63, 1, 0, 0, 0, 0, 0, 0, // the value codec's 319, 9 bytes
            ],
            &[
                23, 1, 11, 0, 0, 0, // kind 23 (RemoveEdgeProperty), required, 11 bytes
                1, 5, 116, 114, 105, 112, 115, // graph "trips"
                88, 2, 107, 109, // id 88, "km"
            ],
            &[
                24, 1, 7, 0, 0, 0, // kind 24 (AddNodeLabel), required, 7 bytes
                0, 19, 4, 67, 105, 116, 121, // the default graph, id 19, "City"
            ],
            &[
                25, 1, 7, 0, 0, 0, // kind 25 (RemoveNodeLabel), required, 7 bytes
                0, 19, 4, 67, 105, 116, 121, // the default graph, id 19, "City"
            ],
            &[
                32, 1, 6, 0, 0, 0, // kind 32 (CreateGraph), required, 6 bytes
                5, 116, 114, 105, 112, 115, // "trips"
            ],
            &[
                33, 1, 6, 0, 0, 0, // kind 33 (DropGraph), required, 6 bytes
                5, 116, 114, 105, 112, 115, // "trips"
            ],
            &[
                40, 1, 8, 0, 0, 0, // kind 40 (PutCatalog), required, 8 bytes
                1, // catalog kind 1 (Schema)
                6, 116, 114, 97, 118, 101, 108, // its payload: "travel"
            ],
            &[
                41, 1, 8, 0, 0, 0, // kind 41 (DropCatalog), required, 8 bytes
                1, // catalog kind 1 (Schema)
                6, 116, 114, 97, 118, 101, 108, // the key: "travel"
            ],
            &[
                64, 1, 23, 0, 0, 0, // kind 64 (InsertTriple), required, 23 bytes
                0, // the default graph
                0, 6, 101, 120, 58, 71, 117, 115, // an IRI, "ex:Gus"
                0, 8, 101, 120, 58, 107, 110, 111, 119, 115, // an IRI, "ex:knows"
                1, 2, 98, 51, // a blank node, "b3"
            ],
            &[
                65, 1, 39, 0, 0, 0, // kind 65 (DeleteTriple), required, 39 bytes
                1, 4, 101, 120, 58, 103, // graph "ex:g"
                1, 2, 98, 51, // a blank node, "b3"
                0, 7, 101, 120, 58, 110, 97, 109, 101, // an IRI, "ex:name"
                2, 3, 77, 105, 97, // a literal, "Mia"
                10, 120, 115, 100, 58, 115, 116, 114, 105, 110, 103, // of "xsd:string"
                1, 2, 110, 108, // in "nl"
            ],
            &[
                66, 1, 5, 0, 0, 0, // kind 66 (CreateRdfGraph), required, 5 bytes
                4, 101, 120, 58, 103, // "ex:g"
            ],
            &[
                67, 1, 6, 0, 0, 0, // kind 67 (DropRdfGraph), required, 6 bytes
                1, 4, 101, 120, 58, 103, // the named graph "ex:g"
            ],
            &[
                68, 1, 1, 0, 0, 0, // kind 68 (ClearRdfGraph), required, 1 byte
                3, // all graphs
            ],
            &[
                69, 1, 7, 0, 0, 0, // kind 69 (CopyRdfGraph), required, 7 bytes
                0, // from the default graph
                1, 4, 101, 120, 58, 98, // to "ex:b"
            ],
            &[
                70, 1, 12, 0, 0, 0, // kind 70 (MoveRdfGraph), required, 12 bytes
                1, 4, 101, 120, 58, 98, // from "ex:b"
                1, 4, 101, 120, 58, 103, // to "ex:g"
            ],
            &[
                71, 1, 7, 0, 0, 0, // kind 71 (AddRdfGraph), required, 7 bytes
                1, 4, 101, 120, 58, 103, // from "ex:g"
                0, // to the default graph
            ],
        ];
        let encoded: Vec<Vec<u8>> = one_of_each().iter().map(framed).collect();
        assert_eq!(encoded, EXPECTED, "{encoded:?}");
    }

    /// The FNV-1a hash of `bytes`, 64 bits.
    fn fnv1a(bytes: &[u8]) -> u64 {
        bytes.iter().fold(0xcbf2_9ce4_8422_2325, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }

    /// Characterization: the bytes of a record of every variant (every
    /// catalog kind, RDF term and target, origin) and of every value kind,
    /// as the first log record encoder wrote them, before the change set's
    /// types became the records' payloads. Their length and FNV-1a hash.
    #[test]
    fn every_variant_and_value_keeps_the_bytes_it_was_first_written_with() {
        let mut bytes = Vec::new();
        for record in every_variant() {
            record.encode_framed(&mut bytes).unwrap();
        }
        let properties: Properties = every_value()
            .into_iter()
            .enumerate()
            .map(|(index, value)| (PropertyKey::new(format!("p{index}")), value))
            .collect();
        data(
            Some("Berlin"),
            DataOp::CreateNode {
                id: NodeId::new(88),
                labels: labels(&["Event", "Trip"]),
                properties: properties.clone(),
            },
        )
        .encode_framed(&mut bytes)
        .unwrap();
        for (key, value) in properties {
            data(
                None,
                DataOp::SetEdgeProperty {
                    id: EdgeId::new(19),
                    key,
                    value,
                },
            )
            .encode_framed(&mut bytes)
            .unwrap();
        }
        assert_eq!(
            (bytes.len(), fnv1a(&bytes)),
            (2338, 13_352_241_720_832_839_041),
            "{bytes:?}"
        );
    }

    /// Characterization: the variant numbers of the enums inside payloads
    /// (origins, RDF graph targets, index keys) and the key of every catalog
    /// kind, which the layouts above pin for `Schema` only.
    #[test]
    fn every_variant_has_a_fixed_number() {
        let payload = |record: LogRecord| framed(&record)[RECORD_HEADER_BYTES..].to_vec();
        let origins: Vec<u8> = [
            Origin::Transaction,
            Origin::Statement,
            Origin::Direct,
            Origin::Schema,
            Origin::Bulk,
        ]
        .into_iter()
        .map(|origin| {
            *payload(LogRecord::GroupBegin(GroupBegin {
                epoch: EpochId::new(3),
                timestamp_ms: 19,
                origin,
            }))
            .last()
            .unwrap()
        })
        .collect();
        assert_eq!(origins, [0, 1, 2, 3, 4]);

        let targets: Vec<Vec<u8>> = [
            RdfGraphTarget::Default,
            RdfGraphTarget::Named("g".into()),
            RdfGraphTarget::AllNamed,
            RdfGraphTarget::All,
        ]
        .into_iter()
        .map(|target| payload(rdf_graph(RdfGraphOp::Clear { target })))
        .collect();
        assert_eq!(targets, [vec![0], vec![1, 1, b'g'], vec![2], vec![3]]);

        let keys: Vec<Vec<u8>> = catalog_keys()
            .into_iter()
            .map(|key| payload(standalone(StandaloneOp::DropCatalog(key))))
            .collect();
        let expected: [&[u8]; 11] = [
            &[1, 6, b't', b'r', b'a', b'v', b'e', b'l'],
            &[2, 4, b'C', b'i', b't', b'y'],
            &[3, 5, b'R', b'O', b'U', b'T', b'E'],
            &[4, 6, b't', b'r', b'a', b'v', b'e', b'l'],
            &[5, 5, b't', b'r', b'i', b'p', b's'],
            &[
                6, 11, b'p', b'e', b'r', b's', b'o', b'n', b'_', b'n', b'a', b'm', b'e',
            ],
            // the default graph, a property index on "id"
            &[7, 0, 0, 2, b'i', b'd'],
            // graph "trips", a vector index on "Doc"."emb"
            &[
                7, 1, 5, b't', b'r', b'i', b'p', b's', 1, 3, b'D', b'o', b'c', 3, b'e', b'm', b'b',
            ],
            // graph "Berlin", a text index on "Doc"."body"
            &[
                7, 1, 6, b'B', b'e', b'r', b'l', b'i', b'n', 2, 3, b'D', b'o', b'c', 4, b'b', b'o',
                b'd', b'y',
            ],
            &[
                8, 11, b'p', b'e', b'r', b's', b'o', b'n', b'_', b'c', b'i', b't', b'y',
            ],
            &[9, 8, b'v', b'i', b's', b'i', b't', b'o', b'r', b's'],
        ];
        assert_eq!(keys, expected);
    }

    /// A catalog record inside a log record is the catalog section's own
    /// record: its kind, then the payload the catalog section stores.
    #[test]
    fn a_catalog_record_is_logged_as_the_catalog_stores_it() {
        for record in catalog_records() {
            let mut catalog = Vec::new();
            record.encode_framed(&mut catalog).unwrap();
            let logged = framed(&standalone(StandaloneOp::PutCatalog(record.clone())));
            assert_eq!(logged[RECORD_HEADER_BYTES], record.kind(), "{record:?}");
            assert_eq!(
                logged[RECORD_HEADER_BYTES + 1..],
                catalog[RECORD_HEADER_BYTES..],
                "{record:?}"
            );
            let mut back = Vec::new();
            read_catalog_records(&mut &catalog[..], &mut |record| {
                back.push(record);
                Ok(())
            })
            .unwrap();
            assert_eq!(
                read_all(&logged).unwrap(),
                [standalone(StandaloneOp::PutCatalog(back.remove(0)))]
            );
        }
    }

    #[test]
    fn values_round_trip_bit_for_bit() {
        let values = every_value();
        let properties: Properties = values
            .iter()
            .enumerate()
            .map(|(index, value)| (PropertyKey::new(format!("p{index}")), value.clone()))
            .collect();
        let mut records = vec![
            data(
                None,
                DataOp::CreateNode {
                    id: NodeId::new(3),
                    labels: labels(&["Event"]),
                    properties: properties.clone(),
                },
            ),
            data(
                Some("Amsterdam"),
                DataOp::CreateEdge {
                    id: EdgeId::new(19),
                    src: NodeId::new(3),
                    dst: NodeId::new(88),
                    edge_type: "VISITS".into(),
                    properties,
                },
            ),
        ];
        for value in &values {
            records.push(data(
                None,
                DataOp::SetNodeProperty {
                    id: NodeId::new(3),
                    key: "stamp".into(),
                    value: value.clone(),
                },
            ));
            records.push(data(
                Some("Amsterdam"),
                DataOp::SetEdgeProperty {
                    id: EdgeId::new(19),
                    key: "stamp".into(),
                    value: value.clone(),
                },
            ));
        }
        let mut bytes = Vec::new();
        for record in &records {
            record.encode_framed(&mut bytes).unwrap();
        }
        let back = read_all(&bytes).unwrap();
        assert_eq!(back.len(), records.len());
        let values_of = |record: &LogRecord| -> Vec<Value> {
            match record {
                LogRecord::Data {
                    op:
                        DataOp::CreateNode { properties, .. } | DataOp::CreateEdge { properties, .. },
                    ..
                } => properties.iter().map(|(_, value)| value.clone()).collect(),
                LogRecord::Data {
                    op:
                        DataOp::SetNodeProperty { value, .. } | DataOp::SetEdgeProperty { value, .. },
                    ..
                } => {
                    vec![value.clone()]
                }
                other => panic!("a record without values: {other:?}"),
            }
        };
        for (record, back) in records.iter().zip(&back) {
            assert_eq!(record.kind(), back.kind());
            let (written, read) = (values_of(record), values_of(back));
            assert_eq!(written.len(), read.len());
            for (written, read) in written.iter().zip(&read) {
                assert!(same(written, read), "{written:?} came back as {read:?}");
            }
            assert_eq!(framed(back), framed(record), "the same bytes again");
        }

        // A value is the value codec's bytes, so a lossy encoding of floats,
        // times or counters cannot hide behind a round trip of its own.
        let nan = Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058));
        let bytes = framed(&data(
            None,
            DataOp::SetNodeProperty {
                id: NodeId::new(3),
                key: "x".into(),
                value: nan.clone(),
            },
        ));
        let codec = encoded_value(&nan);
        let tail = &bytes[bytes.len() - codec.len() - 1..];
        assert_eq!(tail[0], 9, "the length of the codec's 9 bytes");
        assert_eq!(&tail[1..], &codec[..]);
    }

    #[test]
    fn unknown_kinds_are_skipped_when_optional_and_refused_when_required() {
        let group = framed(&one_of_each()[0]);
        for kind in [2u8, 26, 34, 42, 72, 99, 255] {
            let optional = [&[kind, 0, 3, 0, 0, 0, 3, 19, 88][..], &group].concat();
            assert_eq!(
                read_all(&optional).unwrap(),
                [one_of_each()[0].clone()],
                "kind {kind}, optional: skipped"
            );
            let required = [&group[..], &[kind, RECORD_REQUIRED, 0, 0, 0, 0]].concat();
            let error = read_error(&required);
            assert!(
                error.contains(&format!("kind {kind} is required"))
                    && error.contains("log record 1 at byte 17"),
                "kind {kind}, required: {error}"
            );
        }
    }

    /// A catalog record or key of a kind this release does not know, inside
    /// a known log kind, is skipped or refused by the log record's flag.
    #[test]
    fn unknown_catalog_kinds_are_skipped_when_optional_and_refused_when_required() {
        let group = framed(&one_of_each()[0]);
        for (kind, what) in [
            (KIND_PUT_CATALOG, "catalog record"),
            (KIND_DROP_CATALOG, "catalog key"),
        ] {
            let optional = [&[kind, 0, 4, 0, 0, 0, 12, 3, 19, 88][..], &group].concat();
            assert_eq!(
                read_all(&optional).unwrap(),
                [one_of_each()[0].clone()],
                "kind {kind}, optional: skipped"
            );
            let required = [
                &group[..],
                &[kind, RECORD_REQUIRED, 4, 0, 0, 0, 12, 3, 19, 88],
            ]
            .concat();
            let error = read_error(&required);
            assert!(
                error.contains("log record 1 at byte 17")
                    && error.contains(&format!("kind {kind}"))
                    && error.contains(&format!("{what} of kind 12")),
                "kind {kind}, required: {error}"
            );

            for flags in [0, RECORD_REQUIRED] {
                let zero = [kind, flags, 2, 0, 0, 0, 0, 88];
                let error = read_error(&zero);
                assert!(
                    error.contains(&format!("{what} of kind 0")),
                    "kind {kind}, flags {flags}: {error}"
                );
                let empty = [kind, flags, 0, 0, 0, 0];
                let error = read_error(&empty);
                assert!(
                    error.contains(&format!("kind {kind}")) && error.contains("empty"),
                    "kind {kind}, flags {flags}: {error}"
                );
            }
        }
    }

    #[test]
    fn only_unknown_flags_among_bits_0_to_3_are_refused() {
        let group = one_of_each()[0].clone();
        for flags in 0..=u8::MAX {
            let mut bytes = framed(&group);
            bytes[1] = flags;
            let read_bits = flags & 0x0F;
            let known = read_all(&bytes);
            if read_bits <= RECORD_REQUIRED {
                assert_eq!(
                    known.unwrap(),
                    std::slice::from_ref(&group),
                    "flags {flags:#04x}"
                );
            } else {
                let error = known.unwrap_err().to_string();
                assert!(
                    error.contains(&format!("{flags:#04x}")),
                    "flags {flags:#04x}: {error}"
                );
            }
        }
        let mut zero = framed(&group);
        zero[0] = 0;
        assert!(read_error(&zero).contains("kind 0"));
    }

    #[test]
    fn a_truncated_or_overlong_record_is_refused() {
        let mut bytes = Vec::new();
        let mut boundaries = Vec::new();
        for record in one_of_each() {
            record.encode_framed(&mut bytes).unwrap();
            boundaries.push(bytes.len());
        }
        for cut in 1..bytes.len() {
            let mut count = 0;
            let result = read_log_records(&bytes[..cut], &mut |_| {
                count += 1;
                Ok(())
            });
            match boundaries.iter().position(|&end| end == cut) {
                Some(index) => {
                    assert!(result.is_ok(), "cut at {cut}, between records: {result:?}");
                    assert_eq!(
                        count,
                        index + 1,
                        "cut at {cut}: the whole records before it"
                    );
                }
                None => assert!(result.is_err(), "cut at {cut}"),
            }
        }
        let header = read_error(&bytes[..3]);
        assert!(
            header.contains("header") && header.contains("3 of its 6"),
            "{header}"
        );
        let payload = read_error(&bytes[..9]);
        assert!(
            payload.contains("payload") && payload.contains("3 of its 11"),
            "{payload}"
        );

        // A header claiming the largest payload, and one byte more, with
        // nothing behind it: refused without reading or allocating it.
        let largest = MAX_RECORD_BYTES - RECORD_HEADER_BYTES;
        let mut claim = [16u8, RECORD_REQUIRED, 0, 0, 0, 0];
        claim[2..].copy_from_slice(&u32::try_from(largest).unwrap().to_le_bytes());
        let error = read_error(&claim);
        assert!(
            error.contains("ends inside the payload")
                && error.contains(&format!("0 of its {largest}")),
            "{error}"
        );
        claim[2..].copy_from_slice(&u32::try_from(largest + 1).unwrap().to_le_bytes());
        let error = read_error(&claim);
        assert!(
            error.contains(&(largest + 1).to_string()) && error.contains(&largest.to_string()),
            "{error}"
        );
        let error = read_error(&[16, 1, 0xFF, 0xFF, 0xFF, 0xFF]);
        assert!(error.contains("4294967295"), "{error}");

        // Bytes left over after a record's fields.
        let mut left_over = framed(&one_of_each()[4]);
        left_over[2] += 1;
        left_over.push(88);
        let error = read_error(&left_over);
        assert!(
            error.contains("2 of its 3") && error.contains("log record 0 at byte 0"),
            "{error}"
        );
    }

    #[test]
    fn an_empty_payload_holds_no_records() {
        assert!(read_all(&[]).unwrap().is_empty(), "no records");
    }

    /// Lengths and counts inside a payload are checked against the bytes
    /// present: a damaged one is refused, not allocated (the allocation
    /// itself is measured by the `log_record_memory` test).
    #[test]
    fn a_damaged_length_inside_a_payload_is_refused() {
        let damaged = |prefix: &[u8], claim: &[u8], rest: &[u8]| {
            let mut payload = prefix.to_vec();
            payload.extend_from_slice(claim);
            payload.extend_from_slice(rest);
            let mut bytes = vec![KIND_CREATE_NODE, RECORD_REQUIRED];
            bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
            bytes.extend_from_slice(&payload);
            read_error(&bytes)
        };
        // A graph name claiming 2^40 bytes, a label list claiming 2^40
        // labels, a property list claiming 2^40 properties, a label claiming
        // 2^40 bytes, a value claiming 2^40 bytes.
        let huge = [0xFD, 0, 0, 0, 0, 0, 1, 0, 0];
        for (prefix, rest) in [
            (&[1u8][..], &b"Prague"[..]),
            (&[0, 3][..], &b"Person"[..]),
            (&[0, 3, 0][..], &[1, b'x'][..]),
            (&[0, 3, 1][..], &b"Person"[..]),
            (&[0, 3, 0, 1, 1, b'x'][..], &[2, 3, 0, 0, 0, 0, 0, 0, 0][..]),
        ] {
            let error = damaged(prefix, &huge, rest);
            assert!(
                error.contains("log record 0 at byte 0") && error.contains("kind 16"),
                "{error}"
            );
        }
    }

    /// A record whose values would take the framed record past
    /// [`MAX_RECORD_BYTES`] is refused from their sizes, and appends
    /// nothing. (That no value is encoded first is measured by the
    /// `log_record_memory` test.)
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a list of 2^20 values takes minutes under Miri; this module has no unsafe code"
    )]
    fn a_record_over_the_maximum_is_refused_before_it_is_written() {
        // About 1.01 GiB of encoded strings, held as 2^20 references to one
        // string of 1 KiB.
        let line = Value::from("Amsterdam ".repeat(103).as_str());
        let lines = Value::List(Arc::from(vec![line; 1 << 20]));
        assert!(encoded_len(&lines) > MAX_RECORD_BYTES);
        for record in [
            data(
                None,
                DataOp::SetNodeProperty {
                    id: NodeId::new(3),
                    key: "notes".into(),
                    value: lines.clone(),
                },
            ),
            data(
                None,
                DataOp::CreateEdge {
                    id: EdgeId::new(3),
                    src: NodeId::new(19),
                    dst: NodeId::new(88),
                    edge_type: "WROTE".into(),
                    properties: vec![("notes".into(), lines.clone())],
                },
            ),
        ] {
            let mut out = vec![3, 19, 88];
            let error = record.encode_framed(&mut out).unwrap_err();
            assert!(matches!(error, Error::Serialization(_)), "{error:?}");
            let message = error.to_string();
            assert!(
                message.contains(&format!("log record of kind {}", record.kind()))
                    && message.contains(&(MAX_RECORD_BYTES - RECORD_HEADER_BYTES).to_string()),
                "{message}"
            );
            assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
            assert!(
                out.capacity() < 1 << 20,
                "the output never took the value: it grew to {} bytes",
                out.capacity()
            );
        }
    }

    /// A catalog record the catalog section would refuse is refused as a log
    /// record too, so a logged change always fits the next checkpoint.
    #[test]
    fn a_catalog_record_the_catalog_refuses_is_refused() {
        let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap();
        let record = CatalogRecord::Procedure(ProcedureRecord {
            name: "visitors".into(),
            params: Vec::new(),
            returns: Vec::new(),
            body: "RETURN 88;\n".repeat(maximum / 11 + 1),
        });
        let mut out = vec![3, 19, 88];
        let error = standalone(StandaloneOp::PutCatalog(record))
            .encode_framed(&mut out)
            .unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(error.to_string().contains(&maximum.to_string()), "{error}");
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");

        // A logged catalog payload past the catalog's maximum is refused
        // before it is decoded.
        // Zeroed whole, then the header and catalog kind: a byte-wise fill of
        // 2 MiB takes minutes under Miri.
        let mut bytes = vec![0u8; RECORD_HEADER_BYTES + maximum + 2];
        bytes[..2].copy_from_slice(&[KIND_PUT_CATALOG, RECORD_REQUIRED]);
        bytes[2..RECORD_HEADER_BYTES]
            .copy_from_slice(&u32::try_from(maximum + 2).unwrap().to_le_bytes());
        bytes[RECORD_HEADER_BYTES] = 9;
        let error = read_error(&bytes);
        assert!(
            error.contains(&(maximum + 1).to_string()) && error.contains(&maximum.to_string()),
            "{error}"
        );
    }

    #[test]
    fn the_maximum_counts_the_record_header() {
        assert_eq!(
            usize::try_from(MAX_RECORD_PAYLOAD).unwrap() + RECORD_HEADER_BYTES,
            MAX_RECORD_BYTES
        );
    }

    /// The writer's limit is [`MAX_RECORD_BYTES`] exactly: a record of that
    /// many bytes, header included, is written, and one a byte longer is
    /// refused, appending nothing. The graph's name is a string of zeros, so
    /// its gigabyte is mostly pages the system has not handed out yet.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a record of 1 GiB is far too large for Miri; this module has no unsafe code"
    )]
    fn a_record_at_the_maximum_is_written_and_one_byte_more_is_refused() {
        // A CreateNode in a graph of that name: the header, the graph (some,
        // its length a u32: a marker and 4 bytes), id 3, no labels, no
        // properties.
        let around_the_name = RECORD_HEADER_BYTES + 1 + 5 + 1 + 1 + 1;
        let node = |name_bytes: usize| LogRecord::Data {
            graph: Some(String::from_utf8(vec![0; name_bytes]).unwrap()),
            op: DataOp::CreateNode {
                id: NodeId::new(3),
                labels: Labels::new(),
                properties: Vec::new(),
            },
        };

        let at_the_maximum = node(MAX_RECORD_BYTES - around_the_name);
        let mut out = Vec::with_capacity(MAX_RECORD_BYTES);
        at_the_maximum.encode_framed(&mut out).unwrap();
        assert_eq!(out.len(), MAX_RECORD_BYTES, "the record fills the maximum");
        assert_eq!(out[..2], [KIND_CREATE_NODE, RECORD_REQUIRED]);
        assert_eq!(
            out[2..RECORD_HEADER_BYTES],
            MAX_RECORD_PAYLOAD.to_le_bytes()
        );
        drop((at_the_maximum, out));

        let one_byte_more = node(MAX_RECORD_BYTES - around_the_name + 1);
        let mut out = vec![3, 19, 88];
        let error = one_byte_more.encode_framed(&mut out).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(
            error.to_string().contains(&MAX_RECORD_PAYLOAD.to_string()),
            "{error}"
        );
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
    }

    #[test]
    fn apply_errors_come_back_unchanged_and_stop_the_read() {
        let mut bytes = Vec::new();
        for record in one_of_each() {
            record.encode_framed(&mut bytes).unwrap();
        }
        let mut applied = 0;
        let error = read_log_records(&bytes, &mut |_| {
            applied += 1;
            if applied == 3 {
                return Err(Error::Internal("Jules refuses".into()));
            }
            Ok(())
        })
        .unwrap_err();
        assert!(
            matches!(&error, Error::Internal(message) if message == "Jules refuses"),
            "{error:?}"
        );
        assert_eq!(applied, 3, "nothing is read after the error");
    }

    #[test]
    fn a_value_the_codec_refuses_is_refused() {
        let too_deep = (0..=crate::storage::value_codec::MAX_VALUE_DEPTH)
            .fold(Value::Null, |inner, _| Value::List(Arc::from(vec![inner])));
        let mut out = vec![3, 19, 88];
        let error = data(
            None,
            DataOp::SetNodeProperty {
                id: NodeId::new(3),
                key: "nested".into(),
                value: too_deep,
            },
        )
        .encode_framed(&mut out)
        .unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(error.to_string().contains("deeper"), "{error}");
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");

        // A stored value with bytes left over after it.
        let mut bytes = framed(&data(
            None,
            DataOp::SetNodeProperty {
                id: NodeId::new(3),
                key: "x".into(),
                value: Value::Null,
            },
        ));
        let last = bytes.len() - 1;
        bytes[last - 1] = 2; // a value of 2 bytes: the null and a stray byte
        bytes.push(88);
        bytes[2] += 1;
        let error = read_error(&bytes);
        assert!(error.contains("trailing"), "{error}");
    }

    /// A CreateNode of the default graph, id 3, framed by hand, so a list the
    /// writer refuses reaches the reader.
    fn node_framed_by_hand(labels: &[ArcStr], properties: &[(PropertyKey, Value)]) -> Vec<u8> {
        let payload = bincode::serde::encode_to_vec(
            (
                None::<String>,
                3u64,
                NamesRef(labels),
                PropertiesRef(properties),
            ),
            bincode::config::standard(),
        )
        .unwrap();
        let mut bytes = vec![KIND_CREATE_NODE, RECORD_REQUIRED];
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn node_of(labels: Labels, properties: Properties) -> LogRecord {
        data(
            None,
            DataOp::CreateNode {
                id: NodeId::new(3),
                labels,
                properties,
            },
        )
    }

    /// The rule's byte counts are the bytes bincode writes: a string's
    /// length takes 1, 3, 5 or 9 bytes, and a property its key and its
    /// value's encoding as a byte string.
    #[test]
    fn the_list_rule_counts_the_bytes_written() {
        for length in [0, 3, 250, 251, 65_535, 65_536] {
            let text = "x".repeat(length);
            let written = bincode::serde::encode_to_vec(&text, bincode::config::standard())
                .unwrap()
                .len();
            assert_eq!(string_bytes(length), written, "a string of {length} bytes");
        }
        assert_eq!(varint_bytes(0xFFFF_FFFF), 5);
        assert_eq!(varint_bytes(0x1_0000_0000), 9);
        for (key, value) in [
            (String::new(), Value::Null),
            ("city".to_string(), Value::from("Amsterdam")),
            ("x".repeat(251), Value::Bytes(Arc::from(vec![19u8; 300]))),
        ] {
            let written = bincode::serde::encode_to_vec(
                (&key, ValueRef(&value)),
                bincode::config::standard(),
            )
            .unwrap()
            .len();
            assert_eq!(property_bytes(&key, &value), written, "{key:?}: {value:?}");
        }
    }

    /// Labels of a byte each take 24 bytes of slots per byte, so a list of
    /// them holds 65,536 (8 per byte plus 1 MiB) and no more: the writer
    /// writes that list and the reader reads it, and one more label is
    /// refused by both, the writer before it writes a byte.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "lists of 65,536 labels take minutes under Miri; this module has no unsafe code"
    )]
    fn a_list_of_labels_at_the_rule_is_written_and_read_and_one_more_is_refused() {
        let at_the_rule = node_of(Labels::from_elem(ArcStr::new(), 65_536), Vec::new());
        let bytes = framed(&at_the_rule);
        assert_eq!(read_all(&bytes).unwrap(), [at_the_rule]);

        let labels = Labels::from_elem(ArcStr::new(), 65_537);
        let mut out = vec![3, 19, 88];
        let error = node_of(labels.clone(), Vec::new())
            .encode_framed(&mut out)
            .unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(
            error.to_string().contains("a list of 65537 labels"),
            "{error}"
        );
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
        let error = read_error(&node_framed_by_hand(&labels, &[]));
        assert!(error.contains("a list of 65537 labels"), "{error}");
    }

    /// A property with an empty key and a null takes three bytes for its 64
    /// bytes of slot: 26,214 fit, and the writer and the reader refuse the
    /// next one alike.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "lists of 26,214 properties take minutes under Miri; this module has no unsafe code"
    )]
    fn a_list_of_properties_at_the_rule_is_written_and_read_and_one_more_is_refused() {
        let property = (PropertyKey::new(""), Value::Null);
        let at_the_rule = node_of(Labels::new(), vec![property.clone(); 26_214]);
        let bytes = framed(&at_the_rule);
        assert_eq!(read_all(&bytes).unwrap().len(), 1);

        let properties = vec![property; 26_215];
        let error = node_of(Labels::new(), properties.clone())
            .encode_framed(&mut Vec::new())
            .unwrap_err();
        assert!(
            error.to_string().contains("a list of 26215 properties"),
            "{error}"
        );
        let error = data(
            None,
            DataOp::CreateEdge {
                id: EdgeId::new(88),
                src: NodeId::new(3),
                dst: NodeId::new(19),
                edge_type: "KNOWS".into(),
                properties: properties.clone(),
            },
        )
        .encode_framed(&mut Vec::new())
        .unwrap_err();
        assert!(error.to_string().contains("26215 properties"), "{error}");
        let error = read_error(&node_framed_by_hand(&[], &properties));
        assert!(error.contains("a list of 26215 properties"), "{error}");
    }

    /// The rule holds at every point of a list, so a decode never holds more
    /// than it allows: long labels after the excess do not make up for it,
    /// for the writer as for the reader. Written the other way round (the
    /// long label first), the same labels pass.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "lists of 65,537 labels take minutes under Miri; this module has no unsafe code"
    )]
    fn the_list_rule_holds_at_every_point_of_the_list() {
        let mut labels = Labels::from_elem(ArcStr::new(), 65_537);
        labels.push(ArcStr::from("Barcelona".repeat(1_000)));
        let error = node_of(labels.clone(), Vec::new())
            .encode_framed(&mut Vec::new())
            .unwrap_err();
        assert!(error.to_string().contains("65537 labels"), "{error}");
        let error = read_error(&node_framed_by_hand(&labels, &[]));
        assert!(error.contains("65537 labels"), "{error}");

        labels.rotate_right(1);
        let record = node_of(labels, Vec::new());
        assert_eq!(read_all(&framed(&record)).unwrap(), [record]);
    }
}
