//! Catalog records: the schema as a sequence of typed, framed records.
//!
//! The catalog section holds the catalog as records of the types below, one
//! per schema, node type, edge type, graph type, graph type binding, named
//! constraint, index, index name and procedure. They hold no engine types,
//! so the WAL can log the same records. Each record is framed:
//!
//! | Bytes | Field |
//! | --- | --- |
//! | 1 | kind |
//! | 1 | flags: bit 0 is [`RECORD_REQUIRED`]; a reader refuses another of bits 0 to 3 and ignores bits 4 to 7 |
//! | 4 | payload length, u32 little endian, at most [`MAX_CATALOG_RECORD_PAYLOAD`] |
//! | length | payload: bincode (standard configuration) of the kind's record type |
//!
//! | Kind | Record |
//! | --- | --- |
//! | 1 | [`SchemaRecord`] |
//! | 2 | [`NodeTypeRecord`] |
//! | 3 | [`EdgeTypeRecord`] |
//! | 4 | [`GraphTypeRecord`] |
//! | 5 | [`GraphBindingRecord`] |
//! | 6 | [`ConstraintRecord`] |
//! | 7 | [`IndexRecord`] |
//! | 8 | [`IndexNameRecord`] |
//! | 9 | [`ProcedureRecord`] |
//!
//! Kind 0 is never written. A reader skips a record of a kind it does not
//! know when the record's required flag is clear, and refuses the catalog
//! when it is set, so a later release can add records that older readers may
//! ignore. Every kind of this release is written required. A kind's payload
//! never changes: a release that needs other fields adds a kind, and the
//! enums below only get variants appended (bincode writes a variant as its
//! index).
//!
//! A default value is written through the lossless value codec
//! ([`serde_option_value`](super::value_codec::serde_option_value)), so it
//! keeps its kind and bits. A property type is written as its number of
//! `LIST<...>` levels (at most [`MAX_LIST_TYPE_DEPTH`], and at most
//! [`MAX_LIST_LEVELS_PER_RECORD`] over all the property types of a record)
//! and the code of the type inside them, both as bincode u32:
//!
//! | Code | Type | Code | Type |
//! | --- | --- | --- | --- |
//! | 0 | `STRING` | 8 | `ZONED DATETIME` |
//! | 1 | `INT64` | 9 | `DURATION` |
//! | 2 | `FLOAT64` | 10 | `LIST` (of any values) |
//! | 3 | `BOOL` | 11 | `MAP` |
//! | 4 | `DATE` | 12 | `BYTES` |
//! | 5 | `TIME` | 13 | `NODE` |
//! | 6 | `TIMESTAMP` | 14 | `EDGE` |
//! | 7 | `LOCAL DATETIME` | 15 | `ANY` |
//!
//! A reader checks a payload's length against the maximum before it reads
//! the payload, into a buffer that grows with the bytes present, and decodes
//! it with a bincode limit of eight times the maximum. The limit counts what
//! bincode charges: each number it reads at the size of its type (a one-byte
//! length as the 8 bytes of a `usize`), at most 8 bytes per byte read, and
//! each string at its length before the string is allocated. So no payload
//! the writer accepts reaches the limit, and a damaged string length cannot
//! allocate more than the limit. A damaged list length allocates at most
//! 1 MiB ahead of the list's elements (serde's cap), and the value codec
//! checks every count against the bytes left.
//!
//! The limit is not a memory ceiling: it does not count the lists, the
//! `LIST<...>` levels of property types or the default values a payload
//! decodes into. Every element but a level takes memory in proportion to its
//! bytes, at most 120 bytes per payload byte while the payload decodes and
//! the engine converts the record, before the allocator's own overhead: an
//! element is held at most three times, while its list grows (the old buffer
//! and the new one, twice as large) or while the engine copies the list. So
//! three times 24 bytes for a list's empty string (1 payload byte), three
//! times 48 for an empty string pair, two absent endpoints or an empty
//! constraint (2 bytes), three times 88 for a property (5 bytes: an empty
//! name, the levels and the code, the nullable flag and no default), and,
//! while the value codec builds a default value's list of nulls, three times
//! the 40 bytes of each null (1 byte) as the list grows and becomes shared.
//!
//! A `LIST<...>` level takes no payload byte of its own (a type's levels are
//! one count) but 32 bytes of memory: a 16-byte box here, and another in the
//! engine's property type, which the engine builds while the record's box
//! still lives. Its memory cannot be bounded per payload byte, so it is
//! bounded per record: the property types of a record nest at most
//! [`MAX_LIST_LEVELS_PER_RECORD`] levels in all, 32,768, which take at most
//! 1 MiB, as much as serde allocates ahead of the elements of a list whose
//! length is damaged, in any record. A payload of `n` bytes so decodes into
//! at most `120 n` bytes plus 1 MiB: at most 241 MiB for the largest payload.
//! The writer refuses a record past the cap, and the reader counts each
//! type's levels before it reads their code and builds them, so it refuses
//! the record at the type that passes the cap.
//!
//! The framing itself (`RecordFraming`, `encode_framed_record`,
//! `read_framed_records`) knows nothing of the catalog: the log records of
//! the WAL ([`log_record`](super::log_record)) use it with their own kinds
//! and maximum.

use std::fmt;
use std::io::{self, Read};

use serde::de::{DeserializeOwned, DeserializeSeed, Error as _, SeqAccess, Visitor};
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::value_codec::{MAX_PROPERTY_VALUE_DEPTH, encoded_len, nests_too_deep_to_encode};
use crate::types::Value;
use crate::utils::error::{Error, Result};

/// Flag bit 0 of a framed record: a reader that does not know the record's
/// kind must refuse the catalog instead of skipping the record.
pub const RECORD_REQUIRED: u8 = 0x01;

/// The most bytes the payload of one catalog record may hold: 2 MiB.
///
/// The writer refuses a larger record and the reader refuses a header that
/// claims one, before reading it.
pub const MAX_CATALOG_RECORD_PAYLOAD: u32 = 1 << 21;

/// The most `LIST<...>` levels a property type may have:
/// [`MAX_PROPERTY_VALUE_DEPTH`], as a property of a type nested deeper could
/// hold no value a write accepts. The writer refuses a deeper type and the
/// reader refuses one before it builds a level.
pub const MAX_LIST_TYPE_DEPTH: usize = MAX_PROPERTY_VALUE_DEPTH;

/// The most `LIST<...>` levels the property types of one record nest in
/// all: 32,768, so 256 properties of the deepest type.
///
/// At 32 bytes of memory per level, they take at most 1 MiB however few
/// payload bytes hold them (see the module documentation). The writer
/// refuses a record past the cap, and the reader refuses one before it builds
/// the levels of the type that passes it.
pub const MAX_LIST_LEVELS_PER_RECORD: usize = 1 << 15;

// A record holds a property of the deepest type.
const _: () = assert!(MAX_LIST_LEVELS_PER_RECORD >= MAX_LIST_TYPE_DEPTH);

/// The most bytes serde allocates ahead of the elements of a list, from the
/// length the payload claims.
const LIST_PREALLOCATION_LIMIT: usize = 1 << 20;

/// The bincode limit of a catalog payload decode: eight times
/// [`MAX_CATALOG_RECORD_PAYLOAD`].
const CATALOG_DECODE_LIMIT: usize = 1 << 24;

/// The bytes of a record's frame before its payload: kind, flags, length.
pub(crate) const RECORD_HEADER_BYTES: usize = 6;

/// The flag bits this release knows (and the only ones it writes).
const KNOWN_FLAGS: u8 = RECORD_REQUIRED;

/// Bits 0 to 3 change how a record is read, so a reader refuses one it does
/// not know; bits 4 to 7 do not, and a reader ignores them: the split of the
/// directory entries' and the file header's flags, so a later release can add
/// a flag older readers may pass over.
const INCOMPATIBLE_FLAGS: u8 = 0x0F;

const KIND_SCHEMA: u8 = 1;
const KIND_NODE_TYPE: u8 = 2;
const KIND_EDGE_TYPE: u8 = 3;
const KIND_GRAPH_TYPE: u8 = 4;
const KIND_GRAPH_BINDING: u8 = 5;
const KIND_CONSTRAINT: u8 = 6;
const KIND_INDEX: u8 = 7;
const KIND_INDEX_NAME: u8 = 8;
const KIND_PROCEDURE: u8 = 9;

// ── Record types ────────────────────────────────────────────────────

/// The type of a property, as the catalog stores it.
///
/// Written as its `LIST<...>` levels and a code (see the module
/// documentation), not as a nested enum, so a damaged payload cannot nest it
/// deep enough to exhaust the stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyTypeRecord {
    /// `STRING`.
    String,
    /// `INT64`.
    Int64,
    /// `FLOAT64`.
    Float64,
    /// `BOOL`.
    Bool,
    /// `DATE`.
    Date,
    /// `TIME`.
    Time,
    /// `TIMESTAMP`.
    Timestamp,
    /// `LOCAL DATETIME`.
    LocalDatetime,
    /// `ZONED DATETIME`.
    ZonedDatetime,
    /// `DURATION`.
    Duration,
    /// `LIST`, of any values.
    List,
    /// `LIST<type>`: a list of values of the inner type.
    ListOf(Box<PropertyTypeRecord>),
    /// `MAP`.
    Map,
    /// `BYTES`.
    Bytes,
    /// `NODE`: a node reference.
    Node,
    /// `EDGE`: an edge reference.
    Edge,
    /// `ANY`: any value.
    Any,
}

impl PropertyTypeRecord {
    /// The `LIST<...>` levels around the type inside them, and that type's
    /// code.
    fn levels_and_code(&self) -> (usize, u32) {
        let mut levels = 0;
        let mut current = self;
        loop {
            let code = match current {
                Self::ListOf(inner) => {
                    levels += 1;
                    current = inner;
                    continue;
                }
                Self::String => 0,
                Self::Int64 => 1,
                Self::Float64 => 2,
                Self::Bool => 3,
                Self::Date => 4,
                Self::Time => 5,
                Self::Timestamp => 6,
                Self::LocalDatetime => 7,
                Self::ZonedDatetime => 8,
                Self::Duration => 9,
                Self::List => 10,
                Self::Map => 11,
                Self::Bytes => 12,
                Self::Node => 13,
                Self::Edge => 14,
                Self::Any => 15,
            };
            return (levels, code);
        }
    }

    /// The type of `code`, or `None` for a code this release does not know.
    fn of_code(code: u32) -> Option<Self> {
        Some(match code {
            0 => Self::String,
            1 => Self::Int64,
            2 => Self::Float64,
            3 => Self::Bool,
            4 => Self::Date,
            5 => Self::Time,
            6 => Self::Timestamp,
            7 => Self::LocalDatetime,
            8 => Self::ZonedDatetime,
            9 => Self::Duration,
            10 => Self::List,
            11 => Self::Map,
            12 => Self::Bytes,
            13 => Self::Node,
            14 => Self::Edge,
            15 => Self::Any,
            _ => return None,
        })
    }

    /// The type of `levels` `LIST<...>` levels around the type of `code`.
    /// The levels are counted before the code is read or a level is built:
    /// they must not pass [`MAX_LIST_TYPE_DEPTH`] nor `levels_left`, the
    /// levels the record may still hold, which they are then taken from.
    fn of_levels_and_code(
        levels: u32,
        code: u32,
        levels_left: &mut usize,
    ) -> std::result::Result<Self, String> {
        let depth = usize::try_from(levels)
            .ok()
            .filter(|depth| *depth <= MAX_LIST_TYPE_DEPTH)
            .ok_or_else(|| {
                format!(
                    "a property type nested {levels} LIST levels deep, deeper than \
                     {MAX_LIST_TYPE_DEPTH}"
                )
            })?;
        *levels_left = levels_left.checked_sub(depth).ok_or_else(|| {
            format!(
                "the property types nest more than the {MAX_LIST_LEVELS_PER_RECORD} LIST \
                 levels a record may hold"
            )
        })?;
        let element =
            Self::of_code(code).ok_or_else(|| format!("unknown property type code {code}"))?;
        Ok((0..depth).fold(element, |inner, _| Self::ListOf(Box::new(inner))))
    }
}

impl Serialize for PropertyTypeRecord {
    /// Writes the `LIST<...>` levels and the code of the type inside them;
    /// refuses a type nested deeper than [`MAX_LIST_TYPE_DEPTH`].
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let (levels, code) = self.levels_and_code();
        if levels > MAX_LIST_TYPE_DEPTH {
            return Err(S::Error::custom(format_args!(
                "a property type nested {levels} LIST levels deep, deeper than \
                 {MAX_LIST_TYPE_DEPTH}"
            )));
        }
        let levels = u32::try_from(levels).map_err(S::Error::custom)?;
        (levels, code).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PropertyTypeRecord {
    /// Reads what [`serialize`](Self::serialize) wrote; refuses more levels
    /// than [`MAX_LIST_TYPE_DEPTH`] before it builds one, and an unknown
    /// code. A record reads the types of its properties another way, which
    /// also counts their levels against [`MAX_LIST_LEVELS_PER_RECORD`].
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let mut levels_left = MAX_LIST_TYPE_DEPTH;
        TypeSeed {
            levels_left: &mut levels_left,
        }
        .deserialize(deserializer)
    }
}

/// Reads a property type, taking its levels from those its record may still
/// hold.
struct TypeSeed<'a> {
    levels_left: &'a mut usize,
}

impl<'de> DeserializeSeed<'de> for TypeSeed<'_> {
    type Value = PropertyTypeRecord;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<PropertyTypeRecord, D::Error> {
        let (levels, code) = <(u32, u32)>::deserialize(deserializer)?;
        PropertyTypeRecord::of_levels_and_code(levels, code, self.levels_left)
            .map_err(D::Error::custom)
    }
}

/// A property of a node or edge type.
///
/// It has no `Deserialize` of its own: a node or edge type record reads its
/// properties together, counting the `LIST<...>` levels of their types
/// against [`MAX_LIST_LEVELS_PER_RECORD`].
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PropertyRecord {
    /// The property's name.
    pub name: String,
    /// The type its values must have.
    pub data_type: PropertyTypeRecord,
    /// Whether it may be null.
    pub nullable: bool,
    /// The value it gets when a node or edge is created without it.
    #[serde(with = "crate::storage::value_codec::serde_option_value")]
    pub default_value: Option<Value>,
}

/// The fields of a [`PropertyRecord`], in the order it writes them.
const PROPERTY_FIELDS: &[&str] = &["name", "data_type", "nullable", "default_value"];

/// Reads the properties of a node or edge type record, counting the
/// `LIST<...>` levels of their types against [`MAX_LIST_LEVELS_PER_RECORD`]
/// as it goes, each type's levels before they are built.
fn deserialize_properties<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Vec<PropertyRecord>, D::Error> {
    deserializer.deserialize_seq(PropertiesVisitor)
}

struct PropertiesVisitor;

impl<'de> Visitor<'de> for PropertiesVisitor {
    type Value = Vec<PropertyRecord>;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a list of properties")
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<Vec<PropertyRecord>, A::Error> {
        // As serde does for any list: no more than its limit ahead of the
        // elements, whatever length the payload claims.
        let ahead = seq
            .size_hint()
            .unwrap_or(0)
            .min(LIST_PREALLOCATION_LIMIT / std::mem::size_of::<PropertyRecord>());
        let mut properties = Vec::with_capacity(ahead);
        let mut levels_left = MAX_LIST_LEVELS_PER_RECORD;
        while let Some(property) = seq.next_element_seed(PropertySeed {
            levels_left: &mut levels_left,
        })? {
            properties.push(property);
        }
        Ok(properties)
    }
}

/// Reads one property, taking the levels of its type from those its record
/// may still hold.
struct PropertySeed<'a> {
    levels_left: &'a mut usize,
}

impl<'de> DeserializeSeed<'de> for PropertySeed<'_> {
    type Value = PropertyRecord;

    fn deserialize<D: Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> std::result::Result<PropertyRecord, D::Error> {
        deserializer.deserialize_struct("PropertyRecord", PROPERTY_FIELDS, self)
    }
}

impl<'de> Visitor<'de> for PropertySeed<'_> {
    type Value = PropertyRecord;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a property")
    }

    fn visit_seq<A: SeqAccess<'de>>(
        self,
        mut seq: A,
    ) -> std::result::Result<PropertyRecord, A::Error> {
        let missing = |index: usize| A::Error::invalid_length(index, &"a property of 4 fields");
        let name: String = seq.next_element()?.ok_or_else(|| missing(0))?;
        let data_type = seq
            .next_element_seed(TypeSeed {
                levels_left: self.levels_left,
            })?
            .ok_or_else(|| missing(1))?;
        let nullable: bool = seq.next_element()?.ok_or_else(|| missing(2))?;
        let DefaultValue(default_value) = seq.next_element()?.ok_or_else(|| missing(3))?;
        Ok(PropertyRecord {
            name,
            data_type,
            nullable,
            default_value,
        })
    }
}

/// A property's default value, read as [`PropertyRecord`] writes it.
#[derive(Deserialize)]
#[serde(transparent)]
struct DefaultValue(
    #[serde(with = "crate::storage::value_codec::serde_option_value")] Option<Value>,
);

/// A constraint of a node or edge type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeConstraintRecord {
    /// The properties identify a node: unique together and not null.
    PrimaryKey(Vec<String>),
    /// The properties are unique together.
    Unique(Vec<String>),
    /// The property is not null.
    NotNull(String),
    /// A `CHECK` expression.
    Check {
        /// The constraint's name, if it has one.
        name: Option<String>,
        /// The expression, as written.
        expression: String,
    },
}

/// A node type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeTypeRecord {
    /// The type's name, which is also its label.
    pub name: String,
    /// Its properties, in declaration order.
    #[serde(deserialize_with = "deserialize_properties")]
    pub properties: Vec<PropertyRecord>,
    /// Its constraints.
    pub constraints: Vec<TypeConstraintRecord>,
    /// The types it extends.
    pub parent_types: Vec<String>,
    /// The labels of its `KEY (...)` clause in a graph type.
    pub key_labels: Vec<String>,
}

/// One (source, target) pair of node types an edge type connects; `None`
/// is any node type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EndpointPair {
    /// The source node type.
    pub source: Option<String>,
    /// The target node type.
    pub target: Option<String>,
}

/// An edge type.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeTypeRecord {
    /// The type's name, which is also its edge type.
    pub name: String,
    /// Its properties, in declaration order.
    #[serde(deserialize_with = "deserialize_properties")]
    pub properties: Vec<PropertyRecord>,
    /// Its constraints.
    pub constraints: Vec<TypeConstraintRecord>,
    /// The pairs of node types it connects; empty for any.
    pub endpoints: Vec<EndpointPair>,
    /// The labels of its `KEY (...)` clause in a graph type.
    pub key_labels: Vec<String>,
}

/// A graph type: the node and edge types a graph of it may hold.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTypeRecord {
    /// The graph type's name.
    pub name: String,
    /// The node types it lists.
    pub node_types: Vec<String>,
    /// The edge types it lists.
    pub edge_types: Vec<String>,
    /// Whether a graph of it may also hold types it does not list.
    pub open: bool,
}

/// A graph bound to a graph type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphBindingRecord {
    /// The graph's name.
    pub graph: String,
    /// The graph type's name.
    pub graph_type: String,
}

/// What a named constraint requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NamedConstraintKindRecord {
    /// The properties are unique among the label's nodes.
    Unique,
    /// The properties are present and unique together.
    NodeKey,
    /// The properties are present (`NOT NULL`).
    NotNull,
    /// The properties are present (`EXISTS`).
    Exists,
}

/// A named constraint (`CREATE CONSTRAINT`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConstraintRecord {
    /// The constraint's name.
    pub name: String,
    /// The node label it applies to.
    pub label: String,
    /// The properties it constrains.
    pub properties: Vec<String>,
    /// What it requires.
    pub kind: NamedConstraintKindRecord,
}

/// The distance a vector index measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistanceMetricRecord {
    /// Cosine distance.
    Cosine,
    /// Euclidean (L2) distance.
    Euclidean,
    /// Negative dot product.
    DotProduct,
    /// Manhattan (L1) distance.
    Manhattan,
}

/// How a vector index compresses its vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationRecord {
    /// Full precision.
    None,
    /// One byte per component.
    Scalar,
    /// One bit per component.
    Binary,
    /// Product quantization.
    Product {
        /// The number of subvectors.
        num_subvectors: u32,
    },
}

/// What an index indexes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKindRecord {
    /// A property index on a node property key.
    Property {
        /// The property key.
        key: String,
    },
    /// A vector index on a label's property.
    Vector {
        /// The node label.
        label: String,
        /// The property holding the vectors.
        property: String,
        /// The vectors' dimensions.
        dimensions: u32,
        /// The distance it measures.
        metric: DistanceMetricRecord,
        /// The HNSW links per node.
        m: u32,
        /// The HNSW candidate list size while building.
        ef_construction: u32,
        /// How it compresses vectors.
        quantization: QuantizationRecord,
    },
    /// A full-text index on a label's property.
    Text {
        /// The node label.
        label: String,
        /// The indexed property.
        property: String,
    },
}

/// An index of one graph. It holds the definition only: a load builds the
/// index from the data, or restores it from its own section.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexRecord {
    /// The graph's storage key; `None` for the default graph.
    pub graph: Option<String>,
    /// What it indexes.
    pub index: IndexKindRecord,
}

/// The kind of index a `CREATE INDEX` name stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexNameKindRecord {
    /// A hash index, for equality.
    Hash,
    /// A B-tree index, for ranges.
    BTree,
    /// A full-text index.
    FullText,
}

/// The name `CREATE INDEX` gave an index, for `SHOW INDEXES` and
/// `DROP INDEX`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexNameRecord {
    /// The index's name.
    pub name: String,
    /// The node label it indexes.
    pub label: String,
    /// The property it indexes.
    pub property: String,
    /// The kind of index.
    pub kind: IndexNameKindRecord,
}

/// A stored procedure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcedureRecord {
    /// The procedure's name.
    pub name: String,
    /// Its parameters, as (name, type).
    pub params: Vec<(String, String)>,
    /// Its result columns, as (name, type).
    pub returns: Vec<(String, String)>,
    /// Its query, as written.
    pub body: String,
}

/// A schema namespace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchemaRecord {
    /// The schema's name.
    pub name: String,
}

/// What identifies an index of a graph: what it indexes, without its
/// parameters.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IndexKeyRecord {
    /// The property index on a node property key.
    Property {
        /// The property key.
        #[serde(deserialize_with = "text")]
        key: String,
    },
    /// The vector index on a label's property.
    Vector {
        /// The node label.
        #[serde(deserialize_with = "text")]
        label: String,
        /// The property holding the vectors.
        #[serde(deserialize_with = "text")]
        property: String,
    },
    /// The full-text index on a label's property.
    Text {
        /// The node label.
        #[serde(deserialize_with = "text")]
        label: String,
        /// The indexed property.
        #[serde(deserialize_with = "text")]
        property: String,
    },
}

/// What identifies a catalog record: what dropping it names
/// ([`StandaloneOp::DropCatalog`](crate::change::StandaloneOp::DropCatalog)),
/// and what [`CatalogRecord::key`] returns. Its kind is the kind of the
/// catalog records it identifies.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CatalogKey {
    /// A schema namespace, by name (catalog kind 1).
    Schema(String),
    /// A node type, by name (catalog kind 2).
    NodeType(String),
    /// An edge type, by name (catalog kind 3).
    EdgeType(String),
    /// A graph type, by name (catalog kind 4).
    GraphType(String),
    /// The graph type binding of a graph, by the graph's name (catalog kind 5).
    GraphBinding(String),
    /// A named constraint, by name (catalog kind 6).
    Constraint(String),
    /// An index of a graph (catalog kind 7).
    Index {
        /// The graph's storage key; `None` for the default graph.
        graph: Option<String>,
        /// What the index indexes.
        index: IndexKeyRecord,
    },
    /// An index name, by the name (catalog kind 8).
    IndexName(String),
    /// A stored procedure, by name (catalog kind 9).
    Procedure(String),
}

impl CatalogKey {
    /// The kind of the catalog records the key identifies, as
    /// [`CatalogRecord::kind`] numbers them: 1 `Schema` to 9 `Procedure`.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::Schema(_) => KIND_SCHEMA,
            Self::NodeType(_) => KIND_NODE_TYPE,
            Self::EdgeType(_) => KIND_EDGE_TYPE,
            Self::GraphType(_) => KIND_GRAPH_TYPE,
            Self::GraphBinding(_) => KIND_GRAPH_BINDING,
            Self::Constraint(_) => KIND_CONSTRAINT,
            Self::Index { .. } => KIND_INDEX,
            Self::IndexName(_) => KIND_INDEX_NAME,
            Self::Procedure(_) => KIND_PROCEDURE,
        }
    }
}

/// One record of the catalog. Matches over it are exhaustive on purpose: a
/// kind added later must be handled by every reader of this crate.
#[derive(Debug, Clone, PartialEq)]
pub enum CatalogRecord {
    /// A schema namespace (kind 1).
    Schema(SchemaRecord),
    /// A node type (kind 2).
    NodeType(NodeTypeRecord),
    /// An edge type (kind 3).
    EdgeType(EdgeTypeRecord),
    /// A graph type (kind 4).
    GraphType(GraphTypeRecord),
    /// A graph bound to a graph type (kind 5).
    GraphBinding(GraphBindingRecord),
    /// A named constraint (kind 6).
    Constraint(ConstraintRecord),
    /// An index definition (kind 7).
    Index(IndexRecord),
    /// An index name (kind 8).
    IndexName(IndexNameRecord),
    /// A stored procedure (kind 9).
    Procedure(ProcedureRecord),
}

impl CatalogRecord {
    /// The record's kind, the first byte of its frame: 1 `Schema`,
    /// 2 `NodeType`, 3 `EdgeType`, 4 `GraphType`, 5 `GraphBinding`,
    /// 6 `Constraint`, 7 `Index`, 8 `IndexName`, 9 `Procedure`.
    #[must_use]
    pub const fn kind(&self) -> u8 {
        match self {
            Self::Schema(_) => KIND_SCHEMA,
            Self::NodeType(_) => KIND_NODE_TYPE,
            Self::EdgeType(_) => KIND_EDGE_TYPE,
            Self::GraphType(_) => KIND_GRAPH_TYPE,
            Self::GraphBinding(_) => KIND_GRAPH_BINDING,
            Self::Constraint(_) => KIND_CONSTRAINT,
            Self::Index(_) => KIND_INDEX,
            Self::IndexName(_) => KIND_INDEX_NAME,
            Self::Procedure(_) => KIND_PROCEDURE,
        }
    }

    /// What identifies the record: what a drop of it names. A record put
    /// again with the same key replaces this one.
    #[must_use]
    pub fn key(&self) -> CatalogKey {
        match self {
            Self::Schema(record) => CatalogKey::Schema(record.name.clone()),
            Self::NodeType(record) => CatalogKey::NodeType(record.name.clone()),
            Self::EdgeType(record) => CatalogKey::EdgeType(record.name.clone()),
            Self::GraphType(record) => CatalogKey::GraphType(record.name.clone()),
            Self::GraphBinding(record) => CatalogKey::GraphBinding(record.graph.clone()),
            Self::Constraint(record) => CatalogKey::Constraint(record.name.clone()),
            Self::Index(record) => CatalogKey::Index {
                graph: record.graph.clone(),
                index: match &record.index {
                    IndexKindRecord::Property { key } => {
                        IndexKeyRecord::Property { key: key.clone() }
                    }
                    IndexKindRecord::Vector {
                        label, property, ..
                    } => IndexKeyRecord::Vector {
                        label: label.clone(),
                        property: property.clone(),
                    },
                    IndexKindRecord::Text { label, property } => IndexKeyRecord::Text {
                        label: label.clone(),
                        property: property.clone(),
                    },
                },
            },
            Self::IndexName(record) => CatalogKey::IndexName(record.name.clone()),
            Self::Procedure(record) => CatalogKey::Procedure(record.name.clone()),
        }
    }

    /// Appends the record framed: `[kind u8][flags u8][length u32 LE][payload]`.
    /// Every kind of this release is written with [`RECORD_REQUIRED`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when the payload would hold more than
    /// [`MAX_CATALOG_RECORD_PAYLOAD`] bytes (for default values, counted
    /// before one is encoded), a property type nests more than
    /// [`MAX_LIST_TYPE_DEPTH`] `LIST` levels, the property types nest more
    /// than [`MAX_LIST_LEVELS_PER_RECORD`] in all, or the value codec refuses
    /// a default value. `out` is then left as it was.
    pub fn encode_framed(&self, out: &mut Vec<u8>) -> Result<()> {
        encode_framed_record(&CATALOG_FRAMING, self.kind(), RECORD_REQUIRED, out, |out| {
            // The value codec encodes each default value whole before the
            // payload takes it: refuse them from their sizes first.
            out.check_room(self.default_value_bytes())?;
            self.encode_payload(out)
        })
    }

    /// Writes the record's payload, without its frame: bincode of the
    /// record type.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when a property type nests more than
    /// [`MAX_LIST_TYPE_DEPTH`] `LIST` levels, the property types nest more
    /// than [`MAX_LIST_LEVELS_PER_RECORD`] in all (before a byte is written),
    /// the value codec refuses a default value or `out` refuses a write;
    /// `out` may then hold part of the payload.
    pub(crate) fn encode_payload(&self, out: &mut impl io::Write) -> Result<()> {
        let levels = self
            .properties()
            .iter()
            .map(|property| property.data_type.levels_and_code().0)
            .fold(0usize, usize::saturating_add);
        if levels > MAX_LIST_LEVELS_PER_RECORD {
            return Err(Error::Serialization(format!(
                "a {} of kind {} would nest {levels} LIST levels in its property types, more \
                 than the {MAX_LIST_LEVELS_PER_RECORD} it may hold",
                CATALOG_FRAMING.what,
                self.kind()
            )));
        }
        match self {
            Self::Schema(record) => encode_bincode_payload(record, out),
            Self::NodeType(record) => encode_bincode_payload(record, out),
            Self::EdgeType(record) => encode_bincode_payload(record, out),
            Self::GraphType(record) => encode_bincode_payload(record, out),
            Self::GraphBinding(record) => encode_bincode_payload(record, out),
            Self::Constraint(record) => encode_bincode_payload(record, out),
            Self::Index(record) => encode_bincode_payload(record, out),
            Self::IndexName(record) => encode_bincode_payload(record, out),
            Self::Procedure(record) => encode_bincode_payload(record, out),
        }
    }

    /// The properties of a node or edge type; none for the other kinds.
    fn properties(&self) -> &[PropertyRecord] {
        match self {
            Self::NodeType(record) => &record.properties,
            Self::EdgeType(record) => &record.properties,
            Self::Schema(_)
            | Self::GraphType(_)
            | Self::GraphBinding(_)
            | Self::Constraint(_)
            | Self::Index(_)
            | Self::IndexName(_)
            | Self::Procedure(_) => &[],
        }
    }

    /// The bytes the value codec writes for the default values of the
    /// record's properties, counted without encoding them. A value nested
    /// deeper than the value codec encodes ([`nests_too_deep_to_encode`])
    /// counts nothing, so the count recurses no deeper than that: the codec
    /// refuses it as it reaches that depth. Every value the codec takes is
    /// counted, also one nested deeper than a property value may be.
    fn default_value_bytes(&self) -> usize {
        self.properties()
            .iter()
            .filter_map(|property| property.default_value.as_ref())
            .filter(|value| !nests_too_deep_to_encode(value))
            .map(encoded_len)
            .fold(0, usize::saturating_add)
    }

    /// The record of `kind` whose payload is `payload`, or `None` for a kind
    /// this release does not know.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Corruption`] when the payload does not decode as the
    /// kind's record type, claims more memory than the decode limit, or has
    /// bytes left over.
    pub(crate) fn decode_payload(kind: u8, payload: &[u8]) -> Result<Option<Self>> {
        fn decode<T: DeserializeOwned>(payload: &[u8]) -> Result<T> {
            decode_bincode_payload::<T, CATALOG_DECODE_LIMIT>(payload)
        }
        let record = match kind {
            KIND_SCHEMA => Self::Schema(decode(payload)?),
            KIND_NODE_TYPE => Self::NodeType(decode(payload)?),
            KIND_EDGE_TYPE => Self::EdgeType(decode(payload)?),
            KIND_GRAPH_TYPE => Self::GraphType(decode(payload)?),
            KIND_GRAPH_BINDING => Self::GraphBinding(decode(payload)?),
            KIND_CONSTRAINT => Self::Constraint(decode(payload)?),
            KIND_INDEX => Self::Index(decode(payload)?),
            KIND_INDEX_NAME => Self::IndexName(decode(payload)?),
            KIND_PROCEDURE => Self::Procedure(decode(payload)?),
            _ => return Ok(None),
        };
        Ok(Some(record))
    }
}

/// Reads framed catalog records from `reader` until it ends, and calls
/// `apply` with each record of a known kind, in order.
///
/// A record of an unknown kind is skipped when its required flag is clear.
/// A stream that ends between two records ends the catalog. Each payload is
/// read into a buffer that grows with the bytes present, so a damaged length
/// allocates nothing up front.
///
/// # Errors
///
/// Returns [`Error::Corruption`] naming the record (its index, from 0, and
/// the byte it starts at) for:
///
/// - kind 0;
/// - a payload longer than [`MAX_CATALOG_RECORD_PAYLOAD`];
/// - a stream that ends inside a record's header or payload;
/// - a payload that does not decode as its kind's record type, or has bytes
///   left over.
///
/// Returns [`Error::Serialization`] naming the record for what a newer
/// release wrote: a flag among bits 0 to 3 other than [`RECORD_REQUIRED`]
/// (bits 4 to 7 are ignored), or an unknown kind with the required flag set.
///
/// An error of `apply` comes back as it is and stops the read, and so does
/// an error of `reader`: the [`Error`] it carries when it carries one (as a
/// [`ChunkStreamReader`](super::ChunkStreamReader) does), else [`Error::Io`].
pub fn read_catalog_records(
    reader: &mut dyn Read,
    apply: &mut dyn FnMut(CatalogRecord) -> Result<()>,
) -> Result<()> {
    read_framed_records(
        &CATALOG_FRAMING,
        reader,
        &mut |kind, _required, payload| CatalogRecord::decode_payload(kind, payload),
        apply,
    )
}

// ── Framing, shared by every family of records ──────────────────────

/// A family of framed records: what its records are called in errors and
/// how many bytes a payload may hold.
pub(crate) struct RecordFraming {
    /// The records' name in errors, such as `catalog record`.
    pub what: &'static str,
    /// The most bytes a payload may hold.
    pub max_payload: u32,
}

/// The framing of catalog records.
pub(crate) const CATALOG_FRAMING: RecordFraming = RecordFraming {
    what: "catalog record",
    max_payload: MAX_CATALOG_RECORD_PAYLOAD,
};

impl RecordFraming {
    /// Where record `index` is, which starts at byte `offset` of the stream.
    fn record(&self, index: u64, offset: u64) -> String {
        format!("{} {index} at byte {offset}", self.what)
    }

    /// Damage in record `index`, which starts at byte `offset`.
    fn damage(&self, index: u64, offset: u64, message: impl fmt::Display) -> Error {
        Error::corruption(format!("{}: {message}", self.record(index, offset)))
    }

    /// Record `index`, which starts at byte `offset`, holds something a
    /// newer release wrote.
    fn refusal(&self, index: u64, offset: u64, message: impl fmt::Display) -> Error {
        Error::Serialization(format!("{}: {message}", self.record(index, offset)))
    }
}

/// The writer a payload is written through: it appends to the output and
/// refuses a write that would take the payload past its maximum, before it
/// copies a byte of that write.
pub(crate) struct PayloadWriter<'a> {
    out: &'a mut Vec<u8>,
    /// The bytes the payload may still take.
    room: usize,
    /// Whether a write was refused for want of room.
    overflowed: bool,
}

impl PayloadWriter<'_> {
    /// Refuses, as a write past the maximum is refused, a part of `bytes`
    /// bytes the payload has no room left for: a part that is built whole
    /// before it is written is so refused before it is built.
    ///
    /// # Errors
    ///
    /// Returns [`Error::Serialization`] when `bytes` passes the room left;
    /// [`encode_framed_record`] then reports the payload as too large.
    pub(crate) fn check_room(&mut self, bytes: usize) -> Result<()> {
        if bytes > self.room {
            self.overflowed = true;
            return Err(Error::Serialization(
                "the payload passes its maximum".to_string(),
            ));
        }
        Ok(())
    }
}

impl io::Write for PayloadWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.len() > self.room {
            self.overflowed = true;
            return Err(io::Error::other("the payload passes its maximum"));
        }
        self.room -= buf.len();
        self.out.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Appends one record of `framing`: `kind`, `flags`, the payload's length
/// (u32 little endian) and the payload `write_payload` writes. The payload
/// is refused as soon as a write would take it past the family's maximum,
/// so a larger one is never copied whole.
///
/// # Errors
///
/// Returns [`Error::Internal`] for kind 0 or flag bits a reader refuses,
/// [`Error::Serialization`] for a payload longer than the family allows, or
/// the error of `write_payload`. `out` is then left as it was.
pub(crate) fn encode_framed_record(
    framing: &RecordFraming,
    kind: u8,
    flags: u8,
    out: &mut Vec<u8>,
    write_payload: impl FnOnce(&mut PayloadWriter<'_>) -> Result<()>,
) -> Result<()> {
    if kind == 0 || flags & !KNOWN_FLAGS != 0 {
        return Err(Error::Internal(format!(
            "cannot write a {} of kind {kind} with flags {flags:#04x}: a reader refuses kind 0 \
             and flag bits other than bit 0",
            framing.what
        )));
    }
    let start = out.len();
    out.extend_from_slice(&[kind, flags, 0, 0, 0, 0]);
    let mut writer = PayloadWriter {
        out,
        room: usize::try_from(framing.max_payload).unwrap_or(usize::MAX),
        overflowed: false,
    };
    let written = match write_payload(&mut writer) {
        Err(_) if writer.overflowed => Err(Error::Serialization(format!(
            "a {} of kind {kind} would hold more than the {} bytes it may hold",
            framing.what, framing.max_payload
        ))),
        written => written,
    }
    .and_then(|()| {
        let length = out.len() - start - RECORD_HEADER_BYTES;
        // The writer kept the payload within the maximum, a u32.
        let length = u32::try_from(length).map_err(|_| {
            Error::Internal(format!(
                "a {} of kind {kind} holds {length} bytes, past its maximum",
                framing.what
            ))
        })?;
        out[start + 2..start + RECORD_HEADER_BYTES].copy_from_slice(&length.to_le_bytes());
        Ok(())
    });
    if written.is_err() {
        out.truncate(start);
    }
    written
}

/// Appends bincode (standard configuration) of `payload`.
///
/// # Errors
///
/// Returns [`Error::Serialization`] when `payload` refuses to serialize or
/// `out` refuses a write.
pub(crate) fn encode_bincode_payload<T: Serialize>(
    payload: &T,
    out: &mut impl io::Write,
) -> Result<()> {
    bincode::serde::encode_into_std_write(payload, out, bincode::config::standard())
        .map(|_| ())
        .map_err(|error| Error::Serialization(format!("the payload does not encode: {error}")))
}

/// Decodes a whole payload as bincode (standard configuration) of `T`,
/// claiming at most `LIMIT` bytes of memory for the lengths inside it.
///
/// # Errors
///
/// Returns [`Error::Corruption`] when the payload does not decode, claims
/// more than `LIMIT`, or has bytes left over.
pub(crate) fn decode_bincode_payload<T: DeserializeOwned, const LIMIT: usize>(
    payload: &[u8],
) -> Result<T> {
    let config = bincode::config::standard().with_limit::<LIMIT>();
    match bincode::serde::decode_from_slice::<T, _>(payload, config) {
        Ok((record, used)) if used == payload.len() => Ok(record),
        Ok((_, used)) => Err(Error::corruption(format!(
            "the payload decodes from {used} of its {} bytes; the rest is left over",
            payload.len()
        ))),
        Err(bincode::error::DecodeError::LimitExceeded) => Err(Error::corruption(format!(
            "a length inside the payload claims more than the decode limit of {LIMIT} bytes"
        ))),
        Err(error) => Err(Error::corruption(format!(
            "the payload does not decode: {error}"
        ))),
    }
}

/// Reads a string in place: its length is checked against the bytes present
/// before the string is copied out. (A `String` field of a derived
/// `Deserialize` would allocate the length bincode reads before it reads
/// the bytes.)
pub(crate) fn text<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<String, D::Error> {
    deserializer.deserialize_str(TextVisitor)
}

/// Reads an optional string in place (see [`text`]).
pub(crate) fn optional_text<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<String>, D::Error> {
    Option::<Text>::deserialize(deserializer).map(|text| text.map(|Text(text)| text))
}

struct TextVisitor;

impl Visitor<'_> for TextVisitor {
    type Value = String;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a string")
    }

    fn visit_str<E: serde::de::Error>(self, text: &str) -> std::result::Result<String, E> {
        Ok(text.to_owned())
    }
}

/// A string read in place (see [`text`]).
pub(crate) struct Text(pub(crate) String);

impl<'de> Deserialize<'de> for Text {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        text(deserializer).map(Text)
    }
}

/// Reads records of `framing` from `reader` until it ends. `decode` turns a
/// kind, whether the record is required (its flag) and its payload into a
/// record, or `None` for a record it does not know; `apply` takes each
/// record, in order.
///
/// A record `decode` does not know is skipped when its required flag is
/// clear. A record of a known kind can hold something `decode` does not know
/// (a log record holding a catalog record of a later kind): `decode` then
/// returns `None` when the record is not required, and its own error naming
/// what it does not know when it is. A stream that ends between two records
/// ends the read.
///
/// # Errors
///
/// As [`read_catalog_records`] lists: [`Error::Corruption`] naming the
/// record for damage the framing finds, [`Error::Serialization`] naming it
/// for what a newer release wrote, an error of `decode` of those variants
/// with the record named, and the errors of `apply` and `reader` as they are.
pub(crate) fn read_framed_records<T>(
    framing: &RecordFraming,
    reader: &mut dyn Read,
    decode: &mut dyn FnMut(u8, bool, &[u8]) -> Result<Option<T>>,
    apply: &mut dyn FnMut(T) -> Result<()>,
) -> Result<()> {
    let mut payload = Vec::new();
    let mut index = 0u64;
    let mut offset = 0u64;
    loop {
        let mut header = [0u8; RECORD_HEADER_BYTES];
        let filled = fill(reader, &mut header).map_err(reader_error)?;
        if filled == 0 {
            return Ok(());
        }
        if filled < RECORD_HEADER_BYTES {
            return Err(framing.damage(
                index,
                offset,
                format_args!(
                    "the stream ends inside the record header, after {filled} of its \
                     {RECORD_HEADER_BYTES} bytes"
                ),
            ));
        }
        let [kind, flags, length @ ..] = header;
        let length = u32::from_le_bytes(length);
        if flags & INCOMPATIBLE_FLAGS & !KNOWN_FLAGS != 0 {
            return Err(framing.refusal(
                index,
                offset,
                format_args!(
                    "flags {flags:#04x} hold bits among 0 to 3 this release does not know (bit 0, required, \
                     is the only one)"
                ),
            ));
        }
        if kind == 0 {
            return Err(framing.damage(index, offset, "kind 0 is never written"));
        }
        if length > framing.max_payload {
            return Err(framing.damage(
                index,
                offset,
                format_args!(
                    "a payload of {length} bytes, more than the {} a {} may hold",
                    framing.max_payload, framing.what
                ),
            ));
        }
        payload.clear();
        let read = Read::take(&mut *reader, u64::from(length))
            .read_to_end(&mut payload)
            .map_err(reader_error)?;
        if (read as u64) < u64::from(length) {
            return Err(framing.damage(
                index,
                offset,
                format_args!(
                    "the stream ends inside the payload, after {read} of its {length} bytes"
                ),
            ));
        }
        let required = flags & RECORD_REQUIRED != 0;
        let decoded = decode(kind, required, &payload).map_err(|error| match error {
            Error::Serialization(message) => {
                framing.refusal(index, offset, format_args!("kind {kind}: {message}"))
            }
            Error::Corruption(_) => error.wrapped(format_args!(
                "{}: kind {kind}",
                framing.record(index, offset)
            )),
            other => other,
        })?;
        match decoded {
            Some(record) => apply(record)?,
            None if required => {
                return Err(framing.refusal(
                    index,
                    offset,
                    format_args!(
                        "kind {kind} is required, but this release does not know it (a newer \
                         release wrote it)"
                    ),
                ));
            }
            None => {}
        }
        index += 1;
        offset += (RECORD_HEADER_BYTES as u64) + u64::from(length);
    }
}

/// Reads into `buffer` until it is full or the stream ends, retrying an
/// interrupted read; returns the bytes read.
fn fill(reader: &mut dyn Read, buffer: &mut [u8]) -> io::Result<usize> {
    let mut filled = 0;
    while filled < buffer.len() {
        match reader.read(&mut buffer[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => {}
            Err(error) => return Err(error),
        }
    }
    Ok(filled)
}

/// The [`Error`] a reader's I/O error carries, as a chunk stream reader's
/// errors do, else the I/O error itself.
fn reader_error(error: io::Error) -> Error {
    if !error.get_ref().is_some_and(|inner| inner.is::<Error>()) {
        return Error::Io(error);
    }
    let kind = error.kind();
    match error.into_inner().map(|inner| inner.downcast::<Error>()) {
        Some(Ok(inner)) => *inner,
        // Not reached: the check above found an `Error` inside.
        Some(Err(inner)) => Error::Io(io::Error::new(kind, inner)),
        None => Error::Io(io::Error::from(kind)),
    }
}

#[cfg(test)]
mod tests {
    use std::io::{self, Read};

    use serde::Serialize;

    use super::*;
    use crate::types::{Timestamp, Value, ZonedDatetime};
    use crate::utils::error::Error;

    fn names(items: &[&str]) -> Vec<String> {
        items.iter().map(|item| (*item).to_string()).collect()
    }

    fn property(
        name: &str,
        data_type: PropertyTypeRecord,
        nullable: bool,
        default_value: Option<Value>,
    ) -> PropertyRecord {
        PropertyRecord {
            name: name.to_string(),
            data_type,
            nullable,
            default_value,
        }
    }

    fn schema(name: &str) -> CatalogRecord {
        CatalogRecord::Schema(SchemaRecord {
            name: name.to_string(),
        })
    }

    fn framed(record: &CatalogRecord) -> Vec<u8> {
        let mut bytes = Vec::new();
        record.encode_framed(&mut bytes).unwrap();
        bytes
    }

    fn read_all(bytes: &[u8]) -> Result<Vec<CatalogRecord>> {
        let mut records = Vec::new();
        read_catalog_records(&mut &bytes[..], &mut |record| {
            records.push(record);
            Ok(())
        })?;
        Ok(records)
    }

    fn read_error(bytes: &[u8]) -> String {
        read_all(bytes).unwrap_err().to_string()
    }

    fn bincode_of<T: Serialize>(value: &T) -> Vec<u8> {
        bincode::serde::encode_to_vec(value, bincode::config::standard()).unwrap()
    }

    /// Every property type that is not a `LIST<...>`, in code order.
    fn element_types() -> Vec<PropertyTypeRecord> {
        use PropertyTypeRecord as T;
        vec![
            T::String,
            T::Int64,
            T::Float64,
            T::Bool,
            T::Date,
            T::Time,
            T::Timestamp,
            T::LocalDatetime,
            T::ZonedDatetime,
            T::Duration,
            T::List,
            T::Map,
            T::Bytes,
            T::Node,
            T::Edge,
            T::Any,
        ]
    }

    fn list_of(depth: usize, element: PropertyTypeRecord) -> PropertyTypeRecord {
        (0..depth).fold(element, |inner, _| {
            PropertyTypeRecord::ListOf(Box::new(inner))
        })
    }

    fn one_of_each() -> Vec<CatalogRecord> {
        vec![
            schema("travel"),
            CatalogRecord::NodeType(NodeTypeRecord {
                name: "City".into(),
                properties: vec![
                    property("name", PropertyTypeRecord::String, false, None),
                    property(
                        "country",
                        PropertyTypeRecord::String,
                        true,
                        Some(Value::from("NL")),
                    ),
                ],
                constraints: vec![TypeConstraintRecord::Unique(names(&["name"]))],
                parent_types: names(&["Place"]),
                key_labels: names(&["CityKey"]),
            }),
            CatalogRecord::EdgeType(EdgeTypeRecord {
                name: "ROUTE".into(),
                properties: vec![property(
                    "km",
                    PropertyTypeRecord::Int64,
                    true,
                    Some(Value::Int64(88)),
                )],
                constraints: Vec::new(),
                endpoints: vec![EndpointPair {
                    source: Some("City".into()),
                    target: Some("City".into()),
                }],
                key_labels: Vec::new(),
            }),
            CatalogRecord::GraphType(GraphTypeRecord {
                name: "travel".into(),
                node_types: names(&["City"]),
                edge_types: names(&["ROUTE"]),
                open: false,
            }),
            CatalogRecord::GraphBinding(GraphBindingRecord {
                graph: "trips".into(),
                graph_type: "travel".into(),
            }),
            CatalogRecord::Constraint(ConstraintRecord {
                name: "person_email".into(),
                label: "Person".into(),
                properties: names(&["email"]),
                kind: NamedConstraintKindRecord::NotNull,
            }),
            CatalogRecord::Index(IndexRecord {
                graph: None,
                index: IndexKindRecord::Property { key: "id".into() },
            }),
            CatalogRecord::Index(IndexRecord {
                graph: Some("trips".into()),
                index: IndexKindRecord::Vector {
                    label: "Doc".into(),
                    property: "emb".into(),
                    dimensions: 3,
                    metric: DistanceMetricRecord::Cosine,
                    m: 16,
                    ef_construction: 200,
                    quantization: QuantizationRecord::Product { num_subvectors: 8 },
                },
            }),
            CatalogRecord::Index(IndexRecord {
                graph: None,
                index: IndexKindRecord::Text {
                    label: "Doc".into(),
                    property: "body".into(),
                },
            }),
            CatalogRecord::IndexName(IndexNameRecord {
                name: "person_name".into(),
                label: "Person".into(),
                property: "name".into(),
                kind: IndexNameKindRecord::BTree,
            }),
            CatalogRecord::Procedure(ProcedureRecord {
                name: "get_adults".into(),
                params: Vec::new(),
                returns: vec![("name".into(), "STRING".into())],
                body: "MATCH (p:Person) WHERE p.age >= 19 RETURN p.name AS name".into(),
            }),
        ]
    }

    /// Records holding every variant of every enum the records use, with
    /// the default values the 0.5.x encodings lose.
    fn every_variant() -> Vec<CatalogRecord> {
        let mut properties: Vec<PropertyRecord> = element_types()
            .into_iter()
            .enumerate()
            .map(|(code, data_type)| property(&format!("p{code}"), data_type, code % 3 == 0, None))
            .collect();
        properties.push(property(
            "stamps",
            list_of(2, PropertyTypeRecord::ZonedDatetime),
            true,
            None,
        ));
        properties.push(property(
            "nan",
            PropertyTypeRecord::Float64,
            true,
            Some(Value::Float64(f64::from_bits(0x7FF8_0000_0000_0058))),
        ));
        properties.push(property(
            "negative_zero",
            PropertyTypeRecord::Float64,
            true,
            Some(Value::Float64(-0.0)),
        ));
        properties.push(property(
            "departs",
            PropertyTypeRecord::ZonedDatetime,
            true,
            Some(Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(1_696_500_000_123_457),
                7200,
            ))),
        ));
        let mut records = vec![
            CatalogRecord::NodeType(NodeTypeRecord {
                name: "Event".into(),
                properties,
                constraints: vec![
                    TypeConstraintRecord::PrimaryKey(names(&["id"])),
                    TypeConstraintRecord::Unique(names(&["name", "city"])),
                    TypeConstraintRecord::NotNull("name".into()),
                    TypeConstraintRecord::Check {
                        name: None,
                        expression: "seats > 3".into(),
                    },
                    TypeConstraintRecord::Check {
                        name: Some("enough".into()),
                        expression: "seats < 88".into(),
                    },
                ],
                parent_types: names(&["Happening", "Gathering"]),
                key_labels: names(&["EventKey", "Dated"]),
            }),
            CatalogRecord::EdgeType(EdgeTypeRecord {
                name: "VISITS".into(),
                properties: Vec::new(),
                constraints: Vec::new(),
                endpoints: vec![
                    EndpointPair {
                        source: Some("Person".into()),
                        target: None,
                    },
                    EndpointPair {
                        source: None,
                        target: Some("Museum".into()),
                    },
                    EndpointPair {
                        source: None,
                        target: None,
                    },
                ],
                key_labels: names(&["VisitKey"]),
            }),
            CatalogRecord::GraphType(GraphTypeRecord {
                name: "open_city".into(),
                node_types: Vec::new(),
                edge_types: Vec::new(),
                open: true,
            }),
        ];
        for (name, kind) in [
            ("person_unique", NamedConstraintKindRecord::Unique),
            ("person_key", NamedConstraintKindRecord::NodeKey),
            ("person_name", NamedConstraintKindRecord::NotNull),
            ("person_city", NamedConstraintKindRecord::Exists),
        ] {
            records.push(CatalogRecord::Constraint(ConstraintRecord {
                name: name.into(),
                label: "Person".into(),
                properties: names(&["name", "city"]),
                kind,
            }));
        }
        let metrics = [
            DistanceMetricRecord::Cosine,
            DistanceMetricRecord::Euclidean,
            DistanceMetricRecord::DotProduct,
            DistanceMetricRecord::Manhattan,
        ];
        let quantizations = [
            QuantizationRecord::None,
            QuantizationRecord::Scalar,
            QuantizationRecord::Binary,
            QuantizationRecord::Product { num_subvectors: 19 },
        ];
        for (metric, quantization) in metrics.into_iter().zip(quantizations) {
            records.push(CatalogRecord::Index(IndexRecord {
                graph: Some("Berlin".into()),
                index: IndexKindRecord::Vector {
                    label: "Doc".into(),
                    property: "emb".into(),
                    dimensions: 88,
                    metric,
                    m: 19,
                    ef_construction: 88,
                    quantization,
                },
            }));
        }
        for (name, kind) in [
            ("doc_hash", IndexNameKindRecord::Hash),
            ("doc_btree", IndexNameKindRecord::BTree),
            ("doc_text", IndexNameKindRecord::FullText),
        ] {
            records.push(CatalogRecord::IndexName(IndexNameRecord {
                name: name.into(),
                label: "Doc".into(),
                property: "title".into(),
                kind,
            }));
        }
        records.push(CatalogRecord::Procedure(ProcedureRecord {
            name: "visitors".into(),
            params: vec![
                ("city".into(), "STRING".into()),
                ("min".into(), "INT64".into()),
            ],
            returns: vec![
                ("name".into(), "STRING".into()),
                ("visits".into(), "INT64".into()),
            ],
            body: "MATCH (p:Person)-[:VISITS]->(:Museum {city: $city}) RETURN p.name AS name, \
                   count(*) AS visits"
                .into(),
        }));
        records
    }

    #[test]
    fn every_record_kind_round_trips() {
        let mut bytes = Vec::new();
        for record in one_of_each() {
            record.encode_framed(&mut bytes).unwrap();
        }
        let back = read_all(&bytes).unwrap();
        assert_eq!(back, one_of_each());
        let kinds: Vec<u8> = back.iter().map(CatalogRecord::kind).collect();
        assert_eq!(
            kinds,
            [1, 2, 3, 4, 5, 6, 7, 7, 7, 8, 9],
            "one kind per record type, numbered as the module documents"
        );
    }

    /// A record's key names it by what identifies it (a name, the graph of
    /// a binding, the graph and target of an index), with the record's kind,
    /// and two records a put would replace one with the other share it.
    #[test]
    fn every_record_names_its_key() {
        let keys: Vec<CatalogKey> = one_of_each().iter().map(CatalogRecord::key).collect();
        assert_eq!(
            keys,
            [
                CatalogKey::Schema("travel".into()),
                CatalogKey::NodeType("City".into()),
                CatalogKey::EdgeType("ROUTE".into()),
                CatalogKey::GraphType("travel".into()),
                CatalogKey::GraphBinding("trips".into()),
                CatalogKey::Constraint("person_email".into()),
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
                    graph: None,
                    index: IndexKeyRecord::Text {
                        label: "Doc".into(),
                        property: "body".into(),
                    },
                },
                CatalogKey::IndexName("person_name".into()),
                CatalogKey::Procedure("get_adults".into()),
            ]
        );
        for record in one_of_each().iter().chain(&every_variant()) {
            assert_eq!(record.key().kind(), record.kind(), "{record:?}");
        }
        // The vector indexes of `every_variant` differ only in parameters:
        // one key, so a put of one replaces another.
        let vector_keys: Vec<CatalogKey> = every_variant()
            .iter()
            .filter(|record| matches!(record, CatalogRecord::Index(_)))
            .map(CatalogRecord::key)
            .collect();
        assert_eq!(vector_keys.len(), 4);
        assert!(
            vector_keys.iter().all(|key| *key == vector_keys[0]),
            "{vector_keys:?}"
        );
    }

    #[test]
    fn every_variant_and_default_value_round_trips_bit_for_bit() {
        for record in every_variant() {
            let bytes = framed(&record);
            let back = read_all(&bytes).unwrap();
            assert_eq!(back.len(), 1, "{record:?}");
            assert_eq!(
                framed(&back[0]),
                bytes,
                "encodes to the same bytes again: floats by their bits, offsets kept"
            );
            if let CatalogRecord::NodeType(node) = &record {
                let CatalogRecord::NodeType(back_node) = &back[0] else {
                    panic!("a node type came back as {:?}", back[0]);
                };
                let types = |node: &NodeTypeRecord| -> Vec<PropertyTypeRecord> {
                    node.properties
                        .iter()
                        .map(|property| property.data_type.clone())
                        .collect()
                };
                assert_eq!(types(back_node), types(node));
                assert_eq!(back_node.constraints, node.constraints);
            } else {
                assert_eq!(back[0], record);
            }
        }
    }

    #[test]
    fn the_framing_is_fixed() {
        let mut bytes = Vec::new();
        CatalogRecord::Schema(SchemaRecord {
            name: "travel".into(),
        })
        .encode_framed(&mut bytes)
        .unwrap();
        assert_eq!(
            bytes,
            [
                1,
                RECORD_REQUIRED,
                7,
                0,
                0,
                0,
                6,
                b't',
                b'r',
                b'a',
                b'v',
                b'e',
                b'l'
            ]
        );
    }

    /// Characterization: pins the bytes of one record of every kind.
    #[test]
    fn the_record_layouts_are_pinned() {
        #[rustfmt::skip]
        const EXPECTED: &[&[u8]] = &[
            &[
                1, 1, 7, 0, 0, 0, // kind 1 (Schema), required, 7 bytes
                6, 116, 114, 97, 118, 101, 108, // "travel"
            ],
            &[
                2, 1, 59, 0, 0, 0, // kind 2 (NodeType), required, 59 bytes
                4, 67, 105, 116, 121, // "City"
                2, // two properties
                // "name": STRING (0 levels, code 0), not nullable, no default
                4, 110, 97, 109, 101, 0, 0, 0, 0,
                7, 99, 111, 117, 110, 116, 114, 121, // "country"
                0, 0, 1, 1, 7, // STRING, nullable, a default of 7 bytes:
                4, 2, 0, 0, 0, 78, 76, // the value codec's "NL"
                1, 1, 1, 4, 110, 97, 109, 101, // one constraint: Unique(["name"])
                1, 5, 80, 108, 97, 99, 101, // parent types ["Place"]
                1, 7, 67, 105, 116, 121, 75, 101, 121, // key labels ["CityKey"]
            ],
            &[
                3, 1, 39, 0, 0, 0, // kind 3 (EdgeType), required, 39 bytes
                5, 82, 79, 85, 84, 69, // "ROUTE"
                1, // one property
                2, 107, 109, // "km"
                0, 1, 1, 1, 9, // INT64, nullable, a default of 9 bytes:
                2, 88, 0, 0, 0, 0, 0, 0, 0, // the value codec's 88
                0, // no constraints
                // one endpoint pair: (City, City)
                1, 1, 4, 67, 105, 116, 121, 1, 4, 67, 105, 116, 121,
                0, // no key labels
            ],
            &[
                4, 1, 21, 0, 0, 0, // kind 4 (GraphType), required, 21 bytes
                6, 116, 114, 97, 118, 101, 108, // "travel"
                1, 4, 67, 105, 116, 121, // node types ["City"]
                1, 5, 82, 79, 85, 84, 69, // edge types ["ROUTE"]
                0, // closed
            ],
            &[
                5, 1, 13, 0, 0, 0, // kind 5 (GraphBinding), required, 13 bytes
                5, 116, 114, 105, 112, 115, // graph "trips"
                6, 116, 114, 97, 118, 101, 108, // graph type "travel"
            ],
            &[
                6, 1, 28, 0, 0, 0, // kind 6 (Constraint), required, 28 bytes
                12, 112, 101, 114, 115, 111, 110, 95, 101, 109, 97, 105, 108, // "person_email"
                6, 80, 101, 114, 115, 111, 110, // on "Person"
                1, 5, 101, 109, 97, 105, 108, // ["email"]
                2, // NOT NULL
            ],
            &[
                7, 1, 5, 0, 0, 0, // kind 7 (Index), required, 5 bytes
                0, // the default graph
                0, 2, 105, 100, // a property index on "id"
            ],
            &[
                7, 1, 22, 0, 0, 0, // kind 7 (Index), required, 22 bytes
                1, 5, 116, 114, 105, 112, 115, // graph "trips"
                1, // a vector index
                3, 68, 111, 99, 3, 101, 109, 98, // on "Doc"."emb"
                3, 0, 16, 200, // 3 dimensions, cosine, m 16, ef_construction 200
                3, 8, // product quantization, 8 subvectors
            ],
            &[
                7, 1, 11, 0, 0, 0, // kind 7 (Index), required, 11 bytes
                0, // the default graph
                2, // a text index
                3, 68, 111, 99, 4, 98, 111, 100, 121, // on "Doc"."body"
            ],
            &[
                8, 1, 25, 0, 0, 0, // kind 8 (IndexName), required, 25 bytes
                11, 112, 101, 114, 115, 111, 110, 95, 110, 97, 109, 101, // "person_name"
                6, 80, 101, 114, 115, 111, 110, // on "Person"
                4, 110, 97, 109, 101, // "name"
                1, // B-tree
            ],
            &[
                9, 1, 82, 0, 0, 0, // kind 9 (Procedure), required, 82 bytes
                10, 103, 101, 116, 95, 97, 100, 117, 108, 116, 115, // "get_adults"
                0, // no parameters
                1, 4, 110, 97, 109, 101, 6, 83, 84, 82, 73, 78, 71, // returns [("name", "STRING")]
                // the body, 56 bytes
                56, 77, 65, 84, 67, 72, 32, 40, 112, 58, 80, 101, 114, 115, 111, 110, 41, 32, 87,
                72, 69, 82, 69, 32, 112, 46, 97, 103, 101, 32, 62, 61, 32, 49, 57, 32, 82, 69, 84,
                85, 82, 78, 32, 112, 46, 110, 97, 109, 101, 32, 65, 83, 32, 110, 97, 109, 101,
            ],
        ];
        let encoded: Vec<Vec<u8>> = one_of_each().iter().map(framed).collect();
        assert_eq!(encoded, EXPECTED, "{encoded:?}");
    }

    /// The codes of property types and the variant indices of the other
    /// enums are part of the format.
    #[test]
    fn every_variant_has_a_fixed_code() {
        let codes: Vec<Vec<u8>> = element_types().iter().map(bincode_of).collect();
        let expected: Vec<Vec<u8>> = (0u8..16).map(|code| vec![0, code]).collect();
        assert_eq!(codes, expected, "no LIST level, then the element code");
        assert_eq!(
            bincode_of(&list_of(2, PropertyTypeRecord::ZonedDatetime)),
            [2, 8]
        );
        assert_eq!(bincode_of(&list_of(1, PropertyTypeRecord::List)), [1, 10]);

        let first = |bytes: Vec<u8>| bytes[0];
        assert_eq!(
            [
                TypeConstraintRecord::PrimaryKey(Vec::new()),
                TypeConstraintRecord::Unique(Vec::new()),
                TypeConstraintRecord::NotNull(String::new()),
                TypeConstraintRecord::Check {
                    name: None,
                    expression: String::new(),
                },
            ]
            .map(|variant| first(bincode_of(&variant))),
            [0, 1, 2, 3]
        );
        assert_eq!(
            [
                NamedConstraintKindRecord::Unique,
                NamedConstraintKindRecord::NodeKey,
                NamedConstraintKindRecord::NotNull,
                NamedConstraintKindRecord::Exists,
            ]
            .map(|variant| bincode_of(&variant)),
            [[0], [1], [2], [3]]
        );
        assert_eq!(
            [
                DistanceMetricRecord::Cosine,
                DistanceMetricRecord::Euclidean,
                DistanceMetricRecord::DotProduct,
                DistanceMetricRecord::Manhattan,
            ]
            .map(|variant| bincode_of(&variant)),
            [[0], [1], [2], [3]]
        );
        assert_eq!(
            [
                QuantizationRecord::None,
                QuantizationRecord::Scalar,
                QuantizationRecord::Binary,
            ]
            .map(|variant| bincode_of(&variant)),
            [[0], [1], [2]]
        );
        assert_eq!(
            bincode_of(&QuantizationRecord::Product { num_subvectors: 19 }),
            [3, 19]
        );
        assert_eq!(
            [
                IndexKindRecord::Property { key: String::new() },
                IndexKindRecord::Vector {
                    label: String::new(),
                    property: String::new(),
                    dimensions: 3,
                    metric: DistanceMetricRecord::Cosine,
                    m: 19,
                    ef_construction: 88,
                    quantization: QuantizationRecord::None,
                },
                IndexKindRecord::Text {
                    label: String::new(),
                    property: String::new(),
                },
            ]
            .map(|variant| first(bincode_of(&variant))),
            [0, 1, 2]
        );
        assert_eq!(
            [
                IndexNameKindRecord::Hash,
                IndexNameKindRecord::BTree,
                IndexNameKindRecord::FullText,
            ]
            .map(|variant| bincode_of(&variant)),
            [[0], [1], [2]]
        );
    }

    #[test]
    fn unknown_kinds_are_skipped_when_optional_and_refused_when_required() {
        let schema = framed(&schema("Berlin"));
        let optional = [&[99u8, 0, 3, 0, 0, 0, 3, 19, 88][..], &schema].concat();
        let mut seen = Vec::new();
        read_catalog_records(&mut &optional[..], &mut |record| {
            seen.push(record);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen.len(), 1, "only the schema");

        let required = [&[99u8, RECORD_REQUIRED, 0, 0, 0, 0][..], &schema].concat();
        let error = read_catalog_records(&mut &required[..], &mut |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("99") && error.contains("required"),
            "{error}"
        );

        let unknown_flag = [1u8, 0x80, 0, 0, 0, 0];
        assert!(read_catalog_records(&mut &unknown_flag[..], &mut |_| Ok(())).is_err());

        let error = read_error(&[&schema[..], &[99u8, RECORD_REQUIRED, 0, 0, 0, 0]].concat());
        assert!(
            error.contains("record 1 at byte 13"),
            "the second record, after the 13 bytes of the first: {error}"
        );
    }

    #[test]
    fn kind_0_is_never_valid() {
        let schema = framed(&schema("Prague"));
        for flags in [0, RECORD_REQUIRED] {
            let error = read_error(&[&[0u8, flags, 0, 0, 0, 0][..], &schema].concat());
            assert!(
                error.contains("kind 0") && error.contains("record 0 at byte 0"),
                "flags {flags}: {error}"
            );
        }
    }

    /// Damage is a corruption naming the record; what a newer release wrote
    /// is refused as such (a newer Grafeo reads it), never as damage.
    #[test]
    fn damage_is_corruption_and_a_newer_record_is_not() {
        let schema = framed(&schema("Amsterdam"));
        let mut left_over = schema.clone();
        left_over[2] += 1;
        left_over.push(88);
        let mut unknown_flag = schema.clone();
        unknown_flag[1] = 0x04;
        for (case, bytes) in [
            ("kind 0", [&[0u8, 0, 0, 0, 0, 0][..], &schema].concat()),
            ("a cut header", schema[..3].to_vec()),
            ("a cut payload", schema[..schema.len() - 1].to_vec()),
            ("bytes left over", left_over),
            (
                "an overlong payload",
                vec![1u8, RECORD_REQUIRED, 0xFF, 0xFF, 0xFF, 0x7F, 6],
            ),
        ] {
            let error = read_all(&bytes).unwrap_err();
            assert!(
                matches!(&error, Error::Corruption(corruption)
                    if corruption.what.starts_with("catalog record 0 at byte 0: ")),
                "{case}: {error:?}"
            );
        }
        for (case, bytes) in [
            (
                "an unknown required kind",
                [&[99u8, RECORD_REQUIRED, 0, 0, 0, 0][..], &schema].concat(),
            ),
            ("an unknown flag", unknown_flag),
        ] {
            let error = read_all(&bytes).unwrap_err();
            assert!(
                matches!(error, Error::Serialization(_)),
                "{case}: {error:?}"
            );
        }
    }

    #[test]
    fn only_unknown_flags_among_bits_0_to_3_are_refused() {
        for flags in 0..=u8::MAX {
            let mut bytes = framed(&schema("Paris"));
            bytes[1] = flags;
            let known = read_all(&bytes);
            // Bits 4 to 7 are ignored, so only bits 0 to 3 decide.
            let read_bits = flags & 0x0F;
            if read_bits <= RECORD_REQUIRED {
                assert_eq!(known.unwrap(), [schema("Paris")], "flags {flags:#04x}");
            } else {
                let error = known.unwrap_err().to_string();
                assert!(
                    error.contains(&format!("{flags:#04x}")),
                    "flags {flags:#04x}: {error}"
                );
            }
            bytes[0] = 99;
            let unknown = read_all(&bytes);
            match read_bits {
                0 => assert!(unknown.unwrap().is_empty(), "an optional record is skipped"),
                RECORD_REQUIRED => assert!(
                    unknown.unwrap_err().to_string().contains("required"),
                    "a required record is refused"
                ),
                _ => assert!(
                    unknown
                        .unwrap_err()
                        .to_string()
                        .contains(&format!("{flags:#04x}")),
                    "flags {flags:#04x} on an unknown kind"
                ),
            }
        }
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
            let result = read_catalog_records(&mut &bytes[..cut], &mut |_| {
                count += 1;
                Ok(())
            });
            match boundaries.iter().position(|&end| end == cut) {
                Some(index) => {
                    assert!(result.is_ok(), "cut at {cut}, between records");
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
            payload.contains("payload") && payload.contains("3 of its 7"),
            "{payload}"
        );

        let huge = [1u8, RECORD_REQUIRED, 0xFF, 0xFF, 0xFF, 0x7F, 6];
        let error = read_catalog_records(&mut &huge[..], &mut |_| Ok(()))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("2147483647") && error.contains(&MAX_CATALOG_RECORD_PAYLOAD.to_string()),
            "no allocation of 2 GiB: {error}"
        );
    }

    #[test]
    fn an_empty_stream_holds_no_records() {
        assert_eq!(read_all(&[]).unwrap(), []);
    }

    #[test]
    fn a_payload_with_bytes_left_over_is_refused() {
        let mut bytes = framed(&schema("Prague"));
        bytes[2] += 1;
        bytes.push(88);
        let error = read_error(&bytes);
        assert!(
            error.contains("7 of its 8") && error.contains("record 0 at byte 0"),
            "{error}"
        );
    }

    /// The decode limit covers a payload of the maximum length whatever it
    /// holds: a list of empty string pairs claims the most of the limit per
    /// byte (8 bytes for each one-byte length).
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a payload of 2 MiB, a million string pairs, takes hours under Miri; this module has no unsafe code"
    )]
    fn the_largest_payload_decodes_whatever_it_holds_and_one_byte_more_is_refused() {
        let procedure = |pairs: usize| {
            CatalogRecord::Procedure(ProcedureRecord {
                name: "get_adults".into(),
                params: vec![(String::new(), String::new()); pairs],
                returns: Vec::new(),
                body: String::new(),
            })
        };
        let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap();
        // name 11 bytes, params count 5, returns count 1, body 1: 18 bytes, then 2 per pair.
        let pairs = (maximum - 18) / 2;
        let largest = procedure(pairs);
        let bytes = framed(&largest);
        assert_eq!(
            bytes.len(),
            RECORD_HEADER_BYTES + maximum,
            "exactly the maximum"
        );
        assert!(
            read_all(&bytes).unwrap() == [largest],
            "the largest payload round trips"
        );

        let mut out = vec![3, 19, 88];
        let error = procedure(pairs + 1).encode_framed(&mut out).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let error = error.to_string();
        assert!(error.contains(&maximum.to_string()), "{error}");
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");

        let mut over = bytes.clone();
        over[2..RECORD_HEADER_BYTES]
            .copy_from_slice(&(MAX_CATALOG_RECORD_PAYLOAD + 1).to_le_bytes());
        over.push(0);
        let error = read_error(&over);
        assert!(
            error.contains(&(maximum + 1).to_string()) && error.contains(&maximum.to_string()),
            "refused from the header: {error}"
        );
    }

    fn node_type(properties: Vec<PropertyRecord>) -> CatalogRecord {
        CatalogRecord::NodeType(NodeTypeRecord {
            name: "Event".into(),
            properties,
            constraints: Vec::new(),
            parent_types: Vec::new(),
            key_labels: Vec::new(),
        })
    }

    fn edge_type(properties: Vec<PropertyRecord>) -> CatalogRecord {
        CatalogRecord::EdgeType(EdgeTypeRecord {
            name: "Event".into(),
            properties,
            constraints: Vec::new(),
            endpoints: Vec::new(),
            key_labels: Vec::new(),
        })
    }

    /// The most `LIST` levels a record holds, as densely as they encode:
    /// 256 properties with empty names, typed 128 levels deep.
    fn deepest_properties() -> Vec<PropertyRecord> {
        let deepest = property(
            "",
            list_of(MAX_LIST_TYPE_DEPTH, PropertyTypeRecord::String),
            true,
            None,
        );
        vec![deepest; MAX_LIST_LEVELS_PER_RECORD / MAX_LIST_TYPE_DEPTH]
    }

    /// The memory bound of the module documentation, at most 120 bytes per
    /// payload byte and 32 bytes per `LIST` level, so at most 1 MiB for the
    /// levels a record may hold, from its parts: each element a payload
    /// holds, encoded at its smallest, against the memory its types take
    /// (three times an element while its list grows), so a change to a record
    /// type that breaks the bound fails here. Then the densest record under
    /// [`MAX_LIST_LEVELS_PER_RECORD`], 256 properties typed 128 levels deep in
    /// 1,292 payload bytes, decodes to one box per level and stays within the
    /// bound, which it could not without the levels' own term: their memory
    /// does not grow with the payload.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "a record of 32,768 LIST levels takes a minute under Miri; this module has no unsafe code"
    )]
    fn a_payload_decodes_into_at_most_120_bytes_per_byte_and_1_mib_of_list_levels() {
        use std::mem::size_of;

        const PER_BYTE: usize = 120;
        // A level's 16-byte box here, and another in the engine's property
        // type, built while the record's chain still lives.
        const PER_LEVEL: usize = 32;
        fn smallest<T: Serialize>(element: &T) -> usize {
            let mut out = Vec::new();
            encode_bincode_payload(element, &mut out).unwrap();
            out.len()
        }
        let in_a_list = |size: usize| 3 * size;
        let elements = [
            (
                "an empty string",
                smallest(&String::new()),
                in_a_list(size_of::<String>()),
            ),
            (
                "an empty string pair",
                smallest(&(String::new(), String::new())),
                in_a_list(size_of::<(String, String)>()),
            ),
            (
                "two absent endpoints",
                smallest(&EndpointPair {
                    source: None,
                    target: None,
                }),
                in_a_list(size_of::<EndpointPair>()),
            ),
            (
                "an empty key constraint",
                smallest(&TypeConstraintRecord::PrimaryKey(Vec::new())),
                in_a_list(size_of::<TypeConstraintRecord>()),
            ),
            (
                "a property, its levels apart",
                smallest(&property("", PropertyTypeRecord::String, true, None)),
                in_a_list(size_of::<PropertyRecord>()),
            ),
            // A default value's list of nulls, one byte per null.
            ("a null in a list", 1, in_a_list(size_of::<Value>())),
        ];
        for (element, bytes, memory) in elements {
            assert!(
                memory <= PER_BYTE * bytes,
                "{element}: {memory} bytes of memory from {bytes} payload bytes"
            );
        }
        assert_eq!(
            PER_LEVEL,
            2 * size_of::<PropertyTypeRecord>(),
            "a box per level holds one property type"
        );
        let levels = PER_LEVEL * MAX_LIST_LEVELS_PER_RECORD;
        assert_eq!(levels, 1 << 20, "the levels of a record take at most 1 MiB");
        let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap();
        assert_eq!(
            PER_BYTE * maximum + levels,
            241 << 20,
            "the bound for the largest payload"
        );

        let properties = deepest_properties();
        let count = properties.len();
        let bytes = framed(&node_type(properties));
        let payload = bytes.len() - RECORD_HEADER_BYTES;
        assert_eq!(
            payload, 1_292,
            "name 6, count 3, 5 bytes per property, three empty lists"
        );
        let memory = count * in_a_list(size_of::<PropertyRecord>()) + levels;
        assert!(
            memory <= PER_BYTE * payload + levels && memory > PER_BYTE * payload,
            "{memory} bytes of memory from {payload} payload bytes"
        );
        let [CatalogRecord::NodeType(read)] = &read_all(&bytes).unwrap()[..] else {
            panic!("one node type");
        };
        assert_eq!(read.properties.len(), count);
        for property in &read.properties {
            assert_eq!(
                property.data_type.levels_and_code(),
                (MAX_LIST_TYPE_DEPTH, 0),
                "one box per level"
            );
        }
    }

    /// The property types of a record nest at most
    /// [`MAX_LIST_LEVELS_PER_RECORD`] `LIST` levels in all: node and edge
    /// types at the cap round trip, and the writer refuses one level more,
    /// appending nothing.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "records of 32,768 LIST levels take minutes under Miri; this module has no unsafe code"
    )]
    fn list_levels_past_the_record_cap_are_refused_by_the_writer() {
        for record in [
            node_type(deepest_properties()),
            edge_type(deepest_properties()),
        ] {
            assert!(
                read_all(&framed(&record)).unwrap() == [record],
                "a record at the cap round trips"
            );
        }
        let mut past = deepest_properties();
        past.push(property(
            "scores",
            list_of(1, PropertyTypeRecord::Int64),
            true,
            None,
        ));
        for (kind, record) in [(2, node_type(past.clone())), (3, edge_type(past))] {
            let mut out = vec![3, 19, 88];
            let error = record.encode_framed(&mut out).unwrap_err();
            assert!(matches!(error, Error::Serialization(_)), "{error:?}");
            let message = error.to_string();
            assert!(
                message.contains(&format!("kind {kind}"))
                    && message.contains("32769 LIST levels")
                    && message.contains("32768"),
                "{message}"
            );
            assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
        }
    }

    /// The reader refuses a record whose property types nest more than
    /// [`MAX_LIST_LEVELS_PER_RECORD`] levels in all, counting each type's
    /// levels before it reads the code they wrap and builds them: the type
    /// that passes the cap is refused, even with a code no release knows.
    /// The count starts again with each record.
    #[test]
    #[cfg_attr(
        miri,
        ignore = "records of 32,768 LIST levels take minutes under Miri; this module has no unsafe code"
    )]
    fn list_levels_past_the_record_cap_are_refused_by_the_reader_before_they_are_built() {
        let at_cap = [
            framed(&node_type(deepest_properties())),
            framed(&edge_type(deepest_properties())),
        ]
        .concat();
        assert_eq!(
            read_all(&at_cap).unwrap().len(),
            2,
            "each record has its own count"
        );

        // A read that should fail, without printing 32,768 levels if it
        // does not.
        let refused = |bytes: &[u8]| match read_all(bytes) {
            Ok(records) => panic!("{} records read past the cap", records.len()),
            Err(error) => error.to_string(),
        };
        let mut properties = deepest_properties();
        properties.push(property("x", PropertyTypeRecord::Int64, true, None));
        for record in [node_type(properties.clone()), edge_type(properties)] {
            let bytes = framed(&record);
            // The header, the name, the property count (257: 0xFB and a
            // u16), 256 properties of 5 bytes and the name "x".
            let levels_at =
                RECORD_HEADER_BYTES + (1 + "Event".len()) + 3 + 256 * 5 + (1 + "x".len());
            assert_eq!(bytes[levels_at..levels_at + 2], [0, 1], "no level, INT64");

            let mut past = bytes.clone();
            past[levels_at] = 1;
            let error = refused(&past);
            assert!(
                error.contains("32768 LIST levels") && error.contains("record 0 at byte 0"),
                "{error}"
            );

            past[levels_at + 1] = 99;
            let error = refused(&past);
            assert!(
                error.contains("32768 LIST levels") && !error.contains("code 99"),
                "the levels are refused before the code is read: {error}"
            );
        }
    }

    /// A record whose default values take more bytes than a payload holds
    /// is refused from their sizes, before one is encoded: a default value
    /// the value codec refuses, ahead of the oversized one, is never reached.
    #[test]
    fn oversized_default_values_are_refused_before_they_are_encoded() {
        use super::super::value_codec::MAX_VALUE_DEPTH;

        let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap();
        let too_deep =
            (0..=MAX_VALUE_DEPTH).fold(Value::Null, |inner, _| Value::List(vec![inner].into()));
        let record = node_type(vec![
            property("nested", PropertyTypeRecord::List, true, Some(too_deep)),
            property(
                "notes",
                PropertyTypeRecord::String,
                true,
                Some(Value::from("Amsterdam ".repeat(maximum / 10 + 1))),
            ),
        ]);
        let mut out = vec![3, 19, 88];
        let error = record.encode_framed(&mut out).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let message = error.to_string();
        assert!(
            message.contains("catalog record of kind 2") && message.contains(&maximum.to_string()),
            "{message}"
        );
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
    }

    /// The count of default value bytes takes every value the value codec
    /// encodes, also one nested deeper than a property value may be (129
    /// and 130 levels): such a default, too large for a payload, is refused
    /// from its size instead of being encoded whole first. Only a value the
    /// codec refuses counts nothing.
    #[test]
    fn default_values_the_codec_encodes_are_counted_at_every_depth() {
        use super::super::value_codec::MAX_VALUE_DEPTH;

        let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap();
        let nested = |levels: usize, inner: Value| {
            (0..levels).fold(inner, |inner, _| Value::List(vec![inner].into()))
        };
        for levels in [MAX_PROPERTY_VALUE_DEPTH + 1, MAX_VALUE_DEPTH] {
            let default = nested(levels, Value::from("Berlin ".repeat(maximum / 7 + 1)));
            let record = node_type(vec![property(
                "nested",
                PropertyTypeRecord::List,
                true,
                Some(default.clone()),
            )]);
            assert_eq!(
                record.default_value_bytes(),
                encoded_len(&default),
                "a default nested {levels} levels deep is counted"
            );
            let mut out = vec![3, 19, 88];
            let error = record.encode_framed(&mut out).unwrap_err();
            assert!(
                error.to_string().contains(&maximum.to_string()),
                "{levels} levels: {error}"
            );
            assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
        }
        let refused = node_type(vec![property(
            "nested",
            PropertyTypeRecord::List,
            true,
            Some(nested(MAX_VALUE_DEPTH + 1, Value::Null)),
        )]);
        assert_eq!(refused.default_value_bytes(), 0, "the codec refuses it");
    }

    /// A record whose payload would pass the maximum is refused while it is
    /// written: the writer stops at the maximum instead of copying the whole
    /// payload (a procedure body of 22 MiB) and refusing it afterwards.
    #[test]
    fn an_oversized_payload_is_refused_without_copying_it() {
        let maximum = usize::try_from(MAX_CATALOG_RECORD_PAYLOAD).unwrap();
        let record = CatalogRecord::Procedure(ProcedureRecord {
            name: "get_adults".into(),
            params: Vec::new(),
            returns: Vec::new(),
            body: "RETURN 88;\n".repeat(1 << 21),
        });
        let mut out = vec![3, 19, 88];
        let error = record.encode_framed(&mut out).unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        let message = error.to_string();
        assert!(
            message.contains("catalog record of kind 9") && message.contains(&maximum.to_string()),
            "{message}"
        );
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
        assert!(
            out.capacity() <= RECORD_HEADER_BYTES + maximum + 3,
            "the writer stopped at the maximum: the output grew to {} bytes",
            out.capacity()
        );
    }

    #[test]
    fn a_length_inside_a_payload_cannot_claim_more_than_the_decode_limit() {
        // A schema whose name claims 64 MiB, with six bytes present.
        let mut payload = vec![0xFC];
        payload.extend_from_slice(&(1u32 << 26).to_le_bytes());
        payload.extend_from_slice(b"travel");
        let mut bytes = vec![1, RECORD_REQUIRED];
        bytes.extend_from_slice(&u32::try_from(payload.len()).unwrap().to_le_bytes());
        bytes.extend_from_slice(&payload);
        let error = read_error(&bytes);
        assert!(
            error.contains("decode limit") && error.contains("record 0 at byte 0"),
            "{error}"
        );
    }

    #[test]
    fn list_types_nested_deeper_than_the_limit_are_refused() {
        let node = |data_type: PropertyTypeRecord| {
            CatalogRecord::NodeType(NodeTypeRecord {
                name: "Event".into(),
                properties: vec![property("scores", data_type, true, None)],
                constraints: Vec::new(),
                parent_types: Vec::new(),
                key_labels: Vec::new(),
            })
        };
        let deepest = node(list_of(MAX_LIST_TYPE_DEPTH, PropertyTypeRecord::Int64));
        let bytes = framed(&deepest);
        assert_eq!(read_all(&bytes).unwrap(), [deepest]);

        let mut out = vec![3, 19, 88];
        let error = node(list_of(MAX_LIST_TYPE_DEPTH + 1, PropertyTypeRecord::Int64))
            .encode_framed(&mut out)
            .unwrap_err();
        assert!(matches!(error, Error::Serialization(_)), "{error:?}");
        assert!(error.to_string().contains("deeper"), "{error}");
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");

        // The depth is the byte after the header, the name, the property
        // count and the property name.
        let depth_at = RECORD_HEADER_BYTES + (1 + "Event".len()) + 1 + (1 + "scores".len());
        assert_eq!(
            usize::from(bytes[depth_at]),
            MAX_LIST_TYPE_DEPTH,
            "the depth byte"
        );
        let mut deeper = bytes.clone();
        deeper[depth_at] += 1;
        let error = read_error(&deeper);
        assert!(
            error.contains("deeper") && error.contains("record 0 at byte 0"),
            "{error}"
        );

        // A depth of u32::MAX is refused before any level is built.
        let mut deepest_claim = bytes[..depth_at].to_vec();
        deepest_claim.extend_from_slice(&[0xFC, 0xFF, 0xFF, 0xFF, 0xFF]);
        deepest_claim.extend_from_slice(&bytes[depth_at + 1..]);
        let length = u32::try_from(deepest_claim.len() - RECORD_HEADER_BYTES).unwrap();
        deepest_claim[2..RECORD_HEADER_BYTES].copy_from_slice(&length.to_le_bytes());
        assert!(read_error(&deepest_claim).contains("deeper"));
    }

    #[test]
    fn an_unknown_property_type_code_is_refused() {
        let node = CatalogRecord::NodeType(NodeTypeRecord {
            name: "Event".into(),
            properties: vec![property("seen", PropertyTypeRecord::Any, true, None)],
            constraints: Vec::new(),
            parent_types: Vec::new(),
            key_labels: Vec::new(),
        });
        let mut bytes = framed(&node);
        let code_at = RECORD_HEADER_BYTES + (1 + "Event".len()) + 1 + (1 + "seen".len()) + 1;
        assert_eq!(bytes[code_at], 15, "the code of ANY");
        bytes[code_at] = 16;
        let error = read_error(&bytes);
        assert!(error.contains("property type code 16"), "{error}");
    }

    #[test]
    fn apply_errors_come_back_unchanged_and_stop_the_read() {
        let bytes = [
            framed(&schema("Amsterdam")),
            framed(&schema("Berlin")),
            framed(&schema("Paris")),
        ]
        .concat();
        let mut applied = Vec::new();
        let error = read_catalog_records(&mut &bytes[..], &mut |record| {
            applied.push(record);
            if applied.len() == 2 {
                return Err(Error::Internal("Gus refuses Berlin".into()));
            }
            Ok(())
        })
        .unwrap_err();
        assert!(
            matches!(&error, Error::Internal(message) if message == "Gus refuses Berlin"),
            "{error:?}"
        );
        assert_eq!(
            applied,
            [schema("Amsterdam"), schema("Berlin")],
            "Paris is not read"
        );
    }

    /// Serves `bytes`, then fails with `error`.
    struct Failing {
        bytes: Vec<u8>,
        position: usize,
        error: Option<io::Error>,
    }

    impl Read for Failing {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.position == self.bytes.len() {
                return match self.error.take() {
                    Some(error) => Err(error),
                    None => Ok(0),
                };
            }
            let count = buffer.len().min(self.bytes.len() - self.position);
            buffer[..count].copy_from_slice(&self.bytes[self.position..self.position + count]);
            self.position += count;
            Ok(count)
        }
    }

    #[test]
    fn reader_errors_come_back_unchanged() {
        // A grafeo error the reader carries, as a chunk stream reader does.
        let mut inside_a_payload = Failing {
            bytes: framed(&schema("Mia"))[..8].to_vec(),
            position: 0,
            error: Some(io::Error::other(Error::Serialization(
                "stream 0 of graph 0: a piece starts at offset 88".into(),
            ))),
        };
        let error = read_catalog_records(&mut inside_a_payload, &mut |_| Ok(())).unwrap_err();
        assert!(
            matches!(&error, Error::Serialization(message)
                if message == "stream 0 of graph 0: a piece starts at offset 88"),
            "{error:?}"
        );

        // A plain I/O error, met while reading a header.
        let mut at_a_header = Failing {
            bytes: framed(&schema("Mia")),
            position: 0,
            error: Some(io::Error::new(io::ErrorKind::PermissionDenied, "Jules")),
        };
        let mut applied = 0;
        let error = read_catalog_records(&mut at_a_header, &mut |_| {
            applied += 1;
            Ok(())
        })
        .unwrap_err();
        assert!(
            matches!(&error, Error::Io(inner) if inner.kind() == io::ErrorKind::PermissionDenied),
            "{error:?}"
        );
        assert_eq!(applied, 1, "the record before the failure is applied");
    }

    /// Hands out one byte per read, failing with `Interrupted` in between.
    struct Trickle<'a> {
        bytes: &'a [u8],
        interrupt: bool,
    }

    impl Read for Trickle<'_> {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.interrupt = !self.interrupt;
            if self.interrupt {
                return Err(io::ErrorKind::Interrupted.into());
            }
            match (self.bytes.split_first(), buffer.first_mut()) {
                (Some((&byte, rest)), Some(slot)) => {
                    *slot = byte;
                    self.bytes = rest;
                    Ok(1)
                }
                _ => Ok(0),
            }
        }
    }

    #[test]
    fn short_and_interrupted_reads_are_retried() {
        let mut bytes = Vec::new();
        for record in one_of_each() {
            record.encode_framed(&mut bytes).unwrap();
        }
        let mut trickle = Trickle {
            bytes: &bytes,
            interrupt: false,
        };
        let mut back = Vec::new();
        read_catalog_records(&mut trickle, &mut |record| {
            back.push(record);
            Ok(())
        })
        .unwrap();
        assert_eq!(back, one_of_each());
    }

    #[test]
    fn the_framing_refuses_to_write_what_it_would_refuse_to_read() {
        let mut out = vec![3, 19, 88];
        for (kind, flags) in [(0, RECORD_REQUIRED), (1, 0x80), (1, 0x02)] {
            let error = encode_framed_record(&CATALOG_FRAMING, kind, flags, &mut out, |out| {
                Ok(io::Write::write_all(out, &[88])?)
            })
            .unwrap_err();
            assert!(
                matches!(error, Error::Internal(_)),
                "kind {kind}, flags {flags}: {error:?}"
            );
        }
        assert_eq!(out, [3, 19, 88], "a refused record appends nothing");
        encode_framed_record(&CATALOG_FRAMING, 99, 0, &mut out, |out| {
            Ok(io::Write::write_all(out, &[88])?)
        })
        .unwrap();
        assert_eq!(out, [3, 19, 88, 99, 0, 1, 0, 0, 0, 88]);
    }
}
