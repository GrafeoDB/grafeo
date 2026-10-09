//! Shared schema DDL types used by multiple query language parsers.
//!
//! These types represent database schema concepts (indexes, constraints, graph
//! types, node/edge types) that are not specific to any single query language.
//! They live here so parsers like Cypher can use them without depending on the
//! GQL feature flag.

use grafeo_common::utils::error::SourceSpan;

/// A schema statement.
#[derive(Debug, Clone)]
pub enum SchemaStatement {
    /// CREATE NODE TYPE.
    CreateNodeType(CreateNodeTypeStatement),
    /// CREATE EDGE TYPE.
    CreateEdgeType(CreateEdgeTypeStatement),
    /// CREATE VECTOR INDEX.
    CreateVectorIndex(CreateVectorIndexStatement),
    /// DROP NODE TYPE.
    DropNodeType {
        /// Type name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// DROP EDGE TYPE.
    DropEdgeType {
        /// Type name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// CREATE INDEX (property, text, btree).
    CreateIndex(CreateIndexStatement),
    /// DROP INDEX.
    DropIndex {
        /// Index name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// CREATE CONSTRAINT.
    CreateConstraint(CreateConstraintStatement),
    /// DROP CONSTRAINT.
    DropConstraint {
        /// Constraint name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// CREATE GRAPH TYPE.
    CreateGraphType(CreateGraphTypeStatement),
    /// DROP GRAPH TYPE.
    DropGraphType {
        /// Type name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// CREATE SCHEMA.
    CreateSchema {
        /// Schema name.
        name: String,
        /// IF NOT EXISTS flag.
        if_not_exists: bool,
    },
    /// DROP SCHEMA.
    DropSchema {
        /// Schema name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// ALTER NODE TYPE.
    AlterNodeType(AlterTypeStatement),
    /// ALTER EDGE TYPE.
    AlterEdgeType(AlterTypeStatement),
    /// ALTER GRAPH TYPE.
    AlterGraphType(AlterGraphTypeStatement),
    /// CREATE PROCEDURE.
    CreateProcedure(CreateProcedureStatement),
    /// DROP PROCEDURE.
    DropProcedure {
        /// Procedure name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// SHOW CONSTRAINTS: lists all constraints.
    ShowConstraints,
    /// SHOW INDEXES: lists all indexes.
    ShowIndexes,
    /// SHOW NODE TYPES: lists all registered node types.
    ShowNodeTypes,
    /// SHOW EDGE TYPES: lists all registered edge types.
    ShowEdgeTypes,
    /// SHOW GRAPH TYPES: lists all registered graph types.
    ShowGraphTypes,
    /// SHOW GRAPH TYPE `name`: shows details of a specific graph type.
    ShowGraphType(String),
    /// SHOW CURRENT GRAPH TYPE: shows the graph type bound to the current graph.
    ShowCurrentGraphType,
    /// SHOW GRAPHS: lists all named graphs in the database (or in the current schema).
    ShowGraphs,
    /// SHOW SCHEMAS: lists all schema namespaces.
    ShowSchemas,
}

/// A CREATE NODE TYPE statement.
#[derive(Debug, Clone)]
pub struct CreateNodeTypeStatement {
    /// Type name.
    pub name: String,
    /// Property definitions.
    pub properties: Vec<PropertyDefinition>,
    /// Parent types for inheritance (GQL `EXTENDS`).
    pub parent_types: Vec<String>,
    /// IF NOT EXISTS flag.
    pub if_not_exists: bool,
    /// OR REPLACE flag.
    pub or_replace: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A CREATE EDGE TYPE statement.
#[derive(Debug, Clone)]
pub struct CreateEdgeTypeStatement {
    /// Type name.
    pub name: String,
    /// Property definitions.
    pub properties: Vec<PropertyDefinition>,
    /// Allowed source node types (GQL `CONNECTING`).
    pub source_node_types: Vec<String>,
    /// Allowed target node types.
    pub target_node_types: Vec<String>,
    /// IF NOT EXISTS flag.
    pub if_not_exists: bool,
    /// OR REPLACE flag.
    pub or_replace: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An inline element type definition within a graph type body.
#[derive(Debug, Clone)]
pub enum InlineElementType {
    /// Inline node type entry in a `CREATE GRAPH TYPE` body.
    ///
    /// When `is_reference` is true, this entry is a pure reference to an
    /// existing node type in the catalog (ISO/IEC 39075 bare element-type
    /// reference). The executor must not register or overwrite the type;
    /// it only validates existence and adds the name to the graph type's
    /// allowed node types.
    ///
    /// When `is_reference` is false, this is an inline declaration and the
    /// executor registers (or replaces) the catalog entry.
    Node {
        /// Type name.
        name: String,
        /// Property definitions. Empty when `is_reference` is true.
        properties: Vec<PropertyDefinition>,
        /// Key label sets (GG21). Empty when `is_reference` is true.
        key_labels: Vec<String>,
        /// True if this is a bare reference to an existing type.
        is_reference: bool,
    },
    /// Inline edge type entry in a `CREATE GRAPH TYPE` body.
    ///
    /// See `Node` for the reference vs declaration semantics.
    Edge {
        /// Type name.
        name: String,
        /// Property definitions. Empty when `is_reference` is true.
        properties: Vec<PropertyDefinition>,
        /// Key label sets (GG21). Empty when `is_reference` is true.
        key_labels: Vec<String>,
        /// Allowed source node types. Empty when `is_reference` is true.
        source_node_types: Vec<String>,
        /// Allowed target node types. Empty when `is_reference` is true.
        target_node_types: Vec<String>,
        /// True if this is a bare reference to an existing type.
        is_reference: bool,
    },
}

impl InlineElementType {
    /// Returns the type name for this element.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Node { name, .. } | Self::Edge { name, .. } => name,
        }
    }
}

/// A CREATE GRAPH TYPE statement.
#[derive(Debug, Clone)]
pub struct CreateGraphTypeStatement {
    /// Graph type name.
    pub name: String,
    /// Allowed node types (empty = open).
    pub node_types: Vec<String>,
    /// Allowed edge types (empty = open).
    pub edge_types: Vec<String>,
    /// Inline element type definitions (GG03).
    pub inline_types: Vec<InlineElementType>,
    /// Copy type from existing graph (GG04): `LIKE <graph_name>`.
    pub like_graph: Option<String>,
    /// Whether unlisted types are also allowed.
    pub open: bool,
    /// IF NOT EXISTS flag.
    pub if_not_exists: bool,
    /// OR REPLACE flag.
    pub or_replace: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A CREATE VECTOR INDEX statement.
///
/// Creates an index for vector similarity search on a node property.
///
/// # Syntax
///
/// ```text
/// CREATE VECTOR INDEX index_name ON :Label(property)
///   [DIMENSION dim]
///   [METRIC metric_name]
/// ```
///
/// # Example
///
/// ```text
/// CREATE VECTOR INDEX movie_embeddings ON :Movie(embedding)
///   DIMENSION 384
///   METRIC 'cosine'
/// ```
#[derive(Debug, Clone)]
pub struct CreateVectorIndexStatement {
    /// Index name.
    pub name: String,
    /// Node label to index.
    pub node_label: String,
    /// Property containing the vector.
    pub property: String,
    /// Vector dimensions (optional, can be inferred).
    pub dimensions: Option<usize>,
    /// Distance metric (default: cosine).
    pub metric: Option<String>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A CREATE INDEX statement.
///
/// # Syntax
///
/// ```text
/// CREATE INDEX name FOR (n:Label) ON (n.property) [USING TEXT|VECTOR|BTREE]
/// ```
#[derive(Debug, Clone)]
pub struct CreateIndexStatement {
    /// Index name.
    pub name: String,
    /// Index kind (property, text, vector, btree).
    pub index_kind: IndexKind,
    /// Node label to index.
    pub label: String,
    /// Properties to index.
    pub properties: Vec<String>,
    /// Additional options (dimensions, metric for vector indexes).
    pub options: IndexOptions,
    /// IF NOT EXISTS flag.
    pub if_not_exists: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// Kind of index.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// Default property index (hash-based).
    Property,
    /// Full-text search index (BM25).
    Text,
    /// Vector similarity index (HNSW).
    Vector,
    /// B-tree range index.
    BTree,
}

/// Additional options for index creation.
#[derive(Debug, Clone, Default)]
pub struct IndexOptions {
    /// Vector dimensions (for vector indexes).
    pub dimensions: Option<usize>,
    /// Distance metric (for vector indexes).
    pub metric: Option<String>,
}

/// A CREATE CONSTRAINT statement.
///
/// # Syntax
///
/// ```text
/// CREATE CONSTRAINT [name] FOR (n:Label) ON (n.property) UNIQUE|NOT NULL
/// ```
#[derive(Debug, Clone)]
pub struct CreateConstraintStatement {
    /// Constraint name (optional).
    pub name: Option<String>,
    /// Constraint kind.
    pub constraint_kind: ConstraintKind,
    /// Node label this constraint applies to.
    pub label: String,
    /// Properties constrained.
    pub properties: Vec<String>,
    /// IF NOT EXISTS flag.
    pub if_not_exists: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// Kind of constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    /// Unique value constraint.
    Unique,
    /// Composite key constraint (unique combination).
    NodeKey,
    /// Property must not be null.
    NotNull,
    /// Property must exist.
    Exists,
}

/// A property definition in a schema.
#[derive(Debug, Clone)]
pub struct PropertyDefinition {
    /// Property name.
    pub name: String,
    /// Property type.
    pub data_type: String,
    /// Whether the property is nullable.
    pub nullable: bool,
    /// Optional default value: a literal of the property type, spelled
    /// `NULL`, `TRUE`, `FALSE`, a decimal integer, a float with a `.` or an
    /// exponent, or a string between single quotes (its escapes resolved).
    pub default_value: Option<String>,
}

/// What a property type takes as a `DEFAULT`: the kind of literal its
/// default may be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PropertyTypeKind {
    /// `STRING`: a string literal.
    String,
    /// `INT64`: an integer literal.
    Integer,
    /// `FLOAT64`: a float literal, or an integer literal a float holds
    /// exactly.
    Float,
    /// `BOOLEAN`: `TRUE` or `FALSE`.
    Boolean,
    /// `ANY`: any literal.
    Any,
    /// The temporal types, lists, maps, bytes, nodes and edges: no literal a
    /// `DEFAULT` takes is one of their values.
    Other,
}

/// Every name type DDL takes for a property type other than `LIST<...>`, in
/// capitals, with its kind: the property value types Grafeo supports
/// (ISO/IEC 39075:2024 18.7 `<property value type>`, Syntax Rule 4). The
/// first name of each type is the one `SHOW NODE TYPES` lists.
///
/// The catalog of grafeo-engine reads the same names
/// (`PropertyDataType::from_type_name`); a test there keeps the two lists
/// equal.
pub const PROPERTY_TYPE_NAMES: &[(&str, PropertyTypeKind)] = &[
    ("STRING", PropertyTypeKind::String),
    ("VARCHAR", PropertyTypeKind::String),
    ("TEXT", PropertyTypeKind::String),
    ("INT64", PropertyTypeKind::Integer),
    ("INT", PropertyTypeKind::Integer),
    ("INTEGER", PropertyTypeKind::Integer),
    ("BIGINT", PropertyTypeKind::Integer),
    ("FLOAT64", PropertyTypeKind::Float),
    ("FLOAT", PropertyTypeKind::Float),
    ("DOUBLE", PropertyTypeKind::Float),
    ("REAL", PropertyTypeKind::Float),
    ("BOOLEAN", PropertyTypeKind::Boolean),
    ("BOOL", PropertyTypeKind::Boolean),
    ("DATE", PropertyTypeKind::Other),
    ("TIME", PropertyTypeKind::Other),
    ("TIMESTAMP", PropertyTypeKind::Other),
    ("DATETIME", PropertyTypeKind::Other),
    ("ZONED DATETIME", PropertyTypeKind::Other),
    ("ZONED_DATETIME", PropertyTypeKind::Other),
    ("ZONEDDATETIME", PropertyTypeKind::Other),
    ("LOCAL DATETIME", PropertyTypeKind::Other),
    ("LOCAL_DATETIME", PropertyTypeKind::Other),
    ("LOCALDATETIME", PropertyTypeKind::Other),
    ("DURATION", PropertyTypeKind::Other),
    ("INTERVAL", PropertyTypeKind::Other),
    ("LIST", PropertyTypeKind::Other),
    ("ARRAY", PropertyTypeKind::Other),
    ("MAP", PropertyTypeKind::Other),
    ("RECORD", PropertyTypeKind::Other),
    ("BYTES", PropertyTypeKind::Other),
    ("BINARY", PropertyTypeKind::Other),
    ("BLOB", PropertyTypeKind::Other),
    ("NODE", PropertyTypeKind::Other),
    ("EDGE", PropertyTypeKind::Other),
    ("RELATIONSHIP", PropertyTypeKind::Other),
    ("ANY", PropertyTypeKind::Any),
];

/// The kind of the property type called `name` (any case; `LIST<...>`
/// excluded: its kind is [`PropertyTypeKind::Other`]), or `None` when
/// `name` is not a property type Grafeo supports.
#[must_use]
pub fn property_type_kind(name: &str) -> Option<PropertyTypeKind> {
    PROPERTY_TYPE_NAMES
        .iter()
        .find(|(known, _)| known.eq_ignore_ascii_case(name))
        .map(|(_, kind)| *kind)
}

/// The message that refuses `name` as a property type, with the types to
/// use instead.
#[must_use]
pub fn unknown_property_type(name: &str) -> String {
    format!(
        "'{name}' is not a property type: use STRING, INT64, FLOAT64, BOOLEAN, DATE, TIME, \
         TIMESTAMP, ZONED DATETIME, LOCAL DATETIME, DURATION, LIST, LIST<type>, MAP, BYTES, \
         NODE, EDGE or ANY"
    )
}

/// An ALTER NODE TYPE or ALTER EDGE TYPE statement.
#[derive(Debug, Clone)]
pub struct AlterTypeStatement {
    /// Type name to alter.
    pub name: String,
    /// Changes to apply.
    pub alterations: Vec<TypeAlteration>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A single alteration to a node or edge type.
#[derive(Debug, Clone)]
pub enum TypeAlteration {
    /// Add a property to the type.
    AddProperty(PropertyDefinition),
    /// Remove a property from the type.
    DropProperty(String),
}

/// An ALTER GRAPH TYPE statement.
#[derive(Debug, Clone)]
pub struct AlterGraphTypeStatement {
    /// Graph type name to alter.
    pub name: String,
    /// Changes to apply.
    pub alterations: Vec<GraphTypeAlteration>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A single alteration to a graph type.
#[derive(Debug, Clone)]
pub enum GraphTypeAlteration {
    /// Add a node type to the graph type.
    AddNodeType(String),
    /// Remove a node type from the graph type.
    DropNodeType(String),
    /// Add an edge type to the graph type.
    AddEdgeType(String),
    /// Remove an edge type from the graph type.
    DropEdgeType(String),
}

/// A CREATE PROCEDURE statement.
///
/// # Syntax
///
/// ```text
/// CREATE [OR REPLACE] PROCEDURE name(param1 type, ...)
///   RETURNS (col1 type, ...)
///   AS { <GQL query body> }
/// ```
#[derive(Debug, Clone)]
pub struct CreateProcedureStatement {
    /// Procedure name.
    pub name: String,
    /// Parameter definitions.
    pub params: Vec<ProcedureParam>,
    /// Return column definitions.
    pub returns: Vec<ProcedureReturn>,
    /// Raw GQL query body.
    pub body: String,
    /// IF NOT EXISTS flag.
    pub if_not_exists: bool,
    /// OR REPLACE flag.
    pub or_replace: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A stored procedure parameter.
#[derive(Debug, Clone)]
pub struct ProcedureParam {
    /// Parameter name.
    pub name: String,
    /// Type name (e.g. "INT64", "STRING").
    pub param_type: String,
}

/// A stored procedure return column.
#[derive(Debug, Clone)]
pub struct ProcedureReturn {
    /// Column name.
    pub name: String,
    /// Type name.
    pub return_type: String,
}
