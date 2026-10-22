//! SQL/PGQ Abstract Syntax Tree.
//!
//! Represents SQL:2023 GRAPH_TABLE (ISO/IEC 9075-16) queries.
//! The inner MATCH clause reuses GQL AST types since SQL/PGQ
//! uses GQL pattern syntax inside `GRAPH_TABLE(...)`.

use grafeo_common::utils::error::SourceSpan;

// Re-use GQL pattern and expression types for the inner MATCH clause
pub use crate::query::gql::ast::{
    AliasedPattern, BinaryOp, CallStatement, EdgeDirection, EdgePattern, Expression, Literal,
    MatchClause, NodePattern, OrderByClause, OrderByItem, PathPattern, Pattern, ReturnClause,
    ReturnItem, SortOrder as GqlSortOrder, UnaryOp, WhereClause, YieldItem,
};

/// A SQL/PGQ statement.
#[derive(Debug, Clone)]
pub enum Statement {
    /// A `SELECT ... FROM GRAPH_TABLE(...)` query.
    Select(SelectStatement),
    /// A set operation (UNION, INTERSECT, EXCEPT) between two SELECT queries.
    SetOperation(SetOperationStatement),
    /// A `CREATE PROPERTY GRAPH` DDL statement.
    CreatePropertyGraph(CreatePropertyGraphStatement),
    /// A `CALL procedure(args)` statement (SQL:2003+).
    Call(CallStatement),
}

/// The kind of set operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetOperationKind {
    /// UNION: combines results from both queries, deduplicating by default.
    Union,
    /// INTERSECT: returns only rows present in both queries.
    Intersect,
    /// EXCEPT: returns rows from the left query that are not in the right.
    Except,
}

/// A set operation (UNION, INTERSECT, EXCEPT) between two SELECT queries.
#[derive(Debug, Clone)]
pub struct SetOperationStatement {
    /// The left SELECT query.
    pub left: SelectStatement,
    /// The kind of set operation.
    pub operation: SetOperationKind,
    /// Whether ALL was specified (preserve duplicates).
    pub all: bool,
    /// The right SELECT query.
    pub right: SelectStatement,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A complete SQL/PGQ SELECT statement.
#[derive(Debug, Clone)]
pub struct SelectStatement {
    /// Whether SELECT DISTINCT was specified.
    pub distinct: bool,
    /// SELECT list (`None` means `SELECT *`).
    pub select_list: SelectList,
    /// The `GRAPH_TABLE(...)` expression in the FROM clause.
    pub graph_table: GraphTableExpression,
    /// Optional table alias for the GRAPH_TABLE result.
    pub table_alias: Option<String>,
    /// Optional SQL-level WHERE clause (references output columns).
    pub where_clause: Option<Expression>,
    /// Optional GROUP BY expressions.
    pub group_by: Option<Vec<Expression>>,
    /// Optional HAVING clause (filter on aggregated results).
    pub having: Option<Expression>,
    /// Optional ORDER BY clause.
    pub order_by: Option<Vec<SortItem>>,
    /// Optional LIMIT.
    pub limit: Option<u64>,
    /// Optional OFFSET.
    pub offset: Option<u64>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// SELECT list items.
#[derive(Debug, Clone)]
pub enum SelectList {
    /// `SELECT *` - all columns from the GRAPH_TABLE.
    All,
    /// Explicit column list.
    Columns(Vec<SelectItem>),
}

/// A single item in the SELECT list.
#[derive(Debug, Clone)]
pub struct SelectItem {
    /// Column expression (may be qualified: `alias.column`).
    pub expression: Expression,
    /// Optional AS alias.
    pub alias: Option<String>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// The `GRAPH_TABLE(...)` expression.
#[derive(Debug, Clone)]
pub struct GraphTableExpression {
    /// Optional graph name reference (e.g., `GRAPH_TABLE(my_graph, MATCH ...)`).
    pub graph_name: Option<String>,
    /// The inner MATCH clause (GQL pattern syntax).
    pub match_clause: MatchClause,
    /// Optional match clauses (LEFT OUTER JOIN or OPTIONAL MATCH).
    pub optional_matches: Vec<MatchClause>,
    /// Optional WHERE clause inside GRAPH_TABLE (between MATCH and COLUMNS).
    pub where_clause: Option<Expression>,
    /// The COLUMNS clause (projection from graph to table).
    pub columns: ColumnsClause,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// The `COLUMNS(...)` clause inside GRAPH_TABLE.
#[derive(Debug, Clone)]
pub struct ColumnsClause {
    /// Column definitions.
    pub items: Vec<ColumnItem>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A single column in the COLUMNS clause.
#[derive(Debug, Clone)]
pub struct ColumnItem {
    /// The expression (GQL expression syntax, e.g. `a.name`).
    pub expression: Expression,
    /// Column alias (AS name). Required by the SQL:2023 spec.
    pub alias: String,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A sort item in ORDER BY.
#[derive(Debug, Clone)]
pub struct SortItem {
    /// Expression to sort by.
    pub expression: Expression,
    /// Sort direction.
    pub direction: SortDirection,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending order (default).
    Asc,
    /// Descending order.
    Desc,
}

// ==================== Property Graph DDL ====================

/// A `CREATE PROPERTY GRAPH` statement (SQL:2023).
#[derive(Debug, Clone)]
pub struct CreatePropertyGraphStatement {
    /// Graph name.
    pub name: String,
    /// Node table definitions.
    pub node_tables: Vec<NodeTableDefinition>,
    /// Edge table definitions.
    pub edge_tables: Vec<EdgeTableDefinition>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A node table definition in `NODE TABLES (...)`.
#[derive(Debug, Clone)]
pub struct NodeTableDefinition {
    /// Table name (maps to a node label in Grafeo).
    pub name: String,
    /// Column definitions.
    pub columns: Vec<ColumnDefinition>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An edge table definition in `EDGE TABLES (...)`.
#[derive(Debug, Clone)]
pub struct EdgeTableDefinition {
    /// Table name (maps to an edge type in Grafeo).
    pub name: String,
    /// Column definitions.
    pub columns: Vec<ColumnDefinition>,
    /// Source key column name (REFERENCES source_table(column)).
    pub source_table: String,
    /// Source column name in the referenced table.
    pub source_column: String,
    /// Target key column name (REFERENCES target_table(column)).
    pub target_table: String,
    /// Target column name in the referenced table.
    pub target_column: String,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A column definition in a table.
#[derive(Debug, Clone)]
pub struct ColumnDefinition {
    /// Column name.
    pub name: String,
    /// SQL data type.
    pub data_type: SqlDataType,
    /// Whether this column is a primary key.
    pub primary_key: bool,
    /// Foreign key reference (if any).
    pub references: Option<ForeignKeyRef>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A foreign key reference.
#[derive(Debug, Clone)]
pub struct ForeignKeyRef {
    /// Referenced table name.
    pub table: String,
    /// Referenced column name.
    pub column: String,
}

/// SQL data types for column definitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SqlDataType {
    /// BIGINT (64-bit integer).
    Bigint,
    /// INT or INTEGER (32-bit integer).
    Int,
    /// VARCHAR with optional length.
    Varchar(Option<usize>),
    /// BOOLEAN.
    Boolean,
    /// FLOAT (single precision).
    Float,
    /// DOUBLE (double precision).
    Double,
    /// TIMESTAMP.
    Timestamp,
}

impl std::fmt::Display for SqlDataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bigint => write!(f, "BIGINT"),
            Self::Int => write!(f, "INT"),
            Self::Varchar(Some(len)) => write!(f, "VARCHAR({len})"),
            Self::Varchar(None) => write!(f, "VARCHAR"),
            Self::Boolean => write!(f, "BOOLEAN"),
            Self::Float => write!(f, "FLOAT"),
            Self::Double => write!(f, "DOUBLE"),
            Self::Timestamp => write!(f, "TIMESTAMP"),
        }
    }
}
