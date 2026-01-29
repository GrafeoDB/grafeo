//! GQL Abstract Syntax Tree.

use grafeo_common::utils::error::SourceSpan;

/// A GQL statement.
#[derive(Debug, Clone)]
pub enum Statement {
    /// A query statement (MATCH, RETURN, etc.)
    Query(QueryStatement),
    /// A data modification statement (INSERT, DELETE, etc.)
    DataModification(DataModificationStatement),
    /// A schema statement (CREATE NODE TYPE, etc.)
    Schema(SchemaStatement),
}

/// A query statement.
#[derive(Debug, Clone)]
pub struct QueryStatement {
    /// MATCH clauses (regular and optional).
    pub match_clauses: Vec<MatchClause>,
    /// Optional WHERE clause.
    pub where_clause: Option<WhereClause>,
    /// SET clauses for property updates.
    pub set_clauses: Vec<SetClause>,
    /// WITH clauses for query chaining.
    pub with_clauses: Vec<WithClause>,
    /// UNWIND clauses for list expansion.
    pub unwind_clauses: Vec<UnwindClause>,
    /// MERGE clauses for conditional create/match.
    pub merge_clauses: Vec<MergeClause>,
    /// CREATE clauses (Cypher-style data modification within query).
    pub create_clauses: Vec<InsertStatement>,
    /// DELETE clauses (data removal within query).
    pub delete_clauses: Vec<DeleteStatement>,
    /// Required RETURN clause.
    pub return_clause: ReturnClause,
    /// Source span in the original query.
    pub span: Option<SourceSpan>,
}

/// A SET clause.
#[derive(Debug, Clone)]
pub struct SetClause {
    /// Property assignments.
    pub assignments: Vec<PropertyAssignment>,
    /// Label operations (add labels to nodes).
    pub label_operations: Vec<LabelOperation>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A label operation for adding/removing labels.
#[derive(Debug, Clone)]
pub struct LabelOperation {
    /// Variable name.
    pub variable: String,
    /// Labels to add.
    pub labels: Vec<String>,
}

/// A MATCH clause.
#[derive(Debug, Clone)]
pub struct MatchClause {
    /// Whether this is an OPTIONAL MATCH.
    pub optional: bool,
    /// Graph patterns to match, potentially with aliases and path functions.
    pub patterns: Vec<AliasedPattern>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A pattern with optional alias and path function wrapper.
#[derive(Debug, Clone)]
pub struct AliasedPattern {
    /// Optional alias for the pattern (e.g., `p` in `p = (a)-[*]-(b)`).
    pub alias: Option<String>,
    /// Optional path function wrapping the pattern.
    pub path_function: Option<PathFunction>,
    /// The underlying pattern.
    pub pattern: Pattern,
}

/// Path functions for shortest path queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathFunction {
    /// Find the shortest path between two nodes.
    ShortestPath,
    /// Find all shortest paths between two nodes.
    AllShortestPaths,
}

/// A WITH clause for query chaining.
#[derive(Debug, Clone)]
pub struct WithClause {
    /// Whether to use DISTINCT.
    pub distinct: bool,
    /// Items to pass to the next query part.
    pub items: Vec<ReturnItem>,
    /// Optional WHERE clause after WITH.
    pub where_clause: Option<WhereClause>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An UNWIND clause for expanding lists into rows.
#[derive(Debug, Clone)]
pub struct UnwindClause {
    /// The expression to unwind (typically a list).
    pub expression: Expression,
    /// The alias for each element.
    pub alias: String,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A MERGE clause for creating or matching patterns.
#[derive(Debug, Clone)]
pub struct MergeClause {
    /// The pattern to merge.
    pub pattern: Pattern,
    /// Actions to perform on create.
    pub on_create: Option<Vec<PropertyAssignment>>,
    /// Actions to perform on match.
    pub on_match: Option<Vec<PropertyAssignment>>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A graph pattern.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// A node pattern.
    Node(NodePattern),
    /// An edge pattern connecting nodes.
    Path(PathPattern),
}

/// A node pattern like (n:Person).
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Labels to match.
    pub labels: Vec<String>,
    /// Property filters.
    pub properties: Vec<(String, Expression)>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A path pattern like `(a)-[:KNOWS]->(b)`.
#[derive(Debug, Clone)]
pub struct PathPattern {
    /// Source node pattern.
    pub source: NodePattern,
    /// Edge patterns.
    pub edges: Vec<EdgePattern>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An edge pattern like `-[:KNOWS]->` or `-[:KNOWS*1..3]->`.
#[derive(Debug, Clone)]
pub struct EdgePattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Edge types to match.
    pub types: Vec<String>,
    /// Direction of the edge.
    pub direction: EdgeDirection,
    /// Target node pattern.
    pub target: NodePattern,
    /// Variable-length path: minimum hops (None means 1).
    pub min_hops: Option<u32>,
    /// Variable-length path: maximum hops (None means unlimited or same as min).
    pub max_hops: Option<u32>,
    /// Property filters for the edge.
    pub properties: Vec<(String, Expression)>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// Direction of an edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    /// Outgoing edge: ->
    Outgoing,
    /// Incoming edge: <-
    Incoming,
    /// Undirected edge: -
    Undirected,
}

/// A WHERE clause.
#[derive(Debug, Clone)]
pub struct WhereClause {
    /// The filter expression.
    pub expression: Expression,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A RETURN clause.
#[derive(Debug, Clone)]
pub struct ReturnClause {
    /// Whether to return DISTINCT results.
    pub distinct: bool,
    /// Items to return.
    pub items: Vec<ReturnItem>,
    /// Optional ORDER BY.
    pub order_by: Option<OrderByClause>,
    /// Optional SKIP.
    pub skip: Option<Expression>,
    /// Optional LIMIT.
    pub limit: Option<Expression>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An item in a RETURN clause.
#[derive(Debug, Clone)]
pub struct ReturnItem {
    /// The expression to return.
    pub expression: Expression,
    /// Optional alias (AS name).
    pub alias: Option<String>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An ORDER BY clause.
#[derive(Debug, Clone)]
pub struct OrderByClause {
    /// Sort items.
    pub items: Vec<OrderByItem>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A sort item.
#[derive(Debug, Clone)]
pub struct OrderByItem {
    /// The expression to sort by.
    pub expression: Expression,
    /// Sort order.
    pub order: SortOrder,
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order.
    Asc,
    /// Descending order.
    Desc,
}

/// A data modification statement.
#[derive(Debug, Clone)]
pub enum DataModificationStatement {
    /// INSERT statement.
    Insert(InsertStatement),
    /// DELETE statement.
    Delete(DeleteStatement),
    /// SET statement.
    Set(SetStatement),
}

/// An INSERT statement.
#[derive(Debug, Clone)]
pub struct InsertStatement {
    /// Patterns to insert.
    pub patterns: Vec<Pattern>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A DELETE statement.
#[derive(Debug, Clone)]
pub struct DeleteStatement {
    /// Variables to delete.
    pub variables: Vec<String>,
    /// Whether to use DETACH DELETE.
    pub detach: bool,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A SET statement.
#[derive(Debug, Clone)]
pub struct SetStatement {
    /// Property assignments.
    pub assignments: Vec<PropertyAssignment>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A property assignment.
#[derive(Debug, Clone)]
pub struct PropertyAssignment {
    /// Variable name.
    pub variable: String,
    /// Property key.
    pub property: String,
    /// Value expression.
    pub value: Expression,
}

/// A schema statement.
#[derive(Debug, Clone)]
pub enum SchemaStatement {
    /// CREATE NODE TYPE.
    CreateNodeType(CreateNodeTypeStatement),
    /// CREATE EDGE TYPE.
    CreateEdgeType(CreateEdgeTypeStatement),
}

/// A CREATE NODE TYPE statement.
#[derive(Debug, Clone)]
pub struct CreateNodeTypeStatement {
    /// Type name.
    pub name: String,
    /// Property definitions.
    pub properties: Vec<PropertyDefinition>,
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
    /// Source span.
    pub span: Option<SourceSpan>,
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
}

/// An expression.
#[derive(Debug, Clone)]
pub enum Expression {
    /// A literal value.
    Literal(Literal),
    /// A variable reference.
    Variable(String),
    /// A parameter reference ($name).
    Parameter(String),
    /// A property access (var.prop).
    PropertyAccess {
        /// The variable.
        variable: String,
        /// The property name.
        property: String,
    },
    /// A binary operation.
    Binary {
        /// Left operand.
        left: Box<Expression>,
        /// Operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<Expression>,
    },
    /// A unary operation.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<Expression>,
    },
    /// A function call.
    FunctionCall {
        /// Function name.
        name: String,
        /// Arguments.
        args: Vec<Expression>,
        /// Whether DISTINCT is applied to arguments.
        distinct: bool,
    },
    /// A list expression.
    List(Vec<Expression>),
    /// A CASE expression.
    Case {
        /// Optional input expression.
        input: Option<Box<Expression>>,
        /// When clauses.
        whens: Vec<(Expression, Expression)>,
        /// Else clause.
        else_clause: Option<Box<Expression>>,
    },
    /// EXISTS subquery expression - checks if inner query returns results.
    ExistsSubquery {
        /// The inner query pattern to check for existence.
        query: Box<QueryStatement>,
    },
}

/// A literal value.
#[derive(Debug, Clone)]
pub enum Literal {
    /// Null literal.
    Null,
    /// Boolean literal.
    Bool(bool),
    /// Integer literal.
    Integer(i64),
    /// Float literal.
    Float(f64),
    /// String literal.
    String(String),
}

/// A binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Comparison
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,

    // Logical
    /// Logical AND.
    And,
    /// Logical OR.
    Or,

    // Arithmetic
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Modulo.
    Mod,

    // String
    /// String concatenation.
    Concat,
    /// LIKE pattern matching.
    Like,
    /// IN list membership.
    In,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Logical NOT.
    Not,
    /// Unary minus.
    Neg,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
}
