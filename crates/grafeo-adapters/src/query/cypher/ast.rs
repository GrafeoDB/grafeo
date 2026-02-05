//! Cypher Abstract Syntax Tree.
//!
//! This AST represents the openCypher 9.0 query language.

use grafeo_common::utils::error::SourceSpan;

/// A Cypher statement.
#[derive(Debug, Clone)]
pub enum Statement {
    /// A query (reading) statement.
    Query(Query),
    /// A CREATE statement.
    Create(CreateClause),
    /// A MERGE statement.
    Merge(MergeClause),
    /// A DELETE statement.
    Delete(DeleteClause),
    /// A SET statement.
    Set(SetClause),
    /// A REMOVE statement.
    Remove(RemoveClause),
}

/// A complete Cypher query.
#[derive(Debug, Clone)]
pub struct Query {
    /// The query clauses in order.
    pub clauses: Vec<Clause>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A query clause.
#[derive(Debug, Clone)]
pub enum Clause {
    /// MATCH clause.
    Match(MatchClause),
    /// OPTIONAL MATCH clause.
    OptionalMatch(MatchClause),
    /// WHERE clause.
    Where(WhereClause),
    /// WITH clause.
    With(WithClause),
    /// RETURN clause.
    Return(ReturnClause),
    /// UNWIND clause.
    Unwind(UnwindClause),
    /// ORDER BY clause.
    OrderBy(OrderByClause),
    /// SKIP clause.
    Skip(Expression),
    /// LIMIT clause.
    Limit(Expression),
    /// CREATE clause (within a query).
    Create(CreateClause),
    /// MERGE clause (within a query).
    Merge(MergeClause),
    /// DELETE clause.
    Delete(DeleteClause),
    /// SET clause.
    Set(SetClause),
    /// REMOVE clause.
    Remove(RemoveClause),
}

/// A MATCH clause.
#[derive(Debug, Clone)]
pub struct MatchClause {
    /// Graph patterns to match.
    pub patterns: Vec<Pattern>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A graph pattern.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// A node pattern.
    Node(NodePattern),
    /// A path pattern.
    Path(PathPattern),
    /// A named path pattern (p = ...).
    NamedPath {
        /// Path variable name.
        name: String,
        /// Optional path function (shortestPath, allShortestPaths).
        path_function: Option<PathFunction>,
        /// The path pattern.
        pattern: Box<Pattern>,
    },
}

/// Path function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathFunction {
    /// shortestPath - finds a single shortest path.
    ShortestPath,
    /// allShortestPaths - finds all shortest paths.
    AllShortestPaths,
}

/// A node pattern like (n:Person {name: 'Alice'}).
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Labels to match.
    pub labels: Vec<String>,
    /// Property map (literal properties for matching/creating).
    pub properties: Vec<(String, Expression)>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A path pattern like `(a)-[r:KNOWS]->(b)`.
#[derive(Debug, Clone)]
pub struct PathPattern {
    /// Starting node pattern.
    pub start: NodePattern,
    /// Relationship chain.
    pub chain: Vec<RelationshipPattern>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A relationship pattern like -[r:KNOWS*1..3]->
#[derive(Debug, Clone)]
pub struct RelationshipPattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Relationship types to match.
    pub types: Vec<String>,
    /// Direction of the relationship.
    pub direction: Direction,
    /// Variable length pattern (min, max).
    pub length: Option<LengthRange>,
    /// Property map.
    pub properties: Vec<(String, Expression)>,
    /// Target node pattern.
    pub target: NodePattern,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// Direction of a relationship.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Outgoing: ->
    Outgoing,
    /// Incoming: <-
    Incoming,
    /// Undirected: -
    Undirected,
}

/// Variable length range for relationships.
#[derive(Debug, Clone, Copy)]
pub struct LengthRange {
    /// Minimum length (default 1).
    pub min: Option<u32>,
    /// Maximum length (None = unbounded).
    pub max: Option<u32>,
}

/// A WHERE clause.
#[derive(Debug, Clone)]
pub struct WhereClause {
    /// The predicate expression.
    pub predicate: Expression,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A WITH clause for query chaining.
#[derive(Debug, Clone)]
pub struct WithClause {
    /// Whether DISTINCT is specified.
    pub distinct: bool,
    /// Projection items.
    pub items: Vec<ProjectionItem>,
    /// Optional WHERE filter.
    pub where_clause: Option<Box<WhereClause>>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A RETURN clause.
#[derive(Debug, Clone)]
pub struct ReturnClause {
    /// Whether DISTINCT is specified.
    pub distinct: bool,
    /// Projection items (* or explicit list).
    pub items: ReturnItems,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// Items in a RETURN clause.
#[derive(Debug, Clone)]
pub enum ReturnItems {
    /// RETURN *
    All,
    /// Explicit list of items.
    Explicit(Vec<ProjectionItem>),
}

/// A projection item (expression AS alias).
#[derive(Debug, Clone)]
pub struct ProjectionItem {
    /// The expression.
    pub expression: Expression,
    /// Optional alias.
    pub alias: Option<String>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An UNWIND clause.
#[derive(Debug, Clone)]
pub struct UnwindClause {
    /// The list expression to unwind.
    pub expression: Expression,
    /// The variable name for each element.
    pub variable: String,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// An ORDER BY clause.
#[derive(Debug, Clone)]
pub struct OrderByClause {
    /// Sort items.
    pub items: Vec<SortItem>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A sort item.
#[derive(Debug, Clone)]
pub struct SortItem {
    /// The expression to sort by.
    pub expression: Expression,
    /// Sort direction.
    pub direction: SortDirection,
}

/// Sort direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    /// Ascending (default).
    Asc,
    /// Descending.
    Desc,
}

impl Default for SortDirection {
    fn default() -> Self {
        Self::Asc
    }
}

/// A CREATE clause.
#[derive(Debug, Clone)]
pub struct CreateClause {
    /// Patterns to create.
    pub patterns: Vec<Pattern>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A MERGE clause.
#[derive(Debug, Clone)]
pub struct MergeClause {
    /// The pattern to merge.
    pub pattern: Pattern,
    /// ON CREATE actions.
    pub on_create: Option<SetClause>,
    /// ON MATCH actions.
    pub on_match: Option<SetClause>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A DELETE clause.
#[derive(Debug, Clone)]
pub struct DeleteClause {
    /// Whether DETACH DELETE.
    pub detach: bool,
    /// Expressions to delete.
    pub expressions: Vec<Expression>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A SET clause.
#[derive(Debug, Clone)]
pub struct SetClause {
    /// Set items.
    pub items: Vec<SetItem>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A SET item.
#[derive(Debug, Clone)]
pub enum SetItem {
    /// Set a property: n.prop = expr
    Property {
        /// Variable name.
        variable: String,
        /// Property name.
        property: String,
        /// Value expression.
        value: Expression,
    },
    /// Set all properties: n = {props}
    AllProperties {
        /// Variable name.
        variable: String,
        /// Properties map expression.
        properties: Expression,
    },
    /// Add properties: n += {props}
    MergeProperties {
        /// Variable name.
        variable: String,
        /// Properties map expression.
        properties: Expression,
    },
    /// Set labels: n:Label1:Label2
    Labels {
        /// Variable name.
        variable: String,
        /// Labels to add.
        labels: Vec<String>,
    },
}

/// A REMOVE clause.
#[derive(Debug, Clone)]
pub struct RemoveClause {
    /// Remove items.
    pub items: Vec<RemoveItem>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A REMOVE item.
#[derive(Debug, Clone)]
pub enum RemoveItem {
    /// Remove a property: n.prop
    Property {
        /// Variable name.
        variable: String,
        /// Property name.
        property: String,
    },
    /// Remove labels: n:Label1:Label2
    Labels {
        /// Variable name.
        variable: String,
        /// Labels to remove.
        labels: Vec<String>,
    },
}

/// An expression.
#[derive(Debug, Clone)]
pub enum Expression {
    /// A literal value.
    Literal(Literal),
    /// A variable reference.
    Variable(String),
    /// A parameter ($param or {param}).
    Parameter(String),
    /// Property access: expr.prop
    PropertyAccess {
        /// The base expression.
        base: Box<Expression>,
        /// Property name.
        property: String,
    },
    /// Index access: `expr[index]`
    IndexAccess {
        /// The base expression.
        base: Box<Expression>,
        /// Index expression.
        index: Box<Expression>,
    },
    /// Slice access: expr[start..end]
    SliceAccess {
        /// The base expression.
        base: Box<Expression>,
        /// Start index (optional).
        start: Option<Box<Expression>>,
        /// End index (optional).
        end: Option<Box<Expression>>,
    },
    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<Expression>,
        /// Operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<Expression>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<Expression>,
    },
    /// Function call.
    FunctionCall {
        /// Function name.
        name: String,
        /// Whether DISTINCT is specified (for aggregates).
        distinct: bool,
        /// Arguments.
        args: Vec<Expression>,
    },
    /// List literal: [1, 2, 3]
    List(Vec<Expression>),
    /// Map literal: {key: value, ...}
    Map(Vec<(String, Expression)>),
    /// List comprehension: [x IN list WHERE pred | expr]
    ListComprehension {
        /// Variable name.
        variable: String,
        /// Source list.
        list: Box<Expression>,
        /// Optional filter predicate.
        filter: Option<Box<Expression>>,
        /// Optional projection expression.
        projection: Option<Box<Expression>>,
    },
    /// Pattern comprehension: [(a)-->(b) | b.name]
    PatternComprehension {
        /// The pattern.
        pattern: Box<Pattern>,
        /// Optional WHERE clause.
        where_clause: Option<Box<Expression>>,
        /// Projection expression.
        projection: Box<Expression>,
    },
    /// CASE expression.
    Case {
        /// Optional input expression (simple CASE).
        input: Option<Box<Expression>>,
        /// WHEN clauses.
        whens: Vec<(Expression, Expression)>,
        /// ELSE clause.
        else_clause: Option<Box<Expression>>,
    },
    /// EXISTS subquery.
    Exists(Box<Query>),
    /// COUNT subquery.
    CountSubquery(Box<Query>),
}

/// A literal value.
#[derive(Debug, Clone)]
pub enum Literal {
    /// NULL
    Null,
    /// Boolean
    Bool(bool),
    /// Integer
    Integer(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
}

/// Binary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Comparison
    /// =
    Eq,
    /// <>
    Ne,
    /// <
    Lt,
    /// <=
    Le,
    /// >
    Gt,
    /// >=
    Ge,

    // Logical
    /// AND
    And,
    /// OR
    Or,
    /// XOR
    Xor,

    // Arithmetic
    /// +
    Add,
    /// -
    Sub,
    /// *
    Mul,
    /// /
    Div,
    /// %
    Mod,
    /// ^
    Pow,

    // String
    /// String concatenation
    Concat,
    /// STARTS WITH
    StartsWith,
    /// ENDS WITH
    EndsWith,
    /// CONTAINS
    Contains,
    /// =~ (regex match)
    RegexMatch,

    // Collection
    /// IN
    In,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// NOT
    Not,
    /// - (negation)
    Neg,
    /// + (positive, no-op)
    Pos,
    /// IS NULL
    IsNull,
    /// IS NOT NULL
    IsNotNull,
}
