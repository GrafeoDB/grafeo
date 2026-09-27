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
    /// A procedure call statement (CALL ... YIELD).
    Call(CallStatement),
    /// A composite query (UNION, EXCEPT, INTERSECT, OTHERWISE).
    CompositeQuery {
        /// The left query.
        left: Box<Statement>,
        /// The composite operation.
        op: CompositeOp,
        /// The right query.
        right: Box<Statement>,
    },
    /// A session or transaction command.
    SessionCommand(SessionCommand),
    /// EXPLAIN: returns the query plan without executing.
    Explain(Box<Statement>),
    /// PROFILE: executes the query and returns per-operator metrics.
    Profile(Box<Statement>),
}

/// GQL transaction isolation level (ISO/IEC 39075 Section 19).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransactionIsolationLevel {
    /// READ COMMITTED
    ReadCommitted,
    /// SNAPSHOT ISOLATION (also matches REPEATABLE READ)
    SnapshotIsolation,
    /// SERIALIZABLE
    Serializable,
}

/// Target for `SESSION RESET` (ISO/IEC 39075 Section 7.2).
///
/// The spec allows resetting schema, graph, time zone, and parameters
/// independently. `All` resets all characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionResetTarget {
    /// `SESSION RESET [ALL CHARACTERISTICS]`: resets everything.
    All,
    /// `SESSION RESET SCHEMA`: resets session schema only (Section 7.2 GR1).
    Schema,
    /// `SESSION RESET [PROPERTY] GRAPH`: resets session graph only (Section 7.2 GR2).
    Graph,
    /// `SESSION RESET TIME ZONE`: resets time zone (Section 7.2 GR3).
    TimeZone,
    /// `SESSION RESET [ALL] PARAMETERS`: resets all session parameters (Section 7.2 GR4).
    Parameters,
}

/// Session and transaction commands.
#[derive(Debug, Clone)]
pub enum SessionCommand {
    /// `USE GRAPH name`
    UseGraph(String),
    /// `CREATE [PROPERTY] GRAPH name [IF NOT EXISTS] [TYPED type_name] [LIKE source | AS COPY OF source] [OPEN | ANY]`
    CreateGraph {
        /// Graph name.
        name: String,
        /// IF NOT EXISTS flag.
        if_not_exists: bool,
        /// Optional graph type binding.
        typed: Option<String>,
        /// LIKE source_graph: clone schema only.
        like_graph: Option<String>,
        /// AS COPY OF source_graph: clone schema and data.
        copy_of: Option<String>,
        /// ANY GRAPH / OPEN: schema-free graph (no type enforcement).
        open: bool,
    },
    /// `DROP [PROPERTY] GRAPH [IF EXISTS] name`
    DropGraph {
        /// Graph name.
        name: String,
        /// IF EXISTS flag.
        if_exists: bool,
    },
    /// `SESSION SET GRAPH name`
    SessionSetGraph(String),
    /// `SESSION SET SCHEMA name` (ISO/IEC 39075 Section 7.1 GR1: independent from graph)
    SessionSetSchema(String),
    /// `SESSION SET TIME ZONE 'tz'`
    SessionSetTimeZone(String),
    /// `SESSION SET PARAMETER $name = value`
    SessionSetParameter(String, Expression),
    /// `SESSION RESET [ALL | SCHEMA | GRAPH | TIME ZONE | PARAMETER]`
    /// (ISO/IEC 39075 Section 7.2: schema and graph can be reset independently)
    SessionReset(SessionResetTarget),
    /// `SESSION CLOSE`
    SessionClose,
    /// `START TRANSACTION [READ ONLY | READ WRITE] [ISOLATION LEVEL <level>]`
    StartTransaction {
        /// Whether the transaction is read-only (default: false = read-write).
        read_only: bool,
        /// Optional isolation level override.
        isolation_level: Option<TransactionIsolationLevel>,
    },
    /// `COMMIT`
    Commit,
    /// `ROLLBACK`
    Rollback,
    /// `SAVEPOINT name`
    Savepoint(String),
    /// `ROLLBACK TO SAVEPOINT name`
    RollbackToSavepoint(String),
    /// `RELEASE SAVEPOINT name`
    ReleaseSavepoint(String),
    /// `CREATE PROJECTION name LABELS (l1, l2) EDGE_TYPES (e1, e2)`
    CreateProjection {
        /// Projection name.
        name: String,
        /// Node labels to include (empty = all).
        node_labels: Vec<String>,
        /// Edge types to include (empty = all).
        edge_types: Vec<String>,
    },
    /// `DROP PROJECTION name`
    DropProjection {
        /// Projection name.
        name: String,
    },
    /// `SHOW PROJECTIONS`
    ShowProjections,
}

/// Composite query operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositeOp {
    /// UNION (distinct).
    Union,
    /// UNION ALL (keep duplicates).
    UnionAll,
    /// EXCEPT (set difference, distinct).
    Except,
    /// EXCEPT ALL (set difference, keep duplicates).
    ExceptAll,
    /// INTERSECT (set intersection, distinct).
    Intersect,
    /// INTERSECT ALL (set intersection, keep duplicates).
    IntersectAll,
    /// OTHERWISE (fallback if left is empty).
    Otherwise,
    /// NEXT (linear composition: output of left feeds into right).
    Next,
}

/// A CALL procedure statement (ISO GQL Section 15).
///
/// ```text
/// CALL procedure_name(args)
///   [YIELD field [AS alias], ...]
///   [WHERE predicate]
///   [RETURN expr [AS alias], ... [ORDER BY ...] [SKIP n] [LIMIT n]]
/// ```
#[derive(Debug, Clone)]
pub struct CallStatement {
    /// Qualified procedure name, e.g. `["grafeo", "pagerank"]`.
    pub procedure_name: Vec<String>,
    /// Positional arguments passed to the procedure.
    pub arguments: Vec<Expression>,
    /// Optional YIELD clause selecting result columns.
    pub yield_items: Option<Vec<YieldItem>>,
    /// Optional WHERE clause filtering yielded rows.
    pub where_clause: Option<WhereClause>,
    /// Optional RETURN clause with projection, ORDER BY, SKIP, LIMIT.
    pub return_clause: Option<ReturnClause>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A single YIELD item: `field_name [AS alias]`.
#[derive(Debug, Clone)]
pub struct YieldItem {
    /// Column name from the procedure result.
    pub field_name: String,
    /// Optional alias.
    pub alias: Option<String>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// File format for LOAD DATA statements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadFormat {
    /// CSV (comma-separated values).
    Csv,
    /// JSON Lines (one JSON object per line).
    Jsonl,
    /// Apache Parquet columnar format.
    Parquet,
}

/// LOAD DATA clause: reads data from a file and produces rows.
#[derive(Debug, Clone)]
pub struct LoadDataClause {
    /// File path (local filesystem).
    pub path: String,
    /// File format.
    pub format: LoadFormat,
    /// Whether the file has a header row (CSV only).
    pub with_headers: bool,
    /// Variable name to bind each row to.
    pub variable: String,
    /// Optional field terminator override (CSV only).
    pub field_terminator: Option<char>,
    /// Source span.
    pub span: SourceSpan,
}

/// A clause in a query, preserving source order for correct variable scoping.
#[derive(Debug, Clone)]
pub enum QueryClause {
    /// A MATCH clause.
    Match(MatchClause),
    /// An UNWIND clause.
    Unwind(UnwindClause),
    /// A FOR clause (desugared to UNWIND).
    For(UnwindClause),
    /// A CREATE/INSERT clause.
    Create(InsertStatement),
    /// A DELETE clause.
    Delete(DeleteStatement),
    /// A SET clause.
    Set(SetClause),
    /// A MERGE clause.
    Merge(MergeClause),
    /// A LET clause (variable bindings).
    Let(Vec<(String, Expression)>),
    /// An inline CALL { subquery } clause (optional = OPTIONAL CALL { ... }).
    InlineCall {
        /// The inner subquery.
        subquery: QueryStatement,
        /// Whether this is OPTIONAL CALL (left-join semantics).
        optional: bool,
    },
    /// A CALL procedure clause within a query.
    CallProcedure(CallStatement),
    /// A LOAD DATA clause.
    LoadData(LoadDataClause),
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
    /// REMOVE clauses for label/property removal.
    pub remove_clauses: Vec<RemoveClause>,
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
    /// Optional HAVING clause (filters aggregate results).
    pub having_clause: Option<HavingClause>,
    /// Ordered clauses preserving source order (for UNWIND/FOR variable scoping).
    pub ordered_clauses: Vec<QueryClause>,
    /// Source span in the original query.
    pub span: Option<SourceSpan>,
}

/// A HAVING clause for filtering aggregate results.
#[derive(Debug, Clone)]
pub struct HavingClause {
    /// The filter expression.
    pub expression: Expression,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A SET clause.
#[derive(Debug, Clone)]
pub struct SetClause {
    /// Property assignments.
    pub assignments: Vec<PropertyAssignment>,
    /// Map assignments (SET n = {map} or SET n += {map}).
    pub map_assignments: Vec<MapAssignment>,
    /// Label operations (add labels to nodes).
    pub label_operations: Vec<LabelOperation>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// A map assignment for SET n = {map} or SET n += {map}.
#[derive(Debug, Clone)]
pub struct MapAssignment {
    /// Variable name.
    pub variable: String,
    /// Map expression (typically a map literal or variable).
    pub map_expr: Expression,
    /// Whether to replace all properties (true for `=`) or merge (false for `+=`).
    pub replace: bool,
}

/// A label operation for adding/removing labels.
#[derive(Debug, Clone)]
pub struct LabelOperation {
    /// Variable name.
    pub variable: String,
    /// Labels to add.
    pub labels: Vec<String>,
}

/// A REMOVE clause for removing labels or properties.
#[derive(Debug, Clone)]
pub struct RemoveClause {
    /// Label removal operations.
    pub label_operations: Vec<LabelOperation>,
    /// Property removal operations (variable.property pairs).
    pub property_removals: Vec<(String, String)>,
    /// Source span.
    pub span: Option<SourceSpan>,
}

/// Path traversal mode (ISO GQL sec 16.6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathMode {
    /// Allows repeated nodes and edges.
    Walk,
    /// No repeated edges.
    Trail,
    /// No repeated nodes except endpoints.
    Simple,
    /// No repeated nodes at all.
    Acyclic,
}

/// Path search prefix for ISO GQL shortest path queries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PathSearchPrefix {
    /// `ALL`: enumerate all matching paths.
    All,
    /// `ANY`: return any single matching path.
    Any,
    /// `ANY k`: return up to k matching paths.
    AnyK(usize),
    /// `ALL SHORTEST`: return all paths of minimum length.
    AllShortest,
    /// `ANY SHORTEST`: return any single shortest path.
    AnyShortest,
    /// `SHORTEST k`: return the k shortest paths.
    ShortestK(usize),
    /// `SHORTEST k GROUPS`: return paths grouped by length.
    ShortestKGroups(usize),
}

/// Match mode controlling edge/node uniqueness (ISO GQL sec 16.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchMode {
    /// `DIFFERENT EDGES`: each edge matched at most once (like TRAIL).
    DifferentEdges,
    /// `REPEATABLE ELEMENTS`: edges and nodes may repeat (like WALK).
    RepeatableElements,
}

/// A MATCH clause.
#[derive(Debug, Clone)]
pub struct MatchClause {
    /// Whether this is an OPTIONAL MATCH.
    pub optional: bool,
    /// Path mode for traversal (WALK, TRAIL, SIMPLE, ACYCLIC).
    pub path_mode: Option<PathMode>,
    /// Path search prefix (ANY, ALL SHORTEST, SHORTEST k, etc.).
    pub search_prefix: Option<PathSearchPrefix>,
    /// Match mode (DIFFERENT EDGES, REPEATABLE ELEMENTS).
    pub match_mode: Option<MatchMode>,
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
    /// Per-pattern path search prefix (e.g., `p = ANY SHORTEST (...)`).
    pub search_prefix: Option<PathSearchPrefix>,
    /// Per-pattern KEEP clause (ISO GQL sec 16.5).
    pub keep: Option<MatchMode>,
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
    /// Items to pass to the next query part (empty when `is_wildcard` is true).
    pub items: Vec<ReturnItem>,
    /// Whether this is `WITH *` (pass all variables through).
    pub is_wildcard: bool,
    /// Optional WHERE clause after WITH.
    pub where_clause: Option<WhereClause>,
    /// LET bindings attached to this WITH clause (e.g. `WITH n LET x = n.age * 2`).
    pub let_bindings: Vec<(String, Expression)>,
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
    /// Optional variable for 1-based element position (FOR ... WITH ORDINALITY var).
    pub ordinality_var: Option<String>,
    /// Optional variable for 0-based element position (FOR ... WITH OFFSET var).
    pub offset_var: Option<String>,
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
    /// A quantified (parenthesized) path pattern: `((a)-[e]->(b)){2,5}`.
    Quantified {
        /// The inner subpattern to repeat.
        pattern: Box<Pattern>,
        /// Minimum repetitions.
        min: u32,
        /// Maximum repetitions (None = unbounded).
        max: Option<u32>,
        /// G048: Subpath variable declaration, e.g. `(p = (a)-[e]->(b)){2,5}`.
        subpath_var: Option<String>,
        /// G049: Path mode prefix inside parenthesized pattern, e.g. `(TRAIL (a)-[]->(b)){2,5}`.
        path_mode: Option<PathMode>,
        /// G050: WHERE clause inside parenthesized pattern, e.g. `((a)-[e]->(b) WHERE e.w > 5){2,5}`.
        where_clause: Option<Expression>,
    },
    /// A union of alternative path patterns: `(a)-[:T1]->(b) | (a)-[:T2]->(c)`.
    /// Set semantics: duplicates are removed.
    Union(Vec<Pattern>),
    /// A multiset union of alternative path patterns: `(a)-[:T1]->(b) |+| (a)-[:T2]->(c)`.
    /// Bag semantics: duplicates are preserved (G030).
    MultisetUnion(Vec<Pattern>),
}

/// A label expression for GQL IS syntax (e.g., `IS Person | Employee`, `IS Person & !Employee`).
#[derive(Debug, Clone)]
pub enum LabelExpression {
    /// A single label name.
    Label(String),
    /// Conjunction (AND): all labels must match.
    Conjunction(Vec<LabelExpression>),
    /// Disjunction (OR): any label must match.
    Disjunction(Vec<LabelExpression>),
    /// Negation (NOT): label must not match.
    Negation(Box<LabelExpression>),
    /// Wildcard (%): matches any label.
    Wildcard,
}

/// A node pattern like (n:Person) or (n IS Person | Employee).
#[derive(Debug, Clone)]
pub struct NodePattern {
    /// Variable name (optional).
    pub variable: Option<String>,
    /// Labels to match (colon syntax: `:Label1:Label2`).
    pub labels: Vec<String>,
    /// Label expression (IS syntax: `IS Person | Employee`).
    pub label_expression: Option<LabelExpression>,
    /// Property filters.
    pub properties: Vec<(String, Expression)>,
    /// Element pattern WHERE clause: `(n WHERE n.age > 30)`.
    pub where_clause: Option<Expression>,
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
    /// Element pattern WHERE clause: `-[e WHERE e.weight > 5]->`.
    pub where_clause: Option<Expression>,
    /// Questioned edge: `->?` means "optional" (0 or 1 hop).
    pub questioned: bool,
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
    /// Items to return (empty when `is_wildcard` is true).
    pub items: Vec<ReturnItem>,
    /// Whether this is `RETURN *` (return all bound variables).
    pub is_wildcard: bool,
    /// Explicit GROUP BY expressions.
    pub group_by: Vec<Expression>,
    /// Optional ORDER BY.
    pub order_by: Option<OrderByClause>,
    /// Optional SKIP.
    pub skip: Option<Expression>,
    /// Optional LIMIT.
    pub limit: Option<Expression>,
    /// Whether this is a FINISH statement (consume input, return empty).
    pub is_finish: bool,
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
    /// Optional null ordering (NULLS FIRST / NULLS LAST).
    pub nulls: Option<NullsOrdering>,
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order.
    Asc,
    /// Descending order.
    Desc,
}

/// Null ordering for ORDER BY (ISO GQL feature GA03).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullsOrdering {
    /// Nulls sort before all non-null values.
    First,
    /// Nulls sort after all non-null values.
    Last,
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

/// A target for a DELETE statement: either a variable name or a general expression.
#[derive(Debug, Clone)]
pub enum DeleteTarget {
    /// A simple variable reference (e.g., `DELETE n`).
    Variable(String),
    /// A general expression (e.g., `DELETE n.friend`, `DELETE head(collect(n))`).
    Expression(Expression),
}

/// A DELETE statement.
#[derive(Debug, Clone)]
pub struct DeleteStatement {
    /// Targets to delete (variables or expressions).
    pub targets: Vec<DeleteTarget>,
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

// Schema DDL types (shared across parsers, defined in query::schema).
pub use crate::query::schema::{
    AlterGraphTypeStatement, AlterTypeStatement, ConstraintKind, CreateConstraintStatement,
    CreateEdgeTypeStatement, CreateGraphTypeStatement, CreateIndexStatement,
    CreateNodeTypeStatement, CreateProcedureStatement, CreateVectorIndexStatement,
    GraphTypeAlteration, IndexKind, IndexOptions, InlineElementType, ProcedureParam,
    ProcedureReturn, PropertyDefinition, SchemaStatement, TypeAlteration,
};

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
    /// EXISTS subquery expression: checks if inner query returns results.
    ExistsSubquery {
        /// The inner query pattern to check for existence.
        query: Box<QueryStatement>,
    },
    /// COUNT subquery expression: counts rows from inner query.
    CountSubquery {
        /// The inner query pattern to count.
        query: Box<QueryStatement>,
    },
    /// VALUE { subquery }: evaluates subquery and returns scalar result.
    ValueSubquery {
        /// The inner subquery.
        query: Box<QueryStatement>,
    },
    /// A map literal: `{key: value, ...}`.
    Map(Vec<(String, Expression)>),
    /// Index access: `expr[index]`.
    IndexAccess {
        /// The base expression.
        base: Box<Expression>,
        /// The index expression.
        index: Box<Expression>,
    },
    /// Slice access: `expr[start..end]`, `expr[..end]`, `expr[start..]`.
    SliceAccess {
        /// The base expression.
        base: Box<Expression>,
        /// Start index (None means from beginning).
        start: Option<Box<Expression>>,
        /// End index (None means to end).
        end: Option<Box<Expression>>,
    },
    /// LET ... IN ... END expression.
    LetIn {
        /// Variable bindings.
        bindings: Vec<(String, Expression)>,
        /// The body expression.
        body: Box<Expression>,
    },
    /// List comprehension: `[x IN list WHERE predicate | expression]`.
    ListComprehension {
        /// Iteration variable name.
        variable: String,
        /// Source list expression.
        list_expr: Box<Expression>,
        /// Optional filter predicate.
        filter_expr: Option<Box<Expression>>,
        /// Mapping expression for each element.
        map_expr: Box<Expression>,
    },
    /// List predicate: `all/any/none/single(x IN list WHERE predicate)`.
    ListPredicate {
        /// Predicate kind.
        kind: ListPredicateKind,
        /// Iteration variable name.
        variable: String,
        /// Source list expression.
        list_expr: Box<Expression>,
        /// Predicate expression.
        predicate: Box<Expression>,
    },
    /// Reduce accumulator: `reduce(acc = init, x IN list | expression)`.
    Reduce {
        /// Accumulator variable name.
        accumulator: String,
        /// Initial value for the accumulator.
        initial: Box<Expression>,
        /// Iteration variable name.
        variable: String,
        /// Source list expression.
        list: Box<Expression>,
        /// Body expression (references both accumulator and variable).
        expression: Box<Expression>,
    },
}

/// The kind of list predicate function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ListPredicateKind {
    /// `all(x IN list WHERE pred)`: true if pred holds for every element.
    All,
    /// `any(x IN list WHERE pred)`: true if pred holds for at least one.
    Any,
    /// `none(x IN list WHERE pred)`: true if pred holds for none.
    None,
    /// `single(x IN list WHERE pred)`: true if pred holds for exactly one.
    Single,
}

/// A literal value.
#[derive(Debug, Clone, PartialEq)]
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
    /// Typed date literal: `DATE '2024-01-15'`
    Date(String),
    /// Typed time literal: `TIME '14:30:00'`
    Time(String),
    /// Typed duration literal: `DURATION 'P1Y2M'`
    Duration(String),
    /// Typed datetime literal: `DATETIME '2024-01-15T14:30:00Z'`
    Datetime(String),
    /// Typed zoned datetime literal: `ZONED DATETIME '2024-01-15T14:30:00+05:30'`
    ZonedDatetime(String),
    /// Typed zoned time literal: `ZONED TIME '14:30:00+05:30'`
    ZonedTime(String),
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
    /// Logical XOR.
    Xor,

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
    /// Exponentiation (`^`).
    Pow,

    // String
    /// String concatenation.
    Concat,
    /// LIKE pattern matching.
    Like,
    /// IN list membership.
    In,
    /// STARTS WITH prefix matching.
    StartsWith,
    /// ENDS WITH suffix matching.
    EndsWith,
    /// CONTAINS substring matching.
    Contains,
}

/// A unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Logical NOT.
    Not,
    /// Unary minus.
    Neg,
    /// Unary plus (identity).
    Pos,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
}
