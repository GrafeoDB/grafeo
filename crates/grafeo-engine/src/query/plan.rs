//! Logical query plan representation.
//!
//! The logical plan is the intermediate representation between parsed queries
//! and physical execution. Both GQL and Cypher queries are translated to this
//! common representation.

use grafeo_common::types::Value;

/// A logical query plan.
#[derive(Debug, Clone)]
pub struct LogicalPlan {
    /// The root operator of the plan.
    pub root: LogicalOperator,
}

impl LogicalPlan {
    /// Creates a new logical plan with the given root operator.
    pub fn new(root: LogicalOperator) -> Self {
        Self { root }
    }
}

/// A logical operator in the query plan.
#[derive(Debug, Clone)]
pub enum LogicalOperator {
    /// Scan all nodes, optionally filtered by label.
    NodeScan(NodeScanOp),

    /// Scan all edges, optionally filtered by type.
    EdgeScan(EdgeScanOp),

    /// Expand from nodes to neighbors via edges.
    Expand(ExpandOp),

    /// Filter rows based on a predicate.
    Filter(FilterOp),

    /// Project specific columns.
    Project(ProjectOp),

    /// Join two inputs.
    Join(JoinOp),

    /// Aggregate with grouping.
    Aggregate(AggregateOp),

    /// Limit the number of results.
    Limit(LimitOp),

    /// Skip a number of results.
    Skip(SkipOp),

    /// Sort results.
    Sort(SortOp),

    /// Remove duplicate results.
    Distinct(DistinctOp),

    /// Create a new node.
    CreateNode(CreateNodeOp),

    /// Create a new edge.
    CreateEdge(CreateEdgeOp),

    /// Delete a node.
    DeleteNode(DeleteNodeOp),

    /// Delete an edge.
    DeleteEdge(DeleteEdgeOp),

    /// Set properties on a node or edge.
    SetProperty(SetPropertyOp),

    /// Add labels to a node.
    AddLabel(AddLabelOp),

    /// Remove labels from a node.
    RemoveLabel(RemoveLabelOp),

    /// Return results (terminal operator).
    Return(ReturnOp),

    /// Empty result set.
    Empty,

    // ==================== RDF/SPARQL Operators ====================
    /// Scan RDF triples matching a pattern.
    TripleScan(TripleScanOp),

    /// Union of multiple result sets.
    Union(UnionOp),

    /// Left outer join for OPTIONAL patterns.
    LeftJoin(LeftJoinOp),

    /// Anti-join for MINUS patterns.
    AntiJoin(AntiJoinOp),

    /// Bind a variable to an expression.
    Bind(BindOp),

    /// Unwind a list into individual rows.
    Unwind(UnwindOp),

    /// Merge a pattern (match or create).
    Merge(MergeOp),

    /// Find shortest path between nodes.
    ShortestPath(ShortestPathOp),

    // ==================== SPARQL Update Operators ====================
    /// Insert RDF triples.
    InsertTriple(InsertTripleOp),

    /// Delete RDF triples.
    DeleteTriple(DeleteTripleOp),

    /// SPARQL MODIFY operation (DELETE/INSERT WHERE).
    /// Evaluates WHERE once, applies DELETE templates, then INSERT templates.
    Modify(ModifyOp),

    /// Clear a graph (remove all triples).
    ClearGraph(ClearGraphOp),

    /// Create a new named graph.
    CreateGraph(CreateGraphOp),

    /// Drop (remove) a named graph.
    DropGraph(DropGraphOp),

    /// Load data from a URL into a graph.
    LoadGraph(LoadGraphOp),

    /// Copy triples from one graph to another.
    CopyGraph(CopyGraphOp),

    /// Move triples from one graph to another.
    MoveGraph(MoveGraphOp),

    /// Add (merge) triples from one graph to another.
    AddGraph(AddGraphOp),

    // ==================== Vector Search Operators ====================
    /// Scan using vector similarity search.
    VectorScan(VectorScanOp),

    /// Join graph patterns with vector similarity search.
    ///
    /// Computes vector distances between entities from the left input and
    /// a query vector, then joins with similarity scores. Useful for:
    /// - Filtering graph traversal results by vector similarity
    /// - Computing aggregated embeddings and finding similar entities
    /// - Combining multiple vector sources with graph structure
    VectorJoin(VectorJoinOp),
}

/// Scan nodes from the graph.
#[derive(Debug, Clone)]
pub struct NodeScanOp {
    /// Variable name to bind the node to.
    pub variable: String,
    /// Optional label filter.
    pub label: Option<String>,
    /// Child operator (if any, for chained patterns).
    pub input: Option<Box<LogicalOperator>>,
}

/// Scan edges from the graph.
#[derive(Debug, Clone)]
pub struct EdgeScanOp {
    /// Variable name to bind the edge to.
    pub variable: String,
    /// Optional edge type filter.
    pub edge_type: Option<String>,
    /// Child operator (if any).
    pub input: Option<Box<LogicalOperator>>,
}

/// Expand from nodes to their neighbors.
#[derive(Debug, Clone)]
pub struct ExpandOp {
    /// Source node variable.
    pub from_variable: String,
    /// Target node variable to bind.
    pub to_variable: String,
    /// Edge variable to bind (optional).
    pub edge_variable: Option<String>,
    /// Direction of expansion.
    pub direction: ExpandDirection,
    /// Optional edge type filter.
    pub edge_type: Option<String>,
    /// Minimum hops (for variable-length patterns).
    pub min_hops: u32,
    /// Maximum hops (for variable-length patterns).
    pub max_hops: Option<u32>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
    /// Path alias for variable-length patterns (e.g., `p` in `p = (a)-[*1..3]->(b)`).
    /// When set, a path length column will be output under this name.
    pub path_alias: Option<String>,
}

/// Direction for edge expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpandDirection {
    /// Follow outgoing edges.
    Outgoing,
    /// Follow incoming edges.
    Incoming,
    /// Follow edges in either direction.
    Both,
}

/// Join two inputs.
#[derive(Debug, Clone)]
pub struct JoinOp {
    /// Left input.
    pub left: Box<LogicalOperator>,
    /// Right input.
    pub right: Box<LogicalOperator>,
    /// Join type.
    pub join_type: JoinType,
    /// Join conditions.
    pub conditions: Vec<JoinCondition>,
}

/// Join type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join.
    Inner,
    /// Left outer join.
    Left,
    /// Right outer join.
    Right,
    /// Full outer join.
    Full,
    /// Cross join (Cartesian product).
    Cross,
    /// Semi join (returns left rows with matching right rows).
    Semi,
    /// Anti join (returns left rows without matching right rows).
    Anti,
}

/// A join condition.
#[derive(Debug, Clone)]
pub struct JoinCondition {
    /// Left expression.
    pub left: LogicalExpression,
    /// Right expression.
    pub right: LogicalExpression,
}

/// Aggregate with grouping.
#[derive(Debug, Clone)]
pub struct AggregateOp {
    /// Group by expressions.
    pub group_by: Vec<LogicalExpression>,
    /// Aggregate functions.
    pub aggregates: Vec<AggregateExpr>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
    /// HAVING clause filter (applied after aggregation).
    pub having: Option<LogicalExpression>,
}

/// An aggregate expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// Aggregate function.
    pub function: AggregateFunction,
    /// Expression to aggregate.
    pub expression: Option<LogicalExpression>,
    /// Whether to use DISTINCT.
    pub distinct: bool,
    /// Alias for the result.
    pub alias: Option<String>,
    /// Percentile parameter for PERCENTILE_DISC/PERCENTILE_CONT (0.0 to 1.0).
    pub percentile: Option<f64>,
}

/// Aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateFunction {
    /// Count all rows (COUNT(*)).
    Count,
    /// Count non-null values (COUNT(expr)).
    CountNonNull,
    /// Sum values.
    Sum,
    /// Average values.
    Avg,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Collect into list.
    Collect,
    /// Sample standard deviation (STDEV).
    StdDev,
    /// Population standard deviation (STDEVP).
    StdDevPop,
    /// Discrete percentile (PERCENTILE_DISC).
    PercentileDisc,
    /// Continuous percentile (PERCENTILE_CONT).
    PercentileCont,
}

/// Filter rows based on a predicate.
#[derive(Debug, Clone)]
pub struct FilterOp {
    /// The filter predicate.
    pub predicate: LogicalExpression,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Project specific columns.
#[derive(Debug, Clone)]
pub struct ProjectOp {
    /// Columns to project.
    pub projections: Vec<Projection>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// A single projection (column selection or computation).
#[derive(Debug, Clone)]
pub struct Projection {
    /// Expression to compute.
    pub expression: LogicalExpression,
    /// Alias for the result.
    pub alias: Option<String>,
}

/// Limit the number of results.
#[derive(Debug, Clone)]
pub struct LimitOp {
    /// Maximum number of rows to return.
    pub count: usize,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Skip a number of results.
#[derive(Debug, Clone)]
pub struct SkipOp {
    /// Number of rows to skip.
    pub count: usize,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Sort results.
#[derive(Debug, Clone)]
pub struct SortOp {
    /// Sort keys.
    pub keys: Vec<SortKey>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// A sort key.
#[derive(Debug, Clone)]
pub struct SortKey {
    /// Expression to sort by.
    pub expression: LogicalExpression,
    /// Sort order.
    pub order: SortOrder,
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Ascending order.
    Ascending,
    /// Descending order.
    Descending,
}

/// Remove duplicate results.
#[derive(Debug, Clone)]
pub struct DistinctOp {
    /// Input operator.
    pub input: Box<LogicalOperator>,
    /// Optional columns to use for deduplication.
    /// If None, all columns are used.
    pub columns: Option<Vec<String>>,
}

/// Create a new node.
#[derive(Debug, Clone)]
pub struct CreateNodeOp {
    /// Variable name to bind the created node to.
    pub variable: String,
    /// Labels for the new node.
    pub labels: Vec<String>,
    /// Properties for the new node.
    pub properties: Vec<(String, LogicalExpression)>,
    /// Input operator (for chained creates).
    pub input: Option<Box<LogicalOperator>>,
}

/// Create a new edge.
#[derive(Debug, Clone)]
pub struct CreateEdgeOp {
    /// Variable name to bind the created edge to.
    pub variable: Option<String>,
    /// Source node variable.
    pub from_variable: String,
    /// Target node variable.
    pub to_variable: String,
    /// Edge type.
    pub edge_type: String,
    /// Properties for the new edge.
    pub properties: Vec<(String, LogicalExpression)>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Delete a node.
#[derive(Debug, Clone)]
pub struct DeleteNodeOp {
    /// Variable of the node to delete.
    pub variable: String,
    /// Whether to detach (delete connected edges) before deleting.
    pub detach: bool,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Delete an edge.
#[derive(Debug, Clone)]
pub struct DeleteEdgeOp {
    /// Variable of the edge to delete.
    pub variable: String,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Set properties on a node or edge.
#[derive(Debug, Clone)]
pub struct SetPropertyOp {
    /// Variable of the entity to update.
    pub variable: String,
    /// Properties to set (name -> expression).
    pub properties: Vec<(String, LogicalExpression)>,
    /// Whether to replace all properties (vs. merge).
    pub replace: bool,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Add labels to a node.
#[derive(Debug, Clone)]
pub struct AddLabelOp {
    /// Variable of the node to update.
    pub variable: String,
    /// Labels to add.
    pub labels: Vec<String>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Remove labels from a node.
#[derive(Debug, Clone)]
pub struct RemoveLabelOp {
    /// Variable of the node to update.
    pub variable: String,
    /// Labels to remove.
    pub labels: Vec<String>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

// ==================== RDF/SPARQL Operators ====================

/// Scan RDF triples matching a pattern.
#[derive(Debug, Clone)]
pub struct TripleScanOp {
    /// Subject pattern (variable name or IRI).
    pub subject: TripleComponent,
    /// Predicate pattern (variable name or IRI).
    pub predicate: TripleComponent,
    /// Object pattern (variable name, IRI, or literal).
    pub object: TripleComponent,
    /// Named graph (optional).
    pub graph: Option<TripleComponent>,
    /// Input operator (for chained patterns).
    pub input: Option<Box<LogicalOperator>>,
}

/// A component of a triple pattern.
#[derive(Debug, Clone)]
pub enum TripleComponent {
    /// A variable to bind.
    Variable(String),
    /// A constant IRI.
    Iri(String),
    /// A constant literal value.
    Literal(Value),
}

/// Union of multiple result sets.
#[derive(Debug, Clone)]
pub struct UnionOp {
    /// Inputs to union together.
    pub inputs: Vec<LogicalOperator>,
}

/// Left outer join for OPTIONAL patterns.
#[derive(Debug, Clone)]
pub struct LeftJoinOp {
    /// Left (required) input.
    pub left: Box<LogicalOperator>,
    /// Right (optional) input.
    pub right: Box<LogicalOperator>,
    /// Optional filter condition.
    pub condition: Option<LogicalExpression>,
}

/// Anti-join for MINUS patterns.
#[derive(Debug, Clone)]
pub struct AntiJoinOp {
    /// Left input (results to keep if no match on right).
    pub left: Box<LogicalOperator>,
    /// Right input (patterns to exclude).
    pub right: Box<LogicalOperator>,
}

/// Bind a variable to an expression.
#[derive(Debug, Clone)]
pub struct BindOp {
    /// Expression to compute.
    pub expression: LogicalExpression,
    /// Variable to bind the result to.
    pub variable: String,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Unwind a list into individual rows.
///
/// For each input row, evaluates the expression (which should return a list)
/// and emits one row for each element in the list.
#[derive(Debug, Clone)]
pub struct UnwindOp {
    /// The list expression to unwind.
    pub expression: LogicalExpression,
    /// The variable name for each element.
    pub variable: String,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Merge a pattern (match or create).
///
/// MERGE tries to match a pattern in the graph. If found, returns the existing
/// elements (optionally applying ON MATCH SET). If not found, creates the pattern
/// (optionally applying ON CREATE SET).
#[derive(Debug, Clone)]
pub struct MergeOp {
    /// The node to merge.
    pub variable: String,
    /// Labels to match/create.
    pub labels: Vec<String>,
    /// Properties that must match (used for both matching and creation).
    pub match_properties: Vec<(String, LogicalExpression)>,
    /// Properties to set on CREATE.
    pub on_create: Vec<(String, LogicalExpression)>,
    /// Properties to set on MATCH.
    pub on_match: Vec<(String, LogicalExpression)>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Find shortest path between two nodes.
///
/// This operator uses Dijkstra's algorithm to find the shortest path(s)
/// between a source node and a target node, optionally filtered by edge type.
#[derive(Debug, Clone)]
pub struct ShortestPathOp {
    /// Input operator providing source/target nodes.
    pub input: Box<LogicalOperator>,
    /// Variable name for the source node.
    pub source_var: String,
    /// Variable name for the target node.
    pub target_var: String,
    /// Optional edge type filter.
    pub edge_type: Option<String>,
    /// Direction of edge traversal.
    pub direction: ExpandDirection,
    /// Variable name to bind the path result.
    pub path_alias: String,
    /// Whether to find all shortest paths (vs. just one).
    pub all_paths: bool,
}

// ==================== SPARQL Update Operators ====================

/// Insert RDF triples.
#[derive(Debug, Clone)]
pub struct InsertTripleOp {
    /// Subject of the triple.
    pub subject: TripleComponent,
    /// Predicate of the triple.
    pub predicate: TripleComponent,
    /// Object of the triple.
    pub object: TripleComponent,
    /// Named graph (optional).
    pub graph: Option<String>,
    /// Input operator (provides variable bindings).
    pub input: Option<Box<LogicalOperator>>,
}

/// Delete RDF triples.
#[derive(Debug, Clone)]
pub struct DeleteTripleOp {
    /// Subject pattern.
    pub subject: TripleComponent,
    /// Predicate pattern.
    pub predicate: TripleComponent,
    /// Object pattern.
    pub object: TripleComponent,
    /// Named graph (optional).
    pub graph: Option<String>,
    /// Input operator (provides variable bindings).
    pub input: Option<Box<LogicalOperator>>,
}

/// SPARQL MODIFY operation (DELETE/INSERT WHERE).
///
/// Per SPARQL 1.1 Update spec, this operator:
/// 1. Evaluates the WHERE clause once to get bindings
/// 2. Applies DELETE templates using those bindings
/// 3. Applies INSERT templates using the SAME bindings
///
/// This ensures DELETE and INSERT see consistent data.
#[derive(Debug, Clone)]
pub struct ModifyOp {
    /// DELETE triple templates (patterns with variables).
    pub delete_templates: Vec<TripleTemplate>,
    /// INSERT triple templates (patterns with variables).
    pub insert_templates: Vec<TripleTemplate>,
    /// WHERE clause that provides variable bindings.
    pub where_clause: Box<LogicalOperator>,
    /// Named graph context (for WITH clause).
    pub graph: Option<String>,
}

/// A triple template for DELETE/INSERT operations.
#[derive(Debug, Clone)]
pub struct TripleTemplate {
    /// Subject (may be a variable).
    pub subject: TripleComponent,
    /// Predicate (may be a variable).
    pub predicate: TripleComponent,
    /// Object (may be a variable or literal).
    pub object: TripleComponent,
    /// Named graph (optional).
    pub graph: Option<String>,
}

/// Clear all triples from a graph.
#[derive(Debug, Clone)]
pub struct ClearGraphOp {
    /// Target graph (None = default graph, Some("") = all named, Some(iri) = specific graph).
    pub graph: Option<String>,
    /// Whether to silently ignore errors.
    pub silent: bool,
}

/// Create a new named graph.
#[derive(Debug, Clone)]
pub struct CreateGraphOp {
    /// IRI of the graph to create.
    pub graph: String,
    /// Whether to silently ignore if graph already exists.
    pub silent: bool,
}

/// Drop (remove) a named graph.
#[derive(Debug, Clone)]
pub struct DropGraphOp {
    /// Target graph (None = default graph).
    pub graph: Option<String>,
    /// Whether to silently ignore errors.
    pub silent: bool,
}

/// Load data from a URL into a graph.
#[derive(Debug, Clone)]
pub struct LoadGraphOp {
    /// Source URL to load data from.
    pub source: String,
    /// Destination graph (None = default graph).
    pub destination: Option<String>,
    /// Whether to silently ignore errors.
    pub silent: bool,
}

/// Copy triples from one graph to another.
#[derive(Debug, Clone)]
pub struct CopyGraphOp {
    /// Source graph.
    pub source: Option<String>,
    /// Destination graph.
    pub destination: Option<String>,
    /// Whether to silently ignore errors.
    pub silent: bool,
}

/// Move triples from one graph to another.
#[derive(Debug, Clone)]
pub struct MoveGraphOp {
    /// Source graph.
    pub source: Option<String>,
    /// Destination graph.
    pub destination: Option<String>,
    /// Whether to silently ignore errors.
    pub silent: bool,
}

/// Add (merge) triples from one graph to another.
#[derive(Debug, Clone)]
pub struct AddGraphOp {
    /// Source graph.
    pub source: Option<String>,
    /// Destination graph.
    pub destination: Option<String>,
    /// Whether to silently ignore errors.
    pub silent: bool,
}

// ==================== Vector Search Operators ====================

/// Vector similarity scan operation.
///
/// Performs approximate nearest neighbor search using a vector index (HNSW)
/// or brute-force search for small datasets. Returns nodes/edges whose
/// embeddings are similar to the query vector.
///
/// # Example GQL
///
/// ```gql
/// MATCH (m:Movie)
/// WHERE vector_similarity(m.embedding, $query_vector) > 0.8
/// RETURN m.title
/// ```
#[derive(Debug, Clone)]
pub struct VectorScanOp {
    /// Variable name to bind matching entities to.
    pub variable: String,
    /// Name of the vector index to use (None = brute-force).
    pub index_name: Option<String>,
    /// Property containing the vector embedding.
    pub property: String,
    /// Optional label filter (scan only nodes with this label).
    pub label: Option<String>,
    /// The query vector expression.
    pub query_vector: LogicalExpression,
    /// Number of nearest neighbors to return.
    pub k: usize,
    /// Distance metric (None = use index default, typically cosine).
    pub metric: Option<VectorMetric>,
    /// Minimum similarity threshold (filters results below this).
    pub min_similarity: Option<f32>,
    /// Maximum distance threshold (filters results above this).
    pub max_distance: Option<f32>,
    /// Input operator (for hybrid queries combining graph + vector).
    pub input: Option<Box<LogicalOperator>>,
}

/// Vector distance/similarity metric for vector scan operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorMetric {
    /// Cosine similarity (1 - cosine_distance). Best for normalized embeddings.
    Cosine,
    /// Euclidean (L2) distance. Best when magnitude matters.
    Euclidean,
    /// Dot product. Best for maximum inner product search.
    DotProduct,
    /// Manhattan (L1) distance. Less sensitive to outliers.
    Manhattan,
}

/// Join graph patterns with vector similarity search.
///
/// This operator takes entities from the left input and computes vector
/// similarity against a query vector, outputting (entity, distance) pairs.
///
/// # Use Cases
///
/// 1. **Hybrid graph + vector queries**: Find similar nodes after graph traversal
/// 2. **Aggregated embeddings**: Use AVG(embeddings) as query vector
/// 3. **Filtering by similarity**: Join with threshold-based filtering
///
/// # Example
///
/// ```gql
/// // Find movies similar to what the user liked
/// MATCH (u:User {id: $user_id})-[:LIKED]->(liked:Movie)
/// WITH avg(liked.embedding) AS user_taste
/// VECTOR JOIN (m:Movie) ON m.embedding
/// WHERE vector_similarity(m.embedding, user_taste) > 0.7
/// RETURN m.title
/// ```
#[derive(Debug, Clone)]
pub struct VectorJoinOp {
    /// Input operator providing entities to match against.
    pub input: Box<LogicalOperator>,
    /// Variable from input to extract vectors from (for entity-to-entity similarity).
    /// If None, uses `query_vector` directly.
    pub left_vector_variable: Option<String>,
    /// Property containing the left vector (used with `left_vector_variable`).
    pub left_property: Option<String>,
    /// The query vector expression (constant or computed).
    pub query_vector: LogicalExpression,
    /// Variable name to bind the right-side matching entities.
    pub right_variable: String,
    /// Property containing the right-side vector embeddings.
    pub right_property: String,
    /// Optional label filter for right-side entities.
    pub right_label: Option<String>,
    /// Name of vector index on right side (None = brute-force).
    pub index_name: Option<String>,
    /// Number of nearest neighbors per left-side entity.
    pub k: usize,
    /// Distance metric.
    pub metric: Option<VectorMetric>,
    /// Minimum similarity threshold.
    pub min_similarity: Option<f32>,
    /// Maximum distance threshold.
    pub max_distance: Option<f32>,
    /// Variable to bind the distance/similarity score.
    pub score_variable: Option<String>,
}

/// Return results (terminal operator).
#[derive(Debug, Clone)]
pub struct ReturnOp {
    /// Items to return.
    pub items: Vec<ReturnItem>,
    /// Whether to return distinct results.
    pub distinct: bool,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// A single return item.
#[derive(Debug, Clone)]
pub struct ReturnItem {
    /// Expression to return.
    pub expression: LogicalExpression,
    /// Alias for the result column.
    pub alias: Option<String>,
}

/// A logical expression.
#[derive(Debug, Clone)]
pub enum LogicalExpression {
    /// A literal value.
    Literal(Value),

    /// A variable reference.
    Variable(String),

    /// Property access (e.g., n.name).
    Property {
        /// The variable to access.
        variable: String,
        /// The property name.
        property: String,
    },

    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<LogicalExpression>,
        /// Operator.
        op: BinaryOp,
        /// Right operand.
        right: Box<LogicalExpression>,
    },

    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryOp,
        /// Operand.
        operand: Box<LogicalExpression>,
    },

    /// Function call.
    FunctionCall {
        /// Function name.
        name: String,
        /// Arguments.
        args: Vec<LogicalExpression>,
        /// Whether DISTINCT is applied (e.g., COUNT(DISTINCT x)).
        distinct: bool,
    },

    /// List literal.
    List(Vec<LogicalExpression>),

    /// Map literal (e.g., {name: 'Alice', age: 30}).
    Map(Vec<(String, LogicalExpression)>),

    /// Index access (e.g., `list[0]`).
    IndexAccess {
        /// The base expression (typically a list or string).
        base: Box<LogicalExpression>,
        /// The index expression.
        index: Box<LogicalExpression>,
    },

    /// Slice access (e.g., list[1..3]).
    SliceAccess {
        /// The base expression (typically a list or string).
        base: Box<LogicalExpression>,
        /// Start index (None means from beginning).
        start: Option<Box<LogicalExpression>>,
        /// End index (None means to end).
        end: Option<Box<LogicalExpression>>,
    },

    /// CASE expression.
    Case {
        /// Test expression (for simple CASE).
        operand: Option<Box<LogicalExpression>>,
        /// WHEN clauses.
        when_clauses: Vec<(LogicalExpression, LogicalExpression)>,
        /// ELSE clause.
        else_clause: Option<Box<LogicalExpression>>,
    },

    /// Parameter reference.
    Parameter(String),

    /// Labels of a node.
    Labels(String),

    /// Type of an edge.
    Type(String),

    /// ID of a node or edge.
    Id(String),

    /// List comprehension: [x IN list WHERE predicate | expression]
    ListComprehension {
        /// Variable name for each element.
        variable: String,
        /// The source list expression.
        list_expr: Box<LogicalExpression>,
        /// Optional filter predicate.
        filter_expr: Option<Box<LogicalExpression>>,
        /// The mapping expression for each element.
        map_expr: Box<LogicalExpression>,
    },

    /// EXISTS subquery.
    ExistsSubquery(Box<LogicalOperator>),

    /// COUNT subquery.
    CountSubquery(Box<LogicalOperator>),
}

/// Binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    /// Equality comparison (=).
    Eq,
    /// Inequality comparison (<>).
    Ne,
    /// Less than (<).
    Lt,
    /// Less than or equal (<=).
    Le,
    /// Greater than (>).
    Gt,
    /// Greater than or equal (>=).
    Ge,

    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Logical XOR.
    Xor,

    /// Addition (+).
    Add,
    /// Subtraction (-).
    Sub,
    /// Multiplication (*).
    Mul,
    /// Division (/).
    Div,
    /// Modulo (%).
    Mod,

    /// String concatenation.
    Concat,
    /// String starts with.
    StartsWith,
    /// String ends with.
    EndsWith,
    /// String contains.
    Contains,

    /// Collection membership (IN).
    In,
    /// Pattern matching (LIKE).
    Like,
    /// Regex matching (=~).
    Regex,
    /// Power/exponentiation (^).
    Pow,
}

/// Unary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Logical NOT.
    Not,
    /// Numeric negation.
    Neg,
    /// IS NULL check.
    IsNull,
    /// IS NOT NULL check.
    IsNotNull,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_node_scan_plan() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Variable("n".into()),
                alias: None,
            }],
            distinct: false,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".into(),
                label: Some("Person".into()),
                input: None,
            })),
        }));

        // Verify structure
        if let LogicalOperator::Return(ret) = &plan.root {
            assert_eq!(ret.items.len(), 1);
            assert!(!ret.distinct);
            if let LogicalOperator::NodeScan(scan) = ret.input.as_ref() {
                assert_eq!(scan.variable, "n");
                assert_eq!(scan.label, Some("Person".into()));
            } else {
                panic!("Expected NodeScan");
            }
        } else {
            panic!("Expected Return");
        }
    }

    #[test]
    fn test_filter_plan() {
        let plan = LogicalPlan::new(LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: LogicalExpression::Property {
                    variable: "n".into(),
                    property: "name".into(),
                },
                alias: Some("name".into()),
            }],
            distinct: false,
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".into(),
                        property: "age".into(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".into(),
                    label: Some("Person".into()),
                    input: None,
                })),
            })),
        }));

        if let LogicalOperator::Return(ret) = &plan.root {
            if let LogicalOperator::Filter(filter) = ret.input.as_ref() {
                if let LogicalExpression::Binary { op, .. } = &filter.predicate {
                    assert_eq!(*op, BinaryOp::Gt);
                } else {
                    panic!("Expected Binary expression");
                }
            } else {
                panic!("Expected Filter");
            }
        } else {
            panic!("Expected Return");
        }
    }
}
