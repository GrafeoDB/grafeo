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
