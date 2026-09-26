//! Logical query plan representation.
//!
//! The logical plan is the intermediate representation between parsed queries
//! and physical execution. Both GQL and Cypher queries are translated to this
//! common representation.

use std::collections::HashMap;
use std::fmt;

use grafeo_common::types::Value;

/// A count expression for SKIP/LIMIT: either a resolved literal or an unresolved parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CountExpr {
    /// A resolved integer count.
    Literal(usize),
    /// An unresolved parameter reference (e.g., `$limit`).
    Parameter(String),
}

impl CountExpr {
    /// Returns the resolved count, or panics if still a parameter reference.
    ///
    /// Call this only after parameter substitution has run.
    ///
    /// # Panics
    ///
    /// Panics if the expression is an unresolved `Parameter` reference.
    pub fn value(&self) -> usize {
        match self {
            Self::Literal(n) => *n,
            Self::Parameter(name) => panic!("Unresolved parameter: ${name}"),
        }
    }

    /// Returns the resolved count, or an error if still a parameter reference.
    ///
    /// # Errors
    ///
    /// Returns an error string if the expression is an unresolved `Parameter`.
    pub fn try_value(&self) -> Result<usize, String> {
        match self {
            Self::Literal(n) => Ok(*n),
            Self::Parameter(name) => Err(format!("Unresolved SKIP/LIMIT parameter: ${name}")),
        }
    }

    /// Returns the count as f64 for cardinality estimation (defaults to 10 for unresolved params).
    pub fn estimate(&self) -> f64 {
        match self {
            Self::Literal(n) => *n as f64,
            Self::Parameter(_) => 10.0, // reasonable default for unresolved params
        }
    }
}

impl fmt::Display for CountExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Literal(n) => write!(f, "{n}"),
            Self::Parameter(name) => write!(f, "${name}"),
        }
    }
}

impl From<usize> for CountExpr {
    fn from(n: usize) -> Self {
        Self::Literal(n)
    }
}

impl PartialEq<usize> for CountExpr {
    fn eq(&self, other: &usize) -> bool {
        matches!(self, Self::Literal(n) if n == other)
    }
}

/// A logical query plan.
#[derive(Debug, Clone)]
pub struct LogicalPlan {
    /// The root operator of the plan.
    pub root: LogicalOperator,
    /// When true, return the plan tree as text instead of executing.
    pub explain: bool,
    /// When true, execute the query and return per-operator runtime metrics.
    pub profile: bool,
    /// Default parameter values from variable declarations (e.g., GraphQL
    /// `query($limit: Int = 2)`). The processor merges these with caller-supplied
    /// params, giving caller values higher precedence.
    pub default_params: HashMap<String, Value>,
}

impl LogicalPlan {
    /// Creates a new logical plan with the given root operator.
    pub fn new(root: LogicalOperator) -> Self {
        Self {
            root,
            explain: false,
            profile: false,
            default_params: HashMap::new(),
        }
    }

    /// Creates an EXPLAIN plan that returns the plan tree without executing.
    pub fn explain(root: LogicalOperator) -> Self {
        Self {
            root,
            explain: true,
            profile: false,
            default_params: HashMap::new(),
        }
    }

    /// Creates a PROFILE plan that executes and returns per-operator metrics.
    pub fn profile(root: LogicalOperator) -> Self {
        Self {
            root,
            explain: false,
            profile: true,
            default_params: HashMap::new(),
        }
    }
}

/// A logical operator in the query plan.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

    /// SPARQL CONSTRUCT: evaluate WHERE, substitute bindings into template,
    /// output (subject, predicate, object) columns.
    Construct(ConstructOp),

    /// Bind a variable to an expression.
    Bind(BindOp),

    /// Unwind a list into individual rows.
    Unwind(UnwindOp),

    /// Collect grouped key-value rows into a single Map value.
    /// Used for Gremlin `groupCount()` semantics.
    MapCollect(MapCollectOp),

    /// Merge a node pattern (match or create).
    Merge(MergeOp),

    /// Merge a relationship pattern (match or create).
    MergeRelationship(MergeRelationshipOp),

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

    /// Per-row aggregation over a list-valued column (horizontal aggregation, GE09).
    HorizontalAggregate(HorizontalAggregateOp),

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

    /// Scan using full-text search with BM25 scoring.
    TextScan(TextScanOp),

    // ==================== Set Operations ====================
    /// Set difference: rows in left that are not in right.
    Except(ExceptOp),

    /// Set intersection: rows common to all inputs.
    Intersect(IntersectOp),

    /// Fallback: use left result if non-empty, otherwise right.
    Otherwise(OtherwiseOp),

    // ==================== Correlated Subquery ====================
    /// Apply (lateral join): evaluate a subplan per input row.
    Apply(ApplyOp),

    /// Parameter scan: leaf of a correlated inner plan that receives values
    /// from the outer Apply operator. The column names match `ApplyOp.shared_variables`.
    ParameterScan(ParameterScanOp),

    // ==================== DDL Operators ====================
    /// Define a property graph schema (SQL/PGQ DDL).
    CreatePropertyGraph(CreatePropertyGraphOp),

    // ==================== Multi-Way Join ====================
    /// Multi-way join using worst-case optimal join (leapfrog).
    /// Used for cyclic patterns (triangles, cliques) with 3+ relations.
    MultiWayJoin(MultiWayJoinOp),

    // ==================== Procedure Call Operators ====================
    /// Invoke a stored procedure (CALL ... YIELD).
    CallProcedure(CallProcedureOp),

    // ==================== Data Import Operators ====================
    /// Load data from a file (CSV, JSONL, or Parquet), producing one row per record.
    LoadData(LoadDataOp),
}

impl LogicalOperator {
    /// Returns `true` if this operator or any of its children perform mutations.
    #[must_use]
    pub fn has_mutations(&self) -> bool {
        match self {
            // Direct mutation operators
            Self::CreateNode(_)
            | Self::CreateEdge(_)
            | Self::DeleteNode(_)
            | Self::DeleteEdge(_)
            | Self::SetProperty(_)
            | Self::AddLabel(_)
            | Self::RemoveLabel(_)
            | Self::Merge(_)
            | Self::MergeRelationship(_)
            | Self::InsertTriple(_)
            | Self::DeleteTriple(_)
            | Self::Modify(_)
            | Self::ClearGraph(_)
            | Self::CreateGraph(_)
            | Self::DropGraph(_)
            | Self::LoadGraph(_)
            | Self::CopyGraph(_)
            | Self::MoveGraph(_)
            | Self::AddGraph(_)
            | Self::CreatePropertyGraph(_) => true,

            // Operators with an `input` child
            Self::Filter(op) => op.input.has_mutations(),
            Self::Project(op) => op.input.has_mutations(),
            Self::Aggregate(op) => op.input.has_mutations(),
            Self::Limit(op) => op.input.has_mutations(),
            Self::Skip(op) => op.input.has_mutations(),
            Self::Sort(op) => op.input.has_mutations(),
            Self::Distinct(op) => op.input.has_mutations(),
            Self::Unwind(op) => op.input.has_mutations(),
            Self::Bind(op) => op.input.has_mutations(),
            Self::MapCollect(op) => op.input.has_mutations(),
            Self::Return(op) => op.input.has_mutations(),
            Self::HorizontalAggregate(op) => op.input.has_mutations(),
            Self::VectorScan(op) => op.input.as_deref().is_some_and(Self::has_mutations),
            Self::VectorJoin(op) => op.input.has_mutations(),
            Self::TextScan(_) => false,

            // Operators with two children
            Self::Join(op) => op.left.has_mutations() || op.right.has_mutations(),
            Self::LeftJoin(op) => op.left.has_mutations() || op.right.has_mutations(),
            Self::AntiJoin(op) => op.left.has_mutations() || op.right.has_mutations(),
            Self::Except(op) => op.left.has_mutations() || op.right.has_mutations(),
            Self::Intersect(op) => op.left.has_mutations() || op.right.has_mutations(),
            Self::Otherwise(op) => op.left.has_mutations() || op.right.has_mutations(),
            Self::Union(op) => op.inputs.iter().any(|i| i.has_mutations()),
            Self::MultiWayJoin(op) => op.inputs.iter().any(|i| i.has_mutations()),
            Self::Apply(op) => op.input.has_mutations() || op.subplan.has_mutations(),

            // Leaf operators (read-only)
            Self::NodeScan(_)
            | Self::EdgeScan(_)
            | Self::Expand(_)
            | Self::TripleScan(_)
            | Self::ShortestPath(_)
            | Self::Empty
            | Self::ParameterScan(_)
            | Self::CallProcedure(_)
            | Self::LoadData(_) => false,
            Self::Construct(op) => op.input.has_mutations(),
        }
    }

    /// Returns references to the child operators.
    ///
    /// Used by [`crate::query::profile::build_profile_tree`] to walk the logical
    /// plan tree in post-order, matching operators to profiling entries.
    #[must_use]
    pub fn children(&self) -> Vec<&LogicalOperator> {
        match self {
            // Optional single input
            Self::NodeScan(op) => op.input.as_deref().into_iter().collect(),
            Self::EdgeScan(op) => op.input.as_deref().into_iter().collect(),
            Self::TripleScan(op) => op.input.as_deref().into_iter().collect(),
            Self::VectorScan(op) => op.input.as_deref().into_iter().collect(),
            Self::CreateNode(op) => op.input.as_deref().into_iter().collect(),
            Self::InsertTriple(op) => op.input.as_deref().into_iter().collect(),
            Self::DeleteTriple(op) => op.input.as_deref().into_iter().collect(),

            // Single required input
            Self::Expand(op) => vec![&*op.input],
            Self::Filter(op) => vec![&*op.input],
            Self::Project(op) => vec![&*op.input],
            Self::Aggregate(op) => vec![&*op.input],
            Self::Limit(op) => vec![&*op.input],
            Self::Skip(op) => vec![&*op.input],
            Self::Sort(op) => vec![&*op.input],
            Self::Distinct(op) => vec![&*op.input],
            Self::Return(op) => vec![&*op.input],
            Self::Unwind(op) => vec![&*op.input],
            Self::Bind(op) => vec![&*op.input],
            Self::Construct(op) => vec![&*op.input],
            Self::MapCollect(op) => vec![&*op.input],
            Self::ShortestPath(op) => vec![&*op.input],
            Self::Merge(op) => vec![&*op.input],
            Self::MergeRelationship(op) => vec![&*op.input],
            Self::CreateEdge(op) => vec![&*op.input],
            Self::DeleteNode(op) => vec![&*op.input],
            Self::DeleteEdge(op) => vec![&*op.input],
            Self::SetProperty(op) => vec![&*op.input],
            Self::AddLabel(op) => vec![&*op.input],
            Self::RemoveLabel(op) => vec![&*op.input],
            Self::HorizontalAggregate(op) => vec![&*op.input],
            Self::VectorJoin(op) => vec![&*op.input],
            Self::Modify(op) => vec![&*op.where_clause],

            // Two children (left + right)
            Self::Join(op) => vec![&*op.left, &*op.right],
            Self::LeftJoin(op) => vec![&*op.left, &*op.right],
            Self::AntiJoin(op) => vec![&*op.left, &*op.right],
            Self::Except(op) => vec![&*op.left, &*op.right],
            Self::Intersect(op) => vec![&*op.left, &*op.right],
            Self::Otherwise(op) => vec![&*op.left, &*op.right],

            // Two children (input + subplan)
            Self::Apply(op) => vec![&*op.input, &*op.subplan],

            // Vec children
            Self::Union(op) => op.inputs.iter().collect(),
            Self::MultiWayJoin(op) => op.inputs.iter().collect(),

            // Leaf operators
            Self::Empty
            | Self::ParameterScan(_)
            | Self::CallProcedure(_)
            | Self::ClearGraph(_)
            | Self::CreateGraph(_)
            | Self::DropGraph(_)
            | Self::LoadGraph(_)
            | Self::CopyGraph(_)
            | Self::MoveGraph(_)
            | Self::AddGraph(_)
            | Self::CreatePropertyGraph(_)
            | Self::LoadData(_)
            | Self::TextScan(_) => vec![],
        }
    }

    /// Returns a new `LogicalOperator` with each child replaced by `f(child)`.
    ///
    /// Mirrors [`Self::children`] in arm coverage; any new operator variant
    /// must extend both. Child-recursive optimizer passes (e.g. predicate
    /// propagation) call this to descend without enumerating every variant
    /// at every call site, eliminating a "forgot to recurse into the new
    /// variant" bug class.
    #[must_use]
    pub fn map_children<F: FnMut(LogicalOperator) -> LogicalOperator>(self, mut f: F) -> Self {
        match self {
            // Optional single input
            Self::NodeScan(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::NodeScan(op)
            }
            Self::EdgeScan(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::EdgeScan(op)
            }
            Self::TripleScan(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::TripleScan(op)
            }
            Self::VectorScan(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::VectorScan(op)
            }
            Self::CreateNode(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::CreateNode(op)
            }
            Self::InsertTriple(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::InsertTriple(op)
            }
            Self::DeleteTriple(mut op) => {
                op.input = op.input.map(|i| Box::new(f(*i)));
                Self::DeleteTriple(op)
            }

            // Single required input
            Self::Expand(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Expand(op)
            }
            Self::Filter(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Filter(op)
            }
            Self::Project(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Project(op)
            }
            Self::Aggregate(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Aggregate(op)
            }
            Self::Limit(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Limit(op)
            }
            Self::Skip(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Skip(op)
            }
            Self::Sort(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Sort(op)
            }
            Self::Distinct(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Distinct(op)
            }
            Self::Return(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Return(op)
            }
            Self::Unwind(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Unwind(op)
            }
            Self::Bind(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Bind(op)
            }
            Self::Construct(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Construct(op)
            }
            Self::MapCollect(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::MapCollect(op)
            }
            Self::ShortestPath(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::ShortestPath(op)
            }
            Self::Merge(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::Merge(op)
            }
            Self::MergeRelationship(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::MergeRelationship(op)
            }
            Self::CreateEdge(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::CreateEdge(op)
            }
            Self::DeleteNode(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::DeleteNode(op)
            }
            Self::DeleteEdge(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::DeleteEdge(op)
            }
            Self::SetProperty(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::SetProperty(op)
            }
            Self::AddLabel(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::AddLabel(op)
            }
            Self::RemoveLabel(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::RemoveLabel(op)
            }
            Self::HorizontalAggregate(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::HorizontalAggregate(op)
            }
            Self::VectorJoin(mut op) => {
                op.input = Box::new(f(*op.input));
                Self::VectorJoin(op)
            }
            Self::Modify(mut op) => {
                op.where_clause = Box::new(f(*op.where_clause));
                Self::Modify(op)
            }

            // Two children (left + right)
            Self::Join(mut op) => {
                op.left = Box::new(f(*op.left));
                op.right = Box::new(f(*op.right));
                Self::Join(op)
            }
            Self::LeftJoin(mut op) => {
                op.left = Box::new(f(*op.left));
                op.right = Box::new(f(*op.right));
                Self::LeftJoin(op)
            }
            Self::AntiJoin(mut op) => {
                op.left = Box::new(f(*op.left));
                op.right = Box::new(f(*op.right));
                Self::AntiJoin(op)
            }
            Self::Except(mut op) => {
                op.left = Box::new(f(*op.left));
                op.right = Box::new(f(*op.right));
                Self::Except(op)
            }
            Self::Intersect(mut op) => {
                op.left = Box::new(f(*op.left));
                op.right = Box::new(f(*op.right));
                Self::Intersect(op)
            }
            Self::Otherwise(mut op) => {
                op.left = Box::new(f(*op.left));
                op.right = Box::new(f(*op.right));
                Self::Otherwise(op)
            }

            // Two children (input + subplan)
            Self::Apply(mut op) => {
                op.input = Box::new(f(*op.input));
                op.subplan = Box::new(f(*op.subplan));
                Self::Apply(op)
            }

            // Vec children
            Self::Union(mut op) => {
                op.inputs = op.inputs.into_iter().map(&mut f).collect();
                Self::Union(op)
            }
            Self::MultiWayJoin(mut op) => {
                op.inputs = op.inputs.into_iter().map(&mut f).collect();
                Self::MultiWayJoin(op)
            }

            // Leaf operators
            leaf @ (Self::Empty
            | Self::ParameterScan(_)
            | Self::CallProcedure(_)
            | Self::ClearGraph(_)
            | Self::CreateGraph(_)
            | Self::DropGraph(_)
            | Self::LoadGraph(_)
            | Self::CopyGraph(_)
            | Self::MoveGraph(_)
            | Self::AddGraph(_)
            | Self::CreatePropertyGraph(_)
            | Self::LoadData(_)
            | Self::TextScan(_)) => leaf,
        }
    }

    /// Returns a compact display label for this operator, used in PROFILE output.
    #[must_use]
    pub fn display_label(&self) -> String {
        match self {
            Self::NodeScan(op) => {
                let label = op.label.as_deref().unwrap_or("*");
                format!("{}:{}", op.variable, label)
            }
            Self::EdgeScan(op) => {
                let types = if op.edge_types.is_empty() {
                    "*".to_string()
                } else {
                    op.edge_types.join("|")
                };
                format!("{}:{}", op.variable, types)
            }
            Self::Expand(op) => {
                let types = if op.edge_types.is_empty() {
                    "*".to_string()
                } else {
                    op.edge_types.join("|")
                };
                let dir = match op.direction {
                    ExpandDirection::Outgoing => "->",
                    ExpandDirection::Incoming => "<-",
                    ExpandDirection::Both => "--",
                };
                format!(
                    "({from}){dir}[:{types}]{dir}({to})",
                    from = op.from_variable,
                    to = op.to_variable,
                )
            }
            Self::Filter(op) => {
                let hint = match &op.pushdown_hint {
                    Some(PushdownHint::IndexLookup { property }) => {
                        format!(" [index: {property}]")
                    }
                    Some(PushdownHint::RangeScan { property }) => {
                        format!(" [range: {property}]")
                    }
                    Some(PushdownHint::LabelFirst) => " [label-first]".to_string(),
                    None => String::new(),
                };
                format!("{}{hint}", fmt_expr(&op.predicate))
            }
            Self::Project(op) => {
                let cols: Vec<String> = op
                    .projections
                    .iter()
                    .map(|p| match &p.alias {
                        Some(alias) => alias.clone(),
                        None => fmt_expr(&p.expression),
                    })
                    .collect();
                cols.join(", ")
            }
            Self::Join(op) => format!("{:?}", op.join_type),
            Self::Aggregate(op) => {
                let groups: Vec<String> = op.group_by.iter().map(fmt_expr).collect();
                format!("group: [{}]", groups.join(", "))
            }
            Self::Limit(op) => format!("{}", op.count),
            Self::Skip(op) => format!("{}", op.count),
            Self::Sort(op) => {
                let keys: Vec<String> = op
                    .keys
                    .iter()
                    .map(|k| {
                        let dir = match k.order {
                            SortOrder::Ascending => "ASC",
                            SortOrder::Descending => "DESC",
                        };
                        format!("{} {dir}", fmt_expr(&k.expression))
                    })
                    .collect();
                keys.join(", ")
            }
            Self::Distinct(_) => String::new(),
            Self::Return(op) => {
                let items: Vec<String> = op
                    .items
                    .iter()
                    .map(|item| match &item.alias {
                        Some(alias) => alias.clone(),
                        None => fmt_expr(&item.expression),
                    })
                    .collect();
                items.join(", ")
            }
            Self::Union(op) => format!("{} branches", op.inputs.len()),
            Self::MultiWayJoin(op) => {
                format!("{} inputs", op.inputs.len())
            }
            Self::LeftJoin(_) => String::new(),
            Self::AntiJoin(_) => String::new(),
            Self::Unwind(op) => op.variable.clone(),
            Self::Bind(op) => op.variable.clone(),
            Self::MapCollect(op) => op.alias.clone(),
            Self::ShortestPath(op) => {
                format!("{} -> {}", op.source_var, op.target_var)
            }
            Self::Merge(op) => op.variable.clone(),
            Self::MergeRelationship(op) => op.variable.clone(),
            Self::CreateNode(op) => {
                let labels = op.labels.join(":");
                format!("{}:{labels}", op.variable)
            }
            Self::CreateEdge(op) => {
                format!(
                    "[{}:{}]",
                    op.variable.as_deref().unwrap_or("?"),
                    op.edge_type
                )
            }
            Self::DeleteNode(op) => op.variable.clone(),
            Self::DeleteEdge(op) => op.variable.clone(),
            Self::SetProperty(op) => op.variable.clone(),
            Self::AddLabel(op) => {
                let labels = op.labels.join(":");
                format!("{}:{labels}", op.variable)
            }
            Self::RemoveLabel(op) => {
                let labels = op.labels.join(":");
                format!("{}:{labels}", op.variable)
            }
            Self::CallProcedure(op) => op.name.join("."),
            Self::LoadData(op) => format!("{} AS {}", op.path, op.variable),
            Self::Apply(_) => String::new(),
            Self::VectorScan(op) => op.variable.clone(),
            Self::VectorJoin(op) => op.right_variable.clone(),
            Self::TextScan(op) => format!("{}:{}", op.variable, op.label),
            _ => String::new(),
        }
    }
}

impl LogicalOperator {
    /// Formats this operator tree as a human-readable plan for EXPLAIN output.
    pub fn explain_tree(&self) -> String {
        let mut output = String::new();
        self.fmt_tree(&mut output, 0);
        output
    }

    fn fmt_tree(&self, out: &mut String, depth: usize) {
        use std::fmt::Write;

        let indent = "  ".repeat(depth);
        match self {
            Self::NodeScan(op) => {
                let label = op.label.as_deref().unwrap_or("*");
                let _ = writeln!(out, "{indent}NodeScan ({var}:{label})", var = op.variable);
                if let Some(input) = &op.input {
                    input.fmt_tree(out, depth + 1);
                }
            }
            Self::EdgeScan(op) => {
                let types = if op.edge_types.is_empty() {
                    "*".to_string()
                } else {
                    op.edge_types.join("|")
                };
                let _ = writeln!(out, "{indent}EdgeScan ({var}:{types})", var = op.variable);
            }
            Self::Expand(op) => {
                let types = if op.edge_types.is_empty() {
                    "*".to_string()
                } else {
                    op.edge_types.join("|")
                };
                let dir = match op.direction {
                    ExpandDirection::Outgoing => "->",
                    ExpandDirection::Incoming => "<-",
                    ExpandDirection::Both => "--",
                };
                let hops = match (op.min_hops, op.max_hops) {
                    (1, Some(1)) => String::new(),
                    (min, Some(max)) if min == max => format!("*{min}"),
                    (min, Some(max)) => format!("*{min}..{max}"),
                    (min, None) => format!("*{min}.."),
                };
                let _ = writeln!(
                    out,
                    "{indent}Expand ({from}){dir}[:{types}{hops}]{dir}({to})",
                    from = op.from_variable,
                    to = op.to_variable,
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Filter(op) => {
                let hint = match &op.pushdown_hint {
                    Some(PushdownHint::IndexLookup { property }) => {
                        format!(" [index: {property}]")
                    }
                    Some(PushdownHint::RangeScan { property }) => {
                        format!(" [range: {property}]")
                    }
                    Some(PushdownHint::LabelFirst) => " [label-first]".to_string(),
                    None => String::new(),
                };
                let _ = writeln!(
                    out,
                    "{indent}Filter ({expr}){hint}",
                    expr = fmt_expr(&op.predicate)
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Project(op) => {
                let cols: Vec<String> = op
                    .projections
                    .iter()
                    .map(|p| {
                        let expr = fmt_expr(&p.expression);
                        match &p.alias {
                            Some(alias) => format!("{expr} AS {alias}"),
                            None => expr,
                        }
                    })
                    .collect();
                let _ = writeln!(out, "{indent}Project ({cols})", cols = cols.join(", "));
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Join(op) => {
                let _ = writeln!(out, "{indent}Join ({ty:?})", ty = op.join_type);
                op.left.fmt_tree(out, depth + 1);
                op.right.fmt_tree(out, depth + 1);
            }
            Self::Aggregate(op) => {
                let groups: Vec<String> = op.group_by.iter().map(fmt_expr).collect();
                let aggs: Vec<String> = op
                    .aggregates
                    .iter()
                    .map(|a| {
                        let func = format!("{:?}", a.function).to_lowercase();
                        match &a.alias {
                            Some(alias) => format!("{func}(...) AS {alias}"),
                            None => format!("{func}(...)"),
                        }
                    })
                    .collect();
                let _ = writeln!(
                    out,
                    "{indent}Aggregate (group: [{groups}], aggs: [{aggs}])",
                    groups = groups.join(", "),
                    aggs = aggs.join(", "),
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Limit(op) => {
                let _ = writeln!(out, "{indent}Limit ({})", op.count);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Skip(op) => {
                let _ = writeln!(out, "{indent}Skip ({})", op.count);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Sort(op) => {
                let keys: Vec<String> = op
                    .keys
                    .iter()
                    .map(|k| {
                        let dir = match k.order {
                            SortOrder::Ascending => "ASC",
                            SortOrder::Descending => "DESC",
                        };
                        format!("{} {dir}", fmt_expr(&k.expression))
                    })
                    .collect();
                let _ = writeln!(out, "{indent}Sort ({keys})", keys = keys.join(", "));
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Distinct(op) => {
                let _ = writeln!(out, "{indent}Distinct");
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Return(op) => {
                let items: Vec<String> = op
                    .items
                    .iter()
                    .map(|item| {
                        let expr = fmt_expr(&item.expression);
                        match &item.alias {
                            Some(alias) => format!("{expr} AS {alias}"),
                            None => expr,
                        }
                    })
                    .collect();
                let distinct = if op.distinct { " DISTINCT" } else { "" };
                let _ = writeln!(
                    out,
                    "{indent}Return{distinct} ({items})",
                    items = items.join(", ")
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Union(op) => {
                let _ = writeln!(out, "{indent}Union ({n} branches)", n = op.inputs.len());
                for input in &op.inputs {
                    input.fmt_tree(out, depth + 1);
                }
            }
            Self::MultiWayJoin(op) => {
                let vars = op.shared_variables.join(", ");
                let _ = writeln!(
                    out,
                    "{indent}MultiWayJoin ({n} inputs, shared: [{vars}])",
                    n = op.inputs.len()
                );
                for input in &op.inputs {
                    input.fmt_tree(out, depth + 1);
                }
            }
            Self::LeftJoin(op) => {
                if let Some(cond) = &op.condition {
                    let _ = writeln!(out, "{indent}LeftJoin (condition: {cond:?})");
                } else {
                    let _ = writeln!(out, "{indent}LeftJoin");
                }
                op.left.fmt_tree(out, depth + 1);
                op.right.fmt_tree(out, depth + 1);
            }
            Self::AntiJoin(op) => {
                let _ = writeln!(out, "{indent}AntiJoin");
                op.left.fmt_tree(out, depth + 1);
                op.right.fmt_tree(out, depth + 1);
            }
            Self::Unwind(op) => {
                let _ = writeln!(out, "{indent}Unwind ({var})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Bind(op) => {
                let _ = writeln!(out, "{indent}Bind ({var})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::MapCollect(op) => {
                let _ = writeln!(
                    out,
                    "{indent}MapCollect ({key} -> {val} AS {alias})",
                    key = op.key_var,
                    val = op.value_var,
                    alias = op.alias
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Apply(op) => {
                let _ = writeln!(out, "{indent}Apply");
                op.input.fmt_tree(out, depth + 1);
                op.subplan.fmt_tree(out, depth + 1);
            }
            Self::Except(op) => {
                let all = if op.all { " ALL" } else { "" };
                let _ = writeln!(out, "{indent}Except{all}");
                op.left.fmt_tree(out, depth + 1);
                op.right.fmt_tree(out, depth + 1);
            }
            Self::Intersect(op) => {
                let all = if op.all { " ALL" } else { "" };
                let _ = writeln!(out, "{indent}Intersect{all}");
                op.left.fmt_tree(out, depth + 1);
                op.right.fmt_tree(out, depth + 1);
            }
            Self::Otherwise(op) => {
                let _ = writeln!(out, "{indent}Otherwise");
                op.left.fmt_tree(out, depth + 1);
                op.right.fmt_tree(out, depth + 1);
            }
            Self::ShortestPath(op) => {
                let _ = writeln!(
                    out,
                    "{indent}ShortestPath ({from} -> {to})",
                    from = op.source_var,
                    to = op.target_var
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::Merge(op) => {
                let _ = writeln!(out, "{indent}Merge ({var})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::MergeRelationship(op) => {
                let _ = writeln!(out, "{indent}MergeRelationship ({var})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::CreateNode(op) => {
                let labels = op.labels.join(":");
                let _ = writeln!(
                    out,
                    "{indent}CreateNode ({var}:{labels})",
                    var = op.variable
                );
                if let Some(input) = &op.input {
                    input.fmt_tree(out, depth + 1);
                }
            }
            Self::CreateEdge(op) => {
                let var = op.variable.as_deref().unwrap_or("?");
                let _ = writeln!(
                    out,
                    "{indent}CreateEdge ({from})-[{var}:{ty}]->({to})",
                    from = op.from_variable,
                    ty = op.edge_type,
                    to = op.to_variable
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::DeleteNode(op) => {
                let _ = writeln!(out, "{indent}DeleteNode ({var})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::DeleteEdge(op) => {
                let _ = writeln!(out, "{indent}DeleteEdge ({var})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::SetProperty(op) => {
                let props: Vec<String> = op
                    .properties
                    .iter()
                    .map(|(k, _)| format!("{}.{k}", op.variable))
                    .collect();
                let _ = writeln!(
                    out,
                    "{indent}SetProperty ({props})",
                    props = props.join(", ")
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::AddLabel(op) => {
                let labels = op.labels.join(":");
                let _ = writeln!(out, "{indent}AddLabel ({var}:{labels})", var = op.variable);
                op.input.fmt_tree(out, depth + 1);
            }
            Self::RemoveLabel(op) => {
                let labels = op.labels.join(":");
                let _ = writeln!(
                    out,
                    "{indent}RemoveLabel ({var}:{labels})",
                    var = op.variable
                );
                op.input.fmt_tree(out, depth + 1);
            }
            Self::CallProcedure(op) => {
                let _ = writeln!(
                    out,
                    "{indent}CallProcedure ({name})",
                    name = op.name.join(".")
                );
            }
            Self::LoadData(op) => {
                let format_name = match op.format {
                    LoadDataFormat::Csv => "LoadCsv",
                    LoadDataFormat::Jsonl => "LoadJsonl",
                    LoadDataFormat::Parquet => "LoadParquet",
                    _ => "LoadData",
                };
                let headers = if op.with_headers && op.format == LoadDataFormat::Csv {
                    " WITH HEADERS"
                } else {
                    ""
                };
                let _ = writeln!(
                    out,
                    "{indent}{format_name}{headers} ('{path}' AS {var})",
                    path = op.path,
                    var = op.variable,
                );
            }
            Self::TripleScan(op) => {
                let _ = writeln!(
                    out,
                    "{indent}TripleScan ({s} {p} {o})",
                    s = fmt_triple_component(&op.subject),
                    p = fmt_triple_component(&op.predicate),
                    o = fmt_triple_component(&op.object)
                );
                if let Some(input) = &op.input {
                    input.fmt_tree(out, depth + 1);
                }
            }
            Self::VectorScan(op) => {
                let metric = op.metric.map_or("default", |m| match m {
                    VectorMetric::Cosine => "cosine",
                    VectorMetric::Euclidean => "euclidean",
                    VectorMetric::DotProduct => "dot_product",
                    VectorMetric::Manhattan => "manhattan",
                });
                let mode = match op.k {
                    Some(k) => format!("top-{k}"),
                    None => "threshold".to_string(),
                };
                let _ = writeln!(
                    out,
                    "{indent}VectorScan ({var}:{label}.{prop}, {metric}, {mode})",
                    var = op.variable,
                    label = op.label.as_deref().unwrap_or("*"),
                    prop = op.property,
                );
                if let Some(input) = &op.input {
                    input.fmt_tree(out, depth + 1);
                }
            }
            Self::TextScan(op) => {
                let mode = match (op.k, op.threshold) {
                    (Some(k), _) => format!("top-{k}"),
                    (None, Some(t)) => format!("threshold>={t}"),
                    (None, None) => "default-top-100".to_string(),
                };
                let query = fmt_expr(&op.query);
                let _ = writeln!(
                    out,
                    "{indent}TextScan ({var}:{label}.{prop}, query={query}, {mode})",
                    var = op.variable,
                    label = op.label,
                    prop = op.property,
                );
            }
            Self::Empty => {
                let _ = writeln!(out, "{indent}Empty");
            }
            // Remaining operators: show a simple name
            _ => {
                let _ = writeln!(out, "{indent}{:?}", std::mem::discriminant(self));
            }
        }
    }
}

/// Format a logical expression compactly for EXPLAIN output.
fn fmt_expr(expr: &LogicalExpression) -> String {
    match expr {
        LogicalExpression::Variable(name) => name.clone(),
        LogicalExpression::Property { variable, property } => format!("{variable}.{property}"),
        LogicalExpression::Literal(val) => format!("{val}"),
        LogicalExpression::Binary { left, op, right } => {
            format!("{} {op:?} {}", fmt_expr(left), fmt_expr(right))
        }
        LogicalExpression::Unary { op, operand } => {
            format!("{op:?} {}", fmt_expr(operand))
        }
        LogicalExpression::FunctionCall { name, args, .. } => {
            let arg_strs: Vec<String> = args.iter().map(fmt_expr).collect();
            format!("{name}({})", arg_strs.join(", "))
        }
        _ => format!("{expr:?}"),
    }
}

/// Format a triple component for EXPLAIN output.
fn fmt_triple_component(comp: &TripleComponent) -> String {
    match comp {
        TripleComponent::Variable(name) => format!("?{name}"),
        TripleComponent::Iri(iri) => format!("<{iri}>"),
        TripleComponent::Literal(val) => format!("{val}"),
        TripleComponent::LangLiteral { value, lang } => format!("\"{value}\"@{lang}"),
        TripleComponent::BlankNode(label) => format!("_:{label}"),
    }
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
    /// Edge type filter (empty = match all types).
    pub edge_types: Vec<String>,
    /// Child operator (if any).
    pub input: Option<Box<LogicalOperator>>,
}

/// Path traversal mode for variable-length expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum PathMode {
    /// Allows repeated nodes and edges (default).
    #[default]
    Walk,
    /// No repeated edges.
    Trail,
    /// No repeated nodes except endpoints.
    Simple,
    /// No repeated nodes at all.
    Acyclic,
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
    /// Edge type filter (empty = match all types, multiple = match any).
    pub edge_types: Vec<String>,
    /// Minimum hops (for variable-length patterns).
    pub min_hops: u32,
    /// Maximum hops (for variable-length patterns).
    pub max_hops: Option<u32>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
    /// Path alias for variable-length patterns (e.g., `p` in `p = (a)-[*1..3]->(b)`).
    /// When set, a path length column will be output under this name.
    pub path_alias: Option<String>,
    /// Path traversal mode (WALK, TRAIL, SIMPLE, ACYCLIC).
    pub path_mode: PathMode,
}

/// Direction for edge expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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

/// Multi-way join for worst-case optimal joins (leapfrog).
///
/// Unlike binary `JoinOp`, this joins 3+ relations simultaneously
/// using the leapfrog trie join algorithm. Preferred for cyclic patterns
/// (triangles, cliques) where cascading binary joins hit O(N^2).
#[derive(Debug, Clone)]
pub struct MultiWayJoinOp {
    /// Input relations (one per relation in the join).
    pub inputs: Vec<LogicalOperator>,
    /// All pairwise join conditions.
    pub conditions: Vec<JoinCondition>,
    /// Variables shared across multiple inputs (intersection keys).
    pub shared_variables: Vec<String>,
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

/// Whether a horizontal aggregate operates on edges or nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntityKind {
    /// Aggregate over edges in a path.
    Edge,
    /// Aggregate over nodes in a path.
    Node,
}

/// Per-row aggregation over a list-valued column (horizontal aggregation, GE09).
///
/// For each input row, reads a list of entity IDs from `list_column`, accesses
/// `property` on each entity, computes the aggregate, and emits the scalar result.
#[derive(Debug, Clone)]
pub struct HorizontalAggregateOp {
    /// The list column name (e.g., `_path_edges_p`).
    pub list_column: String,
    /// Whether the list contains edge IDs or node IDs.
    pub entity_kind: EntityKind,
    /// The aggregate function to apply.
    pub function: AggregateFunction,
    /// The property to access on each entity.
    pub property: String,
    /// Output alias for the result column.
    pub alias: String,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// An aggregate expression.
#[derive(Debug, Clone)]
pub struct AggregateExpr {
    /// Aggregate function.
    pub function: AggregateFunction,
    /// Expression to aggregate (first/only argument, y for binary set functions).
    pub expression: Option<LogicalExpression>,
    /// Second expression for binary set functions (x for COVAR, CORR, REGR_*).
    pub expression2: Option<LogicalExpression>,
    /// Whether to use DISTINCT.
    pub distinct: bool,
    /// Alias for the result.
    pub alias: Option<String>,
    /// Percentile parameter for PERCENTILE_DISC/PERCENTILE_CONT (0.0 to 1.0).
    pub percentile: Option<f64>,
    /// Separator string for GROUP_CONCAT / LISTAGG (defaults to space for GROUP_CONCAT, comma for LISTAGG).
    pub separator: Option<String>,
}

/// Aggregate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
    /// Sample variance (VAR_SAMP / VARIANCE).
    Variance,
    /// Population variance (VAR_POP).
    VariancePop,
    /// Discrete percentile (PERCENTILE_DISC).
    PercentileDisc,
    /// Continuous percentile (PERCENTILE_CONT).
    PercentileCont,
    /// Concatenate values with separator (GROUP_CONCAT).
    GroupConcat,
    /// Return an arbitrary value from the group (SAMPLE).
    Sample,
    /// Sample covariance (COVAR_SAMP(y, x)).
    CovarSamp,
    /// Population covariance (COVAR_POP(y, x)).
    CovarPop,
    /// Pearson correlation coefficient (CORR(y, x)).
    Corr,
    /// Regression slope (REGR_SLOPE(y, x)).
    RegrSlope,
    /// Regression intercept (REGR_INTERCEPT(y, x)).
    RegrIntercept,
    /// Coefficient of determination (REGR_R2(y, x)).
    RegrR2,
    /// Regression count of non-null pairs (REGR_COUNT(y, x)).
    RegrCount,
    /// Regression sum of squares for x (REGR_SXX(y, x)).
    RegrSxx,
    /// Regression sum of squares for y (REGR_SYY(y, x)).
    RegrSyy,
    /// Regression sum of cross-products (REGR_SXY(y, x)).
    RegrSxy,
    /// Regression average of x (REGR_AVGX(y, x)).
    RegrAvgx,
    /// Regression average of y (REGR_AVGY(y, x)).
    RegrAvgy,
}

/// Hint about how a filter will be executed at the physical level.
///
/// Set during EXPLAIN annotation to communicate pushdown decisions.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PushdownHint {
    /// Equality predicate resolved via a property index.
    IndexLookup {
        /// The indexed property name.
        property: String,
    },
    /// Range predicate resolved via a range/btree index.
    RangeScan {
        /// The indexed property name.
        property: String,
    },
    /// No index available, but label narrows the scan before filtering.
    LabelFirst,
}

/// Filter rows based on a predicate.
#[derive(Debug, Clone)]
pub struct FilterOp {
    /// The filter predicate.
    pub predicate: LogicalExpression,
    /// Input operator.
    pub input: Box<LogicalOperator>,
    /// Optional hint about pushdown strategy (populated by EXPLAIN).
    pub pushdown_hint: Option<PushdownHint>,
}

/// Project specific columns.
#[derive(Debug, Clone)]
pub struct ProjectOp {
    /// Columns to project.
    pub projections: Vec<Projection>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
    /// When true, all input columns are passed through and the explicit
    /// projections are appended as additional output columns. Used by GQL
    /// LET clauses which add bindings without replacing the existing scope.
    pub pass_through_input: bool,
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
    /// Maximum number of rows to return (literal or parameter reference).
    pub count: CountExpr,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Skip a number of results.
#[derive(Debug, Clone)]
pub struct SkipOp {
    /// Number of rows to skip (literal or parameter reference).
    pub count: CountExpr,
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
    /// Optional null ordering (NULLS FIRST / NULLS LAST).
    pub nulls: Option<NullsOrdering>,
}

/// Sort order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SortOrder {
    /// Ascending order.
    Ascending,
    /// Descending order.
    Descending,
}

/// Null ordering for sort operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum NullsOrdering {
    /// Nulls sort before all non-null values.
    First,
    /// Nulls sort after all non-null values.
    Last,
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
    /// Whether the target variable is an edge (vs. node).
    pub is_edge: bool,
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

/// SPARQL dataset restriction from FROM / FROM NAMED clauses.
///
/// When present, restricts which graphs are visible to a triple scan:
/// - `default_graphs`: IRIs whose union forms the default graph (basic patterns).
/// - `named_graphs`: IRIs that enumerate the available named graphs (GRAPH patterns).
#[derive(Debug, Clone, Default)]
pub struct DatasetRestriction {
    /// FROM IRIs: the default graph is the union of these named graphs.
    /// Empty means no FROM clause was specified (unrestricted default graph).
    pub default_graphs: Vec<String>,
    /// FROM NAMED IRIs: only these named graphs are available to GRAPH patterns.
    /// Empty means no FROM NAMED clause was specified (all named graphs visible).
    pub named_graphs: Vec<String>,
}

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
    /// Dataset restriction from SPARQL FROM / FROM NAMED clauses.
    pub dataset: Option<DatasetRestriction>,
}

/// A component of a triple pattern.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum TripleComponent {
    /// A variable to bind.
    Variable(String),
    /// A constant IRI.
    Iri(String),
    /// A constant literal value.
    Literal(Value),
    /// A language-tagged string literal (RDF `rdf:langString`).
    ///
    /// Carries the lexical value and the BCP47 language tag separately so that
    /// the tag survives the translator to planner to RDF store round-trip.
    LangLiteral {
        /// The lexical string value.
        value: String,
        /// BCP47 language tag, e.g. `"fr"`, `"en-GB"`.
        lang: String,
    },
    /// A blank node with a scoped label (used in INSERT DATA).
    BlankNode(String),
}

impl TripleComponent {
    /// Returns the variable name if this component is a `Variable`, or `None`.
    #[must_use]
    pub fn as_variable(&self) -> Option<&str> {
        match self {
            Self::Variable(v) => Some(v),
            _ => None,
        }
    }
}

/// Union of multiple result sets.
#[derive(Debug, Clone)]
pub struct UnionOp {
    /// Inputs to union together.
    pub inputs: Vec<LogicalOperator>,
}

/// Set difference: rows in left that are not in right.
#[derive(Debug, Clone)]
pub struct ExceptOp {
    /// Left input.
    pub left: Box<LogicalOperator>,
    /// Right input (rows to exclude).
    pub right: Box<LogicalOperator>,
    /// If true, preserve duplicates (EXCEPT ALL); if false, deduplicate (EXCEPT DISTINCT).
    pub all: bool,
}

/// Set intersection: rows common to both inputs.
#[derive(Debug, Clone)]
pub struct IntersectOp {
    /// Left input.
    pub left: Box<LogicalOperator>,
    /// Right input.
    pub right: Box<LogicalOperator>,
    /// If true, preserve duplicates (INTERSECT ALL); if false, deduplicate (INTERSECT DISTINCT).
    pub all: bool,
}

/// Fallback operator: use left result if non-empty, otherwise use right.
#[derive(Debug, Clone)]
pub struct OtherwiseOp {
    /// Primary input (preferred).
    pub left: Box<LogicalOperator>,
    /// Fallback input (used only if left produces zero rows).
    pub right: Box<LogicalOperator>,
}

/// Apply (lateral join): evaluate a subplan for each row of the outer input.
///
/// The subplan can reference variables bound by the outer input. Results are
/// concatenated (cross-product per row).
#[derive(Debug, Clone)]
pub struct ApplyOp {
    /// Outer input providing rows.
    pub input: Box<LogicalOperator>,
    /// Subplan to evaluate per outer row.
    pub subplan: Box<LogicalOperator>,
    /// Variables imported from the outer scope into the inner plan.
    /// When non-empty, the planner injects these via `ParameterState`.
    pub shared_variables: Vec<String>,
    /// When true, uses left-join semantics: outer rows with no matching inner
    /// rows are emitted with NULLs for the inner columns (OPTIONAL CALL).
    pub optional: bool,
}

/// Parameter scan: leaf operator for correlated subquery inner plans.
///
/// Emits a single row containing the values injected from the outer Apply.
/// Column names correspond to the outer variables imported via WITH.
#[derive(Debug, Clone)]
pub struct ParameterScanOp {
    /// Column names for the injected parameters.
    pub columns: Vec<String>,
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
    /// Optional variable for 1-based element position (ORDINALITY).
    pub ordinality_var: Option<String>,
    /// Optional variable for 0-based element position (OFFSET).
    pub offset_var: Option<String>,
    /// Input operator.
    pub input: Box<LogicalOperator>,
}

/// Collect grouped key-value rows into a single Map value.
/// Used for Gremlin `groupCount()` semantics.
#[derive(Debug, Clone)]
pub struct MapCollectOp {
    /// Variable holding the map key.
    pub key_var: String,
    /// Variable holding the map value.
    pub value_var: String,
    /// Output variable alias.
    pub alias: String,
    /// Input operator (typically a grouped aggregate).
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

/// Merge a relationship pattern (match or create between two bound nodes).
///
/// MERGE on a relationship tries to find an existing relationship of the given type
/// between the source and target nodes. If found, returns the existing relationship
/// (optionally applying ON MATCH SET). If not found, creates it (optionally applying
/// ON CREATE SET).
#[derive(Debug, Clone)]
pub struct MergeRelationshipOp {
    /// Variable to bind the relationship to.
    pub variable: String,
    /// Source node variable (must already be bound).
    pub source_variable: String,
    /// Target node variable (must already be bound).
    pub target_variable: String,
    /// Relationship type.
    pub edge_type: String,
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
    /// Edge type filter (empty = match all types, multiple = match any).
    pub edge_types: Vec<String>,
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

/// SPARQL CONSTRUCT: evaluate WHERE, substitute bindings into template.
///
/// Produces rows with columns `subject`, `predicate`, `object` by instantiating
/// the template once per binding from the WHERE clause.
#[derive(Debug, Clone)]
pub struct ConstructOp {
    /// Triple templates to instantiate.
    pub templates: Vec<TripleTemplate>,
    /// Input operator (WHERE clause evaluation).
    pub input: Box<LogicalOperator>,
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
    /// Number of nearest neighbors to return (None = threshold mode only).
    pub k: Option<usize>,
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
#[non_exhaustive]
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

/// Text search scan using BM25 inverted index.
#[derive(Debug, Clone)]
pub struct TextScanOp {
    /// Variable to bind matched nodes.
    pub variable: String,
    /// Label of nodes to search.
    pub label: String,
    /// Property holding the text to search.
    pub property: String,
    /// The search query expression (must resolve to a string).
    pub query: LogicalExpression,
    /// Top-k limit (None = threshold mode or default 100).
    pub k: Option<usize>,
    /// Minimum score threshold (None = top-k mode).
    pub threshold: Option<f64>,
    /// Optional column name to bind the BM25 score.
    pub score_column: Option<String>,
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

/// Define a property graph schema (SQL/PGQ DDL).
#[derive(Debug, Clone)]
pub struct CreatePropertyGraphOp {
    /// Graph name.
    pub name: String,
    /// Node table schemas (label name + column definitions).
    pub node_tables: Vec<PropertyGraphNodeTable>,
    /// Edge table schemas (type name + column definitions + references).
    pub edge_tables: Vec<PropertyGraphEdgeTable>,
}

/// A node table in a property graph definition.
#[derive(Debug, Clone)]
pub struct PropertyGraphNodeTable {
    /// Table name (maps to a node label).
    pub name: String,
    /// Column definitions as (name, type_name) pairs.
    pub columns: Vec<(String, String)>,
}

/// An edge table in a property graph definition.
#[derive(Debug, Clone)]
pub struct PropertyGraphEdgeTable {
    /// Table name (maps to an edge type).
    pub name: String,
    /// Column definitions as (name, type_name) pairs.
    pub columns: Vec<(String, String)>,
    /// Source node table name.
    pub source_table: String,
    /// Target node table name.
    pub target_table: String,
}

// ==================== Procedure Call Types ====================

/// A CALL procedure operation.
///
/// ```text
/// CALL grafeo.pagerank({damping: 0.85}) YIELD nodeId, score
/// ```
#[derive(Debug, Clone)]
pub struct CallProcedureOp {
    /// Dotted procedure name, e.g. `["grafeo", "pagerank"]`.
    pub name: Vec<String>,
    /// Argument expressions (constants in Phase 1).
    pub arguments: Vec<LogicalExpression>,
    /// Optional YIELD clause: which columns to expose + aliases.
    pub yield_items: Option<Vec<ProcedureYield>>,
}

/// A single YIELD item in a procedure call.
#[derive(Debug, Clone)]
pub struct ProcedureYield {
    /// Column name from the procedure result.
    pub field_name: String,
    /// Optional alias (YIELD score AS rank).
    pub alias: Option<String>,
}

/// Re-export format enum from the physical operator.
pub use grafeo_core::execution::operators::LoadDataFormat;

/// LOAD DATA operator: reads a file and produces rows.
///
/// With headers (CSV), each row is bound as a `Value::Map` with column names as keys.
/// Without headers (CSV), each row is bound as a `Value::List` of string values.
/// JSONL always produces `Value::Map`. Parquet always produces `Value::Map`.
#[derive(Debug, Clone)]
pub struct LoadDataOp {
    /// File format.
    pub format: LoadDataFormat,
    /// Whether the file has a header row (CSV only, ignored for JSONL/Parquet).
    pub with_headers: bool,
    /// File path (local filesystem).
    pub path: String,
    /// Variable name to bind each row to.
    pub variable: String,
    /// Field separator character (CSV only, default: comma).
    pub field_terminator: Option<char>,
}

/// A logical expression.
#[derive(Debug, Clone)]
#[non_exhaustive]
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

    /// Map literal (e.g., {name: 'Alix', age: 30}).
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

    /// List predicate: all/any/none/single(x IN list WHERE pred).
    ListPredicate {
        /// The kind of list predicate.
        kind: ListPredicateKind,
        /// The iteration variable name.
        variable: String,
        /// The source list expression.
        list_expr: Box<LogicalExpression>,
        /// The predicate to test for each element.
        predicate: Box<LogicalExpression>,
    },

    /// EXISTS subquery.
    ExistsSubquery(Box<LogicalOperator>),

    /// COUNT subquery.
    CountSubquery(Box<LogicalOperator>),

    /// VALUE subquery: returns scalar value from first row of inner query.
    ValueSubquery(Box<LogicalOperator>),

    /// Map projection: `node { .prop1, .prop2, key: expr, .* }`.
    MapProjection {
        /// The base variable name.
        base: String,
        /// Projection entries (property selectors, literal entries, all-properties).
        entries: Vec<MapProjectionEntry>,
    },

    /// reduce() accumulator: `reduce(acc = init, x IN list | expr)`.
    Reduce {
        /// Accumulator variable name.
        accumulator: String,
        /// Initial value for the accumulator.
        initial: Box<LogicalExpression>,
        /// Iteration variable name.
        variable: String,
        /// List to iterate over.
        list: Box<LogicalExpression>,
        /// Body expression evaluated per iteration (references both accumulator and variable).
        expression: Box<LogicalExpression>,
    },

    /// Pattern comprehension: `[(pattern) WHERE pred | expr]`.
    ///
    /// Executes the inner subplan, evaluates the projection for each row,
    /// and collects the results into a list.
    PatternComprehension {
        /// The subplan produced by translating the pattern (+optional WHERE).
        subplan: Box<LogicalOperator>,
        /// The projection expression evaluated for each match.
        projection: Box<LogicalExpression>,
    },
}

/// An entry in a map projection.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum MapProjectionEntry {
    /// `.propertyName`: shorthand for `propertyName: base.propertyName`.
    PropertySelector(String),
    /// `key: expression`: explicit key-value pair.
    LiteralEntry(String, LogicalExpression),
    /// `.*`: include all properties of the base entity.
    AllProperties,
}

/// The kind of list predicate function.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ListPredicateKind {
    /// all(x IN list WHERE pred): true if pred holds for every element.
    All,
    /// any(x IN list WHERE pred): true if pred holds for at least one element.
    Any,
    /// none(x IN list WHERE pred): true if pred holds for no element.
    None,
    /// single(x IN list WHERE pred): true if pred holds for exactly one element.
    Single,
}

/// Binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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
                pushdown_hint: None,
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

    // ========================================================================
    // has_mutations(): the index-scan operators carry an `input` subtree
    // (used to combine graph patterns with vector/text scoring) and must
    // recurse into it so a mutation buried under one is not misclassified
    // as read-only.
    // ========================================================================

    fn read_only_scan() -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".into(),
            label: Some("Article".into()),
            input: None,
        })
    }

    fn mutating_create_node() -> LogicalOperator {
        LogicalOperator::CreateNode(CreateNodeOp {
            variable: "n".into(),
            labels: vec!["Article".into()],
            properties: vec![],
            input: None,
        })
    }

    #[test]
    fn test_text_scan_is_leaf_no_mutations() {
        let op = LogicalOperator::TextScan(TextScanOp {
            variable: "doc".into(),
            label: "Article".into(),
            property: "body".into(),
            query: LogicalExpression::Literal(Value::String("rust".into())),
            k: Some(10),
            threshold: None,
            score_column: None,
        });
        assert!(!op.has_mutations(), "TextScan is a leaf and never mutates");
    }

    #[test]
    fn test_vector_scan_no_input_no_mutations() {
        let op = LogicalOperator::VectorScan(VectorScanOp {
            variable: "doc".into(),
            index_name: None,
            property: "embedding".into(),
            label: Some("Article".into()),
            query_vector: LogicalExpression::Literal(Value::Vector(vec![0.5_f32].into())),
            k: Some(10),
            metric: None,
            min_similarity: None,
            max_distance: None,
            input: None,
        });
        assert!(!op.has_mutations(), "VectorScan with no input is read-only");
    }

    #[test]
    fn test_vector_scan_recurses_into_mutating_input() {
        let op = LogicalOperator::VectorScan(VectorScanOp {
            variable: "doc".into(),
            index_name: None,
            property: "embedding".into(),
            label: Some("Article".into()),
            query_vector: LogicalExpression::Literal(Value::Vector(vec![0.5_f32].into())),
            k: Some(10),
            metric: None,
            min_similarity: None,
            max_distance: None,
            input: Some(Box::new(mutating_create_node())),
        });
        assert!(
            op.has_mutations(),
            "VectorScan must propagate mutations from its input subtree"
        );
    }

    #[test]
    fn test_vector_scan_recurses_into_read_only_input() {
        let op = LogicalOperator::VectorScan(VectorScanOp {
            variable: "doc".into(),
            index_name: None,
            property: "embedding".into(),
            label: Some("Article".into()),
            query_vector: LogicalExpression::Literal(Value::Vector(vec![0.5_f32].into())),
            k: Some(10),
            metric: None,
            min_similarity: None,
            max_distance: None,
            input: Some(Box::new(read_only_scan())),
        });
        assert!(
            !op.has_mutations(),
            "VectorScan with read-only input is read-only"
        );
    }

    #[test]
    fn test_vector_join_recurses_into_mutating_input() {
        let op = LogicalOperator::VectorJoin(VectorJoinOp {
            input: Box::new(mutating_create_node()),
            left_vector_variable: None,
            left_property: None,
            query_vector: LogicalExpression::Literal(Value::Vector(vec![0.5_f32].into())),
            right_variable: "m".into(),
            right_property: "embedding".into(),
            right_label: Some("Movie".into()),
            index_name: None,
            k: 10,
            metric: Some(VectorMetric::Cosine),
            min_similarity: None,
            max_distance: None,
            score_variable: None,
        });
        assert!(
            op.has_mutations(),
            "VectorJoin must recurse into input, was previously hard-coded false"
        );
    }

    #[test]
    fn test_vector_join_with_read_only_input_is_read_only() {
        let op = LogicalOperator::VectorJoin(VectorJoinOp {
            input: Box::new(read_only_scan()),
            left_vector_variable: None,
            left_property: None,
            query_vector: LogicalExpression::Literal(Value::Vector(vec![0.5_f32].into())),
            right_variable: "m".into(),
            right_property: "embedding".into(),
            right_label: Some("Movie".into()),
            index_name: None,
            k: 10,
            metric: Some(VectorMetric::Cosine),
            min_similarity: None,
            max_distance: None,
            score_variable: None,
        });
        assert!(!op.has_mutations());
    }

    // ========================================================================
    // TextScan EXPLAIN/fmt_tree labeling: distinguishes top-k, threshold, and
    // the default-top-100 path (when both k and threshold are None).
    // ========================================================================

    fn text_scan_with_modes(k: Option<usize>, threshold: Option<f64>) -> String {
        let plan = LogicalPlan::new(LogicalOperator::TextScan(TextScanOp {
            variable: "doc".into(),
            label: "Article".into(),
            property: "body".into(),
            query: LogicalExpression::Literal(Value::String("rust".into())),
            k,
            threshold,
            score_column: None,
        }));
        let mut out = String::new();
        plan.root.fmt_tree(&mut out, 0);
        out
    }

    #[test]
    fn test_text_scan_display_top_k_mode() {
        let out = text_scan_with_modes(Some(10), None);
        assert!(out.contains("top-10"), "expected top-10 in:\n{out}");
        assert!(
            !out.contains("threshold"),
            "top-k mode should not say threshold:\n{out}"
        );
    }

    #[test]
    fn test_text_scan_display_threshold_mode() {
        let out = text_scan_with_modes(None, Some(0.5));
        assert!(
            out.contains("threshold>=0.5"),
            "expected threshold>=0.5 in:\n{out}"
        );
        assert!(
            !out.contains("top-"),
            "threshold mode should not say top-:\n{out}"
        );
    }

    #[test]
    fn test_text_scan_display_default_mode_when_both_none() {
        let out = text_scan_with_modes(None, None);
        assert!(
            out.contains("default-top-100"),
            "expected default-top-100 (both k and threshold None) in:\n{out}"
        );
    }

    #[test]
    fn test_text_scan_display_k_takes_precedence_over_threshold() {
        // When both are set, k wins (top-k mode is what the planner actually executes).
        let out = text_scan_with_modes(Some(5), Some(0.3));
        assert!(out.contains("top-5"), "expected top-5 in:\n{out}");
        assert!(
            !out.contains("threshold"),
            "k should take precedence over threshold:\n{out}"
        );
    }

    /// EXPLAIN tree for Project(Filter(Expand(NodeScan))) includes each
    /// operator name, uses 2-space indentation per depth, and calls
    /// `display_label` semantics (labels appear in the tree).
    #[test]
    fn test_explain_tree_basic_operators() {
        let plan = LogicalOperator::Project(ProjectOp {
            projections: vec![Projection {
                expression: LogicalExpression::Property {
                    variable: "b".into(),
                    property: "name".into(),
                },
                alias: Some("name".into()),
            }],
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "b".into(),
                        property: "age".into(),
                    }),
                    op: BinaryOp::Gt,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: Box::new(LogicalOperator::Expand(ExpandOp {
                    from_variable: "a".into(),
                    to_variable: "b".into(),
                    edge_variable: None,
                    direction: ExpandDirection::Outgoing,
                    edge_types: vec!["KNOWS".into()],
                    min_hops: 1,
                    max_hops: Some(1),
                    input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                        variable: "a".into(),
                        label: Some("Person".into()),
                        input: None,
                    })),
                    path_alias: None,
                    path_mode: PathMode::Walk,
                })),
                pushdown_hint: Some(PushdownHint::LabelFirst),
            })),
            pass_through_input: false,
        });

        let tree = plan.explain_tree();

        // Each operator appears with the expected name
        assert!(tree.contains("Project"), "missing Project in:\n{tree}");
        assert!(tree.contains("Filter"), "missing Filter in:\n{tree}");
        assert!(tree.contains("Expand"), "missing Expand in:\n{tree}");
        assert!(tree.contains("NodeScan"), "missing NodeScan in:\n{tree}");

        // Indentation: Project at depth 0, Filter at depth 1 (2 spaces),
        // Expand at depth 2 (4 spaces), NodeScan at depth 3 (6 spaces).
        assert!(tree.starts_with("Project"));
        assert!(
            tree.contains("\n  Filter"),
            "Filter should be indented by 2 spaces"
        );
        assert!(
            tree.contains("\n    Expand"),
            "Expand should be indented by 4 spaces"
        );
        assert!(
            tree.contains("\n      NodeScan"),
            "NodeScan should be indented by 6 spaces"
        );

        // Labels from display_label-style rendering appear: Person label,
        // KNOWS edge type, label-first pushdown hint, projection alias.
        assert!(tree.contains("Person"));
        assert!(tree.contains("KNOWS"));
        assert!(tree.contains("[label-first]"));
        assert!(tree.contains("AS name"));
    }

    /// `has_mutations` recurses through Project/Filter into their inputs.
    #[test]
    fn test_has_mutations_recursive() {
        // Project(Filter(CreateNode)) ⇒ true
        let with_mutation = LogicalOperator::Project(ProjectOp {
            projections: vec![],
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(LogicalOperator::CreateNode(CreateNodeOp {
                    variable: "n".into(),
                    labels: vec!["Person".into()],
                    properties: vec![],
                    input: None,
                })),
                pushdown_hint: None,
            })),
            pass_through_input: false,
        });
        assert!(with_mutation.has_mutations());

        // Project(Filter(NodeScan)) ⇒ false
        let read_only = LogicalOperator::Project(ProjectOp {
            projections: vec![],
            input: Box::new(LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Literal(Value::Bool(true)),
                input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                    variable: "n".into(),
                    label: None,
                    input: None,
                })),
                pushdown_hint: None,
            })),
            pass_through_input: false,
        });
        assert!(!read_only.has_mutations());
    }

    /// Union returns all its branches in order via `children()`, and
    /// Apply returns both input and subplan.
    #[test]
    fn test_children_collection_for_union_and_apply() {
        let leaf = |label: &str| {
            LogicalOperator::NodeScan(NodeScanOp {
                variable: "n".into(),
                label: Some(label.into()),
                input: None,
            })
        };

        let union = LogicalOperator::Union(UnionOp {
            inputs: vec![leaf("Amsterdam"), leaf("Berlin"), leaf("Prague")],
        });
        let children = union.children();
        assert_eq!(children.len(), 3);
        match children[0] {
            LogicalOperator::NodeScan(s) => assert_eq!(s.label.as_deref(), Some("Amsterdam")),
            _ => panic!("Expected NodeScan"),
        }
        match children[2] {
            LogicalOperator::NodeScan(s) => assert_eq!(s.label.as_deref(), Some("Prague")),
            _ => panic!("Expected NodeScan"),
        }

        let apply = LogicalOperator::Apply(ApplyOp {
            input: Box::new(leaf("Person")),
            subplan: Box::new(leaf("Company")),
            shared_variables: vec![],
            optional: false,
        });
        let apply_children = apply.children();
        assert_eq!(apply_children.len(), 2);
        match apply_children[0] {
            LogicalOperator::NodeScan(s) => assert_eq!(s.label.as_deref(), Some("Person")),
            _ => panic!("Expected input NodeScan"),
        }
        match apply_children[1] {
            LogicalOperator::NodeScan(s) => assert_eq!(s.label.as_deref(), Some("Company")),
            _ => panic!("Expected subplan NodeScan"),
        }
    }

    /// Unresolved `CountExpr::Parameter` falls back to a default estimate of 10.0.
    #[test]
    fn test_count_expr_parameter_default() {
        let param = CountExpr::Parameter("limit".to_string());
        assert!((param.estimate() - 10.0).abs() < f64::EPSILON);

        let literal = CountExpr::Literal(42);
        assert!((literal.estimate() - 42.0).abs() < f64::EPSILON);
        assert_eq!(literal.value(), 42);
        assert_eq!(literal.try_value(), Ok(42));

        // try_value returns an error for unresolved parameters,
        // preserving the parameter name in the message.
        let err = param.try_value().unwrap_err();
        assert!(err.contains("$limit"), "error should mention $limit: {err}");

        // Display/Equality sanity
        assert_eq!(format!("{literal}"), "42");
        assert_eq!(format!("{param}"), "$limit");
        assert_eq!(literal, 42usize);
    }

    // ==================== CountExpr ====================

    #[test]
    fn count_expr_literal_value() {
        let count = CountExpr::Literal(42);
        assert_eq!(count.value(), 42);
        assert_eq!(count.try_value(), Ok(42));
        assert!((count.estimate() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn count_expr_parameter_try_value_errors() {
        let count = CountExpr::Parameter("limit".into());
        let err = count.try_value().unwrap_err();
        assert!(err.contains("$limit"));
        // Estimate falls back to default for unresolved parameters.
        assert!((count.estimate() - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "Unresolved parameter: $rows")]
    fn count_expr_parameter_value_panics() {
        let count = CountExpr::Parameter("rows".into());
        let _ = count.value();
    }

    #[test]
    fn count_expr_display_and_conversions() {
        assert_eq!(format!("{}", CountExpr::Literal(7)), "7");
        assert_eq!(format!("{}", CountExpr::Parameter("n".into())), "$n");
        let from_usize: CountExpr = 3usize.into();
        assert_eq!(from_usize, CountExpr::Literal(3));
        assert_eq!(CountExpr::Literal(5), 5usize);
        assert!(CountExpr::Parameter("x".into()) != 5usize);
    }

    // ==================== LogicalPlan constructors ====================

    #[test]
    fn logical_plan_constructors() {
        let leaf = || LogicalOperator::Empty;

        let normal = LogicalPlan::new(leaf());
        assert!(!normal.explain);
        assert!(!normal.profile);
        assert!(normal.default_params.is_empty());

        let explained = LogicalPlan::explain(leaf());
        assert!(explained.explain);
        assert!(!explained.profile);

        let profiled = LogicalPlan::profile(leaf());
        assert!(!profiled.explain);
        assert!(profiled.profile);
    }

    // ==================== Helpers for tests ====================

    fn var(name: &str) -> LogicalExpression {
        LogicalExpression::Variable(name.into())
    }

    fn leaf_empty() -> Box<LogicalOperator> {
        Box::new(LogicalOperator::Empty)
    }

    fn leaf_node_scan(v: &str) -> Box<LogicalOperator> {
        Box::new(LogicalOperator::NodeScan(NodeScanOp {
            variable: v.into(),
            label: None,
            input: None,
        }))
    }

    fn leaf_create_node(v: &str) -> Box<LogicalOperator> {
        Box::new(LogicalOperator::CreateNode(CreateNodeOp {
            variable: v.into(),
            labels: vec!["Person".into()],
            properties: vec![],
            input: None,
        }))
    }

    // ==================== has_mutations ====================

    #[test]
    fn has_mutations_direct_operators_are_mutating() {
        // A representative direct mutation operator.
        let op = LogicalOperator::CreateNode(CreateNodeOp {
            variable: "vincent".into(),
            labels: vec!["Person".into()],
            properties: vec![],
            input: None,
        });
        assert!(op.has_mutations());

        let delete = LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: "vincent".into(),
            detach: true,
            input: leaf_node_scan("vincent"),
        });
        assert!(delete.has_mutations());

        let set_prop = LogicalOperator::SetProperty(SetPropertyOp {
            variable: "mia".into(),
            properties: vec![("city".into(), LogicalExpression::Literal(Value::Null))],
            replace: false,
            is_edge: false,
            input: leaf_node_scan("mia"),
        });
        assert!(set_prop.has_mutations());

        let insert_triple = LogicalOperator::InsertTriple(InsertTripleOp {
            subject: TripleComponent::Iri("s".into()),
            predicate: TripleComponent::Iri("p".into()),
            object: TripleComponent::Iri("o".into()),
            graph: None,
            input: None,
        });
        assert!(insert_triple.has_mutations());

        let clear = LogicalOperator::ClearGraph(ClearGraphOp {
            graph: None,
            silent: false,
        });
        assert!(clear.has_mutations());

        let ddl = LogicalOperator::CreatePropertyGraph(CreatePropertyGraphOp {
            name: "g".into(),
            node_tables: vec![],
            edge_tables: vec![],
        });
        assert!(ddl.has_mutations());
    }

    #[test]
    fn has_mutations_propagates_through_single_input_operators() {
        let base = || {
            LogicalOperator::SetProperty(SetPropertyOp {
                variable: "butch".into(),
                properties: vec![],
                replace: false,
                is_edge: false,
                input: leaf_node_scan("butch"),
            })
        };

        // Filter, Project, Limit, Skip, Sort, Distinct, Unwind, Bind, MapCollect,
        // Return, HorizontalAggregate all wrap the input.
        let filter = LogicalOperator::Filter(FilterOp {
            predicate: var("x"),
            input: Box::new(base()),
            pushdown_hint: None,
        });
        assert!(filter.has_mutations());

        let project = LogicalOperator::Project(ProjectOp {
            projections: vec![],
            input: Box::new(base()),
            pass_through_input: false,
        });
        assert!(project.has_mutations());

        let agg = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![],
            aggregates: vec![],
            input: Box::new(base()),
            having: None,
        });
        assert!(agg.has_mutations());

        let limit = LogicalOperator::Limit(LimitOp {
            count: CountExpr::Literal(10),
            input: Box::new(base()),
        });
        assert!(limit.has_mutations());

        let skip = LogicalOperator::Skip(SkipOp {
            count: CountExpr::Literal(5),
            input: Box::new(base()),
        });
        assert!(skip.has_mutations());

        let sort = LogicalOperator::Sort(SortOp {
            keys: vec![],
            input: Box::new(base()),
        });
        assert!(sort.has_mutations());

        let distinct = LogicalOperator::Distinct(DistinctOp {
            input: Box::new(base()),
            columns: None,
        });
        assert!(distinct.has_mutations());

        let unwind = LogicalOperator::Unwind(UnwindOp {
            expression: var("xs"),
            variable: "x".into(),
            ordinality_var: None,
            offset_var: None,
            input: Box::new(base()),
        });
        assert!(unwind.has_mutations());

        let bind = LogicalOperator::Bind(BindOp {
            expression: var("x"),
            variable: "y".into(),
            input: Box::new(base()),
        });
        assert!(bind.has_mutations());

        let map_collect = LogicalOperator::MapCollect(MapCollectOp {
            key_var: "k".into(),
            value_var: "v".into(),
            alias: "m".into(),
            input: Box::new(base()),
        });
        assert!(map_collect.has_mutations());

        let ret = LogicalOperator::Return(ReturnOp {
            items: vec![],
            distinct: false,
            input: Box::new(base()),
        });
        assert!(ret.has_mutations());

        let hagg = LogicalOperator::HorizontalAggregate(HorizontalAggregateOp {
            list_column: "_path".into(),
            entity_kind: EntityKind::Edge,
            function: AggregateFunction::Sum,
            property: "weight".into(),
            alias: "total".into(),
            input: Box::new(base()),
        });
        assert!(hagg.has_mutations());

        let construct = LogicalOperator::Construct(ConstructOp {
            templates: vec![],
            input: Box::new(base()),
        });
        assert!(construct.has_mutations());
    }

    #[test]
    fn has_mutations_vector_operators_are_readonly() {
        let vscan = LogicalOperator::VectorScan(VectorScanOp {
            variable: "m".into(),
            index_name: None,
            property: "embedding".into(),
            label: None,
            query_vector: LogicalExpression::Literal(Value::Null),
            k: Some(5),
            metric: Some(VectorMetric::Cosine),
            min_similarity: None,
            max_distance: None,
            input: None,
        });
        assert!(!vscan.has_mutations());

        let vjoin = LogicalOperator::VectorJoin(VectorJoinOp {
            input: leaf_node_scan("m"),
            left_vector_variable: None,
            left_property: None,
            query_vector: LogicalExpression::Literal(Value::Null),
            right_variable: "n".into(),
            right_property: "embedding".into(),
            right_label: None,
            index_name: None,
            k: 3,
            metric: None,
            min_similarity: None,
            max_distance: None,
            score_variable: None,
        });
        assert!(!vjoin.has_mutations());
    }

    #[test]
    fn has_mutations_two_children_and_union_apply() {
        let mutating = || *leaf_create_node("jules");
        let read = || *leaf_node_scan("jules");

        let join_readonly = LogicalOperator::Join(JoinOp {
            left: Box::new(read()),
            right: Box::new(read()),
            join_type: JoinType::Inner,
            conditions: vec![],
        });
        assert!(!join_readonly.has_mutations());

        let join_right_mutates = LogicalOperator::Join(JoinOp {
            left: Box::new(read()),
            right: Box::new(mutating()),
            join_type: JoinType::Left,
            conditions: vec![],
        });
        assert!(join_right_mutates.has_mutations());

        let left_join = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(mutating()),
            right: Box::new(read()),
            condition: None,
        });
        assert!(left_join.has_mutations());

        let anti_join = LogicalOperator::AntiJoin(AntiJoinOp {
            left: Box::new(read()),
            right: Box::new(mutating()),
        });
        assert!(anti_join.has_mutations());

        let except = LogicalOperator::Except(ExceptOp {
            left: Box::new(read()),
            right: Box::new(read()),
            all: true,
        });
        assert!(!except.has_mutations());

        let intersect = LogicalOperator::Intersect(IntersectOp {
            left: Box::new(mutating()),
            right: Box::new(read()),
            all: false,
        });
        assert!(intersect.has_mutations());

        let otherwise = LogicalOperator::Otherwise(OtherwiseOp {
            left: Box::new(read()),
            right: Box::new(mutating()),
        });
        assert!(otherwise.has_mutations());

        let union = LogicalOperator::Union(UnionOp {
            inputs: vec![read(), mutating(), read()],
        });
        assert!(union.has_mutations());

        let mwj = LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs: vec![read(), read()],
            conditions: vec![],
            shared_variables: vec!["a".into()],
        });
        assert!(!mwj.has_mutations());

        let apply_readonly = LogicalOperator::Apply(ApplyOp {
            input: Box::new(read()),
            subplan: Box::new(read()),
            shared_variables: vec![],
            optional: false,
        });
        assert!(!apply_readonly.has_mutations());

        let apply_inner_mutates = LogicalOperator::Apply(ApplyOp {
            input: Box::new(read()),
            subplan: Box::new(mutating()),
            shared_variables: vec![],
            optional: true,
        });
        assert!(apply_inner_mutates.has_mutations());
    }

    #[test]
    fn has_mutations_leaf_operators_are_readonly() {
        assert!(!LogicalOperator::Empty.has_mutations());
        assert!(
            !LogicalOperator::ParameterScan(ParameterScanOp {
                columns: vec!["a".into()],
            })
            .has_mutations()
        );
        assert!(
            !LogicalOperator::CallProcedure(CallProcedureOp {
                name: vec!["grafeo".into(), "pagerank".into()],
                arguments: vec![],
                yield_items: None,
            })
            .has_mutations()
        );
        assert!(
            !LogicalOperator::LoadData(LoadDataOp {
                format: LoadDataFormat::Csv,
                with_headers: true,
                path: "/tmp/x.csv".into(),
                variable: "row".into(),
                field_terminator: None,
            })
            .has_mutations()
        );
        assert!(
            !LogicalOperator::TripleScan(TripleScanOp {
                subject: TripleComponent::Variable("s".into()),
                predicate: TripleComponent::Variable("p".into()),
                object: TripleComponent::Variable("o".into()),
                graph: None,
                input: None,
                dataset: None,
            })
            .has_mutations()
        );
    }

    // ==================== children() ====================

    #[test]
    fn children_of_leaf_operators() {
        assert!(LogicalOperator::Empty.children().is_empty());
        assert!(
            LogicalOperator::CallProcedure(CallProcedureOp {
                name: vec!["p".into()],
                arguments: vec![],
                yield_items: None,
            })
            .children()
            .is_empty()
        );
        assert!(
            LogicalOperator::CreateGraph(CreateGraphOp {
                graph: "g".into(),
                silent: false,
            })
            .children()
            .is_empty()
        );
        assert!(
            LogicalOperator::LoadData(LoadDataOp {
                format: LoadDataFormat::Jsonl,
                with_headers: false,
                path: "x.jsonl".into(),
                variable: "r".into(),
                field_terminator: None,
            })
            .children()
            .is_empty()
        );
    }

    #[test]
    fn children_of_optional_input_operators() {
        let ns_no_input = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".into(),
            label: None,
            input: None,
        });
        assert_eq!(ns_no_input.children().len(), 0);

        let ns_with_input = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".into(),
            label: None,
            input: Some(leaf_empty()),
        });
        assert_eq!(ns_with_input.children().len(), 1);

        let edge_scan_in = LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "e".into(),
            edge_types: vec![],
            input: Some(leaf_empty()),
        });
        assert_eq!(edge_scan_in.children().len(), 1);
    }

    #[test]
    fn children_of_two_child_operators() {
        let join = LogicalOperator::Join(JoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
            join_type: JoinType::Cross,
            conditions: vec![],
        });
        assert_eq!(join.children().len(), 2);

        let apply = LogicalOperator::Apply(ApplyOp {
            input: leaf_empty(),
            subplan: leaf_empty(),
            shared_variables: vec![],
            optional: false,
        });
        assert_eq!(apply.children().len(), 2);

        let union = LogicalOperator::Union(UnionOp {
            inputs: vec![*leaf_empty(), *leaf_empty(), *leaf_empty()],
        });
        assert_eq!(union.children().len(), 3);
    }

    #[test]
    fn children_of_modify_returns_where_clause() {
        let modify = LogicalOperator::Modify(ModifyOp {
            delete_templates: vec![],
            insert_templates: vec![],
            where_clause: leaf_empty(),
            graph: None,
        });
        assert_eq!(modify.children().len(), 1);
    }

    // ==================== display_label ====================

    #[test]
    fn display_label_spot_checks() {
        let ns = LogicalOperator::NodeScan(NodeScanOp {
            variable: "vincent".into(),
            label: Some("Person".into()),
            input: None,
        });
        assert_eq!(ns.display_label(), "vincent:Person");

        let ns_no_label = LogicalOperator::NodeScan(NodeScanOp {
            variable: "mia".into(),
            label: None,
            input: None,
        });
        assert_eq!(ns_no_label.display_label(), "mia:*");

        let edge_scan = LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "e".into(),
            edge_types: vec!["KNOWS".into(), "LIKES".into()],
            input: None,
        });
        assert_eq!(edge_scan.display_label(), "e:KNOWS|LIKES");

        let edge_scan_any = LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "e".into(),
            edge_types: vec![],
            input: None,
        });
        assert_eq!(edge_scan_any.display_label(), "e:*");

        let expand = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: vec!["KNOWS".into()],
            min_hops: 1,
            max_hops: Some(1),
            input: leaf_node_scan("a"),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        assert_eq!(expand.display_label(), "(a)->[:KNOWS]->(b)");

        let expand_in = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_variable: None,
            direction: ExpandDirection::Incoming,
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(1),
            input: leaf_node_scan("a"),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        assert_eq!(expand_in.display_label(), "(a)<-[:*]<-(b)");

        let expand_both = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_variable: None,
            direction: ExpandDirection::Both,
            edge_types: vec![],
            min_hops: 1,
            max_hops: Some(1),
            input: leaf_node_scan("a"),
            path_alias: None,
            path_mode: PathMode::Walk,
        });
        assert_eq!(expand_both.display_label(), "(a)--[:*]--(b)");
    }

    #[test]
    fn display_label_filter_pushdown_hints() {
        let make = |hint: Option<PushdownHint>| {
            LogicalOperator::Filter(FilterOp {
                predicate: var("x"),
                input: leaf_empty(),
                pushdown_hint: hint,
            })
        };

        let f_none = make(None);
        let s = f_none.display_label();
        assert!(!s.contains('['));

        let f_index = make(Some(PushdownHint::IndexLookup {
            property: "name".into(),
        }));
        assert!(f_index.display_label().contains("[index: name]"));

        let f_range = make(Some(PushdownHint::RangeScan {
            property: "age".into(),
        }));
        assert!(f_range.display_label().contains("[range: age]"));

        let f_label = make(Some(PushdownHint::LabelFirst));
        assert!(f_label.display_label().contains("[label-first]"));
    }

    #[test]
    fn display_label_projection_join_sort_return() {
        let proj = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: var("n"),
                    alias: Some("person".into()),
                },
                Projection {
                    expression: LogicalExpression::Property {
                        variable: "n".into(),
                        property: "city".into(),
                    },
                    alias: None,
                },
            ],
            input: leaf_empty(),
            pass_through_input: false,
        });
        let s = proj.display_label();
        assert!(s.contains("person"));
        assert!(s.contains("n.city"));

        let join = LogicalOperator::Join(JoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
            join_type: JoinType::Cross,
            conditions: vec![],
        });
        assert_eq!(join.display_label(), "Cross");

        let agg = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![var("city")],
            aggregates: vec![],
            input: leaf_empty(),
            having: None,
        });
        assert_eq!(agg.display_label(), "group: [city]");

        let limit = LogicalOperator::Limit(LimitOp {
            count: CountExpr::Literal(10),
            input: leaf_empty(),
        });
        assert_eq!(limit.display_label(), "10");

        let skip = LogicalOperator::Skip(SkipOp {
            count: CountExpr::Parameter("off".into()),
            input: leaf_empty(),
        });
        assert_eq!(skip.display_label(), "$off");

        let sort = LogicalOperator::Sort(SortOp {
            keys: vec![
                SortKey {
                    expression: var("a"),
                    order: SortOrder::Ascending,
                    nulls: None,
                },
                SortKey {
                    expression: var("b"),
                    order: SortOrder::Descending,
                    nulls: None,
                },
            ],
            input: leaf_empty(),
        });
        let s = sort.display_label();
        assert!(s.contains("a ASC"));
        assert!(s.contains("b DESC"));

        let distinct = LogicalOperator::Distinct(DistinctOp {
            input: leaf_empty(),
            columns: None,
        });
        assert_eq!(distinct.display_label(), "");

        let ret = LogicalOperator::Return(ReturnOp {
            items: vec![
                ReturnItem {
                    expression: var("n"),
                    alias: Some("node".into()),
                },
                ReturnItem {
                    expression: var("m"),
                    alias: None,
                },
            ],
            distinct: true,
            input: leaf_empty(),
        });
        let s = ret.display_label();
        assert!(s.contains("node"));
        assert!(s.contains('m'));
    }

    #[test]
    fn display_label_remaining_operators() {
        let union = LogicalOperator::Union(UnionOp {
            inputs: vec![*leaf_empty(), *leaf_empty()],
        });
        assert_eq!(union.display_label(), "2 branches");

        let mwj = LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs: vec![*leaf_empty(), *leaf_empty(), *leaf_empty()],
            conditions: vec![],
            shared_variables: vec![],
        });
        assert_eq!(mwj.display_label(), "3 inputs");

        let lj = LogicalOperator::LeftJoin(LeftJoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
            condition: None,
        });
        assert_eq!(lj.display_label(), "");

        let aj = LogicalOperator::AntiJoin(AntiJoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
        });
        assert_eq!(aj.display_label(), "");

        let unwind = LogicalOperator::Unwind(UnwindOp {
            expression: var("xs"),
            variable: "item".into(),
            ordinality_var: None,
            offset_var: None,
            input: leaf_empty(),
        });
        assert_eq!(unwind.display_label(), "item");

        let bind = LogicalOperator::Bind(BindOp {
            expression: var("x"),
            variable: "y".into(),
            input: leaf_empty(),
        });
        assert_eq!(bind.display_label(), "y");

        let mapc = LogicalOperator::MapCollect(MapCollectOp {
            key_var: "k".into(),
            value_var: "v".into(),
            alias: "counts".into(),
            input: leaf_empty(),
        });
        assert_eq!(mapc.display_label(), "counts");

        let sp = LogicalOperator::ShortestPath(ShortestPathOp {
            input: leaf_empty(),
            source_var: "a".into(),
            target_var: "b".into(),
            edge_types: vec![],
            direction: ExpandDirection::Outgoing,
            path_alias: "p".into(),
            all_paths: false,
        });
        assert_eq!(sp.display_label(), "a -> b");

        let merge = LogicalOperator::Merge(MergeOp {
            variable: "django".into(),
            labels: vec![],
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            input: leaf_empty(),
        });
        assert_eq!(merge.display_label(), "django");

        let merge_rel = LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".into(),
            source_variable: "a".into(),
            target_variable: "b".into(),
            edge_type: "KNOWS".into(),
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            input: leaf_empty(),
        });
        assert_eq!(merge_rel.display_label(), "r");

        let cnode = LogicalOperator::CreateNode(CreateNodeOp {
            variable: "shosanna".into(),
            labels: vec!["Person".into(), "Hero".into()],
            properties: vec![],
            input: None,
        });
        assert_eq!(cnode.display_label(), "shosanna:Person:Hero");

        let cedge_with = LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some("r".into()),
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_type: "KNOWS".into(),
            properties: vec![],
            input: leaf_empty(),
        });
        assert_eq!(cedge_with.display_label(), "[r:KNOWS]");

        let cedge_without = LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: None,
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_type: "KNOWS".into(),
            properties: vec![],
            input: leaf_empty(),
        });
        assert_eq!(cedge_without.display_label(), "[?:KNOWS]");

        let dnode = LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: "hans".into(),
            detach: false,
            input: leaf_empty(),
        });
        assert_eq!(dnode.display_label(), "hans");

        let dedge = LogicalOperator::DeleteEdge(DeleteEdgeOp {
            variable: "r".into(),
            input: leaf_empty(),
        });
        assert_eq!(dedge.display_label(), "r");

        let set_prop = LogicalOperator::SetProperty(SetPropertyOp {
            variable: "beatrix".into(),
            properties: vec![],
            replace: false,
            is_edge: false,
            input: leaf_empty(),
        });
        assert_eq!(set_prop.display_label(), "beatrix");

        let add_lbl = LogicalOperator::AddLabel(AddLabelOp {
            variable: "n".into(),
            labels: vec!["A".into(), "B".into()],
            input: leaf_empty(),
        });
        assert_eq!(add_lbl.display_label(), "n:A:B");

        let rm_lbl = LogicalOperator::RemoveLabel(RemoveLabelOp {
            variable: "n".into(),
            labels: vec!["A".into()],
            input: leaf_empty(),
        });
        assert_eq!(rm_lbl.display_label(), "n:A");

        let call = LogicalOperator::CallProcedure(CallProcedureOp {
            name: vec!["grafeo".into(), "pagerank".into()],
            arguments: vec![],
            yield_items: None,
        });
        assert_eq!(call.display_label(), "grafeo.pagerank");

        let load = LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: true,
            path: "data.csv".into(),
            variable: "r".into(),
            field_terminator: None,
        });
        assert_eq!(load.display_label(), "data.csv AS r");

        let apply = LogicalOperator::Apply(ApplyOp {
            input: leaf_empty(),
            subplan: leaf_empty(),
            shared_variables: vec![],
            optional: false,
        });
        assert_eq!(apply.display_label(), "");

        let vscan = LogicalOperator::VectorScan(VectorScanOp {
            variable: "m".into(),
            index_name: None,
            property: "embedding".into(),
            label: None,
            query_vector: LogicalExpression::Literal(Value::Null),
            k: Some(5),
            metric: None,
            min_similarity: None,
            max_distance: None,
            input: None,
        });
        assert_eq!(vscan.display_label(), "m");

        let vjoin = LogicalOperator::VectorJoin(VectorJoinOp {
            input: leaf_empty(),
            left_vector_variable: None,
            left_property: None,
            query_vector: LogicalExpression::Literal(Value::Null),
            right_variable: "t".into(),
            right_property: "emb".into(),
            right_label: None,
            index_name: None,
            k: 3,
            metric: None,
            min_similarity: None,
            max_distance: None,
            score_variable: None,
        });
        assert_eq!(vjoin.display_label(), "t");

        // Empty / catch-all branch.
        assert_eq!(LogicalOperator::Empty.display_label(), "");
    }

    // ==================== explain_tree / fmt_tree ====================

    #[test]
    fn explain_tree_covers_all_common_arms() {
        // Build a deeply nested tree that exercises many arms.
        let ns = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".into(),
            label: Some("Person".into()),
            input: Some(Box::new(LogicalOperator::Empty)),
        });
        let out = ns.explain_tree();
        assert!(out.contains("NodeScan (n:Person)"));
        assert!(out.contains("Empty"));

        let ns_star = LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".into(),
            label: None,
            input: None,
        });
        assert!(ns_star.explain_tree().contains("NodeScan (n:*)"));

        let es = LogicalOperator::EdgeScan(EdgeScanOp {
            variable: "e".into(),
            edge_types: vec![],
            input: None,
        });
        assert!(es.explain_tree().contains("EdgeScan (e:*)"));
    }

    #[test]
    fn explain_tree_expand_variants() {
        let mk = |min, max, dir| {
            LogicalOperator::Expand(ExpandOp {
                from_variable: "a".into(),
                to_variable: "b".into(),
                edge_variable: None,
                direction: dir,
                edge_types: vec!["KNOWS".into()],
                min_hops: min,
                max_hops: max,
                input: leaf_node_scan("a"),
                path_alias: None,
                path_mode: PathMode::Walk,
            })
            .explain_tree()
        };

        let s = mk(1, Some(1), ExpandDirection::Outgoing);
        assert!(s.contains("(a)->[:KNOWS]->(b)"));
        let s = mk(2, Some(2), ExpandDirection::Incoming);
        assert!(s.contains("*2"));
        assert!(s.contains("<-"));
        let s = mk(1, Some(3), ExpandDirection::Both);
        assert!(s.contains("*1..3"));
        assert!(s.contains("--"));
        let s = mk(2, None, ExpandDirection::Outgoing);
        assert!(s.contains("*2.."));
    }

    #[test]
    fn explain_tree_filter_with_all_hints() {
        let base = || {
            LogicalOperator::Filter(FilterOp {
                predicate: LogicalExpression::Binary {
                    left: Box::new(LogicalExpression::Property {
                        variable: "n".into(),
                        property: "age".into(),
                    }),
                    op: BinaryOp::Eq,
                    right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
                },
                input: leaf_node_scan("n"),
                pushdown_hint: None,
            })
        };
        let mut f = base();
        if let LogicalOperator::Filter(ref mut op) = f {
            op.pushdown_hint = Some(PushdownHint::IndexLookup {
                property: "age".into(),
            });
        }
        assert!(f.explain_tree().contains("[index: age]"));

        if let LogicalOperator::Filter(ref mut op) = f {
            op.pushdown_hint = Some(PushdownHint::RangeScan {
                property: "age".into(),
            });
        }
        assert!(f.explain_tree().contains("[range: age]"));

        if let LogicalOperator::Filter(ref mut op) = f {
            op.pushdown_hint = Some(PushdownHint::LabelFirst);
        }
        assert!(f.explain_tree().contains("[label-first]"));
    }

    #[test]
    fn explain_tree_projection_aggregate_sort_return() {
        let proj = LogicalOperator::Project(ProjectOp {
            projections: vec![
                Projection {
                    expression: var("n"),
                    alias: Some("who".into()),
                },
                Projection {
                    expression: var("m"),
                    alias: None,
                },
            ],
            input: leaf_empty(),
            pass_through_input: true,
        });
        let s = proj.explain_tree();
        assert!(s.contains("Project"));
        assert!(s.contains("n AS who"));

        let agg = LogicalOperator::Aggregate(AggregateOp {
            group_by: vec![var("city")],
            aggregates: vec![
                AggregateExpr {
                    function: AggregateFunction::Count,
                    expression: None,
                    expression2: None,
                    distinct: false,
                    alias: Some("c".into()),
                    percentile: None,
                    separator: None,
                },
                AggregateExpr {
                    function: AggregateFunction::Sum,
                    expression: Some(var("x")),
                    expression2: None,
                    distinct: false,
                    alias: None,
                    percentile: None,
                    separator: None,
                },
            ],
            input: leaf_empty(),
            having: None,
        });
        let s = agg.explain_tree();
        assert!(s.contains("Aggregate"));
        assert!(s.contains("count(...) AS c"));
        assert!(s.contains("sum(...)"));

        let sort = LogicalOperator::Sort(SortOp {
            keys: vec![SortKey {
                expression: var("age"),
                order: SortOrder::Descending,
                nulls: None,
            }],
            input: leaf_empty(),
        });
        assert!(sort.explain_tree().contains("age DESC"));

        let ret_distinct = LogicalOperator::Return(ReturnOp {
            items: vec![ReturnItem {
                expression: var("n"),
                alias: Some("who".into()),
            }],
            distinct: true,
            input: leaf_empty(),
        });
        let s = ret_distinct.explain_tree();
        assert!(s.contains("Return DISTINCT"));
        assert!(s.contains("n AS who"));

        let limit = LogicalOperator::Limit(LimitOp {
            count: CountExpr::Literal(5),
            input: leaf_empty(),
        });
        assert!(limit.explain_tree().contains("Limit (5)"));

        let skip = LogicalOperator::Skip(SkipOp {
            count: CountExpr::Literal(2),
            input: leaf_empty(),
        });
        assert!(skip.explain_tree().contains("Skip (2)"));

        let distinct = LogicalOperator::Distinct(DistinctOp {
            input: leaf_empty(),
            columns: None,
        });
        assert!(distinct.explain_tree().contains("Distinct"));
    }

    #[test]
    fn explain_tree_joins_and_set_ops() {
        let join = LogicalOperator::Join(JoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
            join_type: JoinType::Inner,
            conditions: vec![],
        });
        assert!(join.explain_tree().contains("Join (Inner)"));

        let left_join_cond = LogicalOperator::LeftJoin(LeftJoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
            condition: Some(var("x")),
        });
        assert!(
            left_join_cond
                .explain_tree()
                .contains("LeftJoin (condition:")
        );

        let left_join_none = LogicalOperator::LeftJoin(LeftJoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
            condition: None,
        });
        let s = left_join_none.explain_tree();
        assert!(s.contains("LeftJoin"));
        assert!(!s.contains("condition:"));

        let anti = LogicalOperator::AntiJoin(AntiJoinOp {
            left: leaf_empty(),
            right: leaf_empty(),
        });
        assert!(anti.explain_tree().contains("AntiJoin"));

        let union = LogicalOperator::Union(UnionOp {
            inputs: vec![*leaf_empty(), *leaf_empty()],
        });
        assert!(union.explain_tree().contains("Union (2 branches)"));

        let mwj = LogicalOperator::MultiWayJoin(MultiWayJoinOp {
            inputs: vec![*leaf_empty(), *leaf_empty()],
            conditions: vec![],
            shared_variables: vec!["a".into(), "b".into()],
        });
        let s = mwj.explain_tree();
        assert!(s.contains("MultiWayJoin"));
        assert!(s.contains("shared: [a, b]"));

        let except_all = LogicalOperator::Except(ExceptOp {
            left: leaf_empty(),
            right: leaf_empty(),
            all: true,
        });
        assert!(except_all.explain_tree().contains("Except ALL"));
        let except = LogicalOperator::Except(ExceptOp {
            left: leaf_empty(),
            right: leaf_empty(),
            all: false,
        });
        assert!(except.explain_tree().contains("Except\n"));

        let inter_all = LogicalOperator::Intersect(IntersectOp {
            left: leaf_empty(),
            right: leaf_empty(),
            all: true,
        });
        assert!(inter_all.explain_tree().contains("Intersect ALL"));
        let inter = LogicalOperator::Intersect(IntersectOp {
            left: leaf_empty(),
            right: leaf_empty(),
            all: false,
        });
        assert!(inter.explain_tree().contains("Intersect\n"));

        let otherwise = LogicalOperator::Otherwise(OtherwiseOp {
            left: leaf_empty(),
            right: leaf_empty(),
        });
        assert!(otherwise.explain_tree().contains("Otherwise"));
    }

    #[test]
    fn explain_tree_unwind_bind_mapcollect_apply_sp() {
        let unwind = LogicalOperator::Unwind(UnwindOp {
            expression: var("xs"),
            variable: "item".into(),
            ordinality_var: None,
            offset_var: None,
            input: leaf_empty(),
        });
        assert!(unwind.explain_tree().contains("Unwind (item)"));

        let bind = LogicalOperator::Bind(BindOp {
            expression: var("x"),
            variable: "y".into(),
            input: leaf_empty(),
        });
        assert!(bind.explain_tree().contains("Bind (y)"));

        let mapc = LogicalOperator::MapCollect(MapCollectOp {
            key_var: "k".into(),
            value_var: "v".into(),
            alias: "m".into(),
            input: leaf_empty(),
        });
        let s = mapc.explain_tree();
        assert!(s.contains("MapCollect"));
        assert!(s.contains("k -> v AS m"));

        let apply = LogicalOperator::Apply(ApplyOp {
            input: leaf_empty(),
            subplan: leaf_empty(),
            shared_variables: vec!["a".into()],
            optional: true,
        });
        assert!(apply.explain_tree().contains("Apply"));

        let sp = LogicalOperator::ShortestPath(ShortestPathOp {
            input: leaf_empty(),
            source_var: "a".into(),
            target_var: "b".into(),
            edge_types: vec![],
            direction: ExpandDirection::Outgoing,
            path_alias: "p".into(),
            all_paths: false,
        });
        assert!(sp.explain_tree().contains("ShortestPath (a -> b)"));
    }

    #[test]
    fn explain_tree_mutations() {
        let merge = LogicalOperator::Merge(MergeOp {
            variable: "vincent".into(),
            labels: vec!["Person".into()],
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            input: leaf_empty(),
        });
        assert!(merge.explain_tree().contains("Merge (vincent)"));

        let merge_rel = LogicalOperator::MergeRelationship(MergeRelationshipOp {
            variable: "r".into(),
            source_variable: "a".into(),
            target_variable: "b".into(),
            edge_type: "KNOWS".into(),
            match_properties: vec![],
            on_create: vec![],
            on_match: vec![],
            input: leaf_empty(),
        });
        assert!(merge_rel.explain_tree().contains("MergeRelationship (r)"));

        let cnode = LogicalOperator::CreateNode(CreateNodeOp {
            variable: "mia".into(),
            labels: vec!["Person".into()],
            properties: vec![],
            input: Some(leaf_empty()),
        });
        let s = cnode.explain_tree();
        assert!(s.contains("CreateNode (mia:Person)"));
        assert!(s.contains("Empty"));

        let cnode_no_input = LogicalOperator::CreateNode(CreateNodeOp {
            variable: "mia".into(),
            labels: vec![],
            properties: vec![],
            input: None,
        });
        assert!(cnode_no_input.explain_tree().contains("CreateNode (mia:)"));

        let cedge = LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: Some("r".into()),
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_type: "KNOWS".into(),
            properties: vec![],
            input: leaf_empty(),
        });
        assert!(
            cedge
                .explain_tree()
                .contains("CreateEdge (a)-[r:KNOWS]->(b)")
        );

        let cedge_anon = LogicalOperator::CreateEdge(CreateEdgeOp {
            variable: None,
            from_variable: "a".into(),
            to_variable: "b".into(),
            edge_type: "KNOWS".into(),
            properties: vec![],
            input: leaf_empty(),
        });
        assert!(cedge_anon.explain_tree().contains("[?:KNOWS]"));

        let dnode = LogicalOperator::DeleteNode(DeleteNodeOp {
            variable: "butch".into(),
            detach: true,
            input: leaf_empty(),
        });
        assert!(dnode.explain_tree().contains("DeleteNode (butch)"));

        let dedge = LogicalOperator::DeleteEdge(DeleteEdgeOp {
            variable: "r".into(),
            input: leaf_empty(),
        });
        assert!(dedge.explain_tree().contains("DeleteEdge (r)"));

        let set_prop = LogicalOperator::SetProperty(SetPropertyOp {
            variable: "n".into(),
            properties: vec![("name".into(), var("x")), ("age".into(), var("y"))],
            replace: false,
            is_edge: false,
            input: leaf_empty(),
        });
        let s = set_prop.explain_tree();
        assert!(s.contains("SetProperty"));
        assert!(s.contains("n.name"));
        assert!(s.contains("n.age"));

        let add_lbl = LogicalOperator::AddLabel(AddLabelOp {
            variable: "n".into(),
            labels: vec!["A".into()],
            input: leaf_empty(),
        });
        assert!(add_lbl.explain_tree().contains("AddLabel (n:A)"));

        let rm_lbl = LogicalOperator::RemoveLabel(RemoveLabelOp {
            variable: "n".into(),
            labels: vec!["A".into(), "B".into()],
            input: leaf_empty(),
        });
        assert!(rm_lbl.explain_tree().contains("RemoveLabel (n:A:B)"));
    }

    #[test]
    fn explain_tree_call_and_load_data() {
        let call = LogicalOperator::CallProcedure(CallProcedureOp {
            name: vec!["grafeo".into(), "pagerank".into()],
            arguments: vec![],
            yield_items: None,
        });
        assert!(
            call.explain_tree()
                .contains("CallProcedure (grafeo.pagerank)")
        );

        let csv = LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: true,
            path: "data.csv".into(),
            variable: "row".into(),
            field_terminator: None,
        });
        let s = csv.explain_tree();
        assert!(s.contains("LoadCsv"));
        assert!(s.contains("WITH HEADERS"));
        assert!(s.contains("data.csv"));
        assert!(s.contains("AS row"));

        let csv_no_hdr = LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Csv,
            with_headers: false,
            path: "data.csv".into(),
            variable: "row".into(),
            field_terminator: None,
        });
        assert!(!csv_no_hdr.explain_tree().contains("WITH HEADERS"));

        let jsonl = LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Jsonl,
            with_headers: false,
            path: "data.jsonl".into(),
            variable: "r".into(),
            field_terminator: None,
        });
        assert!(jsonl.explain_tree().contains("LoadJsonl"));

        let parquet = LogicalOperator::LoadData(LoadDataOp {
            format: LoadDataFormat::Parquet,
            with_headers: false,
            path: "data.parquet".into(),
            variable: "r".into(),
            field_terminator: None,
        });
        assert!(parquet.explain_tree().contains("LoadParquet"));
    }

    #[test]
    fn explain_tree_triple_scan_and_fallback() {
        let ts = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".into()),
            predicate: TripleComponent::Iri("http://ex/p".into()),
            object: TripleComponent::Literal(Value::Int64(5)),
            graph: None,
            input: Some(leaf_empty()),
            dataset: None,
        });
        let s = ts.explain_tree();
        assert!(s.contains("TripleScan"));
        assert!(s.contains("?s"));
        assert!(s.contains("<http://ex/p>"));
        assert!(s.contains("Empty"));

        let ts_no_input = LogicalOperator::TripleScan(TripleScanOp {
            subject: TripleComponent::Variable("s".into()),
            predicate: TripleComponent::Variable("p".into()),
            object: TripleComponent::Variable("o".into()),
            graph: None,
            input: None,
            dataset: None,
        });
        assert!(ts_no_input.explain_tree().contains("TripleScan"));

        // Fallback arm for operators without a specific formatter.
        let graph_op = LogicalOperator::CreateGraph(CreateGraphOp {
            graph: "g".into(),
            silent: false,
        });
        let out = graph_op.explain_tree();
        assert!(!out.is_empty());
    }

    // ==================== fmt_expr helper ====================

    #[test]
    fn fmt_expr_covers_common_variants() {
        let v = var("n");
        assert_eq!(fmt_expr(&v), "n");

        let p = LogicalExpression::Property {
            variable: "n".into(),
            property: "age".into(),
        };
        assert_eq!(fmt_expr(&p), "n.age");

        let lit = LogicalExpression::Literal(Value::Int64(42));
        assert_eq!(fmt_expr(&lit), "42");

        let bin = LogicalExpression::Binary {
            left: Box::new(var("a")),
            op: BinaryOp::Eq,
            right: Box::new(LogicalExpression::Literal(Value::Int64(1))),
        };
        let s = fmt_expr(&bin);
        assert!(s.contains("Eq"));
        assert!(s.contains('a'));

        let un = LogicalExpression::Unary {
            op: UnaryOp::Not,
            operand: Box::new(var("a")),
        };
        let s = fmt_expr(&un);
        assert!(s.contains("Not"));

        let fc = LogicalExpression::FunctionCall {
            name: "toLower".into(),
            args: vec![var("name")],
            distinct: false,
        };
        assert_eq!(fmt_expr(&fc), "toLower(name)");

        // Fallback arm: non-common variant hits the `_ => format!("{expr:?}")` path.
        let list = LogicalExpression::List(vec![var("a")]);
        let out = fmt_expr(&list);
        assert!(out.contains("List") || out.contains('['));
    }

    // ==================== fmt_triple_component helper ====================

    #[test]
    fn fmt_triple_component_variants() {
        assert_eq!(
            fmt_triple_component(&TripleComponent::Variable("s".into())),
            "?s"
        );
        assert_eq!(
            fmt_triple_component(&TripleComponent::Iri("http://ex/p".into())),
            "<http://ex/p>"
        );
        assert!(fmt_triple_component(&TripleComponent::Literal(Value::Int64(10))).contains("10"));
        assert_eq!(
            fmt_triple_component(&TripleComponent::LangLiteral {
                value: "hello".into(),
                lang: "en".into(),
            }),
            "\"hello\"@en"
        );
        assert_eq!(
            fmt_triple_component(&TripleComponent::BlankNode("b0".into())),
            "_:b0"
        );
    }

    // ==================== TripleComponent::as_variable ====================

    #[test]
    fn triple_component_as_variable() {
        assert_eq!(
            TripleComponent::Variable("s".into()).as_variable(),
            Some("s")
        );
        assert_eq!(
            TripleComponent::Iri("http://ex/p".into()).as_variable(),
            None
        );
        assert_eq!(
            TripleComponent::Literal(Value::Int64(1)).as_variable(),
            None
        );
        assert_eq!(TripleComponent::BlankNode("b".into()).as_variable(), None);
        assert_eq!(
            TripleComponent::LangLiteral {
                value: "v".into(),
                lang: "en".into(),
            }
            .as_variable(),
            None
        );
    }
}
