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
            Self::VectorScan(_) | Self::VectorJoin(_) | Self::TextScan(_) => false,

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
                let mode = match op.k {
                    Some(k) => format!("top-{k}"),
                    None => "threshold".to_string(),
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
}
