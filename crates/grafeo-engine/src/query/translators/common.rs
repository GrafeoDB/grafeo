//! Shared utilities for query language translators.
//!
//! Functions here are used by multiple translator modules (GQL, Cypher, etc.)
//! to avoid duplication of identical logic.

use std::cell::RefCell;
use std::collections::HashSet;
#[cfg(any(feature = "graphql", feature = "gremlin", test))]
use std::sync::atomic::{AtomicU32, Ordering};

use crate::query::plan::{
    AggregateFunction, BinaryOp, CountExpr, DistinctOp, FilterOp, JoinOp, JoinType, LeftJoinOp,
    LimitOp, LogicalExpression, LogicalOperator, ParameterScanOp, Projection, ReturnItem, ReturnOp,
    SetPropertyOp, SkipOp, SortKey, SortOp,
};
use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryError, QueryErrorKind, Result};

/// Expands the `RETURN *` that ends a `CALL` subquery into the variables the
/// subquery binds itself, in name order: the variables of the outer row
/// (`outer`) stay where they are, and internal names (`_...`) are not
/// returned. Fails when those variables are not known, so that the subquery
/// names what it returns.
pub(crate) fn expand_subquery_return_star(
    subplan: &mut LogicalOperator,
    outer: Option<&HashSet<String>>,
) -> Result<()> {
    let Some(ret) = final_return_mut(subplan) else {
        return Ok(());
    };
    let [item] = ret.items.as_slice() else {
        return Ok(());
    };
    if !matches!(&item.expression, LogicalExpression::Variable(name) if name == "*") {
        return Ok(());
    }
    let (Some(outer), Some(bound)) = (outer, ret.input.bound_variables(outer)) else {
        return Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "RETURN * in this CALL subquery cannot tell which variables it binds: return them by name",
        )));
    };
    let mut names: Vec<String> = bound
        .into_iter()
        .filter(|name| !name.starts_with('_') && !outer.contains(name))
        .collect();
    names.sort();
    ret.items = names
        .into_iter()
        .map(|name| ReturnItem {
            expression: LogicalExpression::Variable(name),
            alias: None,
        })
        .collect();
    Ok(())
}

/// The `RETURN` that ends `plan`, under the `ORDER BY`, `SKIP`, `LIMIT` or
/// `DISTINCT` that follow it.
fn final_return_mut(plan: &mut LogicalOperator) -> Option<&mut ReturnOp> {
    match plan {
        LogicalOperator::Return(ret) => Some(ret),
        LogicalOperator::Sort(op) => final_return_mut(&mut op.input),
        LogicalOperator::Limit(op) => final_return_mut(&mut op.input),
        LogicalOperator::Skip(op) => final_return_mut(&mut op.input),
        LogicalOperator::Distinct(op) => final_return_mut(&mut op.input),
        _ => None,
    }
}

/// The variables a `CALL` subquery imports (see [`call_imports`]) while its
/// body is translated: none outside one. They stay in scope for the whole
/// body: a `WITH` in it that leaves one out passes it on all the same
/// (openCypher: "a subsequent WITH within the subquery cannot descope an
/// imported variable"; in GQL they are fields of the body's incoming working
/// record, which a statement of the body does not drop). A `WITH` that binds
/// an import's name to anything but the import itself (`WITH 3 AS a`) makes
/// the name a variable of the body, and no longer an import.
#[derive(Debug, Default)]
pub(crate) struct CallImports(RefCell<Vec<String>>);

impl CallImports {
    /// Runs `translate` with `imports` as the imports of the body it
    /// translates, and puts back the imports of the enclosing body
    /// afterwards.
    pub(crate) fn within<T>(&self, imports: Vec<String>, translate: impl FnOnce() -> T) -> T {
        let enclosing = self.0.replace(imports);
        let translated = translate();
        self.0.replace(enclosing);
        translated
    }

    /// The imports a `WITH` that passes on `projected` leaves out, which it
    /// passes on as well. Each projected item is its name and whether it is
    /// the variable of that name itself (`WITH a`, `WITH a AS a`). An import
    /// the `WITH` binds to something else stops being one.
    pub(crate) fn left_out(&self, projected: &[(String, bool)]) -> Vec<String> {
        let mut imports = self.0.borrow_mut();
        imports.retain(|import| {
            projected
                .iter()
                .all(|(name, itself)| name != import || *itself)
        });
        imports
            .iter()
            .filter(|import| projected.iter().all(|(name, _)| name != *import))
            .cloned()
            .collect()
    }
}

/// The variables of a `CALL` subquery's outer row that it imports:
/// `shared_variables` (the `ApplyOp`'s), with `*` replaced by the variables of
/// the outer row, `outer`, in name order (none when they are not known here).
pub(crate) fn call_imports(
    shared_variables: &[String],
    outer: Option<&HashSet<String>>,
) -> Vec<String> {
    if shared_variables.iter().any(|name| name == "*") {
        let mut names: Vec<String> = outer.into_iter().flatten().cloned().collect();
        names.sort();
        return names;
    }
    shared_variables.to_vec()
}

/// The rows of a `WITH` in a `CALL` body (`plan`, its projection, or its
/// aggregation when `aggregates`), with the imports it leaves out, `missing`
/// (see [`CallImports`]), beside its own columns. A projection passes them on
/// as items. An aggregation cannot group by them, since a count over no rows
/// would then be no row instead of one: its rows are joined with a scan of
/// the imported values (one row, the outer row's).
pub(crate) fn with_imports(
    plan: LogicalOperator,
    missing: Vec<String>,
    aggregates: bool,
) -> LogicalOperator {
    if missing.is_empty() {
        return plan;
    }
    match plan {
        LogicalOperator::Project(mut project) if !aggregates && !project.pass_through_input => {
            project
                .projections
                .extend(missing.into_iter().map(|name| Projection {
                    expression: LogicalExpression::Variable(name),
                    alias: None,
                }));
            LogicalOperator::Project(project)
        }
        plan => LogicalOperator::Join(JoinOp {
            left: Box::new(plan),
            right: Box::new(LogicalOperator::ParameterScan(ParameterScanOp {
                columns: missing,
            })),
            join_type: JoinType::Cross,
            conditions: Vec::new(),
        }),
    }
}

/// The error for a `WITH` item that is an expression without a name. As in
/// openCypher, later clauses refer to what a `WITH` passes on by name, and a
/// property read such as `n.name` does not keep `n`.
pub(crate) fn unaliased_with_expression() -> Error {
    Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
        "Expression in WITH must be aliased (use AS)",
    ))
}

// Whether a function name is an aggregate: one list with the planner's check
// of function calls.
pub(crate) use crate::query::functions::is_aggregate_function;

/// Converts a function name to an `AggregateFunction` enum variant.
pub(crate) fn to_aggregate_function(name: &str) -> Option<AggregateFunction> {
    match name.to_uppercase().as_str() {
        "COUNT" => Some(AggregateFunction::Count),
        "SUM" => Some(AggregateFunction::Sum),
        "AVG" => Some(AggregateFunction::Avg),
        "MIN" => Some(AggregateFunction::Min),
        "MAX" => Some(AggregateFunction::Max),
        "COLLECT" => Some(AggregateFunction::Collect),
        "STDEV" | "STDDEV" | "STDDEV_SAMP" => Some(AggregateFunction::StdDev),
        "STDEVP" | "STDDEVP" | "STDDEV_POP" => Some(AggregateFunction::StdDevPop),
        "VARIANCE" | "VAR_SAMP" => Some(AggregateFunction::Variance),
        "VAR_POP" => Some(AggregateFunction::VariancePop),
        "PERCENTILE_DISC" | "PERCENTILEDISC" => Some(AggregateFunction::PercentileDisc),
        "PERCENTILE_CONT" | "PERCENTILECONT" => Some(AggregateFunction::PercentileCont),
        "GROUP_CONCAT" | "GROUPCONCAT" | "LISTAGG" => Some(AggregateFunction::GroupConcat),
        "SAMPLE" => Some(AggregateFunction::Sample),
        "COVAR_SAMP" => Some(AggregateFunction::CovarSamp),
        "COVAR_POP" => Some(AggregateFunction::CovarPop),
        "CORR" => Some(AggregateFunction::Corr),
        "REGR_SLOPE" => Some(AggregateFunction::RegrSlope),
        "REGR_INTERCEPT" => Some(AggregateFunction::RegrIntercept),
        "REGR_R2" => Some(AggregateFunction::RegrR2),
        "REGR_COUNT" => Some(AggregateFunction::RegrCount),
        "REGR_SXX" => Some(AggregateFunction::RegrSxx),
        "REGR_SYY" => Some(AggregateFunction::RegrSyy),
        "REGR_SXY" => Some(AggregateFunction::RegrSxy),
        "REGR_AVGX" => Some(AggregateFunction::RegrAvgx),
        "REGR_AVGY" => Some(AggregateFunction::RegrAvgy),
        _ => None,
    }
}

/// Returns true if the aggregate function is a binary set function (requires two arguments).
pub(crate) fn is_binary_set_function(func: AggregateFunction) -> bool {
    matches!(
        func,
        AggregateFunction::CovarSamp
            | AggregateFunction::CovarPop
            | AggregateFunction::Corr
            | AggregateFunction::RegrSlope
            | AggregateFunction::RegrIntercept
            | AggregateFunction::RegrR2
            | AggregateFunction::RegrCount
            | AggregateFunction::RegrSxx
            | AggregateFunction::RegrSyy
            | AggregateFunction::RegrSxy
            | AggregateFunction::RegrAvgx
            | AggregateFunction::RegrAvgy
    )
}

/// Evaluates GraphQL `@skip` and `@include` directives to determine if a field
/// should be included in the query result.
///
/// Per the GraphQL spec:
/// - `@skip(if: true)` excludes the field
/// - `@skip(if: false)` includes the field
/// - `@include(if: false)` excludes the field
/// - `@include(if: true)` includes the field
///
/// When both directives are present, the field is included only if it passes both
/// checks (`@skip` must not exclude AND `@include` must include).
///
/// Returns `true` if the field should be included, `false` if it should be skipped.
#[cfg(feature = "graphql")]
pub(crate) fn graphql_directives_allow(
    directives: &[grafeo_adapters::query::graphql::ast::Directive],
) -> bool {
    let mut include = true;

    for directive in directives {
        match directive.name.as_str() {
            "skip" => {
                // @skip(if: true) excludes the field
                if let Some(arg) = directive.arguments.iter().find(|a| a.name == "if")
                    && let grafeo_adapters::query::graphql::ast::InputValue::Boolean(val) =
                        &arg.value
                    && *val
                {
                    include = false;
                }
            }
            "include" => {
                // @include(if: false) excludes the field
                if let Some(arg) = directive.arguments.iter().find(|a| a.name == "if")
                    && let grafeo_adapters::query::graphql::ast::InputValue::Boolean(val) =
                        &arg.value
                    && !val
                {
                    include = false;
                }
            }
            _ => {} // Unknown directives are ignored
        }
    }

    include
}

/// Capitalizes the first character of a string.
///
/// Used by GraphQL translators to convert field names to type names.
#[cfg(any(feature = "graphql", test))]
pub(crate) fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}

/// Generates unique variable names with an atomic counter.
///
/// Replaces the duplicated `var_counter: AtomicU32` + `next_var()` pattern
/// used across multiple translators.
#[cfg(any(feature = "graphql", feature = "gremlin", test))]
pub(crate) struct VarGen {
    counter: AtomicU32,
}

#[cfg(any(feature = "graphql", feature = "gremlin", test))]
impl VarGen {
    /// Creates a new variable generator starting from 0.
    pub fn new() -> Self {
        Self {
            counter: AtomicU32::new(0),
        }
    }

    /// Returns the next unique variable name (e.g., `_v0`, `_v1`, ...).
    pub fn next(&self) -> String {
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("_v{n}")
    }

    /// Returns the current counter value without incrementing.
    #[cfg(any(feature = "gremlin", test))]
    pub fn current(&self) -> u32 {
        self.counter.load(Ordering::Relaxed)
    }
}

/// Combines a non-empty vector of predicates into a single AND expression.
///
/// Returns an error if the input is empty. Used by `build_property_predicate`
/// in multiple translators.
pub(crate) fn combine_with_and(predicates: Vec<LogicalExpression>) -> Result<LogicalExpression> {
    LogicalExpression::conjunction(predicates).ok_or_else(|| {
        Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            "Empty property predicate",
        ))
    })
}

/// `hasLabel(variable, label)` for every label, combined with AND, or `None`
/// when `labels` is empty. A node pattern with several labels requires all of
/// them, wherever the node appears in a pattern.
pub(crate) fn has_all_labels(variable: &str, labels: &[String]) -> Option<LogicalExpression> {
    LogicalExpression::conjunction(labels.iter().map(|label| LogicalExpression::FunctionCall {
        name: "hasLabel".into(),
        args: vec![
            LogicalExpression::Variable(variable.to_string()),
            LogicalExpression::Literal(Value::String(label.clone().into())),
        ],
        distinct: false,
    }))
}

// ---------------------------------------------------------------------------
// Whole paths of path patterns
// ---------------------------------------------------------------------------

/// One hop of a path pattern as a translator binds it: the edge it takes and
/// the node it reaches.
#[cfg(any(feature = "gql", feature = "cypher"))]
#[derive(Debug, Clone)]
pub(crate) struct PathHop {
    /// The variable of the edge (of a single-hop edge pattern).
    pub edge: String,
    /// The variable of the node the hop ends at.
    pub target: String,
    /// For a variable-length edge pattern, the path alias of its expand: the
    /// `_path_nodes_` and `_path_edges_` columns of that alias hold the
    /// nodes and edges of the hop, and `edge` is not read.
    pub segment: Option<String>,
}

/// `[id(variable)]`, or `[]` when `variable` is null: the hop of a
/// questioned edge (`->?`) that matched nothing has a null edge and node, and
/// the path leaves the hop out. (A list literal keeps a null item, so the
/// null is tested here.)
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn id_list(variable: &str) -> LogicalExpression {
    let id = LogicalExpression::Id(variable.to_string());
    LogicalExpression::Case {
        operand: None,
        when_clauses: vec![(
            LogicalExpression::Unary {
                op: crate::query::plan::UnaryOp::IsNull,
                operand: Box::new(id.clone()),
            },
            LogicalExpression::List(Vec::new()),
        )],
        else_clause: Some(Box::new(LogicalExpression::List(vec![id]))),
    }
}

/// The ids of the nodes of the path that starts at `source` and takes `hops`,
/// as one list. A hop that is missing (a questioned edge, `->?`, that
/// matched nothing) is left out (see [`id_list`]).
#[cfg(any(feature = "gql", feature = "cypher"))]
fn path_node_ids(source: &str, hops: &[PathHop]) -> LogicalExpression {
    let mut parts = vec![id_list(source)];
    for hop in hops {
        parts.push(match &hop.segment {
            // The first node of a segment is the last node of the hop before
            Some(segment) => LogicalExpression::FunctionCall {
                name: "tail".into(),
                args: vec![LogicalExpression::Variable(format!(
                    "_path_nodes_{segment}"
                ))],
                distinct: false,
            },
            None => id_list(&hop.target),
        });
    }
    concatenation(parts)
}

/// The ids of the edges of the path that takes `hops`, as one list (a
/// missing hop left out, see [`path_node_ids`]).
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn path_edge_ids(hops: &[PathHop]) -> LogicalExpression {
    concatenation(
        hops.iter()
            .map(|hop| match &hop.segment {
                Some(segment) => LogicalExpression::Variable(format!("_path_edges_{segment}")),
                None => id_list(&hop.edge),
            })
            .collect(),
    )
}

/// The lists `parts` joined into one (`a + b + ...`).
#[cfg(any(feature = "gql", feature = "cypher"))]
fn concatenation(parts: Vec<LogicalExpression>) -> LogicalExpression {
    parts
        .into_iter()
        .reduce(|left, right| LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::Add,
            right: Box::new(right),
        })
        .unwrap_or(LogicalExpression::List(Vec::new()))
}

/// The path that starts at `source` and takes `hops`, as one path value.
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn whole_path(source: &str, hops: &[PathHop]) -> LogicalExpression {
    LogicalExpression::FunctionCall {
        name: "path".into(),
        args: vec![path_node_ids(source, hops), path_edge_ids(hops)],
        distinct: false,
    }
}

/// Binds the path variable `alias` to the whole path of a path pattern with
/// more than one edge pattern, which starts at `source` and takes `hops`: the
/// columns `length(p)`, `nodes(p)` and `edges(p)` read (`_path_length_p` and
/// so on) and the path value itself. An expand binds a path for its own edge
/// pattern only (ISO/IEC 39075:2024 16.7: a path variable binds the path of
/// the whole path pattern).
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn bind_whole_path(
    plan: LogicalOperator,
    alias: &str,
    source: &str,
    hops: &[PathHop],
) -> LogicalOperator {
    let projection = |expression: LogicalExpression, name: String| crate::query::plan::Projection {
        expression,
        alias: Some(name),
    };
    LogicalOperator::Project(crate::query::plan::ProjectOp {
        projections: vec![
            projection(path_node_ids(source, hops), format!("_path_nodes_{alias}")),
            projection(path_edge_ids(hops), format!("_path_edges_{alias}")),
            projection(
                LogicalExpression::FunctionCall {
                    name: "size".into(),
                    args: vec![path_edge_ids(hops)],
                    distinct: false,
                },
                format!("_path_length_{alias}"),
            ),
            projection(whole_path(source, hops), alias.to_string()),
        ],
        input: Box::new(plan),
        pass_through_input: true,
    })
}

// ---------------------------------------------------------------------------
// Generated names
// ---------------------------------------------------------------------------

/// The names a translator makes up for one statement: for its anonymous
/// nodes and edges (`_anon_3`) and for helper columns. They are counted per
/// statement, so a statement gets the same names each time it is
/// translated, and none of them is a word the statement spells (see
/// [`written_names`]): a user variable named `_anon_0` stays the user's.
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) struct GeneratedNames {
    /// The number the next name gets.
    next: std::cell::Cell<u32>,
    /// The words of the statement a generated name could be.
    written: HashSet<String>,
}

#[cfg(any(feature = "gql", feature = "cypher"))]
impl GeneratedNames {
    /// The names for the statement `query`.
    pub(crate) fn new(query: &str) -> Self {
        Self {
            next: std::cell::Cell::new(0),
            written: written_names(query),
        }
    }

    /// The next name that starts with `prefix`: `_anon_` gives `_anon_0`,
    /// then `_anon_1`, skipping the ones the statement spells.
    pub(crate) fn next(&self, prefix: &str) -> String {
        loop {
            let number = self.next.get();
            self.next.set(number + 1);
            let name = format!("{prefix}{number}");
            if !self.written.contains(&name) {
                return name;
            }
        }
    }

    /// Whether the statement spells `name`, so that a name made up another
    /// way (Cypher's aggregate columns) must skip it.
    #[cfg(feature = "cypher")]
    pub(crate) fn is_written(&self, name: &str) -> bool {
        self.written.contains(name)
    }
}

/// The words of `query` that start with `_`, which is how every name a
/// translator makes up starts (`_anon_3`): the statement's own variables and
/// aliases among them, wherever they appear, backquoted or not. A word in a
/// string literal or a comment is taken too, which only skips a name.
#[cfg(any(feature = "gql", feature = "cypher"))]
fn written_names(query: &str) -> HashSet<String> {
    query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|word| word.starts_with('_'))
        .map(str::to_string)
        .collect()
}

// ---------------------------------------------------------------------------
// Edges a pattern binds once
// ---------------------------------------------------------------------------

/// An edge pattern of a graph pattern whose edges must all be different:
/// GQL's DIFFERENT EDGES (ISO/IEC 39075:2024 16.4) and every Cypher MATCH
/// (openCypher 9, relationship uniqueness). See [`different_edges`].
#[cfg(any(feature = "gql", feature = "cypher"))]
#[derive(Debug, Clone)]
pub(crate) struct EdgeOccurrence {
    /// The variable the edge pattern binds.
    pub variable: String,
    /// Whether `variable` is a group variable: a quantified (variable-length)
    /// edge pattern binds it to the list of its edges.
    pub group: bool,
    /// The edge types the pattern allows, any type when empty.
    pub types: Vec<String>,
}

#[cfg(any(feature = "gql", feature = "cypher"))]
impl EdgeOccurrence {
    /// Whether this edge pattern and `other` can bind one edge: an edge has
    /// one type, so two patterns that allow no common type never do.
    fn may_bind_the_edge_of(&self, other: &Self) -> bool {
        self.types.is_empty()
            || other.types.is_empty()
            || self
                .types
                .iter()
                .any(|edge_type| other.types.contains(edge_type))
    }

    /// The ids of the edges the pattern binds, as a list. An edge that
    /// matched nothing (GQL's questioned edge, `->?`) is null and binds no
    /// edge: an empty list (see [`id_list`]).
    fn edge_ids(&self) -> LogicalExpression {
        if self.group {
            LogicalExpression::Variable(self.variable.clone())
        } else {
            id_list(&self.variable)
        }
    }
}

/// The edge patterns of `edges` (by index) that must be compared, in groups:
/// patterns that may bind one edge (directly or through other patterns) are
/// compared together, and a pattern no other one may share an edge with
/// needs no check. With `lone_groups`, a group variable on its own is still
/// checked, for an expand whose path may repeat an edge; without, its own
/// edges are known to differ (its expand matches trails).
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn edges_to_compare(edges: &[EdgeOccurrence], lone_groups: bool) -> Vec<Vec<usize>> {
    let mut grouped = vec![false; edges.len()];
    let mut groups = Vec::new();
    for start in 0..edges.len() {
        if grouped[start] {
            continue;
        }
        grouped[start] = true;
        // Every pattern that may share an edge with a member joins
        let mut members = vec![start];
        let mut next = 0;
        while let Some(&current) = members.get(next) {
            next += 1;
            for other in 0..edges.len() {
                if !grouped[other] && edges[current].may_bind_the_edge_of(&edges[other]) {
                    grouped[other] = true;
                    members.push(other);
                }
            }
        }
        if members.len() > 1 || (lone_groups && edges[start].group) {
            members.sort_unstable();
            groups.push(members);
        }
    }
    groups
}

/// The condition that the edges the patterns of each of `groups` (indices
/// into `edges`, see [`edges_to_compare`]) bind are all different, an edge
/// of a group variable included: `all_different` over their ids, one check
/// per group. `None` when there is no group.
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn different_edges(
    edges: &[EdgeOccurrence],
    groups: &[Vec<usize>],
) -> Option<LogicalExpression> {
    let checks = groups
        .iter()
        .map(|members| LogicalExpression::FunctionCall {
            name: "all_different".into(),
            args: vec![concatenation(
                members
                    .iter()
                    .map(|&index| edges[index].edge_ids())
                    .collect(),
            )],
            distinct: false,
        })
        .collect();
    join_and_conjuncts(checks)
}

// ---------------------------------------------------------------------------
// MERGE
// ---------------------------------------------------------------------------

/// The variable of an end node of a MERGE relationship pattern, and `input`
/// with a MERGE of that node after it when the pattern gives the node
/// `labels` or `match_properties`, as in `MERGE (h)-[:R]->(:T {x: 3})`: a node
/// without a `variable` gets the name `fresh` makes. One with neither a
/// variable nor a label or property could be any node, which the
/// relationship MERGE does not support: an error.
///
/// # Errors
///
/// Returns an error for an anonymous node without a label or property.
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn merge_end_node(
    variable: Option<&str>,
    labels: &[String],
    match_properties: Vec<(String, LogicalExpression)>,
    input: LogicalOperator,
    fresh: impl FnOnce() -> String,
) -> Result<(String, LogicalOperator)> {
    let defined = !labels.is_empty() || !match_properties.is_empty();
    let variable = match variable {
        Some(name) => name.to_string(),
        None if defined => fresh(),
        None => {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                "MERGE of a relationship with an anonymous node without a label or property is \
                 not supported: MATCH or MERGE that node first and use its variable",
            )));
        }
    };
    if !defined {
        return Ok((variable, input));
    }
    let merge = LogicalOperator::Merge(crate::query::plan::MergeOp {
        variable: variable.clone(),
        labels: labels.to_vec(),
        match_properties,
        on_create: Vec::new(),
        on_match: Vec::new(),
        on_create_labels: Vec::new(),
        on_match_labels: Vec::new(),
        input: Box::new(input),
    });
    Ok((variable, merge))
}

/// The error for a MERGE of a pattern with more than one relationship, which
/// would have to match or create the whole pattern: only one relationship
/// per MERGE is supported.
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn merge_of_a_longer_path() -> Error {
    Error::Query(QueryError::new(
        QueryErrorKind::Semantic,
        "MERGE of a pattern with more than one relationship is not supported: merge one \
         relationship per MERGE clause",
    ))
}

// ---------------------------------------------------------------------------
// Variable extraction
// ---------------------------------------------------------------------------

/// `all(hop IN edges(path) WHERE predicate)`: a property map or an element
/// pattern `WHERE` on a variable-length edge must hold for every edge of the
/// path. `hop` may be the edge pattern's own variable: inside the predicate it
/// is then the one edge, not the list of the path's edges.
pub(crate) fn every_edge_matches(
    path: String,
    hop: String,
    predicate: LogicalExpression,
) -> LogicalExpression {
    LogicalExpression::ListPredicate {
        kind: crate::query::plan::ListPredicateKind::All,
        variable: hop,
        list_expr: Box::new(LogicalExpression::FunctionCall {
            name: "edges".into(),
            args: vec![LogicalExpression::Variable(path)],
            distinct: false,
        }),
        predicate: Box::new(predicate),
    }
}

/// Dotted access `base.key` into a map value (`n.meta.route`), or into a node
/// or edge that is not a pattern variable (`startNode(r).name`,
/// `head(rs).w`, `x.msg.id` of `{msg: m}`), for the GQL and Cypher
/// translators.
///
/// # Errors
///
/// Returns an error when `base` can be neither a map nor a node or edge (a
/// number, a string), or reads an aggregate (`head(collect(n)).name`, whose
/// aggregate the translators do not take out of a key read), so the read
/// would give null without saying why.
pub(crate) fn map_access(base: LogicalExpression, key: &str) -> Result<LogicalExpression> {
    if calls_an_aggregate(&base) {
        let base = crate::query::planner::common::expression_to_string(&base);
        return Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!(
                "{base} reads an aggregate, so .{key} cannot read from it in the same clause: \
                 name it first (WITH {base} AS x) and read x.{key}"
            ),
        )));
    }
    if !can_be_map(&base) {
        let base = crate::query::planner::common::expression_to_string(&base);
        return Err(Error::Query(QueryError::new(
            QueryErrorKind::Semantic,
            format!(
                "{base} is not a map value, a node or an edge, so .{key} cannot read from it: \
                 read .{key} of a node, an edge or a map value"
            ),
        )));
    }
    Ok(LogicalExpression::MapAccess {
        base: Box::new(base),
        key: key.to_string(),
    })
}

/// Whether an expression can evaluate to a map, a node or an edge: a
/// variable, a parameter, a property, a map literal or projection,
/// `properties(...)`, `startNode(r)` and `endNode(r)`, the first or last item
/// of a list (`head`, `last`), or a key or element of one of those (or of a
/// list literal, or of a list of nodes or edges such as `nodes(p)`).
fn can_be_map(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::Variable(_)
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Map(_)
        | LogicalExpression::MapProjection { .. }
        | LogicalExpression::MapAccess { .. } => true,
        LogicalExpression::IndexAccess { base, .. } => {
            matches!(**base, LogicalExpression::List(_)) || can_be_list(base) || can_be_map(base)
        }
        LogicalExpression::FunctionCall { name, .. } => [
            "properties",
            "startnode",
            "start_node",
            "endnode",
            "end_node",
            "head",
            "last",
        ]
        .iter()
        .any(|function| name.eq_ignore_ascii_case(function)),
        _ => false,
    }
}

/// Whether an expression is a function that returns a list whose items can
/// be maps, nodes or edges (`nodes(p)`, `relationships(p)`, and `tail` or
/// `reverse` of a list).
fn can_be_list(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::FunctionCall { name, .. } => {
            ["nodes", "relationships", "edges", "tail", "reverse"]
                .iter()
                .any(|function| name.eq_ignore_ascii_case(function))
        }
        _ => false,
    }
}

/// Whether an expression calls an aggregate function in a function argument,
/// an index or a key read (`head(collect(n))`, `collect(n)[0]`).
fn calls_an_aggregate(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::FunctionCall { name, args, .. } => {
            is_aggregate_function(name) || args.iter().any(calls_an_aggregate)
        }
        LogicalExpression::IndexAccess { base, index } => {
            calls_an_aggregate(base) || calls_an_aggregate(index)
        }
        LogicalExpression::MapAccess { base, .. } => calls_an_aggregate(base),
        _ => false,
    }
}

pub(crate) fn collect_expression_variables(expr: &LogicalExpression, vars: &mut HashSet<String>) {
    match expr {
        LogicalExpression::Variable(name) => {
            vars.insert(name.clone());
        }
        LogicalExpression::Property { variable, .. }
        | LogicalExpression::Labels(variable)
        | LogicalExpression::Type(variable)
        | LogicalExpression::Id(variable) => {
            vars.insert(variable.clone());
        }
        LogicalExpression::Binary { left, right, .. } => {
            collect_expression_variables(left, vars);
            collect_expression_variables(right, vars);
        }
        LogicalExpression::Unary { operand, .. } => {
            collect_expression_variables(operand, vars);
        }
        LogicalExpression::FunctionCall { args, .. } => {
            for arg in args {
                collect_expression_variables(arg, vars);
            }
        }
        LogicalExpression::List(items) => {
            for item in items {
                collect_expression_variables(item, vars);
            }
        }
        LogicalExpression::Map(pairs) => {
            for (_, value) in pairs {
                collect_expression_variables(value, vars);
            }
        }
        LogicalExpression::IndexAccess { base, index } => {
            collect_expression_variables(base, vars);
            collect_expression_variables(index, vars);
        }
        LogicalExpression::MapAccess { base, .. } => collect_expression_variables(base, vars),
        LogicalExpression::SliceAccess { base, start, end } => {
            collect_expression_variables(base, vars);
            if let Some(s) = start {
                collect_expression_variables(s, vars);
            }
            if let Some(e) = end {
                collect_expression_variables(e, vars);
            }
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            if let Some(op) = operand {
                collect_expression_variables(op, vars);
            }
            for (cond, result) in when_clauses {
                collect_expression_variables(cond, vars);
                collect_expression_variables(result, vars);
            }
            if let Some(else_expr) = else_clause {
                collect_expression_variables(else_expr, vars);
            }
        }
        // The variable a comprehension, list predicate or `reduce` binds is
        // its own: only the other names its body uses come from outside.
        LogicalExpression::ListComprehension {
            variable,
            list_expr,
            filter_expr,
            map_expr,
        } => {
            collect_expression_variables(list_expr, vars);
            let mut body = HashSet::new();
            if let Some(filter) = filter_expr {
                collect_expression_variables(filter, &mut body);
            }
            collect_expression_variables(map_expr, &mut body);
            body.remove(variable);
            vars.extend(body);
        }
        LogicalExpression::ListPredicate {
            variable,
            list_expr,
            predicate,
            ..
        } => {
            collect_expression_variables(list_expr, vars);
            let mut body = HashSet::new();
            collect_expression_variables(predicate, &mut body);
            body.remove(variable);
            vars.extend(body);
        }
        LogicalExpression::MapProjection { base, entries } => {
            vars.insert(base.clone());
            for entry in entries {
                if let crate::query::plan::MapProjectionEntry::LiteralEntry(_, expr) = entry {
                    collect_expression_variables(expr, vars);
                }
            }
        }
        LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            collect_expression_variables(initial, vars);
            collect_expression_variables(list, vars);
            let mut body = HashSet::new();
            collect_expression_variables(expression, &mut body);
            body.remove(accumulator);
            body.remove(variable);
            vars.extend(body);
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            collect_expression_variables(projection, vars);
        }
        LogicalExpression::Literal(_)
        | LogicalExpression::Parameter(_)
        | LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_) => {}
    }
}

// ---------------------------------------------------------------------------
// OPTIONAL MATCH predicate classification
// ---------------------------------------------------------------------------

/// Splits a conjunctive predicate (AND-chain) into individual conjuncts.
pub(crate) fn split_conjuncts(expr: LogicalExpression) -> Vec<LogicalExpression> {
    let mut result = Vec::new();
    split_conjuncts_recursive(expr, &mut result);
    result
}

fn split_conjuncts_recursive(expr: LogicalExpression, out: &mut Vec<LogicalExpression>) {
    if let LogicalExpression::Binary {
        left,
        op: BinaryOp::And,
        right,
    } = expr
    {
        split_conjuncts_recursive(*left, out);
        split_conjuncts_recursive(*right, out);
    } else {
        out.push(expr);
    }
}

/// Result of classifying WHERE predicates for OPTIONAL MATCH.
///
/// The WHERE of an OPTIONAL MATCH is part of its pattern (ISO GQL,
/// openCypher): it decides which matches count and never removes a row of the
/// clauses before it, so each conjunct goes into the optional side or into
/// the join condition.
pub(crate) struct ClassifiedPredicates {
    /// Conjuncts with an `EXISTS`, `COUNT` or value subquery or a pattern
    /// comprehension that read no variable only the optional side binds (see
    /// [`classify_optional_predicates`]): a Filter above the LeftJoin.
    pub post_filters: Vec<LogicalExpression>,
    /// Conjuncts that read no variable only the left side binds (the optional
    /// side's own variables, the variables both sides share, or none): a
    /// filter on the right input of the LeftJoin.
    pub right_filters: Vec<LogicalExpression>,
    /// Conjuncts that read a variable only the left side binds: conditions
    /// of the LeftJoin, which reads the joined row (a left row none of whose
    /// pairs pass them keeps nulls).
    pub cross_filters: Vec<LogicalExpression>,
}

/// Classifies the conjuncts of the WHERE of an OPTIONAL MATCH by the
/// variables they read: `left_vars` are those of the rows the OPTIONAL MATCH
/// goes on from, `right_vars` those of its pattern (see
/// [`ClassifiedPredicates`]).
///
/// A conjunct with a subquery or pattern comprehension keeps the placement it
/// had before: [`collect_expression_variables`] does not see what the
/// subquery reads, so it cannot be placed by its variables, and the join
/// condition cannot run a subquery. One that reads a variable only the
/// optional side binds is a right filter, any other one stays a filter above
/// the join.
pub(crate) fn classify_optional_predicates(
    predicate: LogicalExpression,
    left_vars: &HashSet<String>,
    right_vars: &HashSet<String>,
) -> ClassifiedPredicates {
    let conjuncts = split_conjuncts(predicate);
    let mut post_filters = Vec::new();
    let mut right_filters = Vec::new();
    let mut cross_filters = Vec::new();

    for conjunct in conjuncts {
        let mut referenced = HashSet::new();
        collect_expression_variables(&conjunct, &mut referenced);

        let has_right_only_var = referenced
            .iter()
            .any(|v| right_vars.contains(v) && !left_vars.contains(v));
        let has_left_only_var = referenced
            .iter()
            .any(|v| left_vars.contains(v) && !right_vars.contains(v));

        if has_subquery(&conjunct) {
            let all_in_right = referenced.iter().all(|v| right_vars.contains(v));
            if has_right_only_var && all_in_right {
                right_filters.push(conjunct);
            } else if has_left_only_var && has_right_only_var {
                cross_filters.push(conjunct);
            } else {
                post_filters.push(conjunct);
            }
        } else if has_left_only_var {
            // `p.id IN xs`, `forum.id = x`, `x = 3`: the joined row has the
            // left side's value
            cross_filters.push(conjunct);
        } else {
            // `p.name = 'Gus'`, a condition on a variable both sides share
            // (`f.id = 3`, the optional side's `f` is the same node), or a
            // constant (`3 = 19`): a filter on the optional side's matches
            right_filters.push(conjunct);
        }
    }

    ClassifiedPredicates {
        post_filters,
        right_filters,
        cross_filters,
    }
}

/// Whether `expr` has an `EXISTS`, `COUNT` or value subquery or a pattern
/// comprehension, whose plan may read variables of the row that
/// [`collect_expression_variables`] does not list.
fn has_subquery(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_)
        | LogicalExpression::PatternComprehension { .. } => true,
        LogicalExpression::Literal(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Property { .. }
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Labels(_)
        | LogicalExpression::Type(_)
        | LogicalExpression::Id(_) => false,
        LogicalExpression::Binary { left, right, .. } => has_subquery(left) || has_subquery(right),
        LogicalExpression::Unary { operand, .. } => has_subquery(operand),
        LogicalExpression::FunctionCall { args: items, .. } | LogicalExpression::List(items) => {
            items.iter().any(has_subquery)
        }
        LogicalExpression::Map(pairs) => pairs.iter().any(|(_, value)| has_subquery(value)),
        LogicalExpression::IndexAccess { base, index } => has_subquery(base) || has_subquery(index),
        LogicalExpression::MapAccess { base, .. } => has_subquery(base),
        LogicalExpression::SliceAccess { base, start, end } => {
            has_subquery(base)
                || start.as_deref().is_some_and(has_subquery)
                || end.as_deref().is_some_and(has_subquery)
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(has_subquery)
                || when_clauses
                    .iter()
                    .any(|(condition, result)| has_subquery(condition) || has_subquery(result))
                || else_clause.as_deref().is_some_and(has_subquery)
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            has_subquery(list_expr)
                || filter_expr.as_deref().is_some_and(has_subquery)
                || has_subquery(map_expr)
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => has_subquery(list_expr) || has_subquery(predicate),
        LogicalExpression::MapProjection { entries, .. } => entries.iter().any(|entry| {
            matches!(
                entry,
                crate::query::plan::MapProjectionEntry::LiteralEntry(_, value) if has_subquery(value)
            )
        }),
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => has_subquery(initial) || has_subquery(list) || has_subquery(expression),
    }
}

/// Collects all variables produced by a logical operator's subtree.
pub(crate) fn collect_operator_variables(op: &LogicalOperator, vars: &mut HashSet<String>) {
    match op {
        LogicalOperator::NodeScan(scan) => {
            vars.insert(scan.variable.clone());
            if let Some(input) = &scan.input {
                collect_operator_variables(input, vars);
            }
        }
        LogicalOperator::EdgeScan(scan) => {
            vars.insert(scan.variable.clone());
        }
        LogicalOperator::Expand(expand) => {
            vars.insert(expand.to_variable.clone());
            if let Some(edge_var) = &expand.edge_variable {
                vars.insert(edge_var.clone());
            }
            collect_operator_variables(&expand.input, vars);
        }
        LogicalOperator::Filter(filter) => {
            collect_operator_variables(&filter.input, vars);
        }
        LogicalOperator::Project(proj) => {
            for p in &proj.projections {
                // A variable passed on as it is (`WITH i`) keeps its name.
                match (&p.alias, &p.expression) {
                    (Some(name), _) | (None, LogicalExpression::Variable(name)) => {
                        vars.insert(name.clone());
                    }
                    (None, _) => {}
                }
            }
            collect_operator_variables(&proj.input, vars);
        }
        // A write passes its input's rows on, with the node or edge it
        // creates or merges (`UNWIND ... AS i CREATE (n) WITH i OPTIONAL
        // MATCH (t {k: i})` reads `i` through the write).
        LogicalOperator::CreateNode(create) => {
            vars.insert(create.variable.clone());
            if let Some(input) = &create.input {
                collect_operator_variables(input, vars);
            }
        }
        LogicalOperator::CreateEdge(create) => {
            if let Some(variable) = &create.variable {
                vars.insert(variable.clone());
            }
            collect_operator_variables(&create.input, vars);
        }
        LogicalOperator::Create(create) => {
            vars.extend(create.variables().map(str::to_string));
            if let Some(input) = &create.input {
                collect_operator_variables(input, vars);
            }
        }
        LogicalOperator::Merge(merge) => {
            vars.insert(merge.variable.clone());
            collect_operator_variables(&merge.input, vars);
        }
        LogicalOperator::MergeRelationship(merge) => {
            vars.insert(merge.variable.clone());
            collect_operator_variables(&merge.input, vars);
        }
        LogicalOperator::SetProperty(set) => collect_operator_variables(&set.input, vars),
        LogicalOperator::AddLabel(add) => collect_operator_variables(&add.input, vars),
        LogicalOperator::RemoveLabel(remove) => collect_operator_variables(&remove.input, vars),
        LogicalOperator::DeleteNode(delete) => collect_operator_variables(&delete.input, vars),
        LogicalOperator::DeleteEdge(delete) => collect_operator_variables(&delete.input, vars),
        LogicalOperator::Join(join) => {
            collect_operator_variables(&join.left, vars);
            collect_operator_variables(&join.right, vars);
        }
        LogicalOperator::LeftJoin(lj) => {
            collect_operator_variables(&lj.left, vars);
            collect_operator_variables(&lj.right, vars);
        }
        LogicalOperator::Unwind(unwind) => {
            vars.insert(unwind.variable.clone());
            collect_operator_variables(&unwind.input, vars);
        }
        LogicalOperator::Bind(bind) => {
            vars.insert(bind.variable.clone());
            collect_operator_variables(&bind.input, vars);
        }
        LogicalOperator::Aggregate(agg) => {
            for expr in &agg.group_by {
                collect_expression_variables(expr, vars);
            }
            for agg_expr in &agg.aggregates {
                if let Some(alias) = &agg_expr.alias {
                    vars.insert(alias.clone());
                }
            }
            collect_operator_variables(&agg.input, vars);
        }
        LogicalOperator::Return(ret) => {
            collect_operator_variables(&ret.input, vars);
        }
        LogicalOperator::Limit(limit) => {
            collect_operator_variables(&limit.input, vars);
        }
        LogicalOperator::Skip(skip) => {
            collect_operator_variables(&skip.input, vars);
        }
        LogicalOperator::Sort(sort) => {
            collect_operator_variables(&sort.input, vars);
        }
        LogicalOperator::Distinct(distinct) => {
            collect_operator_variables(&distinct.input, vars);
        }
        _ => {
            // For other operators, do not recurse to avoid false positives.
            // The common cases (NodeScan, Expand, Filter, Join, LeftJoin,
            // Unwind, Project, Aggregate, Return, the writes) are covered
            // above.
        }
    }
}

/// The variables on which a comma-separated part of a MATCH (`part`, the
/// variables it names, starting from the node variable `start`) is joined to
/// the rows of the parts before it, in name order, or none when the part goes
/// on from those rows instead. `clause` holds the variables of the earlier
/// parts of the clause, `input` those of the rows the clause starts from (a
/// MATCH, UNWIND or subquery import before it).
///
/// A part goes on from the rows when it shares no variable with the earlier
/// parts, or when it starts from a variable of the input: its scan reuses the
/// bound start and expands from it, and an expand to a node or edge that is
/// bound already binds a fresh variable checked against it (`close_cycles`).
/// Any other part is matched on its own and joined on every variable it
/// shares with the earlier parts or with the input, so that a part like
/// `(a)<-[:R]-(b)` after `(b:C)` still meets the `a` of the input, unless it
/// reads a value of those rows (see [`comma_part_reads_earlier_rows`]).
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn comma_part_join_variables(
    part: &HashSet<String>,
    start: Option<&str>,
    clause: &HashSet<String>,
    input: &HashSet<String>,
) -> Vec<String> {
    if part.is_disjoint(clause) || start.is_some_and(|start| input.contains(start)) {
        return Vec::new();
    }
    let mut shared: Vec<String> = part
        .iter()
        .filter(|name| clause.contains(*name) || input.contains(*name))
        .cloned()
        .collect();
    shared.sort();
    shared
}

/// Whether `part`, a comma-separated part of a MATCH translated on its own
/// (as one joined to the parts before it is), reads in its filters (its
/// property maps and inline WHERE clauses) a variable that the earlier parts
/// (`clause`) or the input bind and the part itself (`part_vars`) does not:
/// an unwound value, or a node of another part, as in `UNWIND [3, 19] AS w
/// MATCH (b:City), (b)<-[:VISITED {w: w}]-(a)`. On its own the part has no
/// such value and matches nothing, so it goes on from the rows before it
/// instead, like a part that shares no variable with them.
#[cfg(any(feature = "gql", feature = "cypher"))]
pub(crate) fn comma_part_reads_earlier_rows(
    part: &LogicalOperator,
    part_vars: &HashSet<String>,
    clause: &HashSet<String>,
    input: &HashSet<String>,
) -> bool {
    let mut read = HashSet::new();
    collect_filter_reads(part, &mut read);
    read.iter()
        .any(|name| !part_vars.contains(name) && (clause.contains(name) || input.contains(name)))
}

/// Adds the variables the filters of `op`, and of every operator below it,
/// read: the edge conditions of shortest-path searches included.
#[cfg(any(feature = "gql", feature = "cypher"))]
fn collect_filter_reads(op: &LogicalOperator, read: &mut HashSet<String>) {
    match op {
        LogicalOperator::Filter(filter) => collect_expression_variables(&filter.predicate, read),
        LogicalOperator::ShortestPath(path) => {
            if let Some(condition) = &path.edge_condition {
                collect_expression_variables(&condition.predicate, read);
            }
        }
        _ => {}
    }
    for child in op.children() {
        collect_filter_reads(child, read);
    }
}

/// The variables the rows of `rows`, a side of the left join of an OPTIONAL
/// MATCH, hold (see [`LogicalOperator::bound_variables`]): a `WITH` holds
/// only what it projects, so a name it dropped that the OPTIONAL MATCH binds
/// again is the optional part's own (LDBC IC5's `WITH forum, collect(friend)
/// AS friends OPTIONAL MATCH (friend)<-...`), a subquery's import holds the
/// variables it imports (`imports` for `WITH *`), and a shortest path holds
/// its ends and its path. Where they are not known here, every variable bound
/// below `rows`.
fn row_variables(rows: &LogicalOperator, imports: Option<&HashSet<String>>) -> HashSet<String> {
    rows.bound_variables(imports).unwrap_or_else(|| {
        let mut vars = HashSet::new();
        collect_operator_variables(rows, &mut vars);
        vars
    })
}

/// The left join of an OPTIONAL MATCH: `right` matched for each row of
/// `left`, with nulls where it has no match. A filter in `right` that reads a
/// variable only `left` binds (GQL's `(c WHERE c.age > a.age)`, or a property
/// map that reads an imported value) is a condition of the join: it decides
/// which matches count, so it moves there. `imports` is what a subquery's
/// `WITH *` imports (see [`row_variables`]).
pub(crate) fn optional_join(
    left: LogicalOperator,
    right: LogicalOperator,
    imports: Option<&HashSet<String>>,
) -> LogicalOperator {
    let left_vars = row_variables(&left, imports);
    let right_vars = row_variables(&right, imports);
    let mut moved = Vec::new();
    let right = take_left_reading_filters(right, &left_vars, &right_vars, &mut moved);
    LogicalOperator::LeftJoin(LeftJoinOp {
        left: Box::new(left),
        right: Box::new(right),
        condition: join_conjuncts(moved),
    })
}

/// Removes from the filters of `plan` (down its pattern) the conjuncts that
/// read a variable `left_vars` has and `right_vars` does not, adding them to
/// `moved`.
fn take_left_reading_filters(
    plan: LogicalOperator,
    left_vars: &HashSet<String>,
    right_vars: &HashSet<String>,
    moved: &mut Vec<LogicalExpression>,
) -> LogicalOperator {
    match plan {
        LogicalOperator::Filter(mut filter) => {
            let input = take_left_reading_filters(*filter.input, left_vars, right_vars, moved);
            let mut kept = Vec::new();
            for conjunct in split_and(filter.predicate) {
                let mut read = HashSet::new();
                collect_expression_variables(&conjunct, &mut read);
                if read
                    .iter()
                    .any(|name| left_vars.contains(name) && !right_vars.contains(name))
                {
                    moved.push(conjunct);
                } else {
                    kept.push(conjunct);
                }
            }
            match join_conjuncts(kept) {
                Some(predicate) => {
                    filter.predicate = predicate;
                    filter.input = Box::new(input);
                    LogicalOperator::Filter(filter)
                }
                None => input,
            }
        }
        LogicalOperator::Expand(mut expand) => {
            expand.input = Box::new(take_left_reading_filters(
                *expand.input,
                left_vars,
                right_vars,
                moved,
            ));
            LogicalOperator::Expand(expand)
        }
        LogicalOperator::NodeScan(mut scan) => {
            scan.input = scan.input.map(|input| {
                Box::new(take_left_reading_filters(
                    *input, left_vars, right_vars, moved,
                ))
            });
            LogicalOperator::NodeScan(scan)
        }
        // A shortest path from a node whose property map reads an earlier
        // value (`shortestPath((a {id: x})-[*]->(b))`) searches from every
        // candidate; the condition keeps the paths of the row's own.
        LogicalOperator::ShortestPath(mut path) => {
            path.input = Box::new(take_left_reading_filters(
                *path.input,
                left_vars,
                right_vars,
                moved,
            ));
            LogicalOperator::ShortestPath(path)
        }
        // The patterns of a comma list each have their filters.
        LogicalOperator::Join(mut join) => {
            join.left = Box::new(take_left_reading_filters(
                *join.left, left_vars, right_vars, moved,
            ));
            join.right = Box::new(take_left_reading_filters(
                *join.right,
                left_vars,
                right_vars,
                moved,
            ));
            LogicalOperator::Join(join)
        }
        other => other,
    }
}

/// The conjuncts of `predicate` (`a AND b AND c` gives three).
fn split_and(predicate: LogicalExpression) -> Vec<LogicalExpression> {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            let mut conjuncts = split_and(*left);
            conjuncts.extend(split_and(*right));
            conjuncts
        }
        other => vec![other],
    }
}

/// The conjunction of `conjuncts`, `None` for none.
fn join_conjuncts(conjuncts: Vec<LogicalExpression>) -> Option<LogicalExpression> {
    LogicalExpression::conjunction(conjuncts)
}

/// Builds a LeftJoin with properly classified WHERE predicates.
///
/// Given a WHERE predicate that follows an OPTIONAL MATCH (whose join is
/// `left_join`), this function:
/// 1. Collects the variables the rows of each side hold (`imports` is what a
///    subquery's `WITH *` imports, see [`row_variables`])
/// 2. Classifies the conjuncts (see [`classify_optional_predicates`])
/// 3. Pushes those that read no left-only variable as a Filter on the right
///    input
/// 4. Adds those that read a left-only variable to `LeftJoinOp.condition`,
///    which decides which pairs of rows are matches (a left row without one
///    keeps nulls)
/// 5. Returns the LeftJoin and any remaining post-filters to apply above
pub(crate) fn build_left_join_with_predicates(
    left_join: LeftJoinOp,
    predicate: Option<LogicalExpression>,
    imports: Option<&HashSet<String>>,
) -> (LogicalOperator, Option<LogicalExpression>) {
    let LeftJoinOp {
        left,
        right,
        condition,
    } = left_join;
    let (left, right) = (*left, *right);
    let Some(predicate) = predicate else {
        let join = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(left),
            right: Box::new(right),
            condition,
        });
        return (join, None);
    };

    // Collect variables from each side
    let left_vars = row_variables(&left, imports);
    let right_vars = row_variables(&right, imports);

    // Classify
    let classified = classify_optional_predicates(predicate, &left_vars, &right_vars);

    // Build right input with right-only filters pushed down
    let filtered_right = if classified.right_filters.is_empty() {
        right
    } else {
        let right_pred = LogicalExpression::conjunction(classified.right_filters)
            .expect("non-empty right_filters");
        wrap_filter(right, right_pred)
    };

    // The cross-side predicates join the condition the join had: together
    // they decide which pairs of rows are matches.
    let cross_condition = join_conjuncts(
        condition
            .into_iter()
            .chain(classified.cross_filters)
            .collect(),
    );

    let join = LogicalOperator::LeftJoin(LeftJoinOp {
        left: Box::new(left),
        right: Box::new(filtered_right),
        condition: cross_condition,
    });

    // Combine remaining post-filters
    let post_filter = LogicalExpression::conjunction(classified.post_filters);

    (join, post_filter)
}

// ---------------------------------------------------------------------------
// Plan node builder helpers
// ---------------------------------------------------------------------------

/// Adds `SET variable.property = value` on top of `plan`.
///
/// A constant value (a literal or a parameter) joins the [`SetPropertyOp`]
/// at the top when that one sets only constants on the same entity: the
/// values read nothing the earlier assignments write, so setting them
/// together is setting them one after the other, and a list of thousands of
/// assignments (`SET n.p0 = 0, n.p1 = 1, ...`) is one operator, not a chain
/// as long. Any other value gets an operator of its own.
pub(crate) fn push_set_property(
    plan: LogicalOperator,
    variable: &str,
    property: String,
    value: LogicalExpression,
    is_edge: bool,
) -> LogicalOperator {
    let constant = |value: &LogicalExpression| {
        matches!(
            value,
            LogicalExpression::Literal(_) | LogicalExpression::Parameter(_)
        )
    };
    match plan {
        LogicalOperator::SetProperty(mut set)
            if constant(&value)
                && set.variable == variable
                && set.is_edge == is_edge
                && !set.replace
                && set
                    .properties
                    .iter()
                    .all(|(name, value)| name != "*" && constant(value)) =>
        {
            set.properties.push((property, value));
            LogicalOperator::SetProperty(set)
        }
        plan => LogicalOperator::SetProperty(SetPropertyOp {
            variable: variable.to_string(),
            properties: vec![(property, value)],
            replace: false,
            is_edge,
            input: Box::new(plan),
        }),
    }
}

/// Wraps an operator with a filter predicate.
pub(crate) fn wrap_filter(input: LogicalOperator, predicate: LogicalExpression) -> LogicalOperator {
    LogicalOperator::Filter(FilterOp {
        predicate,
        input: Box::new(input),
        pushdown_hint: None,
    })
}

/// Wraps an operator with ORDER BY.
pub(crate) fn wrap_sort(input: LogicalOperator, keys: Vec<SortKey>) -> LogicalOperator {
    LogicalOperator::Sort(SortOp {
        keys,
        input: Box::new(input),
    })
}

/// Wraps an operator with SKIP.
pub(crate) fn wrap_skip(input: LogicalOperator, count: impl Into<CountExpr>) -> LogicalOperator {
    LogicalOperator::Skip(SkipOp {
        count: count.into(),
        input: Box::new(input),
    })
}

/// Wraps an operator with LIMIT.
pub(crate) fn wrap_limit(input: LogicalOperator, count: impl Into<CountExpr>) -> LogicalOperator {
    LogicalOperator::Limit(LimitOp {
        count: count.into(),
        input: Box::new(input),
    })
}

/// Output columns of a query branch, looking through ORDER BY, SKIP, LIMIT and
/// DISTINCT down to its `RETURN`. Each entry is the explicit column name (an
/// alias or a bare variable), or `None` for an unaliased expression, whose name
/// is implementation-defined. Returns `None` when the branch does not end in a
/// `RETURN` or uses `RETURN *` (columns unknown before planning).
fn branch_output_columns(op: &LogicalOperator) -> Option<Vec<Option<String>>> {
    match op {
        LogicalOperator::Return(ret) => {
            if ret
                .items
                .iter()
                .any(|item| matches!(&item.expression, LogicalExpression::Variable(v) if v == "*"))
            {
                return None;
            }
            Some(
                ret.items
                    .iter()
                    .map(|item| match (&item.alias, &item.expression) {
                        (Some(alias), _) => Some(alias.clone()),
                        (None, LogicalExpression::Variable(name)) => Some(name.clone()),
                        (None, _) => None,
                    })
                    .collect(),
            )
        }
        // A RETURN made only of aggregates and grouping keys plans to a bare
        // Aggregate: grouping keys first, then aggregates.
        LogicalOperator::Aggregate(agg) => Some(
            agg.group_by
                .iter()
                .map(|key| match key {
                    LogicalExpression::Variable(name) => Some(name.clone()),
                    _ => None,
                })
                .chain(agg.aggregates.iter().map(|a| a.alias.clone()))
                .collect(),
        ),
        // A nested UNION (`a UNION b UNION c` parses left-deep) was checked when
        // it was built; it outputs its branches' shared columns.
        LogicalOperator::Union(union) => {
            let mut merged: Option<Vec<Option<String>>> = None;
            for columns in union.inputs.iter().filter_map(branch_output_columns) {
                match &mut merged {
                    None => merged = Some(columns),
                    Some(known) => {
                        for (slot, name) in known.iter_mut().zip(columns) {
                            if slot.is_none() {
                                *slot = name;
                            }
                        }
                    }
                }
            }
            merged
        }
        // A nested EXCEPT, INTERSECT or OTHERWISE was checked when it was
        // built; it outputs its branches' columns.
        LogicalOperator::Except(op) => {
            branch_output_columns(&op.left).or_else(|| branch_output_columns(&op.right))
        }
        LogicalOperator::Intersect(op) => {
            branch_output_columns(&op.left).or_else(|| branch_output_columns(&op.right))
        }
        LogicalOperator::Otherwise(op) => {
            branch_output_columns(&op.left).or_else(|| branch_output_columns(&op.right))
        }
        LogicalOperator::Sort(sort) => branch_output_columns(&sort.input),
        LogicalOperator::Limit(limit) => branch_output_columns(&limit.input),
        LogicalOperator::Skip(skip) => branch_output_columns(&skip.input),
        LogicalOperator::Distinct(distinct) => branch_output_columns(&distinct.input),
        LogicalOperator::Filter(filter) => branch_output_columns(&filter.input),
        _ => None,
    }
}

/// Checks that the branches of a user-written set operation (`UNION`,
/// `EXCEPT`, `INTERSECT`, `OTHERWISE`, named by `operation`) are compatible, as
/// GQL (ISO/IEC 39075 14.2) and Cypher require: every branch returns the same
/// number of columns, and where both branches name a column explicitly (alias
/// or bare variable) the names match position by position.
///
/// Branches whose columns cannot be determined here (for example `RETURN *`)
/// are not checked.
///
/// # Errors
///
/// Returns a semantic error naming both column lists on the first mismatch.
pub(crate) fn check_branch_columns(operation: &str, branches: &[LogicalOperator]) -> Result<()> {
    let render = |columns: &[Option<String>]| -> String {
        columns
            .iter()
            .map(|c| c.as_deref().unwrap_or("<expression>"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut expected: Option<Vec<Option<String>>> = None;
    for branch in branches {
        let Some(columns) = branch_output_columns(branch) else {
            continue;
        };
        let Some(first) = &expected else {
            expected = Some(columns);
            continue;
        };
        let names_conflict = first
            .iter()
            .zip(&columns)
            .any(|(a, b)| matches!((a, b), (Some(a), Some(b)) if a != b));
        if first.len() != columns.len() || names_conflict {
            return Err(Error::Query(QueryError::new(
                QueryErrorKind::Semantic,
                format!(
                    "All {operation} branches must return the same columns in the same order: \
                     [{}] vs [{}]",
                    render(first),
                    render(&columns)
                ),
            )));
        }
    }
    Ok(())
}

/// Wraps an operator with DISTINCT.
pub(crate) fn wrap_distinct(input: LogicalOperator) -> LogicalOperator {
    LogicalOperator::Distinct(DistinctOp {
        input: Box::new(input),
        columns: None,
    })
}

/// Checks if a logical expression references any variable in the given set.
pub(crate) fn references_any(expr: &LogicalExpression, names: &[String]) -> bool {
    match expr {
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => false,
        LogicalExpression::Variable(v) => names.iter().any(|n| n == v),
        LogicalExpression::Property { variable, .. }
        | LogicalExpression::Labels(variable)
        | LogicalExpression::Type(variable)
        | LogicalExpression::Id(variable) => names.iter().any(|n| n == variable),
        LogicalExpression::Binary { left, right, .. } => {
            references_any(left, names) || references_any(right, names)
        }
        LogicalExpression::Unary { operand, .. } => references_any(operand, names),
        LogicalExpression::FunctionCall { args, .. } => {
            args.iter().any(|a| references_any(a, names))
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_ref().is_some_and(|e| references_any(e, names))
                || when_clauses
                    .iter()
                    .any(|(cond, val)| references_any(cond, names) || references_any(val, names))
                || else_clause
                    .as_ref()
                    .is_some_and(|e| references_any(e, names))
        }
        LogicalExpression::List(items) => items.iter().any(|i| references_any(i, names)),
        LogicalExpression::Map(entries) => entries.iter().any(|(_, v)| references_any(v, names)),
        LogicalExpression::IndexAccess { base, index } => {
            references_any(base, names) || references_any(index, names)
        }
        LogicalExpression::MapAccess { base, .. } => references_any(base, names),
        LogicalExpression::SliceAccess { base, start, end } => {
            references_any(base, names)
                || start.as_ref().is_some_and(|s| references_any(s, names))
                || end.as_ref().is_some_and(|e| references_any(e, names))
        }
        LogicalExpression::ListComprehension {
            list_expr,
            filter_expr,
            map_expr,
            ..
        } => {
            references_any(list_expr, names)
                || filter_expr
                    .as_ref()
                    .is_some_and(|f| references_any(f, names))
                || references_any(map_expr, names)
        }
        LogicalExpression::ListPredicate {
            list_expr,
            predicate,
            ..
        } => references_any(list_expr, names) || references_any(predicate, names),
        LogicalExpression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            references_any(initial, names)
                || references_any(list, names)
                || references_any(expression, names)
        }
        LogicalExpression::PatternComprehension { projection, .. } => {
            references_any(projection, names)
        }
        LogicalExpression::MapProjection { base, .. } => names.iter().any(|n| n == base),
        // Subqueries have their own scope; outer names are not referenced.
        LogicalExpression::ExistsSubquery(_)
        | LogicalExpression::CountSubquery(_)
        | LogicalExpression::ValueSubquery(_) => false,
    }
}

/// Flattens a tree of AND-joined expressions into a list of conjuncts.
pub(crate) fn flatten_and_conjuncts(expr: &LogicalExpression) -> Vec<&LogicalExpression> {
    match expr {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            let mut parts = flatten_and_conjuncts(left);
            parts.extend(flatten_and_conjuncts(right));
            parts
        }
        other => vec![other],
    }
}

/// Joins a list of expressions with AND. Returns `None` for an empty list.
pub(crate) fn join_and_conjuncts(parts: Vec<LogicalExpression>) -> Option<LogicalExpression> {
    LogicalExpression::conjunction(parts)
}

/// Wraps an operator with RETURN.
pub(crate) fn wrap_return(
    input: LogicalOperator,
    items: Vec<ReturnItem>,
    distinct: bool,
) -> LogicalOperator {
    LogicalOperator::Return(ReturnOp {
        items,
        distinct,
        input: Box::new(input),
    })
}

/// Ends a statement that has no result: one without a `RETURN` (openCypher,
/// ISO GQL), or with GQL's `FINISH`. A `RETURN` of no items, which the
/// planner runs to the end of its input for the writes and which returns no
/// rows and no columns.
pub(crate) fn no_result(input: LogicalOperator) -> LogicalOperator {
    wrap_return(input, Vec::new(), false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::Value;

    // --- edges_to_compare and GeneratedNames ---

    #[cfg(any(feature = "gql", feature = "cypher"))]
    fn edge(variable: &str, group: bool, types: &[&str]) -> EdgeOccurrence {
        EdgeOccurrence {
            variable: variable.to_string(),
            group,
            types: types
                .iter()
                .map(|edge_type| (*edge_type).to_string())
                .collect(),
        }
    }

    /// Edge patterns of types that share none never bind one edge, while an
    /// untyped one may bind the edge of any: it ties patterns of different
    /// types into one check.
    #[cfg(any(feature = "gql", feature = "cypher"))]
    #[test]
    fn edge_patterns_are_compared_when_they_may_bind_one_edge() {
        let knows = edge("a", false, &["KNOWS"]);
        let likes = edge("b", false, &["LIKES"]);
        let any = edge("c", false, &[]);
        let either = edge("d", false, &["LIKES", "KNOWS"]);
        assert_eq!(
            edges_to_compare(&[knows.clone(), likes.clone()], true),
            Vec::<Vec<usize>>::new()
        );
        assert_eq!(
            edges_to_compare(&[knows.clone(), either], true),
            [vec![0, 1]]
        );
        assert_eq!(
            edges_to_compare(&[knows.clone(), likes.clone(), any], true),
            [vec![0, 1, 2]],
            "the untyped pattern may share an edge with both"
        );
        assert_eq!(
            edges_to_compare(&[knows.clone(), likes, knows.clone(), knows], true),
            [vec![0, 2, 3]]
        );
    }

    /// A group variable on its own is checked only where its expand may
    /// repeat an edge.
    #[cfg(any(feature = "gql", feature = "cypher"))]
    #[test]
    fn a_lone_group_is_checked_only_on_request() {
        let group = edge("g", true, &["KNOWS"]);
        let single = edge("s", false, &["LIKES"]);
        assert_eq!(
            edges_to_compare(&[group.clone(), single.clone()], true),
            [vec![0]]
        );
        assert_eq!(
            edges_to_compare(&[group, single], false),
            Vec::<Vec<usize>>::new()
        );
    }

    /// The names of a statement count from 0 and skip the words it spells.
    #[cfg(any(feature = "gql", feature = "cypher"))]
    #[test]
    fn generated_names_skip_the_words_of_the_statement() {
        let names = GeneratedNames::new("MATCH (_anon_0)-->(`_anon_2`) RETURN '_anon_3'");
        let made: Vec<String> = (0..3).map(|_| names.next("_anon_")).collect();
        assert_eq!(made, ["_anon_1", "_anon_4", "_anon_5"]);
        assert_eq!(names.next("_path_"), "_path_6");
    }

    // --- capitalize_first ---

    #[test]
    fn capitalize_first_empty() {
        assert_eq!(capitalize_first(""), "");
    }

    #[test]
    fn capitalize_first_single_char() {
        assert_eq!(capitalize_first("a"), "A");
    }

    #[test]
    fn capitalize_first_already_upper() {
        assert_eq!(capitalize_first("Hello"), "Hello");
    }

    #[test]
    fn capitalize_first_lower() {
        assert_eq!(capitalize_first("person"), "Person");
    }

    // --- VarGen ---

    #[test]
    fn var_gen_starts_at_zero() {
        let vg = VarGen::new();
        assert_eq!(vg.current(), 0);
    }

    #[test]
    fn var_gen_increments() {
        let vg = VarGen::new();
        assert_eq!(vg.next(), "_v0");
        assert_eq!(vg.next(), "_v1");
        assert_eq!(vg.current(), 2);
    }

    // --- is_aggregate_function ---

    #[test]
    fn aggregate_functions_recognized() {
        for name in [
            "count",
            "COUNT",
            "sum",
            "avg",
            "min",
            "max",
            "collect",
            "stdev",
            "stddev",
            "stdevp",
            "stddevp",
            "stddev_samp",
            "STDDEV_POP",
            "variance",
            "VARIANCE",
            "var_samp",
            "VAR_POP",
            "percentile_disc",
            "percentiledisc",
            "percentile_cont",
            "percentilecont",
        ] {
            assert!(is_aggregate_function(name), "{name} should be aggregate");
        }
    }

    #[test]
    fn non_aggregate_functions_rejected() {
        for name in ["toString", "toUpper", "size", "rand", "abs", "coalesce", ""] {
            assert!(
                !is_aggregate_function(name),
                "{name} should not be aggregate"
            );
        }
    }

    // --- to_aggregate_function ---

    #[test]
    fn to_aggregate_all_variants() {
        assert!(matches!(
            to_aggregate_function("count"),
            Some(AggregateFunction::Count)
        ));
        assert!(matches!(
            to_aggregate_function("SUM"),
            Some(AggregateFunction::Sum)
        ));
        assert!(matches!(
            to_aggregate_function("Avg"),
            Some(AggregateFunction::Avg)
        ));
        assert!(matches!(
            to_aggregate_function("MIN"),
            Some(AggregateFunction::Min)
        ));
        assert!(matches!(
            to_aggregate_function("max"),
            Some(AggregateFunction::Max)
        ));
        assert!(matches!(
            to_aggregate_function("collect"),
            Some(AggregateFunction::Collect)
        ));
        assert!(matches!(
            to_aggregate_function("stdev"),
            Some(AggregateFunction::StdDev)
        ));
        assert!(matches!(
            to_aggregate_function("stddev"),
            Some(AggregateFunction::StdDev)
        ));
        assert!(matches!(
            to_aggregate_function("stdevp"),
            Some(AggregateFunction::StdDevPop)
        ));
        assert!(matches!(
            to_aggregate_function("stddevp"),
            Some(AggregateFunction::StdDevPop)
        ));
        assert!(matches!(
            to_aggregate_function("stddev_samp"),
            Some(AggregateFunction::StdDev)
        ));
        assert!(matches!(
            to_aggregate_function("STDDEV_POP"),
            Some(AggregateFunction::StdDevPop)
        ));
        assert!(matches!(
            to_aggregate_function("variance"),
            Some(AggregateFunction::Variance)
        ));
        assert!(matches!(
            to_aggregate_function("VAR_SAMP"),
            Some(AggregateFunction::Variance)
        ));
        assert!(matches!(
            to_aggregate_function("VAR_POP"),
            Some(AggregateFunction::VariancePop)
        ));
        assert!(matches!(
            to_aggregate_function("percentile_disc"),
            Some(AggregateFunction::PercentileDisc)
        ));
        assert!(matches!(
            to_aggregate_function("percentiledisc"),
            Some(AggregateFunction::PercentileDisc)
        ));
        assert!(matches!(
            to_aggregate_function("percentile_cont"),
            Some(AggregateFunction::PercentileCont)
        ));
        assert!(matches!(
            to_aggregate_function("percentilecont"),
            Some(AggregateFunction::PercentileCont)
        ));
    }

    #[test]
    fn to_aggregate_unknown_returns_none() {
        assert!(to_aggregate_function("unknown").is_none());
        assert!(to_aggregate_function("").is_none());
        assert!(to_aggregate_function("size").is_none());
    }

    // --- is_aggregate_function: bivariate / concat / sample ---

    #[test]
    fn aggregate_functions_bivariate_recognized() {
        for name in [
            "group_concat",
            "GROUPCONCAT",
            "listagg",
            "LISTAGG",
            "sample",
            "SAMPLE",
            "covar_samp",
            "COVAR_SAMP",
            "covar_pop",
            "COVAR_POP",
            "corr",
            "CORR",
            "regr_slope",
            "REGR_SLOPE",
            "regr_intercept",
            "REGR_INTERCEPT",
            "regr_r2",
            "REGR_R2",
            "regr_count",
            "REGR_COUNT",
            "regr_sxx",
            "REGR_SXX",
            "regr_syy",
            "REGR_SYY",
            "regr_sxy",
            "REGR_SXY",
            "regr_avgx",
            "REGR_AVGX",
            "regr_avgy",
            "REGR_AVGY",
        ] {
            assert!(is_aggregate_function(name), "{name} should be aggregate");
        }
    }

    // --- to_aggregate_function: bivariate variants ---

    #[test]
    fn to_aggregate_bivariate_variants() {
        assert!(matches!(
            to_aggregate_function("group_concat"),
            Some(AggregateFunction::GroupConcat)
        ));
        assert!(matches!(
            to_aggregate_function("GROUPCONCAT"),
            Some(AggregateFunction::GroupConcat)
        ));
        assert!(matches!(
            to_aggregate_function("LISTAGG"),
            Some(AggregateFunction::GroupConcat)
        ));
        assert!(matches!(
            to_aggregate_function("sample"),
            Some(AggregateFunction::Sample)
        ));
        assert!(matches!(
            to_aggregate_function("COVAR_SAMP"),
            Some(AggregateFunction::CovarSamp)
        ));
        assert!(matches!(
            to_aggregate_function("COVAR_POP"),
            Some(AggregateFunction::CovarPop)
        ));
        assert!(matches!(
            to_aggregate_function("CORR"),
            Some(AggregateFunction::Corr)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SLOPE"),
            Some(AggregateFunction::RegrSlope)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_INTERCEPT"),
            Some(AggregateFunction::RegrIntercept)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_R2"),
            Some(AggregateFunction::RegrR2)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_COUNT"),
            Some(AggregateFunction::RegrCount)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SXX"),
            Some(AggregateFunction::RegrSxx)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SYY"),
            Some(AggregateFunction::RegrSyy)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_SXY"),
            Some(AggregateFunction::RegrSxy)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_AVGX"),
            Some(AggregateFunction::RegrAvgx)
        ));
        assert!(matches!(
            to_aggregate_function("REGR_AVGY"),
            Some(AggregateFunction::RegrAvgy)
        ));
    }

    // --- is_binary_set_function ---

    #[test]
    fn binary_set_functions_recognized() {
        let binary = [
            AggregateFunction::CovarSamp,
            AggregateFunction::CovarPop,
            AggregateFunction::Corr,
            AggregateFunction::RegrSlope,
            AggregateFunction::RegrIntercept,
            AggregateFunction::RegrR2,
            AggregateFunction::RegrCount,
            AggregateFunction::RegrSxx,
            AggregateFunction::RegrSyy,
            AggregateFunction::RegrSxy,
            AggregateFunction::RegrAvgx,
            AggregateFunction::RegrAvgy,
        ];
        for func in binary {
            assert!(is_binary_set_function(func), "{func:?} should be binary");
        }
    }

    #[test]
    fn non_binary_set_functions_rejected() {
        let non_binary = [
            AggregateFunction::Count,
            AggregateFunction::CountNonNull,
            AggregateFunction::Sum,
            AggregateFunction::Avg,
            AggregateFunction::Min,
            AggregateFunction::Max,
            AggregateFunction::Collect,
            AggregateFunction::StdDev,
            AggregateFunction::StdDevPop,
            AggregateFunction::Variance,
            AggregateFunction::VariancePop,
            AggregateFunction::PercentileDisc,
            AggregateFunction::PercentileCont,
            AggregateFunction::GroupConcat,
            AggregateFunction::Sample,
        ];
        for func in non_binary {
            assert!(
                !is_binary_set_function(func),
                "{func:?} should not be binary"
            );
        }
    }

    // --- combine_with_and ---

    #[test]
    fn combine_with_and_empty_returns_error() {
        let result = combine_with_and(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn combine_with_and_single_predicate() {
        let pred = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "name".to_string(),
        };
        let result = combine_with_and(vec![pred.clone()]).unwrap();
        assert!(matches!(result, LogicalExpression::Property { .. }));
    }

    #[test]
    fn combine_with_and_two_predicates() {
        let p1 = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "a".to_string(),
        };
        let p2 = LogicalExpression::Property {
            variable: "n".to_string(),
            property: "b".to_string(),
        };
        let result = combine_with_and(vec![p1, p2]).unwrap();
        assert!(matches!(
            result,
            LogicalExpression::Binary {
                op: BinaryOp::And,
                ..
            }
        ));
    }

    // --- split_conjuncts ---

    #[test]
    fn split_conjuncts_single() {
        let expr = LogicalExpression::Variable("x".into());
        let conjuncts = split_conjuncts(expr);
        assert_eq!(conjuncts.len(), 1);
    }

    #[test]
    fn split_conjuncts_nested_and() {
        // (a AND b) AND c -> [a, b, c]
        let a = LogicalExpression::Variable("a".into());
        let b = LogicalExpression::Variable("b".into());
        let c = LogicalExpression::Variable("c".into());
        let ab = LogicalExpression::Binary {
            left: Box::new(a),
            op: BinaryOp::And,
            right: Box::new(b),
        };
        let abc = LogicalExpression::Binary {
            left: Box::new(ab),
            op: BinaryOp::And,
            right: Box::new(c),
        };
        let conjuncts = split_conjuncts(abc);
        assert_eq!(conjuncts.len(), 3);
    }

    #[test]
    fn split_conjuncts_or_not_split() {
        // a OR b should NOT be split (only AND is split)
        let a = LogicalExpression::Variable("a".into());
        let b = LogicalExpression::Variable("b".into());
        let or_expr = LogicalExpression::Binary {
            left: Box::new(a),
            op: BinaryOp::Or,
            right: Box::new(b),
        };
        let conjuncts = split_conjuncts(or_expr);
        assert_eq!(conjuncts.len(), 1);
    }

    // --- classify_optional_predicates ---

    #[test]
    fn classify_left_only_predicate() {
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["m".into()].into_iter().collect();
        let pred = LogicalExpression::Property {
            variable: "n".into(),
            property: "age".into(),
        };

        // A condition on the rows before the OPTIONAL MATCH decides which
        // matches count: a join condition, never a filter of those rows
        let result = classify_optional_predicates(pred, &left_vars, &right_vars);
        assert_eq!(
            result.cross_filters.len(),
            1,
            "left-only should be a join condition"
        );
        assert!(result.post_filters.is_empty());
        assert!(result.right_filters.is_empty());
    }

    #[test]
    fn classify_shared_only_and_constant_predicates_as_right() {
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["n".into(), "m".into()].into_iter().collect();
        let shared = LogicalExpression::Property {
            variable: "n".into(),
            property: "active".into(),
        };
        let constant = LogicalExpression::Literal(Value::Bool(false));
        let combined = LogicalExpression::Binary {
            left: Box::new(shared),
            op: BinaryOp::And,
            right: Box::new(constant),
        };

        let result = classify_optional_predicates(combined, &left_vars, &right_vars);
        assert_eq!(result.right_filters.len(), 2, "both filter the matches");
        assert!(result.post_filters.is_empty());
        assert!(result.cross_filters.is_empty());
    }

    #[test]
    fn classify_right_only_predicate() {
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["m".into()].into_iter().collect();
        let pred = LogicalExpression::Property {
            variable: "m".into(),
            property: "age".into(),
        };

        let result = classify_optional_predicates(pred, &left_vars, &right_vars);
        assert!(result.post_filters.is_empty());
        assert_eq!(
            result.right_filters.len(),
            1,
            "right-only should be right-filter"
        );
    }

    #[test]
    fn classify_shared_variable_as_right() {
        // Variable `n` is in both left_vars and right_vars (shared).
        // A predicate on `m.age > n.age` has both m and n on right side,
        // so it should be classified as right-filter.
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["n".into(), "m".into()].into_iter().collect();
        let pred = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "m".into(),
                property: "age".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Property {
                variable: "n".into(),
                property: "age".into(),
            }),
        };

        let result = classify_optional_predicates(pred, &left_vars, &right_vars);
        assert!(result.post_filters.is_empty());
        assert_eq!(
            result.right_filters.len(),
            1,
            "shared variable predicate should be right-filter"
        );
    }

    #[test]
    fn classify_mixed_and_predicate() {
        // n.active AND m.age > 30 should split into post-filter and right-filter
        let left_vars: HashSet<String> = ["n".into()].into_iter().collect();
        let right_vars: HashSet<String> = ["n".into(), "m".into()].into_iter().collect();

        let left_pred = LogicalExpression::Property {
            variable: "n".into(),
            property: "active".into(),
        };
        let right_pred = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "m".into(),
                property: "age".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Literal(Value::Int64(30))),
        };
        let combined = LogicalExpression::Binary {
            left: Box::new(left_pred),
            op: BinaryOp::And,
            right: Box::new(right_pred),
        };

        let result = classify_optional_predicates(combined, &left_vars, &right_vars);

        // n.active -> n is in both left and right, so all_in_right = true
        // It should be classified as right_filter since n is available on right side.
        // Actually n.active references only n, which IS in right_vars too.
        // So both predicates should be right_filters.
        // BUT we also need to check: n.active only references n, which is in left_vars.
        // Since n is in BOTH sets, all_in_left AND all_in_right are true.
        // The logic checks all_in_right FIRST, so it goes to right_filters.
        assert_eq!(
            result.right_filters.len() + result.post_filters.len(),
            2,
            "two conjuncts should be classified"
        );
    }

    // --- collect_expression_variables ---

    #[test]
    fn collect_vars_from_property() {
        let expr = LogicalExpression::Property {
            variable: "n".into(),
            property: "age".into(),
        };
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.contains("n"));
        assert_eq!(vars.len(), 1);
    }

    #[test]
    fn collect_vars_from_binary() {
        let expr = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: "a".into(),
                property: "x".into(),
            }),
            op: BinaryOp::Gt,
            right: Box::new(LogicalExpression::Property {
                variable: "b".into(),
                property: "y".into(),
            }),
        };
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.contains("a"));
        assert!(vars.contains("b"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn collect_vars_from_function_call() {
        let expr = LogicalExpression::FunctionCall {
            name: "size".into(),
            args: vec![LogicalExpression::Variable("list".into())],
            distinct: false,
        };
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.contains("list"));
    }

    #[test]
    fn collect_vars_from_literal_is_empty() {
        let expr = LogicalExpression::Literal(Value::Int64(42));
        let mut vars = HashSet::new();
        collect_expression_variables(&expr, &mut vars);
        assert!(vars.is_empty());
    }

    // --- graphql_directives_allow ---
    //
    // These tests require the `graphql` feature so that the
    // `grafeo_adapters::query::graphql::ast` types are available.

    #[cfg(feature = "graphql")]
    mod graphql_directive_tests {
        use super::super::graphql_directives_allow;
        use grafeo_adapters::query::graphql::ast::{Argument, Directive, InputValue};

        fn directive(name: &str, if_value: bool) -> Directive {
            Directive {
                name: name.to_string(),
                arguments: vec![Argument {
                    name: "if".to_string(),
                    value: InputValue::Boolean(if_value),
                }],
            }
        }

        #[test]
        fn test_skip_directive_true_excludes() {
            // @skip(if: true) should exclude the field
            let directives = vec![directive("skip", true)];
            assert!(
                !graphql_directives_allow(&directives),
                "@skip(if: true) should return false (field excluded)"
            );
        }

        #[test]
        fn test_skip_directive_false_includes() {
            // @skip(if: false) should include the field
            let directives = vec![directive("skip", false)];
            assert!(
                graphql_directives_allow(&directives),
                "@skip(if: false) should return true (field included)"
            );
        }

        #[test]
        fn test_include_directive_true_includes() {
            // @include(if: true) should include the field
            let directives = vec![directive("include", true)];
            assert!(
                graphql_directives_allow(&directives),
                "@include(if: true) should return true"
            );
        }

        #[test]
        fn test_include_directive_false_excludes() {
            // @include(if: false) should exclude the field
            let directives = vec![directive("include", false)];
            assert!(
                !graphql_directives_allow(&directives),
                "@include(if: false) should return false"
            );
        }

        #[test]
        fn test_no_directives_includes() {
            // No directives at all should include the field
            let directives: Vec<Directive> = vec![];
            assert!(
                graphql_directives_allow(&directives),
                "empty directives should return true"
            );
        }

        #[test]
        fn test_unknown_directive_ignored() {
            // Unknown directives should be ignored, field included
            let directives = vec![Directive {
                name: "deprecated".to_string(),
                arguments: vec![],
            }];
            assert!(
                graphql_directives_allow(&directives),
                "unknown directive should be ignored, returning true"
            );
        }
    }
}
