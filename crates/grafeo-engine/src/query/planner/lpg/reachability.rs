//! Reachability search for variable-length expands.
//!
//! A variable-length expand in WALK mode emits one row per walk, and many of
//! those rows can repeat an (input row, target) pair: an undirected pattern
//! walks back along the edge it came in on, and longer walks pass the same
//! nodes again, the more so around nodes with many neighbors. When the rows
//! only reach an operator that ignores duplicate rows, the expand emits each
//! node an input row reaches once instead (see
//! [`VariableLengthExpandOperator::with_reachability`](grafeo_core::execution::operators::VariableLengthExpandOperator::with_reachability)),
//! in the order of its first walk, so that operator returns the same rows in
//! the same order. When that operator reads nothing of the input row but the
//! target, the expand emits each node once over all input rows (see
//! [`VariableLengthExpandOperator::with_reachability_across_rows`](grafeo_core::execution::operators::VariableLengthExpandOperator::with_reachability_across_rows)).
//! This module finds those expands; EXPLAIN and PROFILE mark them
//! `[reachability]` and `[reachability: once]`.

use std::collections::HashSet;

use crate::query::plan::{
    AggregateExpr, AggregateFunction, AggregateOp, ExpandOp, LogicalExpression, LogicalOperator,
    MapProjectionEntry, PathMode, SortKey,
};
use crate::query::planner::common::{
    aggregate_column_name, expression_to_string, output_column_name,
};

/// How a variable-length expand runs its reachability search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReachabilityMode {
    /// Each node once per input row.
    PerInputRow,
    /// Each node once over all input rows.
    AcrossInputRows,
}

impl ReachabilityMode {
    /// How EXPLAIN and PROFILE mark an expand that runs in this mode.
    pub(crate) fn marker(self) -> &'static str {
        match self {
            Self::PerInputRow => "[reachability]",
            Self::AcrossInputRows => "[reachability: once]",
        }
    }
}

/// The variable-length expands in `root` that run as a reachability search,
/// and how.
///
/// Such an expand is in WALK mode, and its rows reach a consumer that ignores
/// duplicate rows through filters, projections and RETURNNs only. The consumer
/// is a `RETURN DISTINCT`, a `DISTINCT` over all columns (`WITH DISTINCT`), or
/// an aggregation whose every aggregate ignores duplicates. Nothing on the way
/// reads the expand's edge variable or its path (nor the path's length, nodes
/// or edges), and every expression on the way gives the same value for each
/// copy of a row (no subquery, no `rand()`), so leaving out the copies changes
/// no result. When nothing on the way reads a column bound below the expand
/// either, only the target and values computed from it, the rows of a target
/// are copies whatever input row reached it, and the search runs across input
/// rows.
///
/// The subplan of an Apply (a CALL subquery) is not searched: the planner may
/// plan a copy of it, whose expands a mark on the original would not reach.
pub(crate) fn reachability_expands(root: &LogicalOperator) -> Vec<(&ExpandOp, ReachabilityMode)> {
    let mut found = Vec::new();
    collect(root, &[], &mut found);
    found
}

/// Adds the reachability expands at and below `op` to `found`. `sort_keys`
/// are those of a sort right above `op`, which the planner may add to the
/// items of a `RETURN DISTINCT` (see `plan_sort`).
fn collect<'a>(
    op: &'a LogicalOperator,
    sort_keys: &[SortKey],
    found: &mut Vec<(&'a ExpandOp, ReachabilityMode)>,
) {
    if let Some(expand) = consumed_expand(op, sort_keys) {
        found.push(expand);
    }
    match op {
        LogicalOperator::Sort(sort) => collect(&sort.input, &sort.keys, found),
        LogicalOperator::Apply(apply) => collect(&apply.input, &[], found),
        _ => {
            for child in op.children() {
                collect(child, &[], found);
            }
        }
    }
}

/// The expand whose rows only reach `op`, when `op` ignores duplicate rows
/// and the expand can run as a reachability search for it, and how.
fn consumed_expand<'a>(
    op: &'a LogicalOperator,
    sort_keys: &[SortKey],
) -> Option<(&'a ExpandOp, ReachabilityMode)> {
    let mut reads = HashSet::new();
    // Whether the consumer's key is every column below it, the edge column of
    // the expand among them
    let mut all_columns = false;
    let mut below = match op {
        LogicalOperator::Return(ret) if ret.distinct => {
            let items = ret.items.iter().map(|item| &item.expression);
            let keys = sort_keys.iter().map(|key| &key.expression);
            if !row_reads_all(items.chain(keys), &mut reads) {
                return None;
            }
            ret.input.as_ref()
        }
        LogicalOperator::Distinct(distinct) if distinct.columns.is_none() => {
            all_columns = true;
            distinct.input.as_ref()
        }
        LogicalOperator::Aggregate(agg)
            if agg.aggregates.iter().all(ignores_duplicates) && having_reads_outputs(agg) =>
        {
            let arguments = agg
                .aggregates
                .iter()
                .flat_map(|aggregate| [&aggregate.expression, &aggregate.expression2])
                .flatten();
            if !row_reads_all(agg.group_by.iter().chain(arguments), &mut reads) {
                return None;
            }
            agg.input.as_ref()
        }
        _ => return None,
    };
    // The operators between the consumer and the expand, from the top
    let mut between = Vec::new();
    let expand = loop {
        let next = match below {
            LogicalOperator::Filter(filter) => {
                if !row_reads(&filter.predicate, &mut reads) {
                    return None;
                }
                filter.input.as_ref()
            }
            LogicalOperator::Project(project) => {
                let expressions = project.projections.iter().map(|p| &p.expression);
                if !row_reads_all(expressions, &mut reads) {
                    return None;
                }
                all_columns &= project.pass_through_input;
                project.input.as_ref()
            }
            LogicalOperator::Return(ret) if !ret.distinct => {
                let items = ret.items.iter().map(|item| &item.expression);
                if !row_reads_all(items, &mut reads) {
                    return None;
                }
                all_columns = false;
                ret.input.as_ref()
            }
            LogicalOperator::Expand(expand) => break expand,
            _ => return None,
        };
        between.push(below);
        below = next;
    };
    let searchable = expand.is_variable_length()
        && expand.path_mode == PathMode::Walk
        && !all_columns
        && expand
            .edge_variable
            .as_ref()
            .is_none_or(|edge| !reads.contains(edge))
        && expand
            .path_alias
            .as_ref()
            .is_none_or(|path| !reads_path(path, &reads));
    if !searchable {
        return None;
    }
    let mode = if reads_only_the_target(op, sort_keys, &between, expand) {
        ReachabilityMode::AcrossInputRows
    } else {
        ReachabilityMode::PerInputRow
    };
    Some((expand, mode))
}

/// Whether the `consumer` of `expand`, and the operators `between` them
/// (from the top), read nothing bound below the expand (its input columns,
/// the source among them): only its target and values computed from it.
///
/// Going up from the expand, `derived` holds the names whose value comes from
/// the target alone, and `columns` every name a row holds, so that a name
/// given to a second column (by a projection that keeps its input) counts as
/// read from below.
fn reads_only_the_target(
    consumer: &LogicalOperator,
    sort_keys: &[SortKey],
    between: &[&LogicalOperator],
    expand: &ExpandOp,
) -> bool {
    let Some(mut columns) = expand.input.bound_variables(None) else {
        return false;
    };
    // A target with the name of a column below would be read as either
    if columns.contains(&expand.to_variable) {
        return false;
    }
    columns.insert(expand.to_variable.clone());
    columns.extend(expand.edge_variable.iter().cloned());
    columns.extend(expand.path_alias.iter().cloned());
    let mut derived = HashSet::from([expand.to_variable.clone()]);
    // Whether every column of the row is derived (none is at the expand)
    let mut all_derived = false;
    for op in between.iter().rev() {
        let (outputs, keeps_input): (Vec<(String, bool)>, bool) = match op {
            LogicalOperator::Filter(filter) => {
                if !reads_only(&filter.predicate, &derived) {
                    return false;
                }
                continue;
            }
            LogicalOperator::Project(project) => (
                project
                    .projections
                    .iter()
                    .map(|p| output(p.alias.as_deref(), &p.expression, &derived))
                    .collect(),
                project.pass_through_input,
            ),
            LogicalOperator::Return(ret) => (
                ret.items
                    .iter()
                    .map(|item| output(item.alias.as_deref(), &item.expression, &derived))
                    .collect(),
                false,
            ),
            _ => return false,
        };
        if keeps_input {
            for (name, is_derived) in outputs {
                all_derived &= is_derived;
                if is_derived && !columns.contains(&name) {
                    derived.insert(name.clone());
                } else {
                    derived.remove(&name);
                }
                columns.insert(name);
            }
        } else {
            all_derived = outputs.iter().all(|(_, is_derived)| *is_derived);
            columns = outputs.iter().map(|(name, _)| name.clone()).collect();
            // A name given to two columns is derived only if both are
            derived.clone_from(&columns);
            for (name, is_derived) in &outputs {
                if !is_derived {
                    derived.remove(name);
                }
            }
        }
    }
    match consumer {
        LogicalOperator::Return(ret) => {
            let items: Vec<(String, bool)> = ret
                .items
                .iter()
                .map(|item| output(item.alias.as_deref(), &item.expression, &derived))
                .collect();
            if !items.iter().all(|(_, is_derived)| *is_derived) {
                return false;
            }
            // A sort key reads the returned values, or the row below them;
            // an alias that also names a column below would be ambiguous
            let mut sortable = derived;
            sortable.extend(
                items
                    .into_iter()
                    .map(|(name, _)| name)
                    .filter(|name| !columns.contains(name)),
            );
            sort_keys
                .iter()
                .all(|key| reads_only(&key.expression, &sortable))
        }
        LogicalOperator::Distinct(_) => all_derived,
        LogicalOperator::Aggregate(agg) => {
            let arguments = agg
                .aggregates
                .iter()
                .flat_map(|aggregate| [&aggregate.expression, &aggregate.expression2])
                .flatten();
            agg.group_by
                .iter()
                .chain(arguments)
                .all(|expr| reads_only(expr, &derived))
        }
        _ => false,
    }
}

/// The name of the column `expr` computes, and whether it reads only
/// `derived` names.
fn output(
    alias: Option<&str>,
    expr: &LogicalExpression,
    derived: &HashSet<String>,
) -> (String, bool) {
    (output_column_name(alias, expr), reads_only(expr, derived))
}

/// Whether `expr` is a function of the row that reads only `names`.
fn reads_only(expr: &LogicalExpression, names: &HashSet<String>) -> bool {
    let mut reads = HashSet::new();
    row_reads(expr, &mut reads) && reads.is_subset(names)
}

/// Whether `reads` holds the path `path` or a column the planner derives
/// from it (see `plan_expand`).
fn reads_path(path: &str, reads: &HashSet<String>) -> bool {
    reads.contains(path)
        || ["_path_length_", "_path_nodes_", "_path_edges_"]
            .iter()
            .any(|prefix| reads.contains(&format!("{prefix}{path}")))
}

/// Whether `aggregate` gives the same result for any number of copies of a
/// row: `min` and `max`, and the aggregates that honor `DISTINCT` (see
/// `AggregateState::new`), over distinct values. The statistical ones
/// (`stDev`, `variance`, the percentiles, the covariance and regression
/// family) count every row even with `DISTINCT`, so they are left out.
fn ignores_duplicates(aggregate: &AggregateExpr) -> bool {
    match aggregate.function {
        AggregateFunction::Min | AggregateFunction::Max => true,
        AggregateFunction::Count
        | AggregateFunction::CountNonNull
        | AggregateFunction::Sum
        | AggregateFunction::Avg
        | AggregateFunction::Collect
        | AggregateFunction::GroupConcat => aggregate.distinct && aggregate.expression.is_some(),
        _ => false,
    }
}

/// Whether the HAVING filter of `agg`, if it has one, reads only its group
/// keys and aggregates, the columns it computes; anything else could read a
/// value from one of the rows of a group.
fn having_reads_outputs(agg: &AggregateOp) -> bool {
    let Some(having) = &agg.having else {
        return true;
    };
    let outputs: HashSet<String> = agg
        .group_by
        .iter()
        .map(expression_to_string)
        .chain(agg.aggregates.iter().map(|aggregate| {
            aggregate
                .alias
                .clone()
                .unwrap_or_else(|| aggregate_column_name(aggregate))
        }))
        .collect();
    reads_only(having, &outputs)
}

/// [`row_reads`] over all of `expressions`.
fn row_reads_all<'a>(
    expressions: impl IntoIterator<Item = &'a LogicalExpression>,
    reads: &mut HashSet<String>,
) -> bool {
    expressions
        .into_iter()
        .all(|expression| row_reads(expression, reads))
}

/// Adds the variables `expr` reads to `reads`. False when `expr` is not
/// known to be a function of the row alone: a subquery (it plans more of the
/// graph, which this module does not follow), a function that can change
/// between calls or that `same_on_every_call` does not list, or the `*` of
/// `RETURN *`, which reads every column.
fn row_reads(expr: &LogicalExpression, reads: &mut HashSet<String>) -> bool {
    match expr {
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => true,
        LogicalExpression::Variable(name) if name == "*" => false,
        LogicalExpression::Variable(name)
        | LogicalExpression::Property { variable: name, .. }
        | LogicalExpression::Labels(name)
        | LogicalExpression::Type(name)
        | LogicalExpression::Id(name) => {
            reads.insert(name.clone());
            true
        }
        LogicalExpression::Binary { left, right, .. } => {
            row_reads(left, reads) && row_reads(right, reads)
        }
        LogicalExpression::Unary { operand: base, .. }
        | LogicalExpression::MapAccess { base, .. } => row_reads(base, reads),
        LogicalExpression::IndexAccess { base, index } => {
            row_reads(base, reads) && row_reads(index, reads)
        }
        LogicalExpression::SliceAccess { base, start, end } => {
            row_reads(base, reads)
                && row_reads_all([start, end].into_iter().flatten().map(Box::as_ref), reads)
        }
        LogicalExpression::List(items) => row_reads_all(items, reads),
        LogicalExpression::Map(entries) => {
            row_reads_all(entries.iter().map(|(_, value)| value), reads)
        }
        LogicalExpression::MapProjection { base, entries } => {
            reads.insert(base.clone());
            row_reads_all(
                entries.iter().filter_map(|entry| match entry {
                    MapProjectionEntry::LiteralEntry(_, value) => Some(value),
                    _ => None,
                }),
                reads,
            )
        }
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            row_reads_all(operand.iter().chain(else_clause).map(Box::as_ref), reads)
                && when_clauses
                    .iter()
                    .all(|(when, then)| row_reads(when, reads) && row_reads(then, reads))
        }
        LogicalExpression::FunctionCall { name, args, .. } => {
            same_on_every_call(name, args.len()) && row_reads_all(args, reads)
        }
        // The variable a comprehension, list predicate or `reduce` binds is
        // its own: only the other names its body uses come from the row
        LogicalExpression::ListComprehension {
            variable,
            list_expr,
            filter_expr,
            map_expr,
        } => {
            let mut body = HashSet::new();
            let pure = row_reads(list_expr, reads)
                && row_reads_all(filter_expr.iter().map(Box::as_ref), &mut body)
                && row_reads(map_expr, &mut body);
            body.remove(variable);
            reads.extend(body);
            pure
        }
        LogicalExpression::ListPredicate {
            variable,
            list_expr,
            predicate,
            ..
        } => {
            let mut body = HashSet::new();
            let pure = row_reads(list_expr, reads) && row_reads(predicate, &mut body);
            body.remove(variable);
            reads.extend(body);
            pure
        }
        LogicalExpression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            let mut body = HashSet::new();
            let pure = row_reads(initial, reads)
                && row_reads(list, reads)
                && row_reads(expression, &mut body);
            body.remove(accumulator);
            body.remove(variable);
            reads.extend(body);
            pure
        }
        _ => false,
    }
}

/// Whether the function `name` with `arity` arguments gives the same value
/// for the same arguments: the functions a seek evaluates once
/// ([`deterministic`](super::seek::deterministic)), and those that read an
/// entity's identity or labels (a label in a node pattern becomes a filter on
/// `hasLabel`).
fn same_on_every_call(name: &str, arity: usize) -> bool {
    const ENTITY: [&str; 6] = [
        "id",
        "elementid",
        "element_id",
        "labels",
        "type",
        "haslabel",
    ];
    super::seek::deterministic(name, arity) || ENTITY.iter().any(|f| name.eq_ignore_ascii_case(f))
}

#[cfg(all(test, feature = "gql", feature = "cypher"))]
mod tests {
    use std::collections::{HashMap, HashSet};

    use grafeo_common::types::Value;

    use super::ReachabilityMode::{self, AcrossInputRows, PerInputRow};
    use crate::GrafeoDB;

    /// People 1 to 5 and cities 10 to 14. KNOWS: a triangle Alix, Gus,
    /// Vincent (with the edge from Alix to Gus twice), Gus and Jules to Mia,
    /// and Mia to herself. LIKES: Alix to Jules, Vincent to Amsterdam.
    /// LIVES_IN: Mia, the hub, to every city. `active` is false for Vincent
    /// and Paris, and missing on Mia.
    fn people() -> GrafeoDB {
        let db = GrafeoDB::new_in_memory();
        let node = |label: &str, id: i64, name: &str, active: Option<bool>| {
            let mut props = vec![("id", Value::from(id)), ("name", Value::from(name))];
            if let Some(active) = active {
                props.push(("active", Value::from(active)));
            }
            db.create_node_with_props(&[label], props).unwrap()
        };
        let alix = node("Person", 1, "Alix", Some(true));
        let gus = node("Person", 2, "Gus", Some(true));
        let vincent = node("Person", 3, "Vincent", Some(false));
        let jules = node("Person", 4, "Jules", Some(true));
        let mia = node("Person", 5, "Mia", None);
        let cities: Vec<_> = ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"]
            .into_iter()
            .zip(10..)
            .map(|(name, id)| node("City", id, name, Some(name != "Paris")))
            .collect();
        for (from, to, edge_type) in [
            (alix, gus, "KNOWS"),
            (alix, gus, "KNOWS"),
            (gus, vincent, "KNOWS"),
            (vincent, alix, "KNOWS"),
            (gus, mia, "KNOWS"),
            (jules, mia, "KNOWS"),
            (mia, mia, "KNOWS"),
            (alix, jules, "LIKES"),
            (vincent, cities[0], "LIKES"),
        ] {
            db.create_edge(from, to, edge_type).unwrap();
        }
        for &city in &cities {
            db.create_edge(mia, city, "LIVES_IN").unwrap();
        }
        db
    }

    /// The sources: Alix, Gus and Jules.
    fn params() -> HashMap<String, Value> {
        let ids: Vec<Value> = [1_i64, 2, 4].into_iter().map(Value::from).collect();
        HashMap::from([("ids".to_string(), Value::List(ids.into()))])
    }

    /// The MATCH that binds the sources `n`, in `language`.
    fn sources(language: &str) -> &'static str {
        if language == "gql" {
            "MATCH (n WHERE n.id IN $ids)"
        } else {
            "MATCH (n) WHERE n.id IN $ids"
        }
    }

    /// The rows of `query`, planned with the reachability search where it
    /// applies, or with every walk enumerated (the plan before it).
    fn run(db: &GrafeoDB, language: &str, query: &str, reachability: bool) -> Vec<Vec<Value>> {
        let mut session = db.session();
        session.set_reachability(reachability);
        session
            .execute_language(query, language, Some(params()))
            .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
            .rows()
            .to_vec()
    }

    /// The text of a one-row EXPLAIN or PROFILE result.
    fn plan_text(db: &GrafeoDB, language: &str, query: &str, reachability: bool) -> String {
        let rows = run(db, language, query, reachability);
        rows[0][0].as_str().unwrap().to_string()
    }

    /// The marks of the expand lines of an EXPLAIN plan.
    fn markers(plan: &str) -> Vec<&str> {
        plan.lines()
            .filter(|line| line.trim_start().starts_with("Expand "))
            .filter_map(|line| line.rfind(" [reachability").map(|at| &line[at + 1..]))
            .collect()
    }

    /// Asserts that `query` runs as a reachability search in `mode` (EXPLAIN
    /// and PROFILE say so) and returns the rows of the walks in the same
    /// order; for a `grouped` aggregation, whose groups come in no fixed
    /// order, the same rows in any order.
    fn assert_same_rows(
        db: &GrafeoDB,
        language: &str,
        query: &str,
        mode: ReachabilityMode,
        grouped: bool,
    ) {
        let plan = plan_text(db, language, &format!("EXPLAIN {query}"), true);
        assert_eq!(
            markers(&plan),
            [mode.marker()],
            "{language}: {query}\n{plan}"
        );
        let profile = plan_text(db, language, &format!("PROFILE {query}"), true);
        assert_eq!(
            expand_marker(&profile),
            Some(mode),
            "{language}: {query}\n{profile}"
        );
        let mut searched = run(db, language, query, true);
        let mut walked = run(db, language, query, false);
        if grouped {
            searched.sort_by_key(|row| format!("{row:?}"));
            walked.sort_by_key(|row| format!("{row:?}"));
        }
        assert_eq!(searched, walked, "{language}: {query}");
    }

    /// The PROFILE line of the variable-length expand.
    fn expand_line(profile: &str) -> &str {
        profile
            .lines()
            .find(|line| line.contains("VariableLengthExpand"))
            .unwrap_or_else(|| panic!("no expand in\n{profile}"))
    }

    /// The mode PROFILE marks the variable-length expand with.
    fn expand_marker(profile: &str) -> Option<ReachabilityMode> {
        let line = expand_line(profile);
        [PerInputRow, AcrossInputRows]
            .into_iter()
            .find(|mode| line.contains(mode.marker()))
    }

    /// The `rows=` count of the PROFILE line of the variable-length expand.
    fn expand_rows(profile: &str) -> usize {
        expand_line(profile)
            .split("rows=")
            .nth(1)
            .unwrap()
            .split_whitespace()
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }

    #[test]
    fn duplicate_insensitive_consumers_get_the_rows_of_the_walks_in_order() {
        let db = people();
        let mut checked = 0;
        for language in ["gql", "cypher"] {
            for hops in ["*1..1", "*1..2", "*1..3", "*2..3", "*0..2"] {
                for edge_type in ["", ":KNOWS"] {
                    for (left, right) in [("-[", "]->"), ("<-[", "]-"), ("-[", "]-")] {
                        for filter in ["", " WHERE m.active = true"] {
                            for (ret, mode) in [
                                ("DISTINCT m.id", AcrossInputRows),
                                ("count(DISTINCT m) AS c", AcrossInputRows),
                                ("collect(DISTINCT m.id) AS ids", AcrossInputRows),
                                ("min(m.id) AS low", AcrossInputRows),
                                ("DISTINCT n.id AS n, m.id AS m", PerInputRow),
                            ] {
                                let pattern = format!("(n){left}{edge_type}{hops}{right}(m)");
                                let query = format!(
                                    "{} MATCH {pattern}{filter} RETURN {ret}",
                                    sources(language)
                                );
                                assert_same_rows(&db, language, &query, mode, false);
                                checked += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(checked, 600);
    }

    #[test]
    fn more_shapes_that_search_reachability_keep_their_rows() {
        let db = people();
        for language in ["gql", "cypher"] {
            let source = sources(language);
            for (tail, mode) in [
                // A named edge variable and a path nothing reads
                (
                    "MATCH (n)-[r*1..2]-(m) RETURN DISTINCT m.id",
                    AcrossInputRows,
                ),
                (
                    "MATCH p = (n)-[*1..2]-(m) RETURN DISTINCT m.id",
                    AcrossInputRows,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id, m.name",
                    AcrossInputRows,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id ORDER BY m.id DESC",
                    AcrossInputRows,
                ),
                // The planner adds the sort key to the distinct row
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id ORDER BY m.name",
                    AcrossInputRows,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id AS id ORDER BY id",
                    AcrossInputRows,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id LIMIT 3",
                    AcrossInputRows,
                ),
                // The source is read: each target once per source
                (
                    "MATCH (n)-[*1..3]-(m) WHERE m.id > n.id RETURN DISTINCT m.id",
                    PerInputRow,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT n.id AS n, m.id AS m",
                    PerInputRow,
                ),
                // Back to the source: a filter on id()
                ("MATCH (n)-[*1..3]->(n) RETURN DISTINCT n.id", PerInputRow),
                // Aggregates over distinct values, and min and max
                (
                    "MATCH (n)-[*1..3]-(m) RETURN sum(DISTINCT m.id) AS s, avg(DISTINCT m.id) AS a",
                    AcrossInputRows,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN max(m.id) AS high, min(DISTINCT m.id) AS low",
                    AcrossInputRows,
                ),
                // A label on the target is a filter on hasLabel()
                (
                    "MATCH (n)-[*1..2]-(m:City) RETURN DISTINCT m.name",
                    AcrossInputRows,
                ),
            ] {
                assert_same_rows(&db, language, &format!("{source} {tail}"), mode, false);
            }
            for (tail, mode) in [
                (
                    "MATCH (n)-[*1..3]-(m) RETURN n.id AS n, count(DISTINCT m) AS c",
                    PerInputRow,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN n.id AS n, collect(DISTINCT m.name) AS names, \
                     max(m.id) AS high",
                    PerInputRow,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN m.active AS active, count(DISTINCT m) AS c",
                    AcrossInputRows,
                ),
            ] {
                assert_same_rows(&db, language, &format!("{source} {tail}"), mode, true);
            }
            let every_person = "MATCH (n:Person)-[*1..2]->(m) RETURN DISTINCT m.id";
            assert_same_rows(&db, language, every_person, AcrossInputRows, false);
        }
        let cypher = sources("cypher");
        for (tail, mode) in [
            (
                "MATCH (n)-[*1..3]-(m) WITH DISTINCT m RETURN m.id",
                AcrossInputRows,
            ),
            (
                "MATCH (n)-[*1..3]-(m) WITH m.id AS id RETURN DISTINCT id",
                AcrossInputRows,
            ),
            (
                "MATCH (n)-[*1..3]-(m) WITH DISTINCT m, n RETURN m.id",
                PerInputRow,
            ),
            (
                "MATCH (n)-[*1..3]-(m) WITH m, n.id AS from WHERE m.id <> from \
                 RETURN DISTINCT m.id",
                PerInputRow,
            ),
            // `m` is the source from here on
            (
                "MATCH (n)-[*1..3]-(m) WITH m AS x, n AS m RETURN DISTINCT m.id",
                PerInputRow,
            ),
        ] {
            assert_same_rows(&db, "cypher", &format!("{cypher} {tail}"), mode, false);
        }
        let gql = sources("gql");
        let having =
            "MATCH (n)-[*1..3]-(m) RETURN m.active AS a, count(DISTINCT m) AS c HAVING c > 1";
        assert_same_rows(
            &db,
            "gql",
            &format!("{gql} {having}"),
            AcrossInputRows,
            true,
        );
        for (tail, mode) in [
            (
                "MATCH (n)-[*1..3]-(m) LET k = m.id RETURN DISTINCT k",
                AcrossInputRows,
            ),
            (
                "MATCH (n)-[*1..3]-(m) LET k = n.id RETURN DISTINCT k",
                PerInputRow,
            ),
        ] {
            assert_same_rows(&db, "gql", &format!("{gql} {tail}"), mode, false);
        }
    }

    #[test]
    fn a_target_reached_by_an_earlier_source_leads_a_later_one_on() {
        // Alix -> Gus -> Django -> Mia, and Jules -> Vincent -> Mia -> Butch.
        // With *2..3, Alix reaches Mia last; Jules reaches Mia two edges away
        // and goes on to Butch, whom only Jules reaches
        let db = GrafeoDB::new_in_memory();
        let node = |label: &str, name: &str| {
            db.create_node_with_props(&[label], [("name", Value::from(name))])
                .unwrap()
        };
        let alix = node("Source", "Alix");
        let jules = node("Source", "Jules");
        let [gus, django, vincent, mia, butch] =
            ["Gus", "Django", "Vincent", "Mia", "Butch"].map(|name| node("Person", name));
        for (from, to) in [
            (alix, gus),
            (gus, django),
            (django, mia),
            (jules, vincent),
            (vincent, mia),
            (mia, butch),
        ] {
            db.create_edge(from, to, "KNOWS").unwrap();
        }
        for language in ["gql", "cypher"] {
            let query = "MATCH (s:Source)-[:KNOWS*2..3]->(t) RETURN DISTINCT t.name AS t";
            assert_same_rows(&db, language, query, AcrossInputRows, false);
            let names: Vec<Value> = ["Django", "Mia", "Butch"].map(Value::from).into();
            let rows: Vec<Value> = run(&db, language, query, true)
                .into_iter()
                .map(|mut row| row.remove(0))
                .collect();
            assert_eq!(rows, names, "{language}");
        }
    }

    #[test]
    fn an_unbounded_pattern_on_a_cycle_returns_what_short_walks_reach() {
        // Up to 100 hops round a triangle with a doubled edge: far too many
        // walks to enumerate, while every node is within six hops
        let db = people();
        let query = |hops: &str| {
            format!("MATCH (n) WHERE n.id IN $ids MATCH (n)-[{hops}]->(m) RETURN DISTINCT m.id")
        };
        let plan = plan_text(&db, "cypher", &format!("EXPLAIN {}", query("*")), true);
        assert_eq!(markers(&plan), [AcrossInputRows.marker()], "{plan}");
        assert_eq!(
            run(&db, "cypher", &query("*"), true),
            run(&db, "cypher", &query("*1..6"), false)
        );
    }

    #[test]
    fn walks_stay_enumerated_where_a_copy_of_a_row_can_count() {
        let db = people();
        let source = sources("cypher");
        let mut queries: Vec<(&str, String)> = [
            "MATCH p = (n)-[*1..2]-(m) RETURN DISTINCT p",
            "MATCH p = (n)-[*1..2]-(m) RETURN DISTINCT m.id, size(nodes(p)) AS nodes",
            "MATCH p = (n)-[*1..2]-(m) RETURN DISTINCT m.id, size(relationships(p)) AS edges",
            "MATCH p = (n)-[*1..2]-(m) RETURN DISTINCT m.id, length(p) AS hops",
            "MATCH (n)-[r*1..2]-(m) RETURN DISTINCT m.id, size(r) AS edges",
            "MATCH (n)-[r*1..2]-(m) WITH DISTINCT m, r RETURN m.id",
            "MATCH (n)-[*1..2]-(m) RETURN count(*) AS c",
            "MATCH (n)-[*1..2]-(m) RETURN count(m) AS c",
            "MATCH (n)-[*1..2]-(m) RETURN sum(m.id) AS s",
            "MATCH (n)-[*1..2]-(m) RETURN avg(m.id) AS a",
            "MATCH (n)-[*1..2]-(m) RETURN collect(m.id) AS ids",
            "MATCH (n)-[*1..2]-(m) RETURN count(DISTINCT m) AS c, count(*) AS walks",
            "MATCH (n)-[*1..2]-(m) RETURN m.id",
            "MATCH (n)-[*1..2]-(m) WITH m LIMIT 3 RETURN DISTINCT m.id",
            "MATCH (n)-[*1..2]-(m)-[:LIVES_IN]->(c) RETURN DISTINCT c.id",
            "MATCH (n)-[*1..2]-(m) RETURN DISTINCT m.id, rand() < 2.0 AS r",
            "MATCH (n)-[*1..2]-(m) RETURN DISTINCT m.id ORDER BY rand()",
            "MATCH (n)-[*1..2]-(m) WHERE EXISTS { MATCH (m)-[:LIVES_IN]->() } RETURN DISTINCT m.id",
            "MATCH (n)-[*1..2]-(m) RETURN DISTINCT *",
            "MATCH (n)-[*1..2 {since: 1}]-(m) RETURN DISTINCT m.id",
        ]
        .into_iter()
        .map(|tail| ("cypher", format!("{source} {tail}")))
        .collect();
        let gql = sources("gql");
        for tail in [
            "MATCH TRAIL (n)-[*1..2]-(m) RETURN DISTINCT m.id",
            "MATCH (n)-[*1..2]-(m) RETURN m.active AS a, count(DISTINCT m) AS c HAVING c > n.id",
        ] {
            queries.push(("gql", format!("{gql} {tail}")));
        }
        for (language, query) in queries {
            let plan = plan_text(&db, language, &format!("EXPLAIN {query}"), true);
            assert!(plan.contains("Expand (n)"), "{language}: {query}\n{plan}");
            assert!(markers(&plan).is_empty(), "{language}: {query}\n{plan}");
        }
    }

    #[test]
    fn profile_counts_the_rows_the_search_emits() {
        let db = people();
        for language in ["gql", "cypher"] {
            let source = sources(language);
            let walks = run(
                &db,
                language,
                &format!("{source} MATCH (n)-[*1..3]-(m) RETURN n.id AS n, m.id AS m"),
                true,
            );
            let pairs: HashSet<String> = walks.iter().map(|row| format!("{row:?}")).collect();
            let targets: HashSet<String> =
                walks.iter().map(|row| format!("{:?}", row[1])).collect();
            assert!(walks.len() > pairs.len(), "the walks repeat pairs");
            assert!(pairs.len() > targets.len(), "the sources share targets");

            // Each target once over all sources
            let once = format!("PROFILE {source} MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id");
            let profile = plan_text(&db, language, &once, true);
            assert_eq!(expand_marker(&profile), Some(AcrossInputRows), "{profile}");
            assert_eq!(expand_rows(&profile), targets.len(), "{profile}");
            // Each target once per source
            let per_source =
                format!("PROFILE {source} MATCH (n)-[*1..3]-(m) RETURN DISTINCT n.id, m.id");
            let profile = plan_text(&db, language, &per_source, true);
            assert_eq!(expand_marker(&profile), Some(PerInputRow), "{profile}");
            assert_eq!(expand_rows(&profile), pairs.len(), "{profile}");
            // Every walk, without the search
            let profile = plan_text(&db, language, &once, false);
            assert_eq!(expand_marker(&profile), None, "{profile}");
            assert_eq!(expand_rows(&profile), walks.len(), "{profile}");
        }
    }

    #[test]
    fn aggregates_that_count_copies_despite_distinct_keep_their_walks() {
        // These aggregates ignore DISTINCT, so every walk counts
        let db = people();
        let queries = [
            ("cypher", "stDev(DISTINCT m.id)"),
            ("cypher", "stDevP(DISTINCT m.id)"),
            ("cypher", "variance(DISTINCT m.id)"),
            ("cypher", "percentileDisc(DISTINCT m.id, 0.5)"),
            ("cypher", "percentileCont(DISTINCT m.id, 0.5)"),
            ("gql", "stddev_samp(DISTINCT m.id)"),
            ("gql", "var_samp(DISTINCT m.id)"),
            ("gql", "percentile_disc(DISTINCT m.id, 0.5)"),
        ];
        for (language, aggregate) in queries {
            let query = format!(
                "{} MATCH (n)-[*1..3]-(m) RETURN {aggregate} AS x",
                sources(language)
            );
            let plan = plan_text(&db, language, &format!("EXPLAIN {query}"), true);
            assert_eq!(
                markers(&plan),
                Vec::<&str>::new(),
                "{language}: {query}\n{plan}"
            );
            assert_eq!(
                run(&db, language, &query, true),
                run(&db, language, &query, false),
                "{language}: {query}"
            );
        }
    }

    /// The rows of `query` in `session`, with and without the search, and
    /// the mode EXPLAIN shows.
    fn rows_in(
        session: &mut crate::session::Session,
        language: &str,
        query: &str,
    ) -> (Vec<Vec<Value>>, Vec<Vec<Value>>, Vec<String>) {
        let mut execute = |query: &str, reachability: bool| {
            session.set_reachability(reachability);
            session
                .execute_language(query, language, Some(params()))
                .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
                .rows()
                .to_vec()
        };
        let explain = execute(&format!("EXPLAIN {query}"), true);
        let plan = explain[0][0].as_str().unwrap().to_string();
        let marks = markers(&plan).into_iter().map(str::to_string).collect();
        (execute(query, true), execute(query, false), marks)
    }

    #[test]
    fn an_open_transaction_searches_its_own_writes() {
        let db = people();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for write in [
            // A new person, linked to Gus and living in Prague
            "MATCH (g:Person {id: 2}), (c:City {id: 13}) \
             CREATE (h:Person {id: 6, name: 'Hans', active: true})-[:KNOWS]->(g), (h)-[:LIVES_IN]->(c)",
            // An edge deleted
            "MATCH (:Person {id: 1})-[r:LIKES]->(:Person {id: 4}) DELETE r",
            // A node deleted with its edges
            "MATCH (v:Person {id: 3}) DETACH DELETE v",
        ] {
            session.execute_cypher(write).unwrap();
        }
        for language in ["gql", "cypher"] {
            let source = sources(language);
            for (tail, mode) in [
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id",
                    AcrossInputRows,
                ),
                (
                    "MATCH (n)-[*1..3]-(m) RETURN DISTINCT n.id, m.id",
                    PerInputRow,
                ),
                (
                    "MATCH (n)-[*2..3]->(m) RETURN count(DISTINCT m) AS c",
                    AcrossInputRows,
                ),
            ] {
                let query = format!("{source} {tail}");
                let (searched, walked, marks) = rows_in(&mut session, language, &query);
                assert_eq!(marks, [mode.marker()], "{language}: {query}");
                assert_eq!(searched, walked, "{language}: {query}");
            }
            // The transaction's writes count: Hans is there, Vincent is gone
            let query = format!("{source} MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id");
            let (searched, _, _) = rows_in(&mut session, language, &query);
            assert!(searched.contains(&vec![Value::from(6_i64)]), "{language}");
            assert!(!searched.contains(&vec![Value::from(3_i64)]), "{language}");
        }
        session.rollback().unwrap();
    }

    #[test]
    fn a_past_epoch_is_searched_as_it_was() {
        let db = people();
        let epoch = db.current_epoch();
        db.execute_cypher("MATCH (:Person {id: 2})-[r:KNOWS]->(:Person {id: 5}) DELETE r")
            .unwrap();
        db.execute_cypher(
            "MATCH (g:Person {id: 2}) CREATE (:Person {id: 6, name: 'Hans'})-[:KNOWS]->(g)",
        )
        .unwrap();
        let mut session = db.session();
        for (tail, mode) in [
            (
                "MATCH (n)-[*1..3]-(m) RETURN DISTINCT m.id",
                AcrossInputRows,
            ),
            (
                "MATCH (n)-[*1..3]-(m) RETURN DISTINCT n.id, m.id",
                PerInputRow,
            ),
        ] {
            let query = format!("{} {tail}", sources("gql"));
            let mut at_epoch = |reachability: bool| {
                session.set_reachability(reachability);
                session
                    .execute_at_epoch_with_params(&query, epoch, Some(params()))
                    .unwrap()
                    .rows()
                    .to_vec()
            };
            let (searched, walked) = (at_epoch(true), at_epoch(false));
            assert_eq!(searched, walked, "{query}");
            let plan = plan_text(&db, "gql", &format!("EXPLAIN {query}"), true);
            assert_eq!(markers(&plan), [mode.marker()], "{query}");
            // The past differs from the present: no Hans then
            assert_ne!(searched, run(&db, "gql", &query, true), "{query}");
        }
    }

    #[test]
    fn many_input_rows_and_output_chunks_keep_their_rows() {
        // 3,000 sources, more than the 1,024 rows a chunk of input holds
        // before the expand sorts it by source, and outputs over several
        // chunks of 2,048 rows
        let db = GrafeoDB::new_in_memory();
        let nodes: Vec<_> = (0..3000_i64)
            .map(|k| {
                db.create_node_with_props(&["Src"], [("k", Value::from(k))])
                    .unwrap()
            })
            .collect();
        for (i, &node) in nodes.iter().enumerate() {
            db.create_edge(node, nodes[(i * 7 + 1) % nodes.len()], "LINK")
                .unwrap();
            db.create_edge(node, nodes[(i + 1) % nodes.len()], "LINK")
                .unwrap();
        }
        for language in ["gql", "cypher"] {
            for (query, mode) in [
                (
                    "MATCH (s:Src)-[*1..2]-(t) RETURN DISTINCT t.k",
                    AcrossInputRows,
                ),
                (
                    "MATCH (s:Src)-[*1..2]-(t) RETURN DISTINCT s.k, t.k",
                    PerInputRow,
                ),
            ] {
                assert_same_rows(&db, language, query, mode, false);
                assert!(run(&db, language, query, true).len() >= 3000, "{query}");
            }
        }
        // Sources in descending order of `k`, so the sort reorders them
        let query =
            "MATCH (s:Src) WITH s ORDER BY s.k DESC MATCH (s)-[*1..2]-(t) RETURN DISTINCT s.k, t.k";
        assert_same_rows(&db, "cypher", query, PerInputRow, false);
    }
}
