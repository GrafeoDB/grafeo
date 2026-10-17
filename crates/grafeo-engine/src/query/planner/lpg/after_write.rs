//! Clauses that read after a write in the same statement.
//!
//! A clause reads the whole clause before it, as ISO GQL and openCypher
//! define a statement, but the planner passes rows on one at a time: in
//! `UNWIND range(1, 4) AS i MERGE (h:Hub) SET h.c = i RETURN h.c` the `RETURN`
//! of the row `i` would read `h.c` before the rows after it set it. So the
//! planner reads the whole writing input of a clause first where the clause
//! reads the graph (a `RETURN`, a `WITH` or `LET` that reads a property, a
//! `WHERE`, `ORDER BY`, aggregate or `UNWIND` that does), where it cuts the
//! rows short (`LIMIT`, which would leave the later rows unwritten), and for a
//! `MERGE` ([`reads_after_the_write`]; the planner puts an eager step before
//! them with `read_first_after_a_write`). A `MATCH`, `OPTIONAL MATCH` and
//! `CALL` after a write read their input first on their own (see
//! `plan_node_scan`, `plan_left_join` and `plan_apply`).
//!
//! One write gets one eager step: [`writes_pending`] tells whether the rows
//! of an input can still come out while it writes, which is no longer so
//! above a clause that read it whole. A clause that passes the rows on
//! without reading the graph (`WITH h, i`) leaves that to the clause after
//! it. EXPLAIN marks the clauses [`reads_after_the_write`] picks with
//! [`MARKER`] (see [`clauses_after_a_write`]).
//!
//! The right side of a join and a subquery read the store, not the rows of
//! the write, so they see the write only if they read the store when they
//! run: the planner looks nothing up for them while planning (see
//! [`right_side_runs_after_a_write`] and [`subquery_runs_after_a_write`]).

use grafeo_core::execution::operators::Operator;

use super::Result;
use crate::query::plan::{
    AggregateOp, ApplyOp, LogicalExpression, LogicalOperator, ProjectOp, ReturnOp, SortOp, UnwindOp,
};

/// What EXPLAIN adds to a clause the planner reads after a write (see
/// [`reads_after_the_write`]).
pub(crate) const MARKER: &str = "[after the write]";

/// Whether the rows of `op` can come out while its plan still writes: a
/// write passes each row on as it writes it, and so does an operator above
/// it that passes the rows on without reading the graph (a `WITH` of
/// variables, a filter on values, `SKIP`, an expand from a node of the row,
/// which reads only what the rows before its own wrote). An aggregate, a
/// sort, and a clause that reads after the write ([`reads_after_the_write`])
/// read their whole input before their first row, and so do a `MATCH` and a
/// `CALL` whose input writes (`plan_node_scan`, `plan_apply`) and a join (an
/// `OPTIONAL MATCH` too), which reads its whole right side for its hash
/// table and a writing left side first (`plan_join`, `plan_left_join`):
/// above them every write below is done. Anything else that writes counts
/// as pending.
pub(crate) fn writes_pending(op: &LogicalOperator) -> bool {
    match op {
        LogicalOperator::CreateNode(_)
        | LogicalOperator::CreateEdge(_)
        | LogicalOperator::Create(_)
        | LogicalOperator::DeleteNode(_)
        | LogicalOperator::DeleteEdge(_)
        | LogicalOperator::SetProperty(_)
        | LogicalOperator::AddLabel(_)
        | LogicalOperator::RemoveLabel(_)
        | LogicalOperator::Merge(_)
        | LogicalOperator::MergeRelationship(_) => true,
        LogicalOperator::Return(ret) => !return_reads(ret) && writes_pending(&ret.input),
        LogicalOperator::Project(project) => {
            !projection_reads(project) && writes_pending(&project.input)
        }
        LogicalOperator::Filter(filter) => {
            !reads_the_graph(&filter.predicate) && writes_pending(&filter.input)
        }
        LogicalOperator::Unwind(unwind) => !unwind_reads(unwind) && writes_pending(&unwind.input),
        LogicalOperator::Skip(skip) => writes_pending(&skip.input),
        LogicalOperator::Distinct(distinct) => writes_pending(&distinct.input),
        LogicalOperator::Expand(expand) => writes_pending(&expand.input),
        LogicalOperator::ShortestPath(path) => writes_pending(&path.input),
        LogicalOperator::Aggregate(_)
        | LogicalOperator::Sort(_)
        | LogicalOperator::Limit(_)
        | LogicalOperator::NodeScan(_)
        | LogicalOperator::LeftJoin(_)
        | LogicalOperator::Join(_) => false,
        LogicalOperator::Apply(apply) => apply.subplan.has_mutations(),
        other => other.has_mutations(),
    }
}

/// Whether the planner reads the whole input of the clause `op` before the
/// clause runs, because the input still writes as its rows come out
/// ([`writes_pending`]) and the clause reads the graph, cuts the rows short
/// or merges: a `RETURN` with items, a `WITH` (`Project`), `WHERE`, `ORDER
/// BY`, aggregate or `UNWIND` that reads the graph ([`reads_the_graph`]), a
/// `LIMIT` and a `MERGE`.
pub(crate) fn reads_after_the_write(op: &LogicalOperator) -> bool {
    let (reads, input) = match op {
        LogicalOperator::Return(ret) => (return_reads(ret), &ret.input),
        LogicalOperator::Project(project) => (projection_reads(project), &project.input),
        LogicalOperator::Filter(filter) => (reads_the_graph(&filter.predicate), &filter.input),
        LogicalOperator::Aggregate(aggregate) => (aggregate_reads(aggregate), &aggregate.input),
        LogicalOperator::Sort(sort) => (sort_reads(sort), &sort.input),
        LogicalOperator::Unwind(unwind) => (unwind_reads(unwind), &unwind.input),
        LogicalOperator::Limit(limit) => (true, &limit.input),
        LogicalOperator::Merge(merge) => (true, &merge.input),
        LogicalOperator::MergeRelationship(merge) => (true, &merge.input),
        _ => return false,
    };
    reads && writes_pending(input)
}

/// Whether the right side of a join or an `OPTIONAL MATCH` with this `left`
/// side runs after a write of the statement: when the left side writes,
/// which the join reads whole first (see `plan_join` and `plan_left_join`).
/// The right side does not read the left side's rows, so it reads the store
/// as the write left it only if it reads the store when it runs (see
/// [`Planner::reads_the_store_as_planned`](super::Planner::reads_the_store_as_planned)):
/// `MATCH (h:Hub) SET h.c = 4 WITH h OPTIONAL MATCH (t:Hub {c: 4})` finds
/// the hub, which no lookup before the `SET` finds.
pub(crate) fn right_side_runs_after_a_write(left: &LogicalOperator) -> bool {
    left.has_mutations()
}

/// Whether the subquery of `apply` runs after a write of the statement: when
/// its input writes (read whole first, see `plan_apply`), or when it writes
/// itself, as it runs again for the next row after the write of the row
/// before (see [`right_side_runs_after_a_write`]).
pub(crate) fn subquery_runs_after_a_write(apply: &ApplyOp) -> bool {
    apply.input.has_mutations() || apply.subplan.has_mutations()
}

impl super::Planner {
    /// Plans `op`, which runs after a write of its statement when
    /// `after_a_write` (or when the operator being planned does): then
    /// nothing in its plan is decided by what the planner reads of the store
    /// while planning (see [`Self::reads_the_store_as_planned`]).
    pub(super) fn plan_after_a_write(
        &self,
        after_a_write: bool,
        op: &LogicalOperator,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let enclosing = self.after_a_write.get();
        self.after_a_write.set(enclosing || after_a_write);
        let planned = self.plan_operator(op);
        self.after_a_write.set(enclosing);
        planned
    }

    /// Whether the operator being planned reads the store as it is while the
    /// plan is built: not when it runs after a write of its statement (the
    /// right side of a join, see [`right_side_runs_after_a_write`], and a
    /// subquery, see [`subquery_runs_after_a_write`], or one per row after a
    /// write, see `lift_subqueries`). What the planner reads of the store
    /// itself, instead of an operator that reads it when it runs (the nodes
    /// of an index or label-first lookup, the zone maps), needs it.
    pub(super) fn reads_the_store_as_planned(&self) -> bool {
        !self.after_a_write.get()
    }
}

/// The clauses in `root` the planner reads after a write (see
/// [`reads_after_the_write`]), for EXPLAIN to mark. The subplan of an Apply
/// (a `CALL` subquery) is not searched: the planner plans its `RETURN` as a
/// projection that passes nodes on unread.
pub(crate) fn clauses_after_a_write(root: &LogicalOperator) -> Vec<&LogicalOperator> {
    let mut found = Vec::new();
    collect(root, &mut found);
    found
}

/// Adds the clauses at and below `op` the planner reads after a write.
fn collect<'a>(op: &'a LogicalOperator, found: &mut Vec<&'a LogicalOperator>) {
    if reads_after_the_write(op) {
        found.push(op);
    }
    match op {
        LogicalOperator::Apply(apply) => collect(&apply.input, found),
        _ => {
            for child in op.children() {
                collect(child, found);
            }
        }
    }
}

/// Whether a `RETURN` reads the graph: any item does, since a node or edge
/// it returns is read when the row is returned.
pub(crate) fn return_reads(ret: &ReturnOp) -> bool {
    !ret.items.is_empty()
}

/// Whether a `WITH` or `LET` reads the graph (a variable passes on as it
/// is, a node or edge as its ID).
pub(crate) fn projection_reads(project: &ProjectOp) -> bool {
    project
        .projections
        .iter()
        .any(|projection| reads_the_graph(&projection.expression))
}

/// Whether an aggregate reads the graph when a row arrives: a group key or
/// an aggregated value does (`collect(n)` collects the node's ID).
pub(crate) fn aggregate_reads(aggregate: &AggregateOp) -> bool {
    aggregate.group_by.iter().any(reads_the_graph)
        || aggregate.aggregates.iter().any(|expr| {
            expr.expression
                .iter()
                .chain(&expr.expression2)
                .any(reads_the_graph)
        })
}

/// Whether an `ORDER BY` reads the graph: a sort key does.
pub(crate) fn sort_reads(sort: &SortOp) -> bool {
    sort.keys.iter().any(|key| reads_the_graph(&key.expression))
}

/// Whether an `UNWIND` reads the graph: its list does.
pub(crate) fn unwind_reads(unwind: &UnwindOp) -> bool {
    reads_the_graph(&unwind.expression)
}

/// Whether evaluating `expression` may read the graph: a property, labels,
/// a type, a subquery or pattern, a map projection, an index or key into a
/// value that may be a node (`n['name']`), any function. Values, parameters,
/// variables (a node or edge as its ID), `id()` and operators over them do
/// not.
pub(crate) fn reads_the_graph(expression: &LogicalExpression) -> bool {
    match expression {
        LogicalExpression::Literal(_)
        | LogicalExpression::Parameter(_)
        | LogicalExpression::Variable(_)
        | LogicalExpression::Id(_) => false,
        LogicalExpression::Binary { left, right, .. } => {
            reads_the_graph(left) || reads_the_graph(right)
        }
        LogicalExpression::Unary { operand, .. } => reads_the_graph(operand),
        LogicalExpression::List(items) => items.iter().any(reads_the_graph),
        LogicalExpression::Map(entries) => entries.iter().any(|(_, value)| reads_the_graph(value)),
        LogicalExpression::Case {
            operand,
            when_clauses,
            else_clause,
        } => {
            operand.as_deref().is_some_and(reads_the_graph)
                || when_clauses
                    .iter()
                    .any(|(when, then)| reads_the_graph(when) || reads_the_graph(then))
                || else_clause.as_deref().is_some_and(reads_the_graph)
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::{
        AggregateExpr, AggregateFunction, ApplyOp, BinaryOp, CreateNodeOp, ExpandDirection,
        ExpandOp, FilterOp, JoinOp, JoinType, LeftJoinOp, LimitOp, MergeOp, NodeScanOp, PathMode,
        Projection, ReturnItem, SetPropertyOp,
    };
    use grafeo_common::types::Value;

    fn variable(name: &str) -> LogicalExpression {
        LogicalExpression::Variable(name.to_string())
    }

    fn property(variable: &str, name: &str) -> LogicalExpression {
        LogicalExpression::Property {
            variable: variable.to_string(),
            property: name.to_string(),
        }
    }

    /// `UNWIND`-like rows, then `SET h.c = i`: a write.
    fn set() -> LogicalOperator {
        LogicalOperator::SetProperty(SetPropertyOp {
            variable: "h".to_string(),
            properties: vec![("c".to_string(), variable("i"))],
            replace: false,
            is_edge: false,
            input: Box::new(LogicalOperator::CreateNode(CreateNodeOp {
                variable: "h".to_string(),
                labels: vec!["Hub".to_string()],
                properties: Vec::new(),
                input: None,
            })),
        })
    }

    /// A node scan of `variable` with no input: a read.
    fn scan(variable: &str) -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: variable.to_string(),
            label: None,
            input: None,
        })
    }

    fn project(projections: Vec<LogicalExpression>, input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Project(ProjectOp {
            projections: projections
                .into_iter()
                .map(|expression| Projection {
                    expression,
                    alias: None,
                })
                .collect(),
            input: Box::new(input),
            pass_through_input: false,
        })
    }

    fn filter(predicate: LogicalExpression, input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(input),
            pushdown_hint: None,
        })
    }

    fn ret(items: Vec<LogicalExpression>, input: LogicalOperator) -> LogicalOperator {
        LogicalOperator::Return(ReturnOp {
            items: items
                .into_iter()
                .map(|expression| ReturnItem {
                    expression,
                    alias: None,
                })
                .collect(),
            distinct: false,
            input: Box::new(input),
        })
    }

    /// `count(*)` over `input`.
    fn count_star(input: LogicalOperator) -> LogicalOperator {
        aggregate(AggregateFunction::Count, None, input)
    }

    /// `sum(expression)` over `input`.
    fn sum(expression: LogicalExpression, input: LogicalOperator) -> LogicalOperator {
        aggregate(AggregateFunction::Sum, Some(expression), input)
    }

    fn aggregate(
        function: AggregateFunction,
        expression: Option<LogicalExpression>,
        input: LogicalOperator,
    ) -> LogicalOperator {
        LogicalOperator::Aggregate(AggregateOp {
            group_by: Vec::new(),
            aggregates: vec![AggregateExpr {
                function,
                expression,
                expression2: None,
                distinct: false,
                alias: Some("total".to_string()),
                percentile: None,
                separator: None,
            }],
            input: Box::new(input),
            having: None,
        })
    }

    #[test]
    fn values_variables_and_ids_read_nothing() {
        for expression in [
            LogicalExpression::Literal(Value::Int64(3)),
            LogicalExpression::Parameter("p".to_string()),
            variable("n"),
            LogicalExpression::Id("n".to_string()),
            LogicalExpression::Binary {
                left: Box::new(variable("i")),
                op: BinaryOp::Add,
                right: Box::new(LogicalExpression::Literal(Value::Int64(19))),
            },
            LogicalExpression::List(vec![variable("i"), variable("n")]),
        ] {
            assert!(!reads_the_graph(&expression), "{expression:?}");
        }
        for expression in [
            property("h", "c"),
            LogicalExpression::Labels("n".to_string()),
            LogicalExpression::Binary {
                left: Box::new(property("h", "c")),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::Int64(4))),
            },
            LogicalExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![variable("i")],
                distinct: false,
            },
            LogicalExpression::IndexAccess {
                base: Box::new(variable("n")),
                index: Box::new(LogicalExpression::Literal("name".into())),
            },
        ] {
            assert!(reads_the_graph(&expression), "{expression:?}");
        }
    }

    /// A clause that reads the graph right after a write is read after it,
    /// and above it nothing writes any more; a `WITH` of variables passes
    /// the pending write on to the clause after it.
    #[test]
    fn the_first_clause_that_reads_takes_the_write() {
        let with_variables = project(vec![variable("h"), variable("i")], set());
        assert!(writes_pending(&with_variables));
        assert!(!reads_after_the_write(&with_variables));

        let returned = ret(vec![variable("i"), property("h", "c")], with_variables);
        assert!(reads_after_the_write(&returned));
        assert!(!writes_pending(&returned));

        let filtered = filter(property("h", "c"), set());
        assert!(reads_after_the_write(&filtered));
        let returned_after_filter = ret(vec![property("h", "c")], filtered);
        assert!(
            !reads_after_the_write(&returned_after_filter),
            "one eager step per write"
        );

        // A filter on values reads nothing: the `RETURN` after it does.
        let on_values = filter(variable("i"), set());
        assert!(!reads_after_the_write(&on_values));
        assert!(reads_after_the_write(&ret(vec![variable("i")], on_values)));
    }

    #[test]
    fn an_aggregate_reads_after_the_write_when_its_values_read_the_graph() {
        assert!(reads_after_the_write(&sum(property("h", "c"), set())));
        assert!(!reads_after_the_write(&count_star(set())));
        assert!(!reads_after_the_write(&sum(variable("i"), set())));
        assert!(
            !writes_pending(&count_star(set())),
            "an aggregate reads its whole input first"
        );
    }

    /// A `LIMIT` and a `MERGE` read after a write whatever they read: one
    /// would cut the write short, the other merges what it wrote.
    #[test]
    fn a_limit_and_a_merge_read_after_the_write() {
        let limited = LogicalOperator::Limit(LimitOp {
            count: 1.into(),
            input: Box::new(project(vec![variable("i")], set())),
        });
        assert!(reads_after_the_write(&limited));
        let merged = LogicalOperator::Merge(MergeOp {
            variable: "t".to_string(),
            labels: vec!["P".to_string()],
            match_properties: Vec::new(),
            on_create: Vec::new(),
            on_match: Vec::new(),
            on_create_labels: Vec::new(),
            on_match_labels: Vec::new(),
            input: Box::new(project(vec![variable("i")], set())),
        });
        assert!(reads_after_the_write(&merged));
        assert!(writes_pending(&merged), "a MERGE writes as it goes");
    }

    /// A `MATCH`, an `OPTIONAL MATCH` and a `CALL` read their writing input
    /// first on their own, so a `MERGE` or `RETURN` after them is not read
    /// again; a `CALL` that writes writes per row.
    #[test]
    fn a_pattern_or_call_after_a_write_reads_it_whole() {
        let matched = LogicalOperator::Expand(ExpandOp {
            from_variable: "a".to_string(),
            to_variable: "b".to_string(),
            edge_variable: None,
            direction: ExpandDirection::Outgoing,
            edge_types: Vec::new(),
            min_hops: 1,
            max_hops: Some(1),
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "a".to_string(),
                label: None,
                input: Some(Box::new(project(vec![variable("i")], set()))),
            })),
            path_alias: None,
            path_mode: PathMode::Walk,
            quantified: false,
        });
        assert!(!writes_pending(&matched));
        assert!(!reads_after_the_write(&ret(
            vec![property("b", "c")],
            matched
        )));

        let optional = LogicalOperator::LeftJoin(LeftJoinOp {
            left: Box::new(set()),
            right: Box::new(scan("t")),
            condition: None,
        });
        assert!(!writes_pending(&optional));

        let reading_call = LogicalOperator::Apply(ApplyOp {
            input: Box::new(set()),
            subplan: Box::new(scan("t")),
            shared_variables: Vec::new(),
            optional: false,
            unit: false,
        });
        assert!(!writes_pending(&reading_call));
        let writing_call = LogicalOperator::Apply(ApplyOp {
            input: Box::new(scan("s")),
            subplan: Box::new(set()),
            shared_variables: Vec::new(),
            optional: false,
            unit: true,
        });
        assert!(writes_pending(&writing_call));
    }

    /// A join reads its whole right side (the hash table) before its first
    /// row, and its whole left side first when that one writes (`plan_join`,
    /// `plan_left_join`): above a join no write below it is pending,
    /// whichever side writes, so the `RETURN` above it is not read again and
    /// EXPLAIN marks nothing.
    #[test]
    fn above_a_join_no_write_is_pending() {
        for (left, right) in [(set(), scan("t")), (scan("t"), set())] {
            let inner = LogicalOperator::Join(JoinOp {
                left: Box::new(left.clone()),
                right: Box::new(right.clone()),
                join_type: JoinType::Inner,
                conditions: Vec::new(),
            });
            let optional = LogicalOperator::LeftJoin(LeftJoinOp {
                left: Box::new(left),
                right: Box::new(right),
                condition: None,
            });
            for join in [inner, optional] {
                assert!(!writes_pending(&join), "{join:?}");
                let returned = ret(vec![property("h", "c")], join);
                assert!(!reads_after_the_write(&returned), "{returned:?}");
                assert!(clauses_after_a_write(&returned).is_empty(), "{returned:?}");
            }
        }
    }

    /// A statement that only reads has no pending write anywhere, so no
    /// clause of it is read after a write.
    #[test]
    fn a_statement_that_only_reads_has_no_clause_after_a_write() {
        let root = LogicalOperator::Limit(LimitOp {
            count: 3.into(),
            input: Box::new(ret(
                vec![variable("n"), property("n", "c")],
                filter(property("n", "c"), scan("n")),
            )),
        });
        assert!(clauses_after_a_write(&root).is_empty());
        assert!(!writes_pending(&root));
    }

    /// EXPLAIN marks the clauses of the statement, not those of a `CALL`
    /// subquery's body.
    #[test]
    fn the_clauses_of_a_subquery_body_are_not_listed() {
        let body = ret(vec![property("h", "c")], set());
        let root = ret(
            vec![variable("i")],
            LogicalOperator::Apply(ApplyOp {
                input: Box::new(set()),
                subplan: Box::new(body),
                shared_variables: Vec::new(),
                optional: false,
                unit: false,
            }),
        );
        let found = clauses_after_a_write(&root);
        assert_eq!(found.len(), 1);
        assert!(
            matches!(
                found[0],
                LogicalOperator::Return(ret)
                    if matches!(&ret.items[0].expression, LogicalExpression::Variable(name) if name == "i")
            ),
            "the statement's own RETURN: {found:?}"
        );
    }
}
