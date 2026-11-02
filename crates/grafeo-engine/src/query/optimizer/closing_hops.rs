//! Hops that close a cycle look up the node they come back to.

use super::cycles::same;
use crate::query::plan::{
    BinaryOp, BuiltinFunction, Callee, ExpandOp, FilterOp, LogicalExpression, LogicalOperator,
    PushdownHint, has_subquery,
};
use crate::query::planner::lpg::seek::split_conjuncts;

/// Marks each filter that an expand's target is a node its input binds, such
/// as the one [`close_cycles`](super::cycles::close_cycles) puts over the last
/// hop of `(a)-->(b)-->(c)-->(a)`: the expand then looks up the edges to that
/// node instead of listing every edge of the row. A lookup costs 120 to 190 ns flat
/// with the degree, a list about 18 ns per entry: below about 8 edges the lookup costs
/// a little more than the walk, and it loses at most that much.
///
/// The filter stays. It answers for a row whose target is no node, and an
/// expand that does not take the hint is merely slower.
pub(super) fn look_up_closing_hops(op: LogicalOperator) -> LogicalOperator {
    match op.map_children(look_up_closing_hops) {
        LogicalOperator::Filter(filter) => marked(filter),
        other => other,
    }
}

/// `filter` with the hint when it checks the target of the expand right below
/// it. Over other filters, the check alone moves down to the expand, as a
/// filter of its own: comparing two nodes raises nothing, and the rest of
/// `filter` stays above the filters written before it, which keep guarding it.
fn marked(mut filter: FilterOp) -> LogicalOperator {
    if filter.pushdown_hint.is_some() {
        return LogicalOperator::Filter(filter);
    }
    if let LogicalOperator::Expand(expand) = filter.input.as_ref() {
        filter.pushdown_hint = closing_target(&filter.predicate, expand)
            .map(|target| PushdownHint::ExpandInto { target });
        return LogicalOperator::Filter(filter);
    }
    // Moving the check past other filters needs a target known to be a node:
    // comparing a node with another kind of value may raise.
    let Some((fresh, target)) = expand_below_filters(&filter.input).and_then(|expand| {
        closing_target(&filter.predicate, expand)
            .filter(|target| binds_a_node(&expand.input, target))
            .map(|target| (expand.to_variable.clone(), target))
    }) else {
        return LogicalOperator::Filter(filter);
    };
    let check = same(&fresh, target.clone());
    let input = with_check_at_the_expand(*filter.input, check, target.clone());
    match without_check(filter.predicate, &fresh, &target) {
        Some(predicate) => LogicalOperator::Filter(FilterOp {
            predicate,
            input: Box::new(input),
            pushdown_hint: None,
        }),
        None => input,
    }
}

/// `predicate` without the conjuncts that say `fresh` and `target` are one.
pub(super) fn without_check(
    predicate: LogicalExpression,
    fresh: &str,
    target: &str,
) -> Option<LogicalExpression> {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => match (
            without_check(*left, fresh, target),
            without_check(*right, fresh, target),
        ) {
            (Some(left), Some(right)) => Some(LogicalExpression::Binary {
                left: Box::new(left),
                op: BinaryOp::And,
                right: Box::new(right),
            }),
            (Some(kept), None) | (None, Some(kept)) => Some(kept),
            (None, None) => None,
        },
        other => (!says_same(&other, fresh, target)).then_some(other),
    }
}

/// Whether `conjunct` says `fresh` and `target` are one, either way round.
fn says_same(conjunct: &LogicalExpression, fresh: &str, target: &str) -> bool {
    compared_variables(conjunct)
        .is_some_and(|(a, b)| (a == fresh && b == target) || (a == target && b == fresh))
}

/// The two variables `conjunct` says are one: `a = b` or `id(a) = id(b)`.
pub(super) fn compared_variables(conjunct: &LogicalExpression) -> Option<(&str, &str)> {
    let LogicalExpression::Binary {
        left,
        op: BinaryOp::Eq,
        right,
    } = conjunct
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (LogicalExpression::Variable(a), LogicalExpression::Variable(b)) => Some((a, b)),
        (left, right) => Some((id_of(left)?, id_of(right)?)),
    }
}

/// The variable of `id(variable)`.
pub(super) fn id_of(expression: &LogicalExpression) -> Option<&str> {
    match expression {
        LogicalExpression::Id(variable) => Some(variable),
        LogicalExpression::FunctionCall {
            callee: Callee::Builtin(BuiltinFunction::Id),
            args,
            ..
        } => match args.as_slice() {
            [LogicalExpression::Variable(variable)] => Some(variable),
            _ => None,
        },
        _ => None,
    }
}

/// `op`, a chain of filters over an expand, with `check` as a hinted filter
/// right over the expand.
fn with_check_at_the_expand(
    op: LogicalOperator,
    check: LogicalExpression,
    target: String,
) -> LogicalOperator {
    match op {
        LogicalOperator::Filter(mut filter) => {
            filter.input = Box::new(with_check_at_the_expand(*filter.input, check, target));
            LogicalOperator::Filter(filter)
        }
        expand => LogicalOperator::Filter(FilterOp {
            predicate: check,
            input: Box::new(expand),
            pushdown_hint: Some(PushdownHint::ExpandInto { target }),
        }),
    }
}

/// The expand below `op`, a chain of filters without a hint or subquery.
pub(super) fn expand_below_filters(op: &LogicalOperator) -> Option<&ExpandOp> {
    match op {
        LogicalOperator::Expand(expand) => Some(expand),
        LogicalOperator::Filter(filter)
            if filter.pushdown_hint.is_none() && !has_subquery(&filter.predicate) =>
        {
            expand_below_filters(&filter.input)
        }
        _ => None,
    }
}

/// The input variable that `predicate` says is the target of `expand`: one
/// single hop to a variable it binds itself, over an input that writes nothing.
fn closing_target(predicate: &LogicalExpression, expand: &ExpandOp) -> Option<String> {
    if expand.is_variable_length() || expand.path_alias.is_some() || expand.input.has_mutations() {
        return None;
    }
    let bound = expand.input.bound_variables(None)?;
    if bound.contains(&expand.to_variable) {
        return None;
    }
    let mut conjuncts = Vec::new();
    split_conjuncts(predicate, &mut conjuncts);
    conjuncts.into_iter().find_map(|conjunct| {
        let (a, b) = compared_variables(conjunct)?;
        let target = if a == expand.to_variable {
            b
        } else if b == expand.to_variable {
            a
        } else {
            return None;
        };
        bound.contains(target).then(|| target.to_string())
    })
}

/// Whether `op` binds `name` to a node: a node scan or the far end of an expand
/// binds it, through filters, joins and projections that pass it on.
pub(super) fn binds_a_node(op: &LogicalOperator, name: &str) -> bool {
    match op {
        LogicalOperator::NodeScan(scan) => {
            scan.variable == name
                || scan
                    .input
                    .as_deref()
                    .is_some_and(|input| binds_a_node(input, name))
        }
        LogicalOperator::Expand(expand) => {
            expand.to_variable == name || binds_a_node(&expand.input, name)
        }
        LogicalOperator::Filter(filter) => binds_a_node(&filter.input, name),
        LogicalOperator::Sort(sort) => binds_a_node(&sort.input, name),
        LogicalOperator::Limit(limit) => binds_a_node(&limit.input, name),
        LogicalOperator::Skip(skip) => binds_a_node(&skip.input, name),
        LogicalOperator::Distinct(distinct) => binds_a_node(&distinct.input, name),
        LogicalOperator::Join(join) => {
            binds_a_node(&join.left, name) || binds_a_node(&join.right, name)
        }
        LogicalOperator::LeftJoin(join) => {
            binds_a_node(&join.left, name) || binds_a_node(&join.right, name)
        }
        LogicalOperator::Project(project) => {
            let projected = project.projections.iter().find_map(|projection| {
                match (&projection.alias, &projection.expression) {
                    (Some(alias), expression) if alias == name => Some(expression),
                    (None, expression @ LogicalExpression::Variable(variable))
                        if variable == name =>
                    {
                        Some(expression)
                    }
                    _ => None,
                }
            });
            match projected {
                Some(LogicalExpression::Variable(variable)) => {
                    binds_a_node(&project.input, variable)
                }
                Some(_) => false,
                None => project.pass_through_input && binds_a_node(&project.input, name),
            }
        }
        _ => false,
    }
}

#[cfg(all(test, feature = "gql"))]
mod tests {
    use crate::query::optimizer::{PlannerRule, PlannerRules};
    use crate::session::Session;

    const TRIANGLES: &str = "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) \
                             RETURN a.name, b.name, c.name";

    /// Alix, Gus and Vincent know each other in a circle, two edges from
    /// Vincent to Alix among them; Mia knows herself; other edges go nowhere.
    fn people(session: &Session) {
        session
            .execute(
                "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(g:Person {name: 'Gus'}),
                        (g)-[:KNOWS]->(v:Person {name: 'Vincent'}),
                        (v)-[:KNOWS]->(a),
                        (v)-[:KNOWS]->(a),
                        (v)-[:LIKES]->(a),
                        (g)-[:KNOWS]->(:Person {name: 'Jules'}),
                        (m:Person {name: 'Mia'})-[:KNOWS]->(m),
                        (a)-[:KNOWS]->(m)",
            )
            .unwrap();
    }

    fn plain_and_usual(db: &crate::GrafeoDB) -> (Session, Session) {
        let mut plain = db.session();
        plain.set_planner_rules(PlannerRules::all().with(PlannerRule::ExpandInto, false));
        (plain, db.session())
    }

    fn plan(session: &Session, query: &str) -> String {
        format!(
            "{:?}",
            session.execute(&format!("EXPLAIN {query}")).unwrap().rows()
        )
    }

    /// What a query answers or fails with, rows in the order they came.
    fn answer(session: &Session, query: &str) -> String {
        match session.execute(query) {
            Ok(result) => format!("{:?}", result.rows()),
            Err(error) => format!("{error}"),
        }
    }

    /// With the intersection off: on, a triangle this small intersects instead.
    #[test]
    fn a_hop_back_to_a_bound_node_shows_as_a_lookup() {
        let db = crate::GrafeoDB::new_in_memory();
        let (plain, mut usual) = plain_and_usual(&db);
        people(&usual);
        usual.set_planner_rules(PlannerRules::all().with(PlannerRule::IntersectExpand, false));
        assert!(plan(&usual, TRIANGLES).contains("[expand into: "));
        assert!(!plan(&plain, TRIANGLES).contains("expand into"));
    }

    #[test]
    fn a_hop_to_a_new_node_lists_its_edges() {
        let db = crate::GrafeoDB::new_in_memory();
        let session = db.session();
        people(&session);
        let query = "MATCH (a:Person)-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN a, b, c";
        assert!(!plan(&session, query).contains("expand into"));
    }

    /// The same rows whatever the hop's shape, and in the same order where the two run one plan
    /// but for the hop: the search may order the plans differently, as order is no part of the
    /// contract without an `ORDER BY`.
    #[test]
    fn a_lookup_answers_what_the_list_answers() {
        let db = crate::GrafeoDB::new_in_memory();
        let (plain, usual) = plain_and_usual(&db);
        people(&usual);
        let fixed_order = PlannerRules::all()
            .with(PlannerRule::JoinOrder, false)
            .with(PlannerRule::IntersectExpand, false);
        let mut lists = db.session();
        lists.set_planner_rules(fixed_order.with(PlannerRule::ExpandInto, false));
        let mut looks_up = db.session();
        looks_up.set_planner_rules(fixed_order);
        let sorted = |session: &Session, query: &str| match session.execute(query) {
            Ok(result) => {
                let mut rows: Vec<String> =
                    result.rows().iter().map(|row| format!("{row:?}")).collect();
                rows.sort();
                rows.join("\n")
            }
            Err(error) => format!("{error}"),
        };
        for query in [
            TRIANGLES,
            // Two edges back are two rows.
            "MATCH (a)-[:KNOWS]->(b)-[e:KNOWS]->(a) RETURN a.name, b.name, e",
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[e]->(a) RETURN a.name, b.name, c.name, type(e)",
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[e:KNOWS|LIKES]->(a) RETURN c.name, type(e)",
            // Against the edge, either way, and a self-loop once.
            "MATCH (a)-[:KNOWS]->(b)<-[:KNOWS]-(c)<-[:KNOWS]-(a) RETURN a.name, b.name, c.name",
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]-(a) RETURN a.name, b.name, c.name",
            "MATCH (a)-[:KNOWS]-(b)-[:KNOWS]-(a) RETURN a.name, b.name",
            "MATCH (a)-[e:KNOWS]->(a) RETURN a.name",
            // A label on the end it comes back to, and more filters around it.
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a:Person) RETURN a.name",
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) WHERE a.name <> c.name RETURN a.name",
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) WHERE b.name = 'Gus' RETURN c.name",
            // Two statements, an OPTIONAL hop, and no such type.
            "MATCH (a)-[:KNOWS]->(b) MATCH (b)-[:KNOWS]->(a) RETURN a.name, b.name",
            "MATCH (a:Person) OPTIONAL MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN a.name, b.name",
            "MATCH (a)-[:KNOWS]->(b)-[:NO_SUCH_TYPE]->(a) RETURN a.name",
            // A count over the hop counts its edges.
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(a) RETURN count(*)",
            "MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(a) RETURN a.name, count(*)",
            // The target is no node, or is null.
            "UNWIND [1, 2] AS a MATCH (x:Person)-[:KNOWS]->(a) RETURN x.name",
            "OPTIONAL MATCH (a:NoSuchLabel) MATCH (x:Person)-[:KNOWS]->(a) RETURN x.name",
        ] {
            assert_eq!(sorted(&plain, query), sorted(&usual, query), "{query}");
            assert_eq!(
                answer(&lists, query),
                answer(&looks_up, query),
                "{query}: the order"
            );
        }
    }

    /// The hop reads the edges back to its target, not the hub's thousand
    /// others: the operator takes the hint, and the rows stay the same.
    #[test]
    fn a_lookup_does_not_read_the_other_edges() {
        let db = crate::GrafeoDB::new_in_memory();
        let (plain, usual) = plain_and_usual(&db);
        usual
            .execute(
                "INSERT (a:Person {name: 'Alix'})-[:KNOWS]->(h:Hub {name: 'Gus'}),
                        (h)-[:KNOWS]->(a)",
            )
            .unwrap();
        usual
            .execute("MATCH (h:Hub) UNWIND range(1, 1000) AS i INSERT (h)-[:KNOWS]->(:Item)")
            .unwrap();
        let query = "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(h)-[:KNOWS]->(a) RETURN h.name";
        let read = |session: &Session| {
            let result = session.execute(query).unwrap();
            assert_eq!(result.row_count(), 1);
            result.rows_scanned().unwrap()
        };
        let (listed, looked_up) = (read(&plain), read(&usual));
        assert!(listed >= 1000, "the list read {listed}");
        assert!(looked_up < 100, "the lookup read {looked_up}");
    }

    /// The lookup sees what the transaction sees: its own new and removed
    /// edges, and not another's open ones.
    #[test]
    fn a_lookup_reads_as_its_transaction_reads() {
        let db = crate::GrafeoDB::new_in_memory();
        let (mut plain, mut usual) = plain_and_usual(&db);
        people(&usual);
        let query = "MATCH (a)-[:KNOWS]->(b)-[e:KNOWS]->(a) RETURN a.name, b.name, e";
        let before = answer(&usual, query);

        for session in [&mut plain, &mut usual] {
            session.begin_transaction().unwrap();
            session
                .execute(
                    "MATCH (g:Person {name: 'Gus'}), (a:Person {name: 'Alix'})
                     INSERT (g)-[:KNOWS]->(a)",
                )
                .unwrap();
            let own = answer(session, query);
            assert_ne!(own, before, "its own edge");
            session.rollback().unwrap();
        }
        assert_eq!(answer(&usual, query), before);

        let writer = db.session();
        writer
            .execute(
                "MATCH (m:Person {name: 'Mia'})-[e:KNOWS]->(m)
                 DELETE e",
            )
            .unwrap();
        let after = answer(&usual, query);
        assert_ne!(after, before, "a removed edge");
        assert_eq!(after, answer(&plain, query));
    }
}
