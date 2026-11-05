//! The last two hops of a cycle intersect two adjacency lists.

use super::closing_hops::{binds_a_node, without_check};
use super::cycles::same;
use super::{IncomingEdges, Optimizer, cannot_raise};
use crate::query::plan::{
    ExpandDirection, ExpandOp, FilterOp, LogicalExpression, LogicalOperator, PushdownHint,
};

impl Optimizer {
    /// Marks each closing hop right over another hop to run with it as one
    /// intersection, where the columns are free: its rows come by node. A closing
    /// hop looks up every far end of the hop below; the intersection gallops from
    /// the shorter of each pair of runs, or hashes the shorter list on a store that
    /// cannot read them where they lie. Checks between the hops move above it, as they
    /// cannot raise, and stay below the rest of the closing filter.
    pub(super) fn intersect_closing_hops(
        &self,
        op: LogicalOperator,
        free: bool,
    ) -> LogicalOperator {
        match op {
            LogicalOperator::Filter(filter) => {
                let filter = if free { self.marked(filter) } else { filter };
                LogicalOperator::Filter(filter)
                    .map_children(|child| self.intersect_closing_hops(child, free))
            }
            LogicalOperator::Return(mut ret) => {
                let star = ret.items.iter().any(
                    |item| matches!(&item.expression, LogicalExpression::Variable(name) if name == "*"),
                );
                ret.input = Box::new(self.intersect_closing_hops(*ret.input, !star));
                LogicalOperator::Return(ret)
            }
            LogicalOperator::Project(mut project) => {
                let free = free || !project.pass_through_input;
                project.input = Box::new(self.intersect_closing_hops(*project.input, free));
                LogicalOperator::Project(project)
            }
            LogicalOperator::Aggregate(mut aggregate) => {
                aggregate.input = Box::new(self.intersect_closing_hops(*aggregate.input, true));
                LogicalOperator::Aggregate(aggregate)
            }
            op @ (LogicalOperator::Union(_)
            | LogicalOperator::Except(_)
            | LogicalOperator::Intersect(_)
            | LogicalOperator::Otherwise(_)) => {
                op.map_children(|child| self.intersect_closing_hops(child, false))
            }
            other => other.map_children(|child| self.intersect_closing_hops(child, free)),
        }
    }

    /// `filter` as an intersection, when it closes a cycle over a hop that reaches
    /// the closing hop through label checks at most.
    fn marked(&self, mut filter: FilterOp) -> FilterOp {
        let Some(PushdownHint::ExpandInto { target }) = &filter.pushdown_hint else {
            return filter;
        };
        let target = target.clone();
        let LogicalOperator::Expand(closing) = filter.input.as_ref() else {
            return filter;
        };
        let mut below = closing.input.as_ref();
        while let LogicalOperator::Filter(check) = below {
            if !cannot_raise(&check.predicate) {
                return filter;
            }
            below = &check.input;
        }
        let LogicalOperator::Expand(near) = below else {
            return filter;
        };
        // They meet at the near hop's far end: the closing hop goes on from it to
        // the target, or ends there, from a node bound before.
        let goes_on = near.to_variable == closing.from_variable;
        let ends_there = near.to_variable == target;
        if !(goes_on || ends_there) || !self.reads_both_ways(near, closing, goes_on) {
            return filter;
        }
        // The list it intersects with is read from a node bound before the near hop.
        let other = if goes_on {
            &target
        } else {
            &closing.from_variable
        };
        if !binds_a_node(&near.input, other) {
            return filter;
        }
        // The checks between the hops read rows both hops have made.
        let LogicalOperator::Expand(closing) = filter.input.as_mut() else {
            return filter;
        };
        let mut lifted = Vec::new();
        while let LogicalOperator::Filter(check) = closing.input.as_mut() {
            let input = std::mem::replace(check.input.as_mut(), LogicalOperator::Empty);
            lifted.push(check.predicate.clone());
            *closing.input = input;
        }
        let fresh = closing.to_variable.clone();
        let Some(lifted) = LogicalExpression::conjunction(lifted.into_iter().rev()) else {
            filter.pushdown_hint = Some(PushdownHint::IntersectExpand { target });
            return filter;
        };
        // The closing check alone goes below them: what else the filter says may raise,
        // and they were written first.
        let intersection = LogicalOperator::Filter(FilterOp {
            predicate: same(&fresh, target.clone()),
            pushdown_hint: Some(PushdownHint::IntersectExpand {
                target: target.clone(),
            }),
            input: filter.input,
        });
        let checks = FilterOp {
            predicate: lifted,
            pushdown_hint: None,
            input: Box::new(intersection),
        };
        match without_check(filter.predicate, &fresh, &target) {
            Some(predicate) => FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(LogicalOperator::Filter(checks)),
            },
            None => checks,
        }
    }

    /// Whether the two hops are single hops the store can read from their two
    /// bound nodes, over an input that writes nothing.
    fn reads_both_ways(&self, near: &ExpandOp, closing: &ExpandOp, goes_on: bool) -> bool {
        let single =
            |expand: &ExpandOp| !expand.is_variable_length() && expand.path_alias.is_none();
        let reads = |direction: ExpandDirection| {
            direction != ExpandDirection::Incoming || self.incoming_edges == IncomingEdges::Kept
        };
        single(near)
            && single(closing)
            && !near.input.has_mutations()
            && reads(near.direction)
            // A hop that goes on is read from the node it comes back to.
            && reads(match closing.direction {
                ExpandDirection::Outgoing if goes_on => ExpandDirection::Incoming,
                ExpandDirection::Incoming if goes_on => ExpandDirection::Outgoing,
                direction => direction,
            })
    }
}
