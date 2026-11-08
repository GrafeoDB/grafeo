//! Hops whose far end a key pins find that end first.

use grafeo_core::execution::operators::{KeyLabels, SeekKey};

use super::Optimizer;
use super::closing_hops::expand_below_filters;
use crate::query::plan::{ExpandOp, FilterOp, LogicalOperator, PushdownHint};
use crate::query::planner::lpg::seek::{LookupLabels, constant_seek};

impl Optimizer {
    /// Marks each filter that pins the target of the expand below it by a key
    /// known without a row: `(c)-[:E]-(b {id: $y})`. The expand then finds the
    /// nodes that hold the key once, before it reads its input, and goes on
    /// from there (see `ExpandOperator::with_target_key`): no node ends the
    /// query, one node is a lookup per row, and several filter each row's list
    /// without reading a property. A key too common to be worth it lists as
    /// usual, so this loses at most the cost of finding that out.
    ///
    /// The filter stays and decides, as over a node seek. Rows and columns are
    /// those of the plain plan, in its order, so an expand that does not take
    /// the hint is merely slower.
    ///
    /// Not under SQL/PGQ's `=`, which raises across kinds: the far end found by its key would
    /// leave a reached value of another kind out of the filter that raises on it.
    pub(super) fn look_up_pinned_far_ends(&self, op: LogicalOperator) -> LogicalOperator {
        if !self.key_reads_decide_rows() {
            return op;
        }
        self.pin_far_ends(op, false)
    }

    /// [`look_up_pinned_far_ends`](Self::look_up_pinned_far_ends) of `op`, whose chain of filters
    /// is taken as one from its top.
    fn pin_far_ends(&self, op: LogicalOperator, below_a_filter: bool) -> LogicalOperator {
        let is_filter = matches!(op, LogicalOperator::Filter(_));
        let mut op = op.map_children(|child| self.pin_far_ends(child, is_filter));
        if !below_a_filter && let LogicalOperator::Filter(top) = &mut op {
            self.pin_far_end_of_chain(top);
        }
        op
    }

    /// Hints at the far end of the expand below the chain of filters `top` heads, on the filter
    /// that holds the key: the lowest one that pins it, by the label checks of the whole chain, as
    /// a check above that filter requires its label as much as one below.
    #[inline(never)]
    fn pin_far_end_of_chain(&self, top: &mut FilterOp) {
        let Some((depth, hint)) = self.far_end_key(top) else {
            return;
        };
        let mut filter = top;
        for _ in 0..depth {
            let LogicalOperator::Filter(below) = filter.input.as_mut() else {
                return;
            };
            filter = below;
        }
        filter.pushdown_hint = Some(hint);
    }

    /// The hint [`pin_far_end_of_chain`](Self::pin_far_end_of_chain) sets, with how many filters
    /// below `top` it goes.
    fn far_end_key(&self, top: &FilterOp) -> Option<(usize, PushdownHint)> {
        let mut chain = vec![top];
        let mut below = top.input.as_ref();
        while let LogicalOperator::Filter(filter) = below {
            chain.push(filter);
            below = &filter.input;
        }
        let LogicalOperator::Expand(expand) = below else {
            return None;
        };
        if !hop_can_take_a_key(expand) {
            return None;
        }
        let labels = LookupLabels::of(top, &expand.to_variable);
        let required: Vec<String> = labels.required().iter().map(ToString::to_string).collect();
        let has_index = |property: &str| self.index_serves(&labels, property);
        chain.iter().enumerate().rev().find_map(|(depth, filter)| {
            if filter.pushdown_hint.is_some() || expand_below_filters(&filter.input).is_none() {
                return None;
            }
            let seek = constant_seek(&filter.predicate, &expand.to_variable, has_index)?;
            let hint = match seek.key {
                SeekKey::Id => PushdownHint::ExpandToKey {
                    key: "id()".to_string(),
                    labels: KeyLabels {
                        read: Vec::new(),
                        required: required.clone(),
                    },
                },
                SeekKey::Property(property) => PushdownHint::ExpandToKey {
                    labels: KeyLabels {
                        read: self
                            .index_read(&labels, &property)
                            .unwrap_or_default()
                            .iter()
                            .map(ToString::to_string)
                            .collect(),
                        required: required.clone(),
                    },
                    key: property,
                },
            };
            Some((depth, hint))
        })
    }
}

/// The variable of the far end of the expand below the chain of filters `top` heads, when a key
/// could pin it: one hop that can take a key, below filters with no hint and no subquery.
pub(crate) fn far_end_variable(top: &FilterOp) -> Option<String> {
    expand_below_filters(&top.input)
        .filter(|expand| hop_can_take_a_key(expand))
        .map(|expand| expand.to_variable.clone())
}

/// Whether `expand` is one hop to a variable it binds itself, over an input
/// that writes nothing: a key found once would miss what the input writes.
fn hop_can_take_a_key(expand: &ExpandOp) -> bool {
    !expand.is_variable_length()
        && expand.path_alias.is_none()
        && !expand.input.has_mutations()
        && expand
            .input
            .bound_variables(None)
            .is_some_and(|bound| !bound.contains(&expand.to_variable))
}
