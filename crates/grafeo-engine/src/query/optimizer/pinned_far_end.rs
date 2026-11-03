//! Hops whose far end a key pins find that end first.

use grafeo_core::execution::operators::SeekKey;

use super::Optimizer;
use super::closing_hops::expand_below_filters;
use crate::query::plan::{ExpandOp, LogicalOperator, PushdownHint};
use crate::query::planner::lpg::seek::constant_seek;

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
    pub(super) fn look_up_pinned_far_ends(&self, op: LogicalOperator) -> LogicalOperator {
        let mut op = op.map_children(|child| self.look_up_pinned_far_ends(child));
        if let LogicalOperator::Filter(filter) = &mut op
            && filter.pushdown_hint.is_none()
        {
            filter.pushdown_hint = expand_below_filters(&filter.input)
                .filter(|expand| hop_can_take_a_key(expand))
                .and_then(|expand| {
                    let has_index = |property: &str| self.indexed_properties.contains(property);
                    constant_seek(&filter.predicate, &expand.to_variable, has_index)
                })
                .map(|seek| PushdownHint::ExpandToKey {
                    key: match seek.key {
                        SeekKey::Id => "id()".to_string(),
                        SeekKey::Property(property) => property,
                    },
                });
        }
        op
    }
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
