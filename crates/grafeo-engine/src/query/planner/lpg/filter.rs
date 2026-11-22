//! Filter planning with zone map pre-filtering and index lookups.
//!
//! The order of the rewrite/pushdown attempts in [`Planner::plan_filter`][pf]
//! is load-bearing, not a performance tweak:
//!
//! 1. Subquery rewrites (`extract_complex_exists` for semi-joins, then the
//!    subqueries planned per row, `plan_filter_with_subqueries`) must run
//!    first. The later steps assume a predicate whose subqueries
//!    the edge check answers; once a rewrite fires, the rest of the method
//!    is bypassed.
//! 2. Zone-map short-circuit fires before any index lookup so we can
//!    skip opening an index file at all when summary statistics prove
//!    emptiness.
//! 3. A scan that runs for each input row is looked up per row by an ID or
//!    an indexed key (the seek, `seek.rs`), or else joined to the input by
//!    hash on equal values (`value_join.rs`). Property-index and
//!    range-index attempts on a scan without input come next, before hybrid
//!    pushdown because they are cheaper and strictly more specific. The
//!    zone maps and the property-index path read the store while planning,
//!    so a filter that runs after a write of its statement takes neither
//!    (see `after_write`): the seek looks its constant key up when it runs.
//! 4. Compound hybrid (`compound_hybrid_route`) is tried
//!    before single-sided vector/text pushdown so an `AND` over both
//!    kinds doesn't get torn apart into a scan + filter. These index
//!    searches, too, replace only a scan without input: they find the
//!    nodes alone, and a scan per input row keeps that row.
//! 5. The generic `FilterOperator` is the last resort; whatever fell
//!    through still runs correctly, just without an index.
//!
//! Changing this order needs a correctness argument, not just a
//! benchmark.
//!
//! [pf]: super::Planner::plan_filter

use std::collections::HashSet;

use grafeo_common::collections::GrafeoSet;

use super::seek::LookupRead;
use super::subquery::reads_outer_values;
use grafeo_common::types::{NodeId, TemporalComparison};
use grafeo_core::execution::operators::find_through_label_indexes;
use grafeo_core::execution::operators::{KeyColumnChecks, KeyColumnIndexes, KeyColumnScanOperator};
use grafeo_core::graph::rowgroup::KeyColumnQuery;

use super::expand::PendingEdgeCondition;
use super::seek::{CheckedScan, LookupLabels, SeekChoice};
use super::value_join::ValueJoin;
use super::{
    Arc, BinaryOp, EmptyOperator, Error, ExpressionPredicate, FilterOp, FilterOperator, GraphStore,
    HashJoinOperator, HashMap, LogicalExpression, LogicalOperator, NodeListOperator, NodeScanOp,
    Operator, PhysicalJoinType, RangeScanOperator, Result, UnaryOp, Value,
};
use crate::query::plan::PushdownHint;
#[cfg(feature = "vector-index")]
use crate::query::plan::{BuiltinFunction, Callee};

/// A route [`plan_filter`](super::Planner::plan_filter) builds a filter by, with what its
/// guard found that the build reads (see [`filter_route`]). The
/// variants are in the order their guards are checked.
pub(super) enum FilterRoute<'a> {
    /// An `EXISTS` subquery as a semi-join (`NOT EXISTS`: an anti-join), and the rest of the
    /// predicate.
    ExistsJoin {
        subquery: &'a LogicalOperator,
        negated: bool,
        remaining: Option<LogicalExpression>,
    },
    /// Subqueries that run per row of the input (see `subquery.rs`).
    LiftedSubqueries,
    /// The closing equality of an intersection, which the intersection meets already: the input
    /// alone.
    IntersectionClosing,
    /// The empty result, when the zone maps prove that no row matches.
    ZoneMapEmpty,
    /// Conditions on every edge of a variable-length path, checked during the expand's search.
    EdgeConditionIntoPath {
        path: &'a str,
        taken: Vec<PendingEdgeCondition>,
        kept: Vec<LogicalExpression>,
    },
    /// A lookup of the node per input row, by a key or an ID.
    NodeSeek {
        below: CheckedScan<'a>,
        labels: LookupLabels<'a>,
        seek: SeekChoice<'a>,
    },
    /// The scanned nodes joined to the input's rows on equal values.
    ValueJoin(ValueJoin<'a>),
    /// An index lookup, or a read of a key's column, for equalities on the scanned node: each
    /// property, its value and what a lookup of the property reads.
    PropertyIndex {
        checked: CheckedScan<'a>,
        conditions: Vec<(String, Value, LookupRead<'a>)>,
    },
    /// The same for a `var.prop IN [literals]` conjunct, with what a lookup of `property`
    /// reads.
    InListIndex {
        checked: CheckedScan<'a>,
        labels: LookupLabels<'a>,
        property: &'a str,
        values: Vec<Value>,
        read: LookupRead<'a>,
    },
    /// A range scan of the property the predicate bounds.
    RangeIndex {
        scan: &'a NodeScanOp,
        property: String,
        range: grafeo_core::graph::ValueRange,
    },
    /// A vector and a text index search, joined.
    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    CompoundHybrid {
        scan: &'a NodeScanOp,
        label: &'a str,
        vector: ExtractedVectorPredicate,
        text: super::filter_hybrid::ExtractedTextPredicate,
        is_or: bool,
    },
    /// A vector index search.
    #[cfg(feature = "vector-index")]
    VectorIndex {
        scan: &'a NodeScanOp,
        label: &'a str,
        extracted: ExtractedVectorPredicate,
    },
    /// A text index search.
    #[cfg(feature = "text-index")]
    TextIndex {
        scan: &'a NodeScanOp,
        label: &'a str,
        extracted: super::filter_hybrid::ExtractedTextPredicate,
    },
    /// A scan of the nodes of the labels of an `OR` the predicate requires.
    LabelUnion {
        scan: &'a NodeScanOp,
        labels: Vec<String>,
    },
    /// The predicate checked on each row of the planned input.
    Generic,
}

#[cfg(any(test, prepared_plans))]
impl FilterRoute<'_> {
    /// The variant's name, as a test records the routes taken and a recipe names a route it
    /// does not cover.
    pub(super) fn name(&self) -> &'static str {
        match self {
            Self::ExistsJoin { .. } => "ExistsJoin",
            Self::LiftedSubqueries => "LiftedSubqueries",
            Self::IntersectionClosing => "IntersectionClosing",
            Self::ZoneMapEmpty => "ZoneMapEmpty",
            Self::EdgeConditionIntoPath { .. } => "EdgeConditionIntoPath",
            Self::NodeSeek { .. } => "NodeSeek",
            Self::ValueJoin(_) => "ValueJoin",
            Self::PropertyIndex { .. } => "PropertyIndex",
            Self::InListIndex { .. } => "InListIndex",
            Self::RangeIndex { .. } => "RangeIndex",
            #[cfg(all(feature = "vector-index", feature = "text-index"))]
            Self::CompoundHybrid { .. } => "CompoundHybrid",
            #[cfg(feature = "vector-index")]
            Self::VectorIndex { .. } => "VectorIndex",
            #[cfg(feature = "text-index")]
            Self::TextIndex { .. } => "TextIndex",
            Self::LabelUnion { .. } => "LabelUnion",
            Self::Generic => "Generic",
        }
    }
}

/// What [`filter_route`] reads besides the plan: the planner that plans the filter, or what a
/// prepared plan's recipe holds, which knows the rules and no store.
pub(super) trait FilterRouteReads {
    /// What a guard answers that needs more than this holds (nothing, for the planner).
    type Unknown;

    /// The planner's rules.
    fn rules(&self) -> crate::query::optimizer::PlannerRules;

    /// Whether the filter reads the store as it is while the plan is built (see `after_write`).
    fn reads_the_store_as_planned(&self) -> bool;

    /// The planner, for a guard's reads of the store's schema and statistics, of the session and
    /// of the planner's own state.
    ///
    /// # Errors
    ///
    /// [`Self::Unknown`] when this holds no planner.
    fn planner(&self) -> std::result::Result<&super::Planner, Self::Unknown>;
}

impl FilterRouteReads for super::Planner {
    type Unknown = std::convert::Infallible;

    fn rules(&self) -> crate::query::optimizer::PlannerRules {
        self.rules
    }

    fn reads_the_store_as_planned(&self) -> bool {
        !self.after_a_write.get()
    }

    fn planner(&self) -> std::result::Result<&super::Planner, Self::Unknown> {
        Ok(self)
    }
}

/// What a guard of [`filter_route`] answers: `None` when its route does not apply, else the
/// route, or what `R` answers when the guard needs more than `R` holds.
pub(super) type Guarded<'a, R> =
    Option<std::result::Result<FilterRoute<'a>, <R as FilterRouteReads>::Unknown>>;

/// The planner of a guard's reads, or the guard's answer that they cannot tell.
macro_rules! planner_of {
    ($reads:expr) => {
        match $reads.planner() {
            Ok(planner) => planner,
            Err(unknown) => return Some(Err(unknown)),
        }
    };
}
pub(super) use planner_of;

/// The route `filter` is built by: of the routes after `after`, a route whose build declined,
/// the first in the order of [`FilterRoute`]'s variants whose guard holds. It plans nothing and
/// writes no planner state: a guard reads the plan, the rules, the store's schema and
/// statistics, and may fill the cache of edge type names.
///
/// Each guard checks the plan's shape before it asks `reads` for anything else, so a route
/// that cannot apply by its shape answers so whatever `reads` holds.
///
/// Only the builds of [`FilterRoute::ZoneMapEmpty`], [`FilterRoute::PropertyIndex`] and
/// [`FilterRoute::InListIndex`] decline: what they read of the store decides.
///
/// # Errors
///
/// What `reads` answers when a guard needs more than it holds.
pub(super) fn filter_route<'a, R: FilterRouteReads>(
    reads: &R,
    filter: &'a FilterOp,
    after: Option<&FilterRoute<'a>>,
) -> std::result::Result<FilterRoute<'a>, R::Unknown> {
    use super::Planner;
    if after.is_none() {
        let input_writes = filter.input.has_mutations();
        if let Some(route) = Planner::subquery_route(reads, filter, input_writes) {
            return route;
        }

        // An intersection makes only rows that meet at its target (see
        // `plan_intersection`): its closing equality, the whole predicate of the
        // filter the optimizer marks (`intersect_hops`), is not checked again.
        if let Some(PushdownHint::IntersectExpand { target }) = &filter.pushdown_hint
            && let LogicalOperator::Expand(closing) = filter.input.as_ref()
            && matches!(closing.input.as_ref(), LogicalOperator::Expand(_))
            && crate::query::optimizer::without_check(
                filter.predicate.clone(),
                &closing.to_variable,
                target,
            )
            .is_none()
        {
            return Ok(FilterRoute::IntersectionClosing);
        }

        // Check zone maps for simple property predicates before scanning
        // If zone map says "definitely no matches", we can short-circuit.
        // Not below a mutation, nor after one (see `after_write`): zone maps
        // describe the store as it is before this query runs, not values the
        // query writes first.
        if !input_writes
            && reads
                .rules()
                .has(crate::query::optimizer::PlannerRule::ZoneMaps)
            && reads.reads_the_store_as_planned()
        {
            return Ok(FilterRoute::ZoneMapEmpty);
        }
    }

    if !matches!(
        after,
        Some(FilterRoute::PropertyIndex { .. } | FilterRoute::InListIndex { .. })
    ) {
        // A condition on every edge of a variable-length path: checked during
        // the expand's search
        if let Some(route) = Planner::edge_condition_route(reads, filter) {
            return route;
        }
        // A key from the input row, or an ID: look the node up per row
        if let Some(route) = Planner::node_seek_route(reads, filter) {
            return route;
        }
        // Values of the input row equal to values of the node: scan once and
        // join on them
        if let Some(route) = Planner::value_join_route(reads, filter) {
            return route;
        }
        // Try to use property index for equality predicates on indexed properties
        if let Some(route) = Planner::property_index_route(reads, filter) {
            return route;
        }
    }

    // Try to use property index for `var.prop IN [literals]` predicates,
    // unioning per-value index lookups instead of falling back to a full
    // scan + filter. Empirically this is the difference between ~10ms
    // and multiple seconds on a few thousand reviews when an `id` index
    // exists.
    if !matches!(after, Some(FilterRoute::InListIndex { .. }))
        && let Some(route) = Planner::in_list_route(reads, filter)
    {
        return route;
    }

    // Try to use range optimization for range predicates (>, <, >=, <=)
    if let Some(route) = Planner::range_route(reads, filter) {
        return route;
    }

    // Try compound hybrid first (both vector AND/OR text in same predicate)
    #[cfg(all(feature = "vector-index", feature = "text-index"))]
    if let Some(route) = Planner::compound_hybrid_route(reads, filter) {
        return route;
    }

    // Try to use vector index for similarity/distance predicates
    #[cfg(feature = "vector-index")]
    if let Some(route) = Planner::vector_index_route(reads, filter) {
        return route;
    }

    // Try to use text index for text_score/text_match predicates
    #[cfg(feature = "text-index")]
    if let Some(route) = Planner::text_index_route(reads, filter) {
        return route;
    }

    // An `OR` of labels: the scan reads the nodes of those labels only
    Planner::label_union_route(reads, filter).unwrap_or(Ok(FilterRoute::Generic))
}

impl super::Planner {
    /// Whether node `id` is visible to this query and, when `label` is given,
    /// carries it. An index lookup checks its few results this way, instead of
    /// collecting every node with the label.
    fn visible_with_label(&self, id: NodeId, label: Option<&str>) -> Result<bool> {
        let view = self.read_view();
        Ok(match label {
            Some(label) => self.store.node_has_label(id, label, view)? == Some(true),
            None => self.store.is_node_visible(id, view)?,
        })
    }

    /// Plans a filter operator by its route (see [`filter_route`]).
    ///
    /// Uses zone map pre-filtering to potentially skip scans when predicates
    /// definitely won't match any data. Also uses property indexes when available
    /// for O(1) lookups instead of full scans. Complex EXISTS/NOT EXISTS subqueries
    /// are rewritten as semi-joins or anti-joins.
    pub(super) fn plan_filter(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // A LIMIT directly above this filter cuts the range scan this filter becomes, and no
        // other: a scan below another operator under it (an expand, a second filter) feeds rows
        // that may still be dropped, so it reads them all.
        let limit = self.limit_hint.take();
        let Ok(mut route) = filter_route(self, filter, None);
        // A hop that knows its target: this filter stays over it and decides. The subquery
        // routes come before it and plan without it.
        if !matches!(
            route,
            FilterRoute::ExistsJoin { .. } | FilterRoute::LiftedSubqueries
        ) {
            #[cfg(not(test))]
            self.note_hop_target(filter, &filter.predicate);
            #[cfg(test)]
            if self.note_hop_target(filter, &filter.predicate) {
                self.filter_routes_taken
                    .borrow_mut()
                    .push("hop target".to_string());
            }
        }
        loop {
            #[cfg(test)]
            self.filter_routes_taken
                .borrow_mut()
                .push(route.name().to_string());
            // A build whose reads of the store say no declines (`None`): the next route that
            // applies is built instead.
            let built = match route {
                FilterRoute::ExistsJoin {
                    subquery,
                    negated,
                    remaining,
                } => {
                    return self.plan_exists_as_semi_join(
                        &filter.input,
                        subquery,
                        negated,
                        remaining,
                    );
                }
                FilterRoute::LiftedSubqueries => return self.plan_filter_with_subqueries(filter),
                FilterRoute::IntersectionClosing => return self.plan_input(&filter.input),
                FilterRoute::ZoneMapEmpty => self.plan_filter_with_zone_maps(filter)?,
                FilterRoute::EdgeConditionIntoPath { path, taken, kept } => {
                    return self.plan_filter_into_variable_length_expand(filter, path, taken, kept);
                }
                FilterRoute::NodeSeek {
                    below,
                    labels,
                    seek,
                } => return self.plan_filter_with_node_seek(filter, below, &labels, seek),
                FilterRoute::ValueJoin(join) => {
                    return self.plan_filter_with_value_join(filter, join);
                }
                FilterRoute::PropertyIndex {
                    ref checked,
                    ref conditions,
                } => self.plan_filter_with_property_index(filter, checked, conditions)?,
                FilterRoute::InListIndex {
                    ref checked,
                    ref labels,
                    property,
                    ref values,
                    ref read,
                } => self.plan_filter_with_in_list_index(
                    filter, checked, labels, property, values, read,
                )?,
                FilterRoute::RangeIndex {
                    scan,
                    property,
                    range,
                } => {
                    return self
                        .plan_filter_with_range_index(filter, scan, &property, range, limit);
                }
                #[cfg(all(feature = "vector-index", feature = "text-index"))]
                FilterRoute::CompoundHybrid {
                    scan,
                    label,
                    vector,
                    text,
                    is_or,
                } => {
                    return self
                        .plan_filter_compound_hybrid(filter, scan, label, &vector, &text, is_or);
                }
                #[cfg(feature = "vector-index")]
                FilterRoute::VectorIndex {
                    scan,
                    label,
                    extracted,
                } => return self.plan_filter_with_vector_index(scan, label, &extracted),
                #[cfg(feature = "text-index")]
                FilterRoute::TextIndex {
                    scan,
                    label,
                    extracted,
                } => return self.plan_filter_with_text_index(scan, label, &extracted),
                FilterRoute::LabelUnion { scan, labels } => {
                    return self.plan_filter_with_label_union(filter, scan, labels);
                }
                FilterRoute::Generic => {
                    let (input_op, columns) = self.plan_input(&filter.input)?;
                    return self.filter_over(filter, &filter.predicate, input_op, columns);
                }
            };
            if let Some(built) = built {
                return Ok(built);
            }
            #[cfg(test)]
            self.filter_routes_taken
                .borrow_mut()
                .push(format!("{} declined", route.name()));
            let Ok(next) = filter_route(self, filter, Some(&route));
            route = next;
        }
    }

    /// The subquery routes of `filter`, whose input writes when `input_writes`.
    fn subquery_route<'a, R: FilterRouteReads>(
        reads: &R,
        filter: &'a FilterOp,
        input_writes: bool,
    ) -> Guarded<'a, R> {
        // Check for complex EXISTS/NOT EXISTS patterns and rewrite as semi/anti join.
        // Simple single-hop EXISTS patterns from a node of the row are handled
        // by the fast path in convert_expression() -> extract_exists_pattern().
        // Not below a write: the join reads the subquery before its input, so
        // the subquery would miss what the input writes; per row (below), it
        // runs after the whole input.
        // A predicate without a subquery (most) skips both checks, which list
        // the input's variables: every filter is planned on every execution.
        if !crate::query::plan::has_subquery(&filter.predicate) {
            return None;
        }
        let planner = planner_of!(reads);
        let outer = filter.input.bound_variables(None);
        if !input_writes
            && let Some((subquery, negated, remaining)) =
                planner.extract_complex_exists(&filter.predicate, outer.as_ref())
        {
            return Some(Ok(FilterRoute::ExistsJoin {
                subquery,
                negated,
                remaining,
            }));
        }

        // EXISTS and COUNT subqueries the edge check cannot answer run per
        // row of the input (see `subquery.rs`), and below a write all of
        // them. With the variables the input's rows hold, a subquery that
        // shares none of them is lifted too (counted once instead of per
        // row).
        let input_columns: Option<Vec<String>> = outer.map(|names| names.into_iter().collect());
        planner
            .has_subquery_to_lift(&filter.predicate, input_columns.as_deref(), input_writes)
            .then_some(Ok(FilterRoute::LiftedSubqueries))
    }

    /// The empty result of `filter` when its zone maps prove that no row matches; `None` when
    /// a row may match, or the zone maps cannot tell.
    pub(super) fn plan_filter_with_zone_maps(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        if let Some(false) = self.check_zone_map_for_predicate(&filter.predicate, &filter.input) {
            // Zone map says no matches possible - return empty result
            let (_, columns) = self.plan_input(&filter.input)?;
            return Ok(Some(self.empty_over(columns)));
        }
        Ok(None)
    }

    /// No rows, with the `columns` of the input it takes the place of.
    pub(super) fn empty_over(&self, columns: Vec<String>) -> (Box<dyn Operator>, Vec<String>) {
        let schema = self.derive_schema_from_columns(&columns);
        (Box::new(EmptyOperator::new(schema)), columns)
    }

    /// The filter `filter` over its planned input, as `predicate` (its own, or a recipe's filled
    /// with a call's values) checked on each row. A predicate that reads the graph after a write
    /// (`... SET h.c = i WITH h WHERE h.c = 4`) reads what the whole write left (see
    /// `after_write`).
    pub(super) fn filter_over(
        &self,
        filter: &FilterOp,
        predicate: &LogicalExpression,
        mut input_op: Box<dyn Operator>,
        columns: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        if super::after_write::reads_the_graph(predicate) {
            input_op = super::mutation::read_first_after_a_write(input_op, &filter.input);
        }

        // Build variable to column index mapping
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();

        // Convert logical expression to filter expression
        let filter_expr = self.convert_expression(predicate)?;

        // Create the predicate
        let predicate = ExpressionPredicate::new(
            filter_expr,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            self.read_view(),
        )
        .with_session_context(self.session_context.clone());

        // Create the filter operator
        let operator = Box::new(FilterOperator::new(input_op, Box::new(predicate)));

        Ok((operator, columns))
    }

    /// [`FilterRoute::EdgeConditionIntoPath`] for the `all(hop IN edges(path) WHERE condition)`
    /// conjuncts of a filter right over the variable-length expand that binds `path`, whose
    /// condition reads the one edge and the input row only (a conjunct that reads anything else
    /// stays in the filter). `None` when no conjunct qualifies.
    fn edge_condition_route<'a, R: FilterRouteReads>(
        reads: &R,
        filter: &'a FilterOp,
    ) -> Guarded<'a, R> {
        let LogicalOperator::Expand(expand) = filter.input.as_ref() else {
            return None;
        };
        let path = expand.path_alias.as_deref()?;
        if !expand.is_variable_length() {
            return None;
        }
        if planner_of!(reads).reachability_mode(expand).is_some() {
            return None;
        }
        let row = expand.input.bound_variables(None)?;
        let mut conjuncts = Vec::new();
        split_conjuncts(&filter.predicate, &mut conjuncts);
        let mut taken = Vec::new();
        let mut kept = Vec::new();
        for conjunct in conjuncts {
            match edge_condition_of(&conjunct, path, &row) {
                Some((hop, predicate)) => taken.push(PendingEdgeCondition {
                    path: path.to_string(),
                    hops: vec![hop],
                    predicate,
                }),
                None => kept.push(conjunct),
            }
        }
        if taken.is_empty() {
            return None;
        }
        Some(Ok(FilterRoute::EdgeConditionIntoPath { path, taken, kept }))
    }

    /// Plans the conditions `taken` of a filter right over the variable-length expand that binds
    /// `path` as a check of each edge during its search, and the `kept` conjuncts above it: a
    /// path stops at its first edge that fails, instead of being enumerated whole and dropped
    /// after. That holds the same paths, as each condition reads the one edge and the input row
    /// only.
    fn plan_filter_into_variable_length_expand(
        &self,
        filter: &FilterOp,
        path: &str,
        taken: Vec<PendingEdgeCondition>,
        kept: Vec<LogicalExpression>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        self.pending_edge_conditions.borrow_mut().extend(taken);
        let planned = match join_conjuncts(kept) {
            Some(predicate) => self.plan_filter(&FilterOp {
                predicate,
                input: filter.input.clone(),
                pushdown_hint: filter.pushdown_hint.clone(),
            }),
            None => self.plan_input(&filter.input),
        };
        // The expand took its conditions; drop any it did not reach.
        self.pending_edge_conditions
            .borrow_mut()
            .retain(|condition| condition.path != path);
        planned
    }

    /// Plans a filter whose predicate has `EXISTS` or `COUNT` subqueries that
    /// run per row: the parts of an `AND` without one are a filter of their
    /// own first (which may use an index), then each subquery adds its count
    /// to the row and the rest of the predicate reads it.
    fn plan_filter_with_subqueries(
        &self,
        filter: &FilterOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let mut conjuncts = Vec::new();
        split_conjuncts(&filter.predicate, &mut conjuncts);
        let input_columns: Option<Vec<String>> = filter
            .input
            .bound_variables(None)
            .map(|names| names.into_iter().collect());
        let input_writes = filter.input.has_mutations();
        // Right after a write the plain conjuncts filter after the
        // subqueries, whose joins read the whole input first: a filter of
        // their own below would read the rows as the write passes them on
        // (or need a second read of the whole input, see `after_write`).
        let pending = super::after_write::writes_pending(&filter.input);
        let (with_subqueries, plain): (Vec<_>, Vec<_>) =
            conjuncts.into_iter().partition(|conjunct| {
                pending
                    || self.has_subquery_to_lift(conjunct, input_columns.as_deref(), input_writes)
            });
        // The labels a conjunct with subqueries requires (`(n:A AND EXISTS
        // {...}) OR (n:B AND ...)`) filter first too, so a scan of every node
        // reads only those labels' nodes and the subqueries run for them
        // alone; the check is implied, the rows are the same.
        let mut plain = plain;
        if !pending && let LogicalOperator::NodeScan(scan) = filter.input.as_ref() {
            plain.extend(
                with_subqueries.iter().filter_map(|conjunct| {
                    super::scan::implied_label_check(conjunct, &scan.variable)
                }),
            );
        }
        let (input_op, columns) = match join_conjuncts(plain) {
            Some(predicate) => self.plan_filter(&FilterOp {
                predicate,
                input: filter.input.clone(),
                pushdown_hint: filter.pushdown_hint.clone(),
            })?,
            None => self.plan_input(&filter.input)?,
        };
        let predicate = join_conjuncts(with_subqueries)
            .ok_or_else(|| Error::Internal("filter without a subquery to lift".to_string()))?;
        self.filter_rest(input_op, columns, &predicate, input_writes)
    }

    /// Filters `input` by `predicate`, whose `EXISTS` and `COUNT` subqueries
    /// the edge check cannot answer run per row first (see `subquery.rs`).
    fn filter_rest(
        &self,
        input: Box<dyn Operator>,
        columns: Vec<String>,
        predicate: &LogicalExpression,
        input_writes: bool,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (predicate, input, columns) =
            self.lift_subqueries(predicate, input, columns, input_writes)?;
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let predicate = ExpressionPredicate::new(
            self.convert_expression(&predicate)?,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            self.read_view(),
        )
        .with_session_context(self.session_context.clone());
        Ok((
            Box::new(FilterOperator::new(input, Box::new(predicate))),
            columns,
        ))
    }

    /// Extracts an EXISTS or NOT EXISTS subquery from a filter predicate for
    /// semi-join rewriting.
    ///
    /// Returns `(subquery, is_negated, remaining_predicate)`.
    ///
    /// At the top level (standalone EXISTS / NOT EXISTS), only complex patterns
    /// that cannot use the inline fast path in `extract_exists_pattern()` are
    /// extracted. Within AND trees, ALL EXISTS patterns are extracted regardless
    /// of complexity: when multiple EXISTS subqueries share a WHERE clause, every
    /// one must go through the semi-join path so the recursive handler in
    /// `plan_exists_as_semi_join` can chain them.
    ///
    /// The Cypher translator builds left-leaning AND trees from sequential WHERE
    /// predicates, so EXISTS nodes sit at the right child of AND nodes at various
    /// depths. The recursive semi-join handler (`plan_exists_as_semi_join`) calls
    /// this function again on each remaining predicate, peeling off one EXISTS
    /// per level until only scalar predicates remain.
    ///
    /// `outer` holds the variables of the rows the predicate filters (`None`:
    /// not known), see [`Self::exists_fast_path_fits`].
    fn extract_complex_exists<'a>(
        &self,
        predicate: &'a LogicalExpression,
        outer: Option<&HashSet<String>>,
    ) -> Option<(&'a LogicalOperator, bool, Option<LogicalExpression>)> {
        match predicate {
            LogicalExpression::ExistsSubquery(subplan) => {
                // Top-level EXISTS: only use semi-join for complex patterns.
                // Simple single-hop patterns use the fast path in convert_expression().
                if self.exists_fast_path_fits(subplan, outer) || reads_outer_values(subplan) {
                    None
                } else {
                    Some((subplan.as_ref(), false, None))
                }
            }
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subplan) = operand.as_ref() {
                    if self.exists_fast_path_fits(subplan, outer) || reads_outer_values(subplan) {
                        None
                    } else {
                        Some((subplan.as_ref(), true, None))
                    }
                } else {
                    None
                }
            }
            LogicalExpression::Binary {
                op: BinaryOp::And,
                left,
                right,
            } => {
                // In AND context, extract ANY EXISTS (simple or complex).
                // When multiple EXISTS appear in the same WHERE, extracting the
                // first one (even if simple) lets the recursive semi-join handler
                // find and extract the remaining complex ones from the rest.
                if let Some((subplan, negated)) = Self::extract_exists_from_expr(left)
                    && !reads_outer_values(subplan)
                {
                    return Some((subplan, negated, Some(right.as_ref().clone())));
                }
                if let Some((subplan, negated)) = Self::extract_exists_from_expr(right)
                    && !reads_outer_values(subplan)
                {
                    return Some((subplan, negated, Some(left.as_ref().clone())));
                }
                // Recurse into left subtree (handles left-leaning AND trees where
                // EXISTS nodes are buried deeper than immediate children)
                if let Some((subplan, negated, inner_remaining)) =
                    self.extract_complex_exists(left, outer)
                {
                    let remaining = match inner_remaining {
                        Some(inner) => LogicalExpression::Binary {
                            left: Box::new(inner),
                            op: BinaryOp::And,
                            right: right.clone(),
                        },
                        None => right.as_ref().clone(),
                    };
                    return Some((subplan, negated, Some(remaining)));
                }
                // Recurse into right subtree
                if let Some((subplan, negated, inner_remaining)) =
                    self.extract_complex_exists(right, outer)
                {
                    let remaining = match inner_remaining {
                        Some(inner) => LogicalExpression::Binary {
                            left: left.clone(),
                            op: BinaryOp::And,
                            right: Box::new(inner),
                        },
                        None => left.as_ref().clone(),
                    };
                    return Some((subplan, negated, Some(remaining)));
                }
                None
            }
            _ => None,
        }
    }

    /// Whether an `EXISTS` subplan takes the fast path for rows that bind
    /// `outer` (`None`: not known): its pattern must start from a node of the
    /// row, its edge must be new to it, and so must the other end of a single
    /// edge. Anything else is a semi-join, which matches every variable the
    /// subquery shares with the row by name.
    ///
    /// A path may end at a node of the row: the fast path then searches from
    /// the start until it reaches that node, each node at most once, where
    /// the semi-join would enumerate every walk from every node, which on a
    /// graph with cycles are more than any memory holds.
    fn exists_fast_path_fits(
        &self,
        subplan: &LogicalOperator,
        outer: Option<&HashSet<String>>,
    ) -> bool {
        let Ok(check) = self.extract_exists_pattern(subplan) else {
            return false;
        };
        outer.is_some_and(|outer| {
            outer.contains(&check.start_var)
                && (!check.one_edge || !outer.contains(&check.end_var))
                && check
                    .edge_var
                    .as_ref()
                    .is_none_or(|edge| !outer.contains(edge))
        })
    }

    /// Helper: extracts EXISTS or NOT EXISTS from a single expression node.
    /// Returns `(subplan, is_negated)`.
    fn extract_exists_from_expr(expr: &LogicalExpression) -> Option<(&LogicalOperator, bool)> {
        match expr {
            LogicalExpression::ExistsSubquery(subplan) => Some((subplan.as_ref(), false)),
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                if let LogicalExpression::ExistsSubquery(subplan) = operand.as_ref() {
                    Some((subplan.as_ref(), true))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Plans a complex EXISTS/NOT EXISTS as a hash-based semi-join or anti-join.
    ///
    /// The inner subquery is planned as a full operator tree via `plan_operator()`.
    /// Variables shared between the outer input and inner subquery become equi-join
    /// keys. Uses `HashJoinOperator` for efficient O(N + M) evaluation.
    ///
    /// When the inner subquery contains a `ParameterScan` (indicating correlated
    /// variable references from the outer scope), a correlated `ApplyOperator`
    /// with EXISTS mode is used instead of a hash-based semi-join.
    fn plan_exists_as_semi_join(
        &self,
        outer_input: &LogicalOperator,
        subquery: &LogicalOperator,
        is_negated: bool,
        remaining_predicate: Option<LogicalExpression>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let input_writes = outer_input.has_mutations();
        let (left_op, left_columns) = self.plan_operator(outer_input)?;
        let (right_op, right_columns) = self.without_profiling(|| self.plan_operator(subquery))?;

        // Semi/anti join only outputs left (outer) columns
        let output_columns = left_columns.clone();

        // Find shared variables for equi-join keys
        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&output_columns);

        let join_type = if is_negated {
            PhysicalJoinType::Anti
        } else {
            PhysicalJoinType::Semi
        };

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            join_type,
            output_schema,
        ));

        // If there's a remaining predicate (from AND splitting), check if it
        // contains more EXISTS subqueries that need semi-join rewriting.
        if let Some(ref remaining) = remaining_predicate {
            // Recursively handle nested EXISTS in the remaining predicate
            let outer: HashSet<String> = output_columns.iter().cloned().collect();
            if let Some((nested_sub, nested_neg, nested_rest)) =
                self.extract_complex_exists(remaining, Some(&outer))
            {
                return self.plan_exists_as_semi_join_with_input(
                    join_op,
                    &output_columns,
                    nested_sub,
                    nested_neg,
                    nested_rest,
                    input_writes,
                );
            }

            return self.filter_rest(join_op, output_columns, remaining, input_writes);
        }

        Ok((join_op, output_columns))
    }

    /// Plans an EXISTS/NOT EXISTS semi-join with an already-planned outer input.
    ///
    /// Used when the remaining predicate from a prior semi-join contains additional
    /// EXISTS subqueries that also need semi-join rewriting (nested NOT EXISTS).
    fn plan_exists_as_semi_join_with_input(
        &self,
        left_op: Box<dyn Operator>,
        left_columns: &[String],
        subquery: &LogicalOperator,
        is_negated: bool,
        remaining_predicate: Option<LogicalExpression>,
        input_writes: bool,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let (right_op, right_columns) = self.without_profiling(|| self.plan_operator(subquery))?;

        let output_columns = left_columns.to_vec();

        let mut probe_keys = Vec::new();
        let mut build_keys = Vec::new();
        for (right_idx, right_col) in right_columns.iter().enumerate() {
            if let Some(left_idx) = left_columns.iter().position(|c| c == right_col) {
                probe_keys.push(left_idx);
                build_keys.push(right_idx);
            }
        }

        let output_schema = self.derive_schema_from_columns(&output_columns);
        let join_type = if is_negated {
            PhysicalJoinType::Anti
        } else {
            PhysicalJoinType::Semi
        };

        let join_op: Box<dyn Operator> = Box::new(HashJoinOperator::new(
            left_op,
            right_op,
            probe_keys,
            build_keys,
            join_type,
            output_schema,
        ));

        // Recursively handle any further EXISTS in the remaining predicate
        if let Some(ref remaining) = remaining_predicate {
            let outer: HashSet<String> = output_columns.iter().cloned().collect();
            if let Some((nested_sub, nested_neg, nested_rest)) =
                self.extract_complex_exists(remaining, Some(&outer))
            {
                return self.plan_exists_as_semi_join_with_input(
                    join_op,
                    &output_columns,
                    nested_sub,
                    nested_neg,
                    nested_rest,
                    input_writes,
                );
            }

            return self.filter_rest(join_op, output_columns, remaining, input_writes);
        }

        Ok((join_op, output_columns))
    }

    /// Checks zone maps for a predicate to see if we can skip the scan entirely.
    ///
    /// A property comparison is checked against the node zone map only when its
    /// variable is provably a node in `input`, and against the edge zone map
    /// only when it is provably a single edge (see [`binding_kind`]); anything
    /// else (maps, projected values, values written earlier in the query) is
    /// not checked.
    ///
    /// Returns:
    /// - `Some(false)` if zone map proves no matches possible (can skip)
    /// - `Some(true)` if zone map says matches might exist
    /// - `None` if zone map check not applicable
    pub(super) fn check_zone_map_for_predicate(
        &self,
        predicate: &LogicalExpression,
        input: &LogicalOperator,
    ) -> Option<bool> {
        use grafeo_core::graph::lpg::CompareOp;

        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                // Check for AND/OR first (compound conditions)
                match op {
                    BinaryOp::And => {
                        let left_result = self.check_zone_map_for_predicate(left, input);
                        let right_result = self.check_zone_map_for_predicate(right, input);

                        return match (left_result, right_result) {
                            // If either side definitely won't match, the AND won't match
                            (Some(false), _) | (_, Some(false)) => Some(false),
                            // If both might match, might match overall
                            (Some(true), Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    BinaryOp::Or => {
                        let left_result = self.check_zone_map_for_predicate(left, input);
                        let right_result = self.check_zone_map_for_predicate(right, input);

                        return match (left_result, right_result) {
                            // Both sides definitely won't match
                            (Some(false), Some(false)) => Some(false),
                            // At least one side might match
                            (Some(true), _) | (_, Some(true)) => Some(true),
                            // Otherwise, can't determine
                            _ => None,
                        };
                    }
                    _ => {}
                }

                // Simple property comparison: n.property op value
                let (variable, property, compare_op, value) = match (left.as_ref(), right.as_ref())
                {
                    (
                        LogicalExpression::Property { variable, property },
                        LogicalExpression::Literal(val),
                    ) => {
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Lt,
                            BinaryOp::Le => CompareOp::Le,
                            BinaryOp::Gt => CompareOp::Gt,
                            BinaryOp::Ge => CompareOp::Ge,
                            _ => return None,
                        };
                        (variable, property.clone(), cmp, val.clone())
                    }
                    (
                        LogicalExpression::Literal(val),
                        LogicalExpression::Property { variable, property },
                    ) => {
                        // Flip comparison for reversed operands
                        let cmp = match op {
                            BinaryOp::Eq => CompareOp::Eq,
                            BinaryOp::Ne => CompareOp::Ne,
                            BinaryOp::Lt => CompareOp::Gt, // val < prop means prop > val
                            BinaryOp::Le => CompareOp::Ge,
                            BinaryOp::Gt => CompareOp::Lt,
                            BinaryOp::Ge => CompareOp::Le,
                            _ => return None,
                        };
                        (variable, property.clone(), cmp, val.clone())
                    }
                    _ => return None,
                };

                // Node statistics only describe nodes: consult the zone map that
                // matches what the variable is bound to, or nothing at all.
                let key: grafeo_common::types::PropertyKey = property.into();
                let nodes = match binding_kind(input, variable)? {
                    BindingKind::Node => true,
                    BindingKind::Edge => false,
                };
                // A zone map orders values as the store does, which can
                // differ from the query's comparison of datetimes and times.
                if !self.store_column_orders_like_query(nodes, key.as_str(), &value) {
                    return None;
                }
                if nodes {
                    Some(
                        self.store
                            .node_property_might_match(&key, compare_op, &value),
                    )
                } else {
                    Some(
                        self.store
                            .edge_property_might_match(&key, compare_op, &value),
                    )
                }
            }

            _ => None,
        }
    }

    /// [`FilterRoute::PropertyIndex`] for an equality check on a property of the nodes of a
    /// simple NodeScan: an index can look up the matching nodes directly instead of scanning
    /// all nodes, or a read of the key's column finds them as it runs. It holds when some
    /// equality has probes (see [`equality_probes`]) and an index to look it up in, or a key's
    /// column to read (see [`Self::plan_first_column_read`]).
    ///
    /// An index's lookup not after a write of the statement (see
    /// [`Self::reads_the_store_as_planned`]): it finds the nodes while
    /// planning. A read of the key's column reads as it runs.
    pub(super) fn property_index_route<'a, R: FilterRouteReads>(
        reads: &R,
        filter: &'a FilterOp,
    ) -> Guarded<'a, R> {
        // A scan without input, directly below or below filters that check
        // the scanned node alone (`MATCH (m {id: $id}) WHERE m:Post OR
        // m:Comment`): the lookup takes their place, and the nodes it finds
        // are checked against them too.
        let checked = super::seek::checked_scan(filter)?;
        let scan = checked.scan;
        if scan.input.is_some() {
            return None;
        }
        let planner = planner_of!(reads);
        if !planner.reads_by_key() {
            return None;
        }

        // Extract property equality conditions from the predicate and from
        // the checks below it: simple (n.prop = val) and compound (n.a = 1
        // AND n.b = 2). A check that holds the key, below a check of another
        // label (`(n:Graph:Method {id: $id})` scanning Method), is looked up
        // here, so that the label check runs on the nodes found, not in a
        // filter of its own above the lookup.
        let mut conditions = Self::extract_equality_conditions(&filter.predicate, &scan.variable);
        for (_, check) in &checked.checks {
            Self::collect_equality_conditions(check, &scan.variable, &mut conditions);
        }
        let labels = LookupLabels::of(filter, &scan.variable);

        let conditions: Vec<(String, Value, LookupRead<'a>)> = conditions
            .into_iter()
            .map(|(property, value)| {
                let read = labels.lookup_read(planner.store.as_ref(), &property, planner.rules);
                (property, value, read)
            })
            .collect();

        // The build looks the indexed conditions up first, while planning, then reads the first
        // key's column it can: it declines only when no index answers and no column can be read.
        let readable = conditions.iter().any(|(_, value, read)| {
            let read_by_key = match read {
                LookupRead::Index(_) => planner.reads_the_store_as_planned(),
                LookupRead::IndexesAndRemainder { .. } | LookupRead::Column { .. } => true,
                LookupRead::Scan => false,
            };
            read_by_key
                && equality_probes(value, planner.session_context.temporal_comparison).is_some()
        });
        readable.then_some(Ok(FilterRoute::PropertyIndex {
            checked,
            conditions,
        }))
    }

    /// Plans [`FilterRoute::PropertyIndex`]: the nodes the indexes find for the `conditions`
    /// on the nodes of the scan of `checked`, else a read of a key's column.
    ///
    /// Returns `Ok(Some((operator, columns)))` if optimization was applied,
    /// `Ok(None)` when no index answers and no column can be read, or `Err` on error.
    pub(super) fn plan_filter_with_property_index(
        &self,
        filter: &FilterOp,
        checked: &CheckedScan<'_>,
        conditions: &[(String, Value, LookupRead<'_>)],
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let (scan_variable, scan_label) =
            (checked.scan.variable.clone(), checked.scan.label.clone());

        // The nodes an index finds may be equal to a condition's value under
        // `=` (#535), the smallest set of any indexed condition, looked up at
        // the query's view. Without one the scan and the filter run as
        // planned.
        let mut candidates: Option<Vec<NodeId>> = None;
        {
            for (property, value, read) in conditions {
                // An index under the rules: `IndexLookup` on and an index that serves.
                let LookupRead::Index(read) = read else {
                    continue;
                };
                if !self.reads_the_store_as_planned() {
                    continue;
                }
                let Some(found) = self.find_maybe_equal(read, property, value)? else {
                    continue;
                };
                // Every lookup reads, also one whose set is not the smallest.
                self.read_counter().add(found.len());
                if candidates
                    .as_ref()
                    .is_none_or(|best| found.len() < best.len())
                {
                    candidates = Some(found);
                }
            }
        }
        let Some(matching_nodes) = candidates else {
            return self.plan_first_column_read(filter, checked, conditions);
        };

        // MVCC visibility: filter out nodes not visible at the current epoch/tx
        // (rolled-back or uncommitted nodes could leak through otherwise), and
        // nodes without the scan's label.
        let mut seen = Vec::with_capacity(matching_nodes.len());
        for id in matching_nodes {
            if self.visible_with_label(id, scan_label.as_deref())? {
                seen.push(id);
            }
        }
        let matching_nodes = seen;

        let columns = vec![scan_variable];

        // Absorbed-scan PROFILE entries (see `record_absorbed_scan_entry`):
        // the scan, then the checks, as the plan's tree lists them.
        self.record_absorbed_scan_entry("NodeScan", checked.operator);
        for (check, _) in &checked.checks {
            self.record_absorbed_scan_entry("Filter", check);
        }

        // The index finds candidates: the checks and the whole predicate
        // decide each, in the order the chain of filters runs them, with the
        // same `=` as a scan.
        let predicate = whole_predicate(filter, checked);
        let found = self.found_nodes(matching_nodes, &columns, &predicate)?;
        Ok(Some((found, columns)))
    }

    /// [`FilterRoute::LabelUnion`] for a filter whose predicate requires one of several labels
    /// of the node a scan of every node reads (see [`label_union`](super::scan::label_union)).
    fn label_union_route<'a, R: FilterRouteReads>(
        reads: &R,
        filter: &'a FilterOp,
    ) -> Guarded<'a, R> {
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return None;
        };
        let planner = planner_of!(reads);
        let labels = super::scan::label_union(&filter.predicate, scan, planner.store.as_ref())?;
        Some(Ok(FilterRoute::LabelUnion { scan, labels }))
    }

    /// Plans [`FilterRoute::LabelUnion`] as a scan of the nodes of `labels`, each once and in
    /// ID order, below the whole predicate, as the query's view sees the labels.
    fn plan_filter_with_label_union(
        &self,
        filter: &FilterOp,
        scan: &NodeScanOp,
        labels: Vec<String>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let labels_read = labels.join("|");
        let scan_op: Box<dyn Operator> = Box::new(
            super::ScanOperator::with_any_label(
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                labels,
                self.read_view(),
            )
            .with_read_counter(self.read_counter()),
        );
        // PROFILE names the labels the scan reads.
        let read = LogicalOperator::NodeScan(NodeScanOp {
            variable: scan.variable.clone(),
            label: Some(labels_read),
            input: None,
        });
        let (scan_op, columns) =
            self.maybe_profile(Ok((scan_op, vec![scan.variable.clone()])), &read)?;
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let predicate = ExpressionPredicate::new(
            self.convert_expression(&filter.predicate)?,
            variable_columns,
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            self.read_view(),
        )
        .with_session_context(self.session_context.clone());
        Ok((
            Box::new(FilterOperator::new(scan_op, Box::new(predicate))),
            columns,
        ))
    }

    /// The nodes an index lookup found that meet `predicate`, as the single
    /// column `columns`: the lookup finds every node `=` may find equal, the
    /// predicate decides.
    fn found_nodes(
        &self,
        nodes: Vec<NodeId>,
        columns: &[String],
        predicate: &LogicalExpression,
    ) -> Result<Box<dyn Operator>> {
        let predicate = self.predicate_over(columns, predicate)?;
        Ok(Box::new(
            NodeListOperator::new(nodes, 2048).with_predicate(predicate),
        ))
    }

    /// `predicate` over rows of `columns`.
    fn predicate_over(
        &self,
        columns: &[String],
        predicate: &LogicalExpression,
    ) -> Result<Box<dyn grafeo_core::execution::operators::Predicate>> {
        let variable_columns: HashMap<String, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, name)| (name.clone(), i))
            .collect();
        let filter_expr = self.convert_expression(predicate)?;
        Ok(Box::new(
            ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                self.read_view(),
            )
            .with_session_context(self.session_context.clone()),
        ))
    }

    /// [`FilterRoute::InListIndex`] for a filter with a `var.prop IN [literals]` conjunct (the
    /// whole predicate, or a conjunct of its `AND` chain, #456) that a lookup of the property
    /// can read: the nodes it finds for the values are the candidates, and the whole predicate
    /// decides each. `None` when not applicable: the input isn't a `NodeScan` below such
    /// filters (see [`checked_scan`](super::seek::checked_scan)), no conjunct is such an `IN`
    /// list, an index's lookup would run after a write of the statement, or a value has no
    /// probes for a read of the column (see [`equality_probes`]).
    pub(super) fn in_list_route<'a, R: FilterRouteReads>(
        reads: &R,
        filter: &'a FilterOp,
    ) -> Guarded<'a, R> {
        // A scan without input, directly below or below filters that check the scanned node
        // alone, which the lookup takes over: the list may be in any of them.
        let checked = super::seek::checked_scan(filter)?;
        let scan = checked.scan;
        if scan.input.is_some() {
            return None;
        }
        let planner = planner_of!(reads);
        if !planner.reads_by_key() {
            return None;
        }

        let labels = LookupLabels::of(filter, &scan.variable);
        let lookup_read = |p: &str| labels.lookup_read(planner.store.as_ref(), p, planner.rules);
        let (property, values) = std::iter::once(&filter.predicate)
            .chain(checked.checks.iter().map(|(_, check)| *check))
            .find_map(|predicate| {
                indexed_in_list(predicate, &scan.variable, |p| {
                    lookup_read(p) != LookupRead::Scan
                })
            })?;
        let read = lookup_read(property);
        // `prop IN []` is always false. Any other list is looked up in an index as the store is
        // planned (an index finds the nodes while planning), or read in the key's column, which
        // reads the values' probes.
        let readable = values.is_empty()
            || match read {
                LookupRead::Index(_) => planner.reads_the_store_as_planned(),
                _ => values.iter().all(|value| {
                    equality_probes(value, planner.session_context.temporal_comparison).is_some()
                }),
            };
        readable.then_some(Ok(FilterRoute::InListIndex {
            checked,
            labels,
            property,
            values,
            read,
        }))
    }

    /// Plans [`FilterRoute::InListIndex`]: the nodes a lookup of `property` that reads `read`
    /// among the nodes of the scan of `checked`, by the `labels` of its chain, finds for
    /// `values`. `Ok(None)` when the list is as long as the rows it is read among, or the index
    /// cannot answer a value.
    pub(super) fn plan_filter_with_in_list_index(
        &self,
        filter: &FilterOp,
        checked: &CheckedScan<'_>,
        labels: &LookupLabels<'_>,
        property: &str,
        values: &[Value],
        read: &LookupRead<'_>,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let (scan_variable, scan_label) =
            (checked.scan.variable.clone(), checked.scan.label.clone());
        if values.is_empty() {
            // `prop IN []` is always false: return an empty result without
            // touching the store.
            self.record_absorbed_scan_entry("NodeScan", checked.operator);
            for (check, _) in &checked.checks {
                self.record_absorbed_scan_entry("Filter", check);
            }
            let columns = vec![scan_variable];
            let empty = Box::new(NodeListOperator::new(Vec::new(), 2048));
            return Ok(Some((empty, columns)));
        }

        // A list as long as the rows it is read among costs more to prepare than the scan's filter
        // costs to run.
        if matches!(read, LookupRead::Column { .. })
            && values.len() >= labels.nodes_at_most(self.store.as_ref())?
        {
            return Ok(None);
        }

        let LookupRead::Index(indexed) = read else {
            // A read of the column, as it runs, in one pass for every value: `IN` decides each
            // node.
            let columns = vec![scan_variable];
            let predicate = whole_predicate(filter, checked);
            let checks = KeyColumnChecks::Always(self.predicate_over(&columns, &predicate)?);
            let scan = self.key_column_scan(read, property, values, scan_label.as_deref(), checks);
            let Some(scan) = scan else {
                return Ok(None);
            };
            self.record_absorbed_scan_entry("NodeScan", checked.operator);
            for (check, _) in &checked.checks {
                self.record_absorbed_scan_entry("Filter", check);
            }
            return Ok(Some((scan, columns)));
        };

        // Per-value index lookup of the nodes `=` may find equal (#535),
        // deduplicate (the same node could match duplicate values, and we
        // don't want to emit it twice).
        let mut seen: GrafeoSet<_> = GrafeoSet::default();
        let mut matching_nodes = Vec::new();
        for value in values {
            let Some(found) = self.find_maybe_equal(indexed, property, value)? else {
                return Ok(None);
            };
            self.read_counter().add(found.len());
            for node_id in found {
                if seen.insert(node_id) {
                    matching_nodes.push(node_id);
                }
            }
        }

        // MVCC visibility and the label constraint, as in the equality fast path.
        let mut seen = Vec::with_capacity(matching_nodes.len());
        for id in matching_nodes {
            if self.visible_with_label(id, scan_label.as_deref())? {
                seen.push(id);
            }
        }
        let matching_nodes = seen;

        // Absorbed-scan PROFILE entries: see `record_absorbed_scan_entry`.
        self.record_absorbed_scan_entry("NodeScan", checked.operator);
        for (check, _) in &checked.checks {
            self.record_absorbed_scan_entry("Filter", check);
        }

        let columns = vec![scan_variable];
        // The lookup finds candidates: the checks and `IN` decide each, in the order the chain of
        // filters runs them, with the same `=` as a scan.
        let predicate = whole_predicate(filter, checked);
        let found = self.found_nodes(matching_nodes, &columns, &predicate)?;
        Ok(Some((found, columns)))
    }

    /// Extracts equality conditions (property = literal) from a predicate.
    ///
    /// Handles both simple predicates and AND chains:
    /// - `n.name = "Alix"` → `[("name", "Alix")]`
    /// - `n.name = "Alix" AND n.age = 30` → `[("name", "Alix"), ("age", 30)]`
    pub(super) fn extract_equality_conditions(
        predicate: &LogicalExpression,
        target_variable: &str,
    ) -> Vec<(String, Value)> {
        let mut conditions = Vec::new();
        Self::collect_equality_conditions(predicate, target_variable, &mut conditions);
        conditions
    }

    /// Recursively collects equality conditions from AND expressions.
    pub(super) fn collect_equality_conditions(
        expr: &LogicalExpression,
        target_variable: &str,
        conditions: &mut Vec<(String, Value)>,
    ) {
        match expr {
            // Handle AND: recurse into both sides
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => {
                Self::collect_equality_conditions(left, target_variable, conditions);
                Self::collect_equality_conditions(right, target_variable, conditions);
            }

            // Handle equality: extract property and value
            LogicalExpression::Binary {
                left,
                op: BinaryOp::Eq,
                right,
            } => {
                if let Some((var, prop, val)) = Self::extract_property_equality(left, right)
                    && var == target_variable
                {
                    conditions.push((prop, val));
                }
            }

            _ => {}
        }
    }

    /// Extracts (variable, property, value) from a property equality expression.
    ///
    /// Returns `None` for NULL literals: `property = NULL` is UNKNOWN in
    /// three-valued logic and must never be optimized into an index lookup.
    pub(super) fn extract_property_equality(
        left: &LogicalExpression,
        right: &LogicalExpression,
    ) -> Option<(String, String, Value)> {
        match (left, right) {
            (
                LogicalExpression::Property { variable, property },
                LogicalExpression::Literal(val),
            ) if !val.is_null() => Some((variable.clone(), property.clone(), val.clone())),
            (
                LogicalExpression::Literal(val),
                LogicalExpression::Property { variable, property },
            ) if !val.is_null() => Some((variable.clone(), property.clone(), val.clone())),
            _ => None,
        }
    }

    /// Pushes a synthetic profile entry for a scan absorbed into a fast-path
    /// physical operator (e.g. `NodeListOperator`).
    ///
    /// `build_profile_tree` walks the logical tree in post-order and panics
    /// if it runs out of entries. When a fast path collapses a `Filter` over
    /// a `NodeScan` into one physical op, the scan's entry is missing
    /// without this call. The entry is marked absorbed: the profile tree
    /// leaves it out, gives its children to the absorbing operator and names
    /// it in that operator's detail (#461), whose stats hold the work.
    ///
    /// Pass the source operator's `name()` (e.g. `"NodeScan"`, `"EdgeScan"`)
    /// so future absorbers don't get mislabeled. A seek also absorbs the
    /// filters between it and the scan (see `seek.rs`): each gets an entry
    /// named `"Filter"`.
    pub(super) fn record_absorbed_scan_entry(&self, op_name: &str, absorbed: &LogicalOperator) {
        if !self.profiling.get() {
            return;
        }
        let entry =
            crate::query::profile::ProfileEntry::absorbed(op_name, absorbed.display_label());
        self.profile_entries.borrow_mut().push(entry);
    }

    /// Pushes the profile entry of a physical operator a fast path builds in
    /// place of `replaced` without profiling it on its own (a seek, a hash
    /// join): unlike an absorbed scan it keeps a line in the profile tree,
    /// with whatever stats it has (none), so PROFILE shows that it runs.
    pub(super) fn record_kept_entry(&self, op_name: &str, replaced: &LogicalOperator) {
        if !self.profiling.get() {
            return;
        }
        let (entry, _stats) =
            crate::query::profile::ProfileEntry::new(op_name, replaced.display_label());
        self.profile_entries.borrow_mut().push(entry);
    }

    /// [`FilterRoute::RangeIndex`] for a filter of range queries on properties.
    ///
    /// Applies when:
    /// - The input is a simple `NodeScan` (no nested operations).
    /// - The predicate contains range comparisons (`>`, `<`, `>=`, `<=`).
    /// - The same variable and property are being filtered.
    ///
    /// Handles both simple range predicates (`n.age > 30`) and `BETWEEN`
    /// patterns (`n.age >= 30 AND n.age <= 50`).
    fn range_route<'a, R: FilterRouteReads>(reads: &R, filter: &'a FilterOp) -> Guarded<'a, R> {
        if !reads
            .rules()
            .has(crate::query::optimizer::PlannerRule::IndexRange)
        {
            return None;
        }
        // Only optimize if input is a simple NodeScan (not nested). The range
        // lookup reads the values at the query's view.
        let scan = match filter.input.as_ref() {
            LogicalOperator::NodeScan(scan) if scan.input.is_none() => scan,
            _ => return None,
        };

        let (property, range) =
            planner_of!(reads).filter_range(&filter.predicate, &scan.variable)?;
        Some(Ok(FilterRoute::RangeIndex {
            scan,
            property,
            range,
        }))
    }

    /// Plans [`FilterRoute::RangeIndex`]: a range scan of `property` over `range` in place of
    /// `scan`. `limit` is the count of a LIMIT directly above `filter`, which the scan may stop
    /// at.
    fn plan_filter_with_range_index(
        &self,
        filter: &FilterOp,
        scan: &NodeScanOp,
        property: &str,
        range: grafeo_core::graph::ValueRange,
        limit: Option<usize>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        self.record_absorbed_scan_entry("NodeScan", &filter.input);
        self.plan_range_filter(&scan.variable, &scan.label, property, range, limit)
    }

    /// The property a filter over a scan of `scan_variable` bounds, and the
    /// range it bounds it to, when the range scan can read it: a `BETWEEN`
    /// pattern (`n.age >= 30 AND n.age <= 50`) first, then one comparison
    /// (`n.age > 30`), on a column that orders as the query compares. `None`
    /// otherwise. The range scan and the range fold read a filter so.
    pub(super) fn filter_range(
        &self,
        predicate: &LogicalExpression,
        scan_variable: &str,
    ) -> Option<(String, grafeo_core::graph::ValueRange)> {
        let bounds = Self::range_bounds(predicate, scan_variable)?;
        for value in [&bounds.min, &bounds.max].into_iter().flatten() {
            if !self.store_column_orders_like_query(true, &bounds.property, value) {
                return None;
            }
        }
        let range = self.value_range(
            bounds.min.as_ref(),
            bounds.max.as_ref(),
            bounds.min_inclusive,
            bounds.max_inclusive,
        );
        Some((bounds.property, range))
    }

    /// The bounds `predicate` puts on a property of `scan_variable`: a `BETWEEN` pattern
    /// (`n.age >= 30 AND n.age <= 50`) or one comparison with a literal (`n.age > 30`).
    pub(super) fn range_bounds(
        predicate: &LogicalExpression,
        scan_variable: &str,
    ) -> Option<RangeBounds> {
        if let Some((variable, property, min, max, min_inclusive, max_inclusive)) =
            Self::extract_between_predicate(predicate)
            && variable == scan_variable
        {
            return Some(RangeBounds {
                property,
                min: Some(min),
                max: Some(max),
                min_inclusive,
                max_inclusive,
            });
        }
        let (variable, property, op, value) = Self::extract_range_predicate(predicate)?;
        if variable != scan_variable {
            return None;
        }
        let (min, max, min_inclusive, max_inclusive) = match op {
            BinaryOp::Lt => (None, Some(value), false, false),
            BinaryOp::Le => (None, Some(value), false, true),
            BinaryOp::Gt => (Some(value), None, false, false),
            BinaryOp::Ge => (Some(value), None, true, false),
            _ => return None,
        };
        Some(RangeBounds {
            property,
            min,
            max,
            min_inclusive,
            max_inclusive,
        })
    }

    /// The range between `min` and `max` under the statement's rule for two
    /// kinds: ISO GQL orders them, openCypher does not.
    fn value_range(
        &self,
        min: Option<&Value>,
        max: Option<&Value>,
        min_inclusive: bool,
        max_inclusive: bool,
    ) -> grafeo_core::graph::ValueRange {
        let across_kinds = if self.session_context.language.follows_opencypher() {
            grafeo_core::execution::operators::value_utils::AcrossKinds::Apart
        } else {
            grafeo_core::execution::operators::value_utils::AcrossKinds::Ordered
        };
        grafeo_core::graph::ValueRange::new(min, max, min_inclusive, max_inclusive, across_kinds)
    }

    /// The nodes an index lookup finds that `=` may find equal to `value`
    /// under the query's temporal comparison (see [`equality_probes`]), each
    /// once and ascending, through the indexes on `labels` (see
    /// [`find_through_label_indexes`]); `None` without an index to read, or when no set of
    /// lookups covers them.
    fn find_maybe_equal(
        &self,
        labels: &[String],
        property: &str,
        value: &Value,
    ) -> Result<Option<Vec<NodeId>>> {
        let Some(probes) = equality_probes(value, self.session_context.temporal_comparison) else {
            return Ok(None);
        };
        let mut found = Vec::new();
        for probe in probes {
            let Some(ids) = find_through_label_indexes(
                self.store.as_ref(),
                labels,
                property,
                &probe,
                self.read_view(),
                self.equal_across_kinds(),
            )?
            else {
                return Ok(None);
            };
            found.extend(ids);
        }
        found.sort_unstable();
        found.dedup();
        Ok(Some(found))
    }

    /// What the statement's `=` gives two values of two kinds: an error in SQL/PGQ, false in GQL
    /// and Cypher. A read by key answers as the scan's filter would (see
    /// [`EqualAcrossKinds`](grafeo_core::execution::operators::value_utils::EqualAcrossKinds)).
    pub(super) fn equal_across_kinds(
        &self,
    ) -> grafeo_core::execution::operators::value_utils::EqualAcrossKinds {
        use grafeo_core::execution::operators::value_utils::EqualAcrossKinds;
        if self.session_context.language == grafeo_common::types::StatementLanguage::SqlPgq {
            EqualAcrossKinds::Raises
        } else {
            EqualAcrossKinds::False
        }
    }

    /// Whether a rule lets a filter's equality on a key read other than its scan.
    fn reads_by_key(&self) -> bool {
        use crate::query::optimizer::PlannerRule;
        self.rules.has(PlannerRule::IndexLookup) || self.rules.has(PlannerRule::KeyColumnLookup)
    }

    /// The read of a key's column for the first of `conditions` that has no index of its own but
    /// indexes for part of its nodes, else for the first with none, in written order; `None` when
    /// none can be read so. On a group whose answer is exact the `=` is not checked again, unless
    /// the statement writes: a row it wrote after the group was read is read again, as a scan's
    /// filter does.
    #[inline(never)]
    fn plan_first_column_read(
        &self,
        filter: &FilterOp,
        checked: &super::seek::CheckedScan<'_>,
        conditions: &[(String, Value, LookupRead<'_>)],
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let variable = &checked.scan.variable;
        let columns = vec![variable.clone()];
        let whole = whole_predicate(filter, checked);
        for partly_indexed in [true, false] {
            for (property, value, read) in conditions {
                if !matches!(
                    read,
                    LookupRead::IndexesAndRemainder { .. } | LookupRead::Column { .. }
                ) || matches!(read, LookupRead::IndexesAndRemainder { .. }) != partly_indexed
                {
                    continue;
                }
                let whole_check = self.predicate_over(&columns, &whole)?;
                let writes = self
                    .statement_writes
                    .get()
                    .is_none_or(|statement| statement.writes);
                let checks = if writes {
                    KeyColumnChecks::Always(whole_check)
                } else {
                    let rest = without_what_the_read_answered(&whole, variable, property, value)
                        .map(|rest| self.predicate_over(&columns, &rest))
                        .transpose()?;
                    KeyColumnChecks::UnlessExact {
                        whole: whole_check,
                        rest,
                    }
                };
                let scan = self.key_column_scan(
                    read,
                    property,
                    std::slice::from_ref(value),
                    checked.scan.label.as_deref(),
                    checks,
                );
                if let Some(scan) = scan {
                    self.record_absorbed_scan_entry("NodeScan", checked.operator);
                    for (check, _) in &checked.checks {
                        self.record_absorbed_scan_entry("Filter", check);
                    }
                    return Ok(Some((scan, columns)));
                }
            }
        }
        Ok(None)
    }

    /// The operator that reads what `read` says for `values`, as it runs; `None` when `read` says
    /// something else, or a value has no lookup (see [`equality_probes`]). `label` is the scan's,
    /// which an index read does not check.
    fn key_column_scan(
        &self,
        read: &LookupRead<'_>,
        property: &str,
        values: &[Value],
        label: Option<&str>,
        checks: KeyColumnChecks,
    ) -> Option<Box<dyn Operator>> {
        let (indexed, required, one_of): (&[String], _, _) = match read {
            LookupRead::IndexesAndRemainder {
                read,
                required,
                one_of,
            } => (read, required, one_of),
            LookupRead::Column { required, one_of } => (&[], required, one_of),
            LookupRead::Index(_) | LookupRead::Scan => return None,
        };
        let mut probes = Vec::with_capacity(values.len());
        for value in values {
            probes.extend(equality_probes(
                value,
                self.session_context.temporal_comparison,
            )?);
        }
        let except: Vec<&str> = indexed.iter().map(String::as_str).collect();
        let indexes = (!indexed.is_empty()).then(|| KeyColumnIndexes {
            labels: indexed.to_vec(),
            values: probes.clone(),
            label: label.map(ToString::to_string),
        });
        let query = KeyColumnQuery::new(required, one_of, &except, property, probes)
            .with_equal_across_kinds(self.equal_across_kinds());
        let mut scan = KeyColumnScanOperator::new(
            Arc::clone(&self.store) as Arc<dyn GraphStore>,
            query,
            self.read_view(),
            checks,
        )
        .with_read_counter(self.read_counter());
        if let Some(indexes) = indexes {
            scan = scan.with_indexes(indexes);
        }
        Some(Box::new(scan))
    }

    /// Whether the store's range lookups and zone maps order `value` against
    /// stored values as the query compares them. They read a local datetime
    /// at UTC against a zoned one, which is the query's reading only in a GQL
    /// session without a time zone, and order a time without an offset
    /// against one with an offset by more than its instant, so a time is
    /// never left to them. The same holds for a time or datetime inside a
    /// list, a map or a path, whose elements they compare the same way.
    fn store_orders_like_query(&self, value: &Value) -> bool {
        // SQL/PGQ raises for two kinds, which only its filter meets: its
        // ordered reads go through `store_column_orders_like_query`, which
        // proves the column holds no value of another kind.
        if self.session_context.language == grafeo_common::types::StatementLanguage::SqlPgq {
            return false;
        }
        self.store_orders_elements_like_query(value)
    }

    /// Whether the store's range lookups and zone maps of `property`'s column
    /// order the values as the query compares them against `value`: as
    /// [`store_orders_like_query`](Self::store_orders_like_query), and for
    /// SQL/PGQ, whose filter raises for two kinds, only when every value the
    /// column holds is of `value`'s kind or a null, so no row can raise.
    fn store_column_orders_like_query(&self, nodes: bool, property: &str, value: &Value) -> bool {
        use grafeo_core::execution::operators::value_utils::ValueKinds;

        if self.session_context.language == grafeo_common::types::StatementLanguage::SqlPgq {
            // A null operand gives null and never raises.
            if matches!(value, Value::Null) {
                return true;
            }
            if !self.store_orders_elements_like_query(value) {
                return false;
            }
            let kind = ValueKinds::of(value);
            return kind.is_one()
                && self
                    .store
                    .column_holds_only_kind(nodes, property, kind)
                    .unwrap_or(false);
        }
        self.store_orders_like_query(value)
    }

    /// [`store_orders_like_query`](Self::store_orders_like_query) for
    /// `value` and every value it holds.
    fn store_orders_elements_like_query(&self, value: &Value) -> bool {
        Self::elements_order_like_query(value, Some(self.session_context.temporal_comparison))
            .unwrap_or(false)
    }

    /// [`store_orders_elements_like_query`](Self::store_orders_elements_like_query) under the
    /// session's temporal rule `temporal`: `None` when the answer depends on that rule and none
    /// is given (a timestamp or zoned datetime, which a time does not settle first).
    pub(super) fn elements_order_like_query(
        value: &Value,
        temporal: Option<TemporalComparison>,
    ) -> Option<bool> {
        let all = |values: &mut dyn Iterator<Item = &Value>| {
            let mut known = true;
            for value in values {
                match Self::elements_order_like_query(value, temporal) {
                    Some(true) => {}
                    Some(false) => return Some(false),
                    None => known = false,
                }
            }
            known.then_some(true)
        };
        match value {
            Value::Time(_) => Some(false),
            Value::Timestamp(_) | Value::ZonedDatetime(_) => {
                temporal.map(|temporal| temporal == TemporalComparison::default())
            }
            Value::List(items) => all(&mut items.iter()),
            Value::Map(entries) => all(&mut entries.values()),
            Value::Path { nodes, edges } => all(&mut nodes.iter().chain(edges.iter())),
            _ => Some(true),
        }
    }

    /// Plans a range filter using [`RangeScanOperator`].
    ///
    /// Phase 4d emits a dedicated range-scan operator that consumes
    /// [`GraphStore::find_nodes_in_range_iter_at`] at the query's
    /// view: the store returns the nodes the view sees. The operator absorbs
    /// the label post-filter that the eager `NodeListOperator` path used to
    /// apply manually.
    ///
    /// Phase 4e: `limit`, the count of a LIMIT directly above the filter
    /// this scan stands for, is forwarded via `with_limit(n)` so the
    /// materialization step terminates after `n` matches. The outer
    /// `LimitOperator` still wraps the result; the hint is a pure
    /// optimization.
    pub(super) fn plan_range_filter(
        &self,
        scan_variable: &str,
        scan_label: &Option<String>,
        property: &str,
        range: grafeo_core::graph::ValueRange,
        limit: Option<usize>,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let mut op = RangeScanOperator::new(
            Arc::clone(&self.store),
            property.to_string(),
            range,
            2048,
            self.read_view(),
        )
        .with_read_counter(self.read_counter());

        if let Some(label) = scan_label {
            op = op.with_label_filter(label.clone());
        }

        if let Some(limit) = limit {
            op = op.with_limit(limit);
        }

        let columns = vec![scan_variable.to_string()];
        Ok((Box::new(op), columns))
    }

    /// Extracts a simple range predicate (>, <, >=, <=) from an expression.
    ///
    /// Returns `(variable, property, operator, value)` if found.
    pub(super) fn extract_range_predicate(
        predicate: &LogicalExpression,
    ) -> Option<(String, String, BinaryOp, Value)> {
        match predicate {
            LogicalExpression::Binary { left, op, right } => {
                match op {
                    BinaryOp::Lt | BinaryOp::Le | BinaryOp::Gt | BinaryOp::Ge => {
                        // Try property on left: n.age > 30
                        if let (
                            LogicalExpression::Property { variable, property },
                            LogicalExpression::Literal(val),
                        ) = (left.as_ref(), right.as_ref())
                        {
                            return Some((variable.clone(), property.clone(), *op, val.clone()));
                        }

                        // Try property on right: 30 < n.age (flip operator)
                        if let (
                            LogicalExpression::Literal(val),
                            LogicalExpression::Property { variable, property },
                        ) = (left.as_ref(), right.as_ref())
                        {
                            let flipped_op = match op {
                                BinaryOp::Lt => BinaryOp::Gt,
                                BinaryOp::Le => BinaryOp::Ge,
                                BinaryOp::Gt => BinaryOp::Lt,
                                BinaryOp::Ge => BinaryOp::Le,
                                _ => return None,
                            };
                            return Some((
                                variable.clone(),
                                property.clone(),
                                flipped_op,
                                val.clone(),
                            ));
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        None
    }

    /// Extracts a BETWEEN pattern from compound predicates.
    ///
    /// Recognizes patterns like:
    /// - `n.age >= 30 AND n.age <= 50`
    /// - `n.age > 30 AND n.age < 50`
    ///
    /// Returns `(variable, property, min_value, max_value, min_inclusive, max_inclusive)`.
    pub(super) fn extract_between_predicate(
        predicate: &LogicalExpression,
    ) -> Option<(String, String, Value, Value, bool, bool)> {
        // Must be an AND expression
        let (left, right) = match predicate {
            LogicalExpression::Binary {
                left,
                op: BinaryOp::And,
                right,
            } => (left.as_ref(), right.as_ref()),
            _ => return None,
        };

        // Extract range predicates from both sides
        let left_range = Self::extract_range_predicate(left);
        let right_range = Self::extract_range_predicate(right);

        let (left_var, left_prop, left_op, left_val) = left_range?;
        let (right_var, right_prop, right_op, right_val) = right_range?;

        // Must be same variable and property
        if left_var != right_var || left_prop != right_prop {
            return None;
        }

        // Determine which is lower bound and which is upper bound
        let (min_val, max_val, min_inc, max_inc) = match (left_op, right_op) {
            // n.x >= min AND n.x <= max
            (BinaryOp::Ge, BinaryOp::Le) => (left_val, right_val, true, true),
            // n.x >= min AND n.x < max
            (BinaryOp::Ge, BinaryOp::Lt) => (left_val, right_val, true, false),
            // n.x > min AND n.x <= max
            (BinaryOp::Gt, BinaryOp::Le) => (left_val, right_val, false, true),
            // n.x > min AND n.x < max
            (BinaryOp::Gt, BinaryOp::Lt) => (left_val, right_val, false, false),
            // Reversed order: n.x <= max AND n.x >= min
            (BinaryOp::Le, BinaryOp::Ge) => (right_val, left_val, true, true),
            // n.x < max AND n.x >= min
            (BinaryOp::Lt, BinaryOp::Ge) => (right_val, left_val, true, false),
            // n.x <= max AND n.x > min
            (BinaryOp::Le, BinaryOp::Gt) => (right_val, left_val, false, true),
            // n.x < max AND n.x > min
            (BinaryOp::Lt, BinaryOp::Gt) => (right_val, left_val, false, false),
            _ => return None,
        };

        Some((left_var, left_prop, min_val, max_val, min_inc, max_inc))
    }
}

/// The first conjunct of `predicate`'s `AND` chain shaped
/// `variable.prop IN list` with `has_index(prop)` and a list of constants: the
/// property and the values to look up. A conjunct under `OR` or `NOT` does
/// not narrow the rows, so it is none.
fn indexed_in_list<'a>(
    predicate: &'a LogicalExpression,
    variable: &str,
    has_index: impl Fn(&str) -> bool + Copy,
) -> Option<(&'a String, Vec<Value>)> {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => indexed_in_list(left, variable, has_index)
            .or_else(|| indexed_in_list(right, variable, has_index)),
        LogicalExpression::Binary {
            left,
            op: BinaryOp::In,
            right,
        } => match left.as_ref() {
            LogicalExpression::Property {
                variable: read,
                property,
            } if read == variable && has_index(property) => {
                in_list_values(right).map(|values| (property, values))
            }
            _ => None,
        },
        _ => None,
    }
}

/// The values the index path of `var.prop IN list` looks up: a list of
/// literals that are not NULL (`None` when an element is something else,
/// which the plan cannot enumerate), or a literal list without its NULLs.
fn in_list_values(list: &LogicalExpression) -> Option<Vec<Value>> {
    match list {
        LogicalExpression::List(items) => items
            .iter()
            .map(|item| match item {
                LogicalExpression::Literal(value) if !value.is_null() => Some(value.clone()),
                _ => None,
            })
            .collect(),
        LogicalExpression::Literal(Value::List(items)) => {
            Some(items.iter().filter(|v| !v.is_null()).cloned().collect())
        }
        _ => None,
    }
}

/// The hint `EXPLAIN` shows for the path [`plan_filter`][pf] takes for a
/// filter over `scan` after the seek, with the indexes of `store`. These
/// paths read a scan without input,
/// in the order the planner tries them: the index lookup of equalities with
/// constants, the index lookup of an `IN` list of constants, also a conjunct
/// of an `AND` chain (no hint for a list without a value that is not NULL: it
/// finds no node, and the planner reads neither index nor label), a read of
/// the key's column for equalities without an index, the label-first scan of
/// the label of equalities when that rule is off,
/// the range scan of a range of constants (with or without an index), a
/// vector or text index search (no hint), and otherwise the scan of the label
/// with the filter above it. A scan per input row takes none of them.
///
/// [pf]: super::Planner::plan_filter
pub(crate) fn scan_hint(
    predicate: &LogicalExpression,
    scan: &NodeScanOp,
    labels: &super::seek::LookupLabels<'_>,
    store: &dyn GraphStore,
    rules: crate::query::optimizer::PlannerRules,
) -> Option<PushdownHint> {
    let has_index = |property: &str| {
        rules.has(crate::query::optimizer::PlannerRule::IndexLookup)
            && labels.served_by(store, property)
    };
    if scan.input.is_some() {
        return None;
    }
    let variable = scan.variable.as_str();
    let conditions = super::Planner::extract_equality_conditions(predicate, variable);
    if let Some((property, _)) = conditions.iter().find(|(p, _)| has_index(p)) {
        return Some(PushdownHint::IndexLookup {
            property: property.clone(),
        });
    }
    if let Some((property, values)) = indexed_in_list(predicate, variable, has_index) {
        return (!values.is_empty()).then(|| PushdownHint::IndexLookup {
            property: property.clone(),
        });
    }
    let partly_indexed = |property: &str| {
        matches!(
            labels.lookup_read(store, property, rules),
            super::seek::LookupRead::IndexesAndRemainder { .. }
        )
    };
    let in_list = indexed_in_list(predicate, variable, partly_indexed);
    if let Some(property) = conditions
        .iter()
        .map(|(property, _)| property)
        .find(|property| partly_indexed(property))
        .or_else(|| in_list.as_ref().map(|(property, _)| *property))
    {
        return Some(PushdownHint::IndexesAndRemainder {
            property: property.clone(),
        });
    }
    let column = |property: &str| {
        matches!(
            labels.lookup_read(store, property, rules),
            super::seek::LookupRead::Column { .. }
        )
    };
    if let Some((property, _)) = conditions.iter().find(|(property, _)| column(property)) {
        return Some(PushdownHint::KeyColumn {
            property: property.clone(),
        });
    }
    if let Some((property, values)) = indexed_in_list(predicate, variable, column) {
        let keeps_the_filter = labels
            .nodes_at_most(store)
            .is_ok_and(|nodes| values.len() >= nodes);
        return (!values.is_empty() && !keeps_the_filter).then(|| PushdownHint::KeyColumn {
            property: property.clone(),
        });
    }
    if !conditions.is_empty() && scan.label.is_some() {
        return Some(PushdownHint::LabelFirst);
    }
    let between = super::Planner::extract_between_predicate(predicate)
        .map(|(read, property, ..)| (read, property));
    let range = || {
        super::Planner::extract_range_predicate(predicate)
            .map(|(read, property, ..)| (read, property))
    };
    if let Some((_, property)) = between
        .filter(|(read, _)| read == variable)
        .or_else(|| range().filter(|(read, _)| read == variable))
        .filter(|_| rules.has(crate::query::optimizer::PlannerRule::IndexRange))
    {
        return Some(PushdownHint::RangeScan { property });
    }
    // A vector or text index on the label that covers a property the filter
    // reads can take it (see `filter_hybrid.rs`), and the search runs instead
    // of the scan.
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    if let Some(label) = scan.label.as_deref()
        && super::scan::search_index_covers(predicate, variable, [label, label], store)
    {
        return None;
    }
    if let Some(labels) = super::scan::label_union(predicate, scan, store) {
        return Some(PushdownHint::LabelUnion { labels });
    }
    scan.label.as_ref().map(|_| PushdownHint::LabelFirst)
}

/// Extracted vector predicate from a filter expression.
///
/// `remaining` carries the predicate that runs in a `FilterOperator` above
/// the `VectorScanOperator`: the comparison itself, which decides each row
/// as the per-row path does. The scan's `min_similarity` and `max_distance`
/// only narrow the candidates, by the index's own arithmetic (`1 - distance`
/// is not the similarity bit for bit, so a row exactly at the threshold
/// could fall on either side), with a margin that keeps every row the
/// comparison keeps.
#[cfg(feature = "vector-index")]
pub(super) struct ExtractedVectorPredicate {
    pub(super) property: String,
    /// Variable bound in the property access (e.g. `n` in `n.embedding`).
    /// Validated against the enclosing NodeScan variable before pushdown.
    pub(super) variable: String,
    pub(super) query_vector: LogicalExpression,
    pub(super) metric: super::VectorMetric,
    pub(super) min_similarity: Option<f32>,
    pub(super) max_distance: Option<f32>,
    /// Predicate to apply above the scan: the comparison, and the
    /// conditions the vector predicate was found next to.
    pub(super) remaining: Option<LogicalExpression>,
}

#[cfg(feature = "vector-index")]
impl ExtractedVectorPredicate {
    /// Builds the predicate of a comparison: `comparison` runs above the
    /// scan and decides every row.
    fn new(
        property: String,
        variable: String,
        query_vector: LogicalExpression,
        metric: super::VectorMetric,
        min_similarity: Option<f32>,
        max_distance: Option<f32>,
        comparison: LogicalExpression,
    ) -> Self {
        Self {
            property,
            variable,
            query_vector,
            metric,
            min_similarity,
            max_distance,
            remaining: Some(comparison),
        }
    }
}

#[cfg(feature = "vector-index")]
impl super::Planner {
    /// [`FilterRoute::VectorIndex`] for a vector similarity/distance predicate that a VectorScan
    /// operator can take.
    ///
    /// Recognizes patterns like:
    /// - `cosine_similarity(n.embedding, $qv) > 0.8`
    /// - `euclidean_distance(n.embedding, $qv) < 0.5`
    pub(super) fn vector_index_route<'a, R: FilterRouteReads>(
        reads: &R,
        filter: &'a super::FilterOp,
    ) -> Guarded<'a, R> {
        // Only push down when input is a full label scan (no nested input):
        // the search finds the nodes alone, so a scan per input row (a later
        // MATCH, an UNWIND) checks the predicate per row instead.
        let LogicalOperator::NodeScan(scan) = filter.input.as_ref() else {
            return None;
        };
        if scan.input.is_some() {
            return None;
        }
        let label = scan.label.as_deref()?;
        let planner = planner_of!(reads);

        // Extract a vector predicate from the filter expression
        let extracted = planner.extract_vector_predicate(&filter.predicate)?;

        // Ensure the predicate references the same variable as the scan being planned
        if extracted.variable != scan.variable {
            return None;
        }

        // Check that a vector index exists (or fall through to brute-force eval)
        if !planner.store.has_vector_index(label, &extracted.property) {
            return None;
        }

        // The pushdown requires a query vector resolvable at plan time. If the
        // expression is something like a property access or non-literal, fall
        // through so per-row Filter evaluation handles it.
        if planner
            .resolve_vector_literal(&extracted.query_vector)
            .is_err()
        {
            return None;
        }
        Some(Ok(FilterRoute::VectorIndex {
            scan,
            label,
            extracted,
        }))
    }

    /// Plans [`FilterRoute::VectorIndex`]: a VectorScan of the nodes of `label` in place of
    /// `scan`, and a filter of the predicates `extracted` leaves above it.
    pub(super) fn plan_filter_with_vector_index(
        &self,
        scan: &NodeScanOp,
        label: &str,
        extracted: &ExtractedVectorPredicate,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        // Build VectorScanOp
        let vector_scan = super::VectorScanOp {
            variable: scan.variable.clone(),
            index_name: Some(format!("{}:{}", label, extracted.property)),
            property: extracted.property.clone(),
            label: Some(label.to_string()),
            query_vector: extracted.query_vector.clone(),
            k: None, // threshold mode: filter by similarity/distance, no top-k limit
            metric: Some(extracted.metric),
            min_similarity: extracted.min_similarity,
            max_distance: extracted.max_distance,
            input: None,
        };

        // Plan through the VectorScan path
        let (scan_op, scan_columns) =
            self.plan_operator(&LogicalOperator::VectorScan(vector_scan))?;

        // If there are remaining predicates (AND with non-vector conditions), wrap in filter
        if let Some(remaining) = &extracted.remaining {
            let variable_columns: HashMap<String, usize> = scan_columns
                .iter()
                .enumerate()
                .map(|(i, name)| (name.clone(), i))
                .collect();
            let filter_expr = self.convert_expression(remaining)?;
            let predicate = ExpressionPredicate::new(
                filter_expr,
                variable_columns,
                Arc::clone(&self.store) as Arc<dyn GraphStore>,
                self.read_view(),
            )
            .with_session_context(self.session_context.clone());
            let filter_op = Box::new(FilterOperator::new(scan_op, Box::new(predicate)));
            Ok((filter_op, scan_columns))
        } else {
            Ok((scan_op, scan_columns))
        }
    }

    /// Recursively extracts a vector predicate from a (potentially compound) expression.
    pub(super) fn extract_vector_predicate(
        &self,
        expr: &LogicalExpression,
    ) -> Option<ExtractedVectorPredicate> {
        match expr {
            LogicalExpression::Binary { left, op, right } => {
                // Try the left side as a vector function call
                if let LogicalExpression::FunctionCall {
                    callee, name, args, ..
                } = left.as_ref()
                    && let Some(extracted) =
                        self.try_extract_vector_fn(*callee, name, args, *op, right)
                {
                    return Some(extracted);
                }
                // AND: recurse into both sides, accumulating remaining predicates
                if *op == BinaryOp::And {
                    if let Some(mut extracted) = self.extract_vector_predicate(left) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: Box::new(prev),
                                op: BinaryOp::And,
                                right: right.clone(),
                            },
                            None => *right.clone(),
                        });
                        return Some(extracted);
                    }
                    if let Some(mut extracted) = self.extract_vector_predicate(right) {
                        extracted.remaining = Some(match extracted.remaining {
                            Some(prev) => LogicalExpression::Binary {
                                left: left.clone(),
                                op: BinaryOp::And,
                                right: Box::new(prev),
                            },
                            None => *left.clone(),
                        });
                        return Some(extracted);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Tries to extract a vector function call predicate.
    ///
    /// Recognizes: `cosine_similarity(n.prop, vec) > threshold`
    /// and the distance variants: `euclidean_distance(n.prop, vec) < threshold`
    fn try_extract_vector_fn(
        &self,
        callee: Callee,
        name: &str,
        args: &[LogicalExpression],
        op: BinaryOp,
        threshold: &LogicalExpression,
    ) -> Option<ExtractedVectorPredicate> {
        if args.len() != 2 {
            return None;
        }

        let (metric, is_similarity) = match callee {
            Callee::Builtin(BuiltinFunction::CosineSimilarity) => {
                (super::VectorMetric::Cosine, true)
            }
            Callee::Builtin(BuiltinFunction::EuclideanDistance) => {
                (super::VectorMetric::Euclidean, false)
            }
            Callee::Builtin(BuiltinFunction::ManhattanDistance) => {
                (super::VectorMetric::Manhattan, false)
            }
            // dot_product is intentionally not pushed down: the physical
            // VectorScanOperator's apply_filters only honors min_similarity
            // when metric == Cosine, so a pushed DotProduct threshold would
            // be silently dropped, returning unfiltered results. Fall
            // through to per-row evaluation instead.
            _ => return None,
        };

        // Strict (>, <) and inclusive (>=, <=) comparisons push down; the
        // comparison itself runs above the scan (see
        // `ExtractedVectorPredicate`).
        let valid_op = if is_similarity {
            matches!(op, BinaryOp::Gt | BinaryOp::Ge)
        } else {
            matches!(op, BinaryOp::Lt | BinaryOp::Le)
        };
        if !valid_op {
            return None;
        }

        // One arg must be a property access, the other a vector literal/parameter
        let (variable, property, query_vector) =
            if let LogicalExpression::Property { variable, property } = &args[0] {
                (variable.clone(), property.clone(), args[1].clone())
            } else if let LogicalExpression::Property { variable, property } = &args[1] {
                (variable.clone(), property.clone(), args[0].clone())
            } else {
                return None;
            };

        // The threshold as vectors measure it, an f32.
        let threshold_val = match threshold {
            LogicalExpression::Literal(value) => value.as_vector_component()?,
            _ => return None,
        };

        let (min_similarity, max_distance) = if is_similarity {
            (Some(threshold_val), None)
        } else {
            (None, Some(threshold_val))
        };

        let comparison = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::FunctionCall {
                callee,
                name: name.to_string(),
                args: args.to_vec(),
                distinct: false,
            }),
            op,
            right: Box::new(threshold.clone()),
        };
        Some(ExtractedVectorPredicate::new(
            property,
            variable,
            query_vector,
            metric,
            min_similarity,
            max_distance,
            comparison,
        ))
    }
}

/// What a variable is bound to, as far as zone-map pruning can prove.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BindingKind {
    Node,
    Edge,
}

/// Finds what `variable` is bound to in `op`, looking only through read-only
/// operators whose output rows carry graph entities unchanged.
///
/// Returns `None` when the binding cannot be proven: projected or unwound
/// values (maps, scalars, a `WITH` rename), aggregates, correlated inputs, and
/// anything below a mutation, because values written earlier in the query are
/// not in the store's zone maps when the plan is built. Callers must treat
/// `None` as "do not prune".
fn binding_kind(op: &LogicalOperator, variable: &str) -> Option<BindingKind> {
    match op {
        LogicalOperator::NodeScan(scan) => {
            if scan.variable == variable {
                Some(BindingKind::Node)
            } else {
                scan.input
                    .as_deref()
                    .and_then(|input| binding_kind(input, variable))
            }
        }
        LogicalOperator::Expand(expand) => {
            if expand.to_variable == variable {
                return Some(BindingKind::Node);
            }
            if expand.edge_variable.as_deref() == Some(variable) {
                // A variable-length expand binds a list of edges.
                let single_hop =
                    !expand.quantified && expand.min_hops == 1 && expand.max_hops == Some(1);
                return single_hop.then_some(BindingKind::Edge);
            }
            if expand.path_alias.as_deref() == Some(variable) {
                return None;
            }
            binding_kind(&expand.input, variable)
        }
        LogicalOperator::Filter(filter) => binding_kind(&filter.input, variable),
        LogicalOperator::Sort(sort) => binding_kind(&sort.input, variable),
        LogicalOperator::Limit(limit) => binding_kind(&limit.input, variable),
        LogicalOperator::Skip(skip) => binding_kind(&skip.input, variable),
        LogicalOperator::Distinct(distinct) => binding_kind(&distinct.input, variable),
        LogicalOperator::Join(join) => {
            binding_kind(&join.left, variable).or_else(|| binding_kind(&join.right, variable))
        }
        LogicalOperator::LeftJoin(join) => {
            binding_kind(&join.left, variable).or_else(|| binding_kind(&join.right, variable))
        }
        _ => None,
    }
}

/// Adds the parts of an `AND` (the predicate itself when it is not one).
fn split_conjuncts(predicate: &LogicalExpression, out: &mut Vec<LogicalExpression>) {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            split_conjuncts(left, out);
            split_conjuncts(right, out);
        }
        other => out.push(other.clone()),
    }
}

/// The `AND` of `conjuncts`, or `None` when there are none.
fn join_conjuncts(conjuncts: Vec<LogicalExpression>) -> Option<LogicalExpression> {
    LogicalExpression::conjunction(conjuncts)
}

/// The hop and the condition of `all(hop IN edges(path) WHERE condition)`
/// when the condition reads nothing but the hop and the variables `row` binds
/// (the input of the expand that binds `path`).
fn edge_condition_of(
    conjunct: &LogicalExpression,
    path: &str,
    row: &std::collections::HashSet<String>,
) -> Option<(String, LogicalExpression)> {
    let LogicalExpression::ListPredicate {
        kind: crate::query::plan::ListPredicateKind::All,
        variable: hop,
        list_expr,
        predicate,
    } = conjunct
    else {
        return None;
    };
    let LogicalExpression::FunctionCall {
        callee: crate::query::plan::Callee::Builtin(crate::query::plan::BuiltinFunction::Edges),
        args,
        ..
    } = list_expr.as_ref()
    else {
        return None;
    };
    if !matches!(args.as_slice(), [LogicalExpression::Variable(name)] if name == path) {
        return None;
    }
    let mut read = Vec::new();
    super::project::collect_vars(predicate, &mut read);
    read.iter()
        .all(|name| name == hop || row.contains(name))
        .then(|| (hop.clone(), predicate.as_ref().clone()))
}

/// The checks below a lookup and the filter's predicate, in the order the chain of filters runs
/// them.
fn whole_predicate(filter: &FilterOp, checked: &super::seek::CheckedScan<'_>) -> LogicalExpression {
    checked
        .checks
        .iter()
        .map(|(_, check)| (*check).clone())
        .chain(std::iter::once(filter.predicate.clone()))
        .reduce(|left, right| LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::And,
            right: Box::new(right),
        })
        .expect("the filter's own predicate")
}

/// `predicate`, the whole of a chain of filters as an `AND` chain, without what an exact read of
/// a key's column has answered: its first conjunct that is the plain `variable.property = value`,
/// and each check of one label, which the read's scope holds at the view. `None` when nothing
/// remains. A conjunct of any other form stays.
fn without_what_the_read_answered(
    predicate: &LogicalExpression,
    variable: &str,
    property: &str,
    value: &Value,
) -> Option<LogicalExpression> {
    fn conjuncts<'e>(expression: &'e LogicalExpression, out: &mut Vec<&'e LogicalExpression>) {
        if let LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } = expression
        {
            conjuncts(left, out);
            conjuncts(right, out);
        } else {
            out.push(expression);
        }
    }
    let mut all = Vec::new();
    conjuncts(predicate, &mut all);
    let is_the_equality = |conjunct: &&LogicalExpression| match conjunct {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::Eq,
            right,
        } => super::Planner::extract_property_equality(left, right).is_some_and(
            |(on, key, equal_to)| on == variable && key == property && equal_to == *value,
        ),
        _ => false,
    };
    let at = all.iter().position(is_the_equality);
    let rest = all
        .iter()
        .enumerate()
        .filter(|(index, conjunct)| {
            Some(*index) != at && super::scan::label_check(conjunct, variable).is_none()
        })
        .map(|(_, conjunct)| (*conjunct).clone());
    LogicalExpression::conjunction(rest)
}

/// The values an index lookup probes for `value`, so that it finds every
/// node `=` finds equal to it under `comparison`: the value itself, and in
/// GQL and SQL/PGQ, which read a local datetime at the session time zone,
/// the datetime of the other kind at the same instant (zoned datetimes are
/// equal, and indexed, by their instant, so one zoned probe covers every
/// offset). `None` for a time in GQL and SQL/PGQ: a local time equals zoned
/// ones at any offset, which no set of probes covers.
fn equality_probes(value: &Value, comparison: TemporalComparison) -> Option<Vec<Value>> {
    use grafeo_common::types::{Timestamp, ZonedDatetime};
    let TemporalComparison::LocalAt(offset) = comparison else {
        return Some(vec![value.clone()]);
    };
    let offset_micros = i64::from(offset) * 1_000_000;
    match value {
        Value::ZonedDatetime(zoned) => Some(vec![
            value.clone(),
            Value::Timestamp(Timestamp::from_micros(
                zoned.as_timestamp().as_micros() + offset_micros,
            )),
        ]),
        Value::Timestamp(local) => Some(vec![
            value.clone(),
            Value::ZonedDatetime(ZonedDatetime::from_timestamp_offset(
                Timestamp::from_micros(local.as_micros() - offset_micros),
                0,
            )),
        ]),
        Value::Time(_) => None,
        _ => Some(vec![value.clone()]),
    }
}

/// The bounds a range predicate puts on a property ([`Planner::range_bounds`](super::Planner)).
pub(super) struct RangeBounds {
    /// The bounded property.
    pub(super) property: String,
    /// The lower bound, if any.
    pub(super) min: Option<Value>,
    /// The upper bound, if any.
    pub(super) max: Option<Value>,
    /// Whether the lower bound is included.
    pub(super) min_inclusive: bool,
    /// Whether the upper bound is included.
    pub(super) max_inclusive: bool,
}
