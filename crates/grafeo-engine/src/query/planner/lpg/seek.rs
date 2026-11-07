//! Node seeks: a filter over a node scan that pins the node by ID or by an
//! indexed property becomes a lookup per input row instead of a scan per row.

use std::collections::HashSet;

use grafeo_common::types::ReadView;
use grafeo_core::execution::operators::{
    ExpressionPredicate, FilterOperator, NodeSeekOperator, Operator, SeekKey, SingleRowOperator,
};
use grafeo_core::graph::GraphStore;

use super::{Arc, HashMap, Result};
use crate::query::optimizer::movable_variables;
use crate::query::plan::BuiltinFunction;
use crate::query::plan::Callee;
use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator, NodeScanOp};
use grafeo_core::execution::operators::Volatility;

/// The seek the planner makes of a filter over a node scan.
pub(crate) struct SeekChoice<'a> {
    /// What the seek looks nodes up by.
    pub key: SeekKey,
    /// The value looked up, evaluated for each input row.
    pub value: &'a LogicalExpression,
    /// For `IN`: the value is a list and each element is looked up.
    pub list: bool,
}

/// Chooses the seek for a filter over `scan`, if one applies.
///
/// A conjunct of the filter must pin the scanned node: `id(v) = e`, or
/// `v.p = e` with `p` indexed, or the `IN` form of either, where `e` does not
/// read `v` (it may read the scan's input, parameters and literals). An ID is
/// preferred over a property, and left out without `id_seeks`. A scan without input with a constant property
/// key is left to the plan-time index lookup, which resolves it once, unless
/// the filter runs `after_a_write` of its statement: that lookup would not
/// see the write (see `Planner::reads_the_store_as_planned`), the seek looks
/// the key up when its one input row arrives.
pub(crate) fn choose_seek<'a>(
    predicate: &'a LogicalExpression,
    scan: &NodeScanOp,
    has_index: impl Fn(&str) -> bool,
    id_seeks: bool,
    after_a_write: bool,
) -> Option<SeekChoice<'a>> {
    let mut conjuncts = Vec::new();
    split_conjuncts(predicate, &mut conjuncts);
    let mut by_property = None;
    for conjunct in conjuncts {
        let Some(choice) = seek_of(conjunct, &scan.variable, &has_index) else {
            continue;
        };
        let Some(variables) = value_variables(choice.value) else {
            continue;
        };
        if variables.contains(&scan.variable) || (scan.input.is_none() && !variables.is_empty()) {
            continue;
        }
        match choice.key {
            SeekKey::Id if id_seeks => return Some(choice),
            SeekKey::Id => {}
            SeekKey::Property(_) => {
                let looked_up_while_planning = scan.input.is_none() && !after_a_write;
                if !looked_up_while_planning && by_property.is_none() {
                    by_property = Some(choice);
                }
            }
        }
    }
    by_property
}

/// [`choose_seek`], for a filter the `rules` allow a seek of: `id()` by
/// [`IdSeek`](crate::query::optimizer::PlannerRule::IdSeek), a property by [`KeySeek`](crate::query::optimizer::PlannerRule::KeySeek).
pub(crate) fn allowed_seek<'a>(
    predicate: &'a LogicalExpression,
    scan: &NodeScanOp,
    labels: &LookupLabels<'_>,
    store: &dyn GraphStore,
    rules: crate::query::optimizer::PlannerRules,
    after_a_write: bool,
) -> Option<SeekChoice<'a>> {
    choose_seek(
        predicate,
        scan,
        |property| {
            rules.has(crate::query::optimizer::PlannerRule::KeySeek)
                && labels.served_by(store, property)
        },
        rules.has(crate::query::optimizer::PlannerRule::IdSeek),
        after_a_write,
    )
}

/// [`allowed_seek`] over a chain of filters: `filter`, then the checks of `below` between it and
/// the scan, each of which a seek for `filter` takes over (see [`checked_scan`]), by the `labels`
/// of the whole chain. `None` also when the rows join faster than the first seek found would look them
/// up: the plan of its filter then decides, as without the others.
pub(crate) fn chain_seek<'a>(
    filter: &'a FilterOp,
    below: &CheckedScan<'a>,
    labels: &LookupLabels<'_>,
    store: &dyn GraphStore,
    rules: crate::query::optimizer::PlannerRules,
    after_a_write: bool,
    writes_per_row: bool,
) -> Option<SeekChoice<'a>> {
    let filters = below
        .checks
        .iter()
        .filter_map(|(operator, _)| match operator {
            LogicalOperator::Filter(check) => Some(check),
            _ => None,
        });
    for holder in std::iter::once(filter).chain(filters) {
        if let Some(seek) = allowed_seek(
            &holder.predicate,
            below.scan,
            labels,
            store,
            rules,
            after_a_write,
        ) {
            let joins = matches!(seek.key, SeekKey::Property(_))
                && super::value_join::beats_a_seek(holder, writes_per_row, store);
            return (!joins).then_some(seek);
        }
    }
    None
}

/// Whether `conjunct` pins `variable` to the nodes a key selects: `id(v) = e`,
/// or `v.p = e` with `p` indexed, or the `IN` form of either, where `e` reads
/// only `input` (the variables of the rows the scan of `v` runs for, none
/// for a scan without input), parameters and literals, through functions a
/// seek evaluates (see [`value_variables`]). The planner then looks the nodes
/// up instead of scanning: a seek (see [`choose_seek`]), or for a constant
/// property key of a scan without input the index lookup while planning.
pub(crate) fn pins(
    conjunct: &LogicalExpression,
    variable: &str,
    has_index: impl Fn(&str) -> bool,
    input: &HashSet<String>,
) -> bool {
    pin_of(conjunct, variable, has_index, input).is_some()
}

/// The seek `conjunct` pins `variable` by, as [`pins`] says.
pub(crate) fn pin_of<'a>(
    conjunct: &'a LogicalExpression,
    variable: &str,
    has_index: impl Fn(&str) -> bool,
    input: &HashSet<String>,
) -> Option<SeekChoice<'a>> {
    seek_of(conjunct, variable, &has_index).filter(|choice| {
        value_variables(choice.value)
            .is_some_and(|variables| !variables.contains(variable) && variables.is_subset(input))
    })
}

/// The seek of a conjunct of `predicate` that pins `variable` to one value
/// known without a row: `id(v) = e` or `v.p = e` with `p` indexed, where `e`
/// reads no variable. An ID is preferred over a property.
pub(crate) fn constant_seek<'a>(
    predicate: &'a LogicalExpression,
    variable: &str,
    has_index: impl Fn(&str) -> bool,
) -> Option<SeekChoice<'a>> {
    let mut conjuncts = Vec::new();
    split_conjuncts(predicate, &mut conjuncts);
    let mut by_property = None;
    for conjunct in conjuncts {
        let Some(choice) = seek_of(conjunct, variable, &has_index) else {
            continue;
        };
        if choice.list || !value_variables(choice.value).is_some_and(|names| names.is_empty()) {
            continue;
        }
        match choice.key {
            SeekKey::Id => return Some(choice),
            SeekKey::Property(_) => by_property = by_property.or(Some(choice)),
        }
    }
    by_property
}

/// The node scan a filter's seek replaces, with the filters between them.
pub(crate) struct CheckedScan<'a> {
    /// The scan below the filter.
    pub scan: &'a NodeScanOp,
    /// The scan, as the operator it is in the plan.
    pub operator: &'a LogicalOperator,
    /// The filters between the filter and the scan, from the bottom up (the
    /// order they run in): each as the operator it is in the plan, and its
    /// predicate.
    pub checks: Vec<(&'a LogicalOperator, &'a LogicalExpression)>,
}

/// The node scan below `filter`, directly or below filters that check the
/// scanned node alone: each reads no other variable and may move (see
/// [`movable_variables`]), like the checks of the other labels of a pattern
/// such as `(t:Graph:TypeDefinition)` (see `scan.rs`). A seek for `filter`
/// replaces the scan, and checks these predicates on the nodes it finds.
pub(crate) fn checked_scan(filter: &FilterOp) -> Option<CheckedScan<'_>> {
    let mut checks = Vec::new();
    let mut below = filter.input.as_ref();
    while let LogicalOperator::Filter(check) = below {
        checks.push((below, &check.predicate));
        below = &check.input;
    }
    let LogicalOperator::NodeScan(scan) = below else {
        return None;
    };
    let checks_the_scan = |predicate: &LogicalExpression| {
        movable_variables(predicate)
            .is_some_and(|variables| variables.iter().all(|v| *v == scan.variable))
    };
    if !checks
        .iter()
        .all(|(_, predicate)| checks_the_scan(predicate))
    {
        return None;
    }
    checks.reverse();
    Some(CheckedScan {
        scan,
        operator: below,
        checks,
    })
}

/// What the label checks of a chain of filters say of the nodes a variable binds, for a lookup
/// to pick its index. Only conjuncts count: under `OR` or `NOT` a node may pass without a label,
/// and a filter above an outer join does not reach below it, as these are one pattern's filters
/// and scan. Read it from the top of the chain: a label check above the filter that holds the key
/// counts as much as one below it.
pub(crate) struct LookupLabels<'a> {
    /// Labels every node has: the label of a scan below the filter, then the labels the filters
    /// check, from the bottom up.
    required: Vec<&'a str>,
    /// For each conjunct `n:A OR n:B`, the labels a node has one of.
    unions: Vec<Vec<&'a str>>,
}

/// What a lookup of a key among nodes with some labels reads. The planner, `EXPLAIN`'s hint and
/// its note read this one answer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LookupRead<'a> {
    /// The indexes of these labels, which together hold every result; none: the key's on every
    /// label.
    Index(Vec<String>),
    /// The indexes of `read`, and a read of the key's column for the rest: nodes with one of
    /// `one_of` (none: any other node) that have none of `read`.
    IndexesAndRemainder {
        read: Vec<String>,
        required: Vec<&'a str>,
        one_of: Vec<&'a str>,
    },
    /// No index: a read of the key's column, among nodes with every label of `required` and one
    /// of `one_of` (none: any node).
    Column {
        required: Vec<&'a str>,
        one_of: Vec<&'a str>,
    },
    /// The scan and its filter.
    Scan,
}

impl<'a> LookupLabels<'a> {
    /// What a lookup of `key` at the plan's start reads in `store` under `rules`. A required label
    /// without an index leaves no index useful, as a node with an indexed label may have it too and
    /// a node with neither would be read anyway. Otherwise the indexed labels of the first union
    /// with one, or with no label needed, of every label with an index on the key. With no index
    /// to read, the key's column.
    pub(crate) fn lookup_read(
        &self,
        store: &dyn GraphStore,
        key: &str,
        rules: crate::query::optimizer::PlannerRules,
    ) -> LookupRead<'a> {
        use crate::query::optimizer::PlannerRule;

        if rules.has(PlannerRule::IndexLookup) && self.served_by(store, key) {
            return LookupRead::Index(self.to_read(store, key));
        }
        if !store.reads_key_columns() {
            return LookupRead::Scan;
        }
        if rules.has(PlannerRule::IndexLookup) {
            for union in &self.unions {
                let (indexed, rest): (Vec<&str>, Vec<&str>) = union
                    .iter()
                    .copied()
                    .partition(|label| store.has_label_index(label, key));
                if !indexed.is_empty() {
                    return LookupRead::IndexesAndRemainder {
                        read: indexed.iter().map(ToString::to_string).collect(),
                        required: self.required.clone(),
                        one_of: rest,
                    };
                }
            }
            if self.required.is_empty() {
                let read: Vec<String> = store
                    .label_index_entries()
                    .into_iter()
                    .filter(|(_, on)| on == key)
                    .map(|(label, _)| label)
                    .collect();
                if !read.is_empty() {
                    return LookupRead::IndexesAndRemainder {
                        read,
                        required: Vec::new(),
                        one_of: Vec::new(),
                    };
                }
            }
        }
        if rules.has(PlannerRule::KeyColumnLookup) {
            return LookupRead::Column {
                required: self.required.clone(),
                one_of: self.unions.first().cloned().unwrap_or_default(),
            };
        }
        LookupRead::Scan
    }

    /// At most how many nodes a read of a key's column looks at: of its rarest required label, the
    /// labels it has one of, or all. Counted at the latest view, as the count picks a route only
    /// and both give the same answer; a store that reads key columns keeps these counts.
    ///
    /// # Errors
    ///
    /// Returns the error of a read the store cannot do.
    pub(crate) fn nodes_at_most(&self, store: &dyn GraphStore) -> Result<usize> {
        let latest = ReadView::latest();
        let count = |label: &&str| store.nodes_by_label_count(label, latest);
        if !self.required.is_empty() {
            return self
                .required
                .iter()
                .map(count)
                .try_fold(usize::MAX, |least, nodes| Ok(least.min(nodes?)));
        }
        match self.unions.first() {
            Some(union) => union.iter().map(count).sum(),
            None => store.node_count(latest),
        }
    }

    /// The labels of the filters from `filter` down, and a scan of `variable` below them. It
    /// tells of the whole chain only with `filter` its top.
    pub(crate) fn of(filter: &'a FilterOp, variable: &str) -> Self {
        let mut filters = vec![&filter.predicate];
        let mut below = filter.input.as_ref();
        while let LogicalOperator::Filter(check) = below {
            filters.push(&check.predicate);
            below = &check.input;
        }
        let mut labels = Self {
            required: Vec::new(),
            unions: Vec::new(),
        };
        if let LogicalOperator::NodeScan(scan) = below
            && scan.variable == variable
            && let Some(label) = scan.label.as_deref()
        {
            labels.required.push(label);
        }
        for predicate in filters.into_iter().rev() {
            labels.add(predicate, variable);
        }
        labels
    }

    /// The labels of `conjuncts`.
    pub(crate) fn of_conjuncts(conjuncts: &'a [LogicalExpression], variable: &str) -> Self {
        let mut labels = Self {
            required: Vec::new(),
            unions: Vec::new(),
        };
        for conjunct in conjuncts {
            labels.add(conjunct, variable);
        }
        labels
    }

    fn add(&mut self, predicate: &'a LogicalExpression, variable: &str) {
        let mut conjuncts = Vec::new();
        split_conjuncts(predicate, &mut conjuncts);
        for conjunct in conjuncts {
            if let Some(label) = super::scan::label_check(conjunct, variable) {
                if !self.required.contains(&label) {
                    self.required.push(label);
                }
            } else if let Some(labels) = super::scan::any_label(conjunct, variable) {
                let mut union: Vec<&str> = Vec::new();
                for label in labels {
                    if !union.contains(&label) {
                        union.push(label);
                    }
                }
                self.unions.push(union);
            }
        }
    }

    /// The labels whose indexes a lookup of `key` reads, of the indexes the two functions tell
    /// of: the first required label with an index, then the first union with an index on each of
    /// its labels, or with none (empty) the key's index on every label. `None` when none of them
    /// serves: a union with a label without an index would miss that label's nodes.
    pub(crate) fn index_for(
        &self,
        key: &str,
        on_label: impl Fn(&str, &str) -> bool,
        on_every_label: impl Fn(&str) -> bool,
    ) -> Option<&[&'a str]> {
        if let Some(label) = self.required.iter().find(|label| on_label(label, key)) {
            return Some(std::slice::from_ref(label));
        }
        if let Some(union) = self
            .unions
            .iter()
            .find(|union| union.iter().all(|label| on_label(label, key)))
        {
            return Some(union);
        }
        on_every_label(key).then_some(&[])
    }

    /// Adds a label every node has, as a scan's.
    pub(crate) fn require(&mut self, label: &'a str) {
        if !self.required.contains(&label) {
            self.required.push(label);
        }
    }

    /// The labels every node has.
    pub(crate) fn required(&self) -> &[&'a str] {
        &self.required
    }

    /// Whether `store` has an index a lookup of `key` can read.
    pub(crate) fn served_by(&self, store: &dyn GraphStore, key: &str) -> bool {
        self.read_in(store, key).is_some()
    }

    /// [`index_for`](Self::index_for), by what `store` holds.
    fn read_in(&self, store: &dyn GraphStore, key: &str) -> Option<&[&'a str]> {
        self.index_for(
            key,
            |label, key| store.has_label_index(label, key),
            |key| store.index_serves(&[], key),
        )
    }

    /// The labels whose indexes a lookup of `key` in `store` reads, as an operator keeps them;
    /// empty for the key's index on every label, and when there is none, as an operator then
    /// finds out.
    pub(crate) fn to_read(&self, store: &dyn GraphStore, key: &str) -> Vec<String> {
        self.read_in(store, key)
            .map(|labels| labels.iter().map(ToString::to_string).collect())
            .unwrap_or_default()
    }
}

/// A key the note of `EXPLAIN` names: see [`key_indexed_on_other_labels`].
pub(crate) struct KeyIndexedElsewhere {
    /// How many filters below the top of the chain the filter that holds the key is.
    pub depth: usize,
    /// The key.
    pub key: String,
    /// The labels with an index on it, sorted.
    pub labels: Vec<String>,
}

/// For `EXPLAIN`: the first key, from the top of the chain of filters `top` down, of a conjunct
/// that would pin `variable` if an index a lookup could read had it, where only indexes on other
/// labels do. The value reads no row unless `reads_rows`, the variables of the input being what
/// it may read, and a list counts only with `lists`: a scan's lookups take both as the operator
/// allows, a hop's only a constant value.
pub(crate) fn key_indexed_on_other_labels(
    top: &FilterOp,
    variable: &str,
    reads_rows: bool,
    lists: bool,
    store: &dyn GraphStore,
) -> Option<KeyIndexedElsewhere> {
    let indexes = store.label_index_entries();
    if indexes.is_empty() {
        return None;
    }
    let labels = LookupLabels::of(top, variable);
    let elsewhere =
        |key: &str| indexes.iter().any(|(_, on)| on == key) && !labels.served_by(store, key);
    let mut filter = top;
    let mut depth = 0;
    loop {
        let mut conjuncts = Vec::new();
        split_conjuncts(&filter.predicate, &mut conjuncts);
        let found = conjuncts.into_iter().find_map(|conjunct| {
            let choice = seek_of(conjunct, variable, &elsewhere)?;
            if choice.list && !lists {
                return None;
            }
            let SeekKey::Property(key) = choice.key else {
                return None;
            };
            let read = value_variables(choice.value)?;
            (!read.contains(variable) && (reads_rows || read.is_empty())).then_some(key)
        });
        if let Some(key) = found {
            let labels = indexes
                .iter()
                .filter(|(_, on)| *on == key)
                .map(|(label, _)| label.clone())
                .collect();
            return Some(KeyIndexedElsewhere { depth, key, labels });
        }
        let LogicalOperator::Filter(below) = filter.input.as_ref() else {
            return None;
        };
        filter = below;
        depth += 1;
    }
}

pub(crate) fn split_conjuncts<'a>(
    expr: &'a LogicalExpression,
    out: &mut Vec<&'a LogicalExpression>,
) {
    match expr {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            split_conjuncts(left, out);
            split_conjuncts(right, out);
        }
        other => out.push(other),
    }
}

/// The seek a single conjunct allows: `key = value`, `value = key` or
/// `key IN value`, where the key is `id(variable)` or an indexed property of
/// `variable`.
fn seek_of<'a>(
    conjunct: &'a LogicalExpression,
    variable: &str,
    has_index: &impl Fn(&str) -> bool,
) -> Option<SeekChoice<'a>> {
    let LogicalExpression::Binary { left, op, right } = conjunct else {
        return None;
    };
    let key = |side: &LogicalExpression| match side {
        LogicalExpression::Id(v) if v == variable => Some(SeekKey::Id),
        LogicalExpression::FunctionCall {
            callee: Callee::Builtin(BuiltinFunction::Id),
            args,
            ..
        } if matches!(args.as_slice(), [LogicalExpression::Variable(v)] if v == variable) => {
            Some(SeekKey::Id)
        }
        LogicalExpression::Property {
            variable: v,
            property,
        } if v == variable && has_index(property) => Some(SeekKey::Property(property.clone())),
        _ => None,
    };
    match op {
        BinaryOp::Eq => key(left)
            .map(|key| SeekChoice {
                key,
                value: right,
                list: false,
            })
            .or_else(|| {
                key(right).map(|key| SeekChoice {
                    key,
                    value: left,
                    list: false,
                })
            }),
        BinaryOp::In => key(left).map(|key| SeekChoice {
            key,
            value: right,
            list: true,
        }),
        _ => None,
    }
}

/// The variables a seek value reads, or `None` for an expression a seek does
/// not evaluate (subqueries, comprehensions, functions whose value changes
/// between calls). The keys of a value join follow the same rule (see
/// `value_join.rs`).
pub(super) fn value_variables(expr: &LogicalExpression) -> Option<HashSet<String>> {
    let mut variables = HashSet::new();
    collect_value_variables(expr, &mut variables).then_some(variables)
}

/// Whether a call of `callee` with `arguments` arguments returns the same
/// value for the same arguments, so a seek can evaluate a key once and probe
/// the index with it while the filter above it reads the same value: the
/// [`Immutable`](Volatility::Immutable) built-in functions. A function that is
/// not Immutable (every clock, session and random function among them) keeps
/// the scan and its filter, so a wrong class costs speed, never rows. The
/// reachability search of variable-length expands uses it too (see
/// `reachability.rs`).
pub(super) fn deterministic(callee: Callee, arguments: usize) -> bool {
    matches!(callee, Callee::Builtin(function)
        if function.volatility(arguments) == Volatility::Immutable)
}

fn collect_value_variables(expr: &LogicalExpression, out: &mut HashSet<String>) -> bool {
    match expr {
        LogicalExpression::Literal(_) | LogicalExpression::Parameter(_) => true,
        LogicalExpression::Variable(name)
        | LogicalExpression::Property { variable: name, .. }
        | LogicalExpression::Id(name)
        | LogicalExpression::Labels(name)
        | LogicalExpression::Type(name) => {
            out.insert(name.clone());
            true
        }
        LogicalExpression::Binary { left, right, .. } => {
            collect_value_variables(left, out) && collect_value_variables(right, out)
        }
        LogicalExpression::Unary { operand, .. } => collect_value_variables(operand, out),
        LogicalExpression::IndexAccess { base, index } => {
            collect_value_variables(base, out) && collect_value_variables(index, out)
        }
        LogicalExpression::MapAccess { base, .. } => collect_value_variables(base, out),
        LogicalExpression::List(items) => {
            items.iter().all(|item| collect_value_variables(item, out))
        }
        LogicalExpression::Map(entries) => entries
            .iter()
            .all(|(_, value)| collect_value_variables(value, out)),
        LogicalExpression::FunctionCall { callee, args, .. } => {
            deterministic(*callee, args.len())
                && args.iter().all(|arg| collect_value_variables(arg, out))
        }
        _ => false,
    }
}

impl super::Planner {
    /// Plans a filter over a node scan (see [`checked_scan`]) as a seek when
    /// [`chain_seek`] finds one: the scan's input rows, each joined with the
    /// nodes its key selects, and a filter on top, which still decides every
    /// row with the checks between the filter and the scan, then the filter's
    /// own predicate.
    pub(super) fn try_plan_filter_with_node_seek(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let Some(below) = checked_scan(filter) else {
            return Ok(None);
        };
        let scan = below.scan;
        // A seek looks a key up when its input row arrives: after a write in
        // the input it would miss what the input writes for later rows (the
        // scan reads its whole input first, see `plan_node_scan`).
        if scan
            .input
            .as_deref()
            .is_some_and(LogicalOperator::has_mutations)
        {
            return Ok(None);
        }
        // A property index looks values up at the query's view; an ID is no
        // index.
        let after_a_write = !self.reads_the_store_as_planned();
        let labels = LookupLabels::of(filter, &scan.variable);
        // Many rows against a small label join faster than they seek.
        let Some(seek) = chain_seek(
            filter,
            &below,
            &labels,
            self.store.as_ref(),
            self.rules,
            after_a_write,
            self.statement_writes
                .get()
                .is_none_or(|writes| writes.per_row),
        ) else {
            return Ok(None);
        };
        let CheckedScan {
            operator, checks, ..
        } = below;

        let (input, mut columns) = match &scan.input {
            Some(input) => self.plan_operator(input)?,
            None => (
                Box::new(SingleRowOperator::new()) as Box<dyn Operator>,
                Vec::new(),
            ),
        };
        // PROFILE entries for the scan and the checks the seek absorbs, in
        // the order the plan's tree lists them (children first).
        self.record_kept_entry("NodeSeek", operator);
        for (check, _) in &checks {
            self.record_absorbed_scan_entry("Filter", check);
        }
        // `first`, the checks, then the filter's predicate: the order the
        // chain of filters runs them in.
        let checked = |first: Option<LogicalExpression>| {
            let conjuncts = first
                .into_iter()
                .chain(checks.iter().map(|(_, predicate)| (*predicate).clone()))
                .chain(std::iter::once(filter.predicate.clone()));
            LogicalExpression::conjunction(conjuncts).expect("the filter's predicate")
        };
        let store = Arc::clone(&self.store) as Arc<dyn GraphStore>;
        let expression =
            |expr: &LogicalExpression, columns: &[String]| -> Result<ExpressionPredicate> {
                let variable_columns: HashMap<String, usize> = columns
                    .iter()
                    .enumerate()
                    .map(|(i, name)| (name.clone(), i))
                    .collect();
                Ok(ExpressionPredicate::new(
                    self.convert_expression(expr)?,
                    variable_columns,
                    Arc::clone(&store),
                    self.read_view(),
                )
                .with_session_context(self.session_context.clone()))
            };

        if columns.contains(&scan.variable) {
            // The input binds the variable already (a correlated subquery):
            // no lookup, the filter checks the bound node, labels included.
            let label = scan.label.as_ref().map(|label| {
                LogicalExpression::builtin(
                    BuiltinFunction::HasLabel,
                    vec![
                        LogicalExpression::Variable(scan.variable.clone()),
                        LogicalExpression::Literal(label.as_str().into()),
                    ],
                )
            });
            let predicate = expression(&checked(label), &columns)?;
            return Ok(Some((
                Box::new(FilterOperator::new(input, Box::new(predicate))),
                columns,
            )));
        }

        let key = expression(seek.value, &columns)?;
        let index_labels = match &seek.key {
            SeekKey::Property(property) => labels.to_read(self.store.as_ref(), property),
            SeekKey::Id => Vec::new(),
        };
        let mut rows = NodeSeekOperator::new(
            Arc::clone(&store),
            input,
            seek.key,
            key,
            scan.label.clone(),
            self.read_view(),
        )
        .with_index_labels(index_labels)
        .with_equal_across_kinds(self.equal_across_kinds())
        .with_read_counter(self.read_counter());
        if seek.list {
            rows = rows.with_list_key();
        }
        columns.push(scan.variable.clone());
        let predicate = expression(&checked(None), &columns)?;
        Ok(Some((
            Box::new(FilterOperator::new(Box::new(rows), Box::new(predicate))),
            columns,
        )))
    }
}

#[cfg(test)]
mod tests {
    use grafeo_common::types::Value;

    use super::*;

    fn has_label(label: &str) -> LogicalExpression {
        LogicalExpression::builtin(
            BuiltinFunction::HasLabel,
            vec![
                LogicalExpression::Variable("t".to_string()),
                LogicalExpression::Literal(Value::from(label)),
            ],
        )
    }

    fn equals(left: LogicalExpression, right: LogicalExpression) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::Eq,
            right: Box::new(right),
        }
    }

    fn property(variable: &str, name: &str) -> LogicalExpression {
        LogicalExpression::Property {
            variable: variable.to_string(),
            property: name.to_string(),
        }
    }

    /// `t.filePath = path`, over the filters with `checks` (top down) over a
    /// scan of `t:Graph`.
    fn keyed_over(checks: Vec<LogicalExpression>) -> FilterOp {
        let scan = LogicalOperator::NodeScan(NodeScanOp {
            variable: "t".to_string(),
            label: Some("Graph".to_string()),
            input: None,
        });
        let input = checks.into_iter().rev().fold(scan, |input, predicate| {
            LogicalOperator::Filter(FilterOp {
                predicate,
                pushdown_hint: None,
                input: Box::new(input),
            })
        });
        FilterOp {
            predicate: equals(
                property("t", "filePath"),
                LogicalExpression::Variable("path".to_string()),
            ),
            pushdown_hint: None,
            input: Box::new(input),
        }
    }

    #[test]
    fn the_checks_of_the_scanned_node_are_listed_in_the_order_they_run() {
        let kind = equals(
            property("t", "kind"),
            LogicalExpression::Literal(Value::from("struct")),
        );
        let filter = keyed_over(vec![has_label("TypeDefinition"), kind.clone()]);
        let checked = checked_scan(&filter).expect("both filters check t alone");
        assert_eq!(checked.scan.label.as_deref(), Some("Graph"));
        assert!(matches!(checked.operator, LogicalOperator::NodeScan(_)));
        let predicates: Vec<String> = checked
            .checks
            .iter()
            .map(|(_, predicate)| format!("{predicate:?}"))
            .collect();
        assert_eq!(
            predicates,
            [
                format!("{kind:?}"),
                format!("{:?}", has_label("TypeDefinition"))
            ]
        );

        let direct = keyed_over(Vec::new());
        assert!(
            checked_scan(&direct).is_some_and(|checked| checked.checks.is_empty()),
            "a filter right on the scan has no checks"
        );
    }

    /// A filter between that reads another variable, holds a subquery or
    /// calls a volatile function is no check of the scanned node alone.
    #[test]
    fn other_filters_between_keep_the_scan() {
        let other_variable = equals(property("t", "owner"), property("f", "name"));
        let volatile = LogicalExpression::Binary {
            left: Box::new(LogicalExpression::FunctionCall {
                callee: crate::query::plan::Callee::named("rand"),
                name: "rand".to_string(),
                args: Vec::new(),
                distinct: false,
            }),
            op: BinaryOp::Lt,
            right: Box::new(property("t", "share")),
        };
        let subquery =
            LogicalExpression::ExistsSubquery(Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "t".to_string(),
                label: Some("TypeDefinition".to_string()),
                input: None,
            })));
        for check in [other_variable, volatile, subquery] {
            let filter = keyed_over(vec![has_label("TypeDefinition"), check.clone()]);
            assert!(checked_scan(&filter).is_none(), "{check:?}");
        }
    }
}
