//! Node scan planning, and the choice of the label a node scan reads.

use super::{
    Arc, BinaryOp, EagerOperator, EpochId, ExpressionPredicate, FilterExpression, FilterOp,
    FilterOperator, GraphStoreSearch, HashMap, LogicalExpression, LogicalOperator, LogicalType,
    NestedLoopJoinOperator, NodeScanOp, Operator, PhysicalJoinType, Result, ScanOperator,
    TransactionId, Value,
};
use crate::query::plan::{BuiltinFunction, Callee, ListPredicateKind};

/// The copy of `filter` whose node scan reads the label with the fewest
/// nodes, or `None` when the scan reads that label already.
///
/// A scan of every node below filters that require a label (`WHERE
/// n:Repository`, `'Repository' IN labels(n)`, alone or in an `AND` chain)
/// scans the label with the fewest nodes of those they require, and the
/// filters still check every label they name.
///
/// A pattern like `(n:Graph:Repository)` is a scan of `Graph` below filters
/// that require the other labels (`hasLabel(n, 'Repository')`), in a chain of
/// filters over the scan. When one of the labels those filters require has
/// fewer nodes than the scan's (`nodes_by_label_count`), the copy scans it and
/// checks the scan's label in the place of its check: the scan reads the
/// fewest nodes, and the rows are the same (a label scan returns its nodes in
/// ID order, whichever label it reads). Of labels with as many nodes, the one
/// written first wins: the scan's label, then the checks from the top filter
/// down.
///
/// The scan keeps its label when a vector or text index on it or on the
/// smaller label covers a property of the scanned node that the filters read:
/// such a filter can be planned as a search of the index of the scan's label
/// (see `filter_hybrid.rs`), and an index search can find other nodes than
/// the filter does.
pub(crate) fn with_smallest_scan_label(
    filter: &FilterOp,
    store: &dyn GraphStoreSearch,
) -> Option<FilterOp> {
    let mut predicates = vec![&filter.predicate];
    let mut below = filter.input.as_ref();
    while let LogicalOperator::Filter(inner) = below {
        predicates.push(&inner.predicate);
        below = &inner.input;
    }
    let LogicalOperator::NodeScan(scan) = below else {
        return None;
    };
    let mut checks = Vec::new();
    for predicate in &predicates {
        label_checks(predicate, &scan.variable, &mut checks);
    }
    let first = *checks.first()?;
    let scan_label = scan.label.as_deref().unwrap_or(first);
    let mut smallest = scan_label;
    let mut fewest = store.nodes_by_label_count(scan_label);
    for label in checks {
        if fewest == 0 {
            break;
        }
        let count = store.nodes_by_label_count(label);
        if count < fewest {
            smallest = label;
            fewest = count;
        }
    }
    if scan.label.as_deref() == Some(smallest) {
        return None;
    }
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    if predicates.iter().any(|predicate| {
        search_index_covers(predicate, &scan.variable, [scan_label, smallest], store)
    }) {
        return None;
    }
    let variable = scan.variable.clone();
    let (smallest, scan_label) = (smallest.to_string(), scan.label.clone());
    let mut reordered = filter.clone();
    match scan_label {
        // The check of the label scanned now checks the label scanned before.
        Some(scan_label) => {
            let checked =
                replace_label_check(&mut reordered.predicate, &variable, &smallest, &scan_label);
            move_scan_label(
                &mut reordered.input,
                &variable,
                &smallest,
                &scan_label,
                !checked,
            );
        }
        // A scan of every node had no label to check: the checks stay.
        None => move_scan_label(&mut reordered.input, &variable, &smallest, "", false),
    }
    Some(reordered)
}

/// Whether a query that reads at `viewing_epoch`, in `transaction_id`, may
/// scan another of a pattern's labels than the one written
/// ([`with_smallest_scan_label`]). The same rule decides whether a query may
/// look values up in a property index.
///
/// Always when the store answers its label scans and lookups exactly at a
/// view (`answers_at_view`, see
/// [`GraphStore::answers_at_view`](grafeo_core::graph::GraphStore::answers_at_view)):
/// they find what a scan with the per-row checks finds, in a transaction and
/// at a past epoch alike. Otherwise as before the store answered at a view:
/// in a transaction, whose per-row reads read the newest values and labels
/// as the store's indexes hold them, and at the current epoch, but not at a
/// past epoch outside a transaction (`current_epoch` is the transaction
/// manager's, `None` without one), where a label scan or an index finds the
/// labels and values of now while the check per row reads those of the
/// epoch.
pub(crate) fn may_choose_scan_label(
    viewing_epoch: EpochId,
    transaction_id: Option<TransactionId>,
    current_epoch: Option<EpochId>,
    answers_at_view: bool,
) -> bool {
    answers_at_view
        || transaction_id.is_some()
        || current_epoch.is_none_or(|current| viewing_epoch >= current)
}

/// The labels of the `OR` of labels that `predicate` requires of the node a
/// scan of every node reads (`n:Tag OR n:Topic`, `'Tag' IN labels(n) OR ...`,
/// `any(l IN labels(n) WHERE l IN ['Tag', 'Topic'])`), when the scan may read
/// only their nodes: it has no label and no input, and the labels hold at most
/// half the nodes (a scan of every node is cheaper than larger lists). Of
/// several such conjuncts of an `AND`, the one whose labels hold the fewest
/// nodes. One label alone is the scan's label instead (see
/// [`with_smallest_scan_label`]).
pub(crate) fn label_union(
    predicate: &LogicalExpression,
    scan: &NodeScanOp,
    store: &dyn GraphStoreSearch,
) -> Option<Vec<String>> {
    if scan.label.is_some() || scan.input.is_some() {
        return None;
    }
    let mut conjuncts = Vec::new();
    and_conjuncts(predicate, &mut conjuncts);
    let mut best: Option<(usize, Vec<&str>)> = None;
    for conjunct in conjuncts {
        let Some(found) = any_label(conjunct, &scan.variable) else {
            continue;
        };
        let mut labels: Vec<&str> = Vec::with_capacity(found.len());
        for label in found {
            if !labels.contains(&label) {
                labels.push(label);
            }
        }
        if labels.len() < 2 {
            continue;
        }
        let nodes: usize = labels
            .iter()
            .map(|label| store.nodes_by_label_count(label))
            .sum();
        if best.as_ref().is_none_or(|(fewest, _)| nodes < *fewest) {
            best = Some((nodes, labels));
        }
    }
    let (nodes, labels) = best?;
    (nodes.saturating_mul(2) <= store.node_count())
        .then(|| labels.into_iter().map(str::to_string).collect())
}

/// The conjuncts of an `AND` chain.
fn and_conjuncts<'a>(predicate: &'a LogicalExpression, out: &mut Vec<&'a LogicalExpression>) {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            and_conjuncts(left, out);
            and_conjuncts(right, out);
        }
        other => out.push(other),
    }
}

/// The labels of which `expression` requires one of `variable`: label checks
/// joined by `OR`, or `any(l IN labels(variable) WHERE ...)` over equalities
/// of `l` and `IN` lists of label names.
fn any_label<'a>(expression: &'a LogicalExpression, variable: &str) -> Option<Vec<&'a str>> {
    if let Some(label) = label_check(expression, variable) {
        return Some(vec![label]);
    }
    match expression {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::Or,
            right,
        } => {
            let mut labels = any_label(left, variable)?;
            labels.extend(any_label(right, variable)?);
            Some(labels)
        }
        LogicalExpression::ListPredicate {
            kind: ListPredicateKind::Any,
            variable: element,
            list_expr,
            predicate,
        } if is_labels_of(list_expr, variable) => element_labels(predicate, element),
        _ => None,
    }
}

/// Whether `expression` is `labels(variable)`.
fn is_labels_of(expression: &LogicalExpression, variable: &str) -> bool {
    matches!(
        expression,
        LogicalExpression::FunctionCall {
            callee: Callee::Builtin(BuiltinFunction::Labels),
            args,
            ..
        } if matches!(args.as_slice(), [LogicalExpression::Variable(of)] if of == variable)
    )
}

/// The label names of which `predicate` requires `element` to be one:
/// `element = 'Tag'`, `element IN ['Tag', 'Topic']`, joined by `OR`.
fn element_labels<'a>(predicate: &'a LogicalExpression, element: &str) -> Option<Vec<&'a str>> {
    let LogicalExpression::Binary { left, op, right } = predicate else {
        return None;
    };
    let is_element = |side: &LogicalExpression| matches!(side, LogicalExpression::Variable(name) if name == element);
    let name = |side: &'a LogicalExpression| match side {
        LogicalExpression::Literal(Value::String(label)) => Some(label.as_str()),
        _ => None,
    };
    match op {
        BinaryOp::Or => {
            let mut labels = element_labels(left, element)?;
            labels.extend(element_labels(right, element)?);
            Some(labels)
        }
        BinaryOp::Eq if is_element(left) => name(right).map(|label| vec![label]),
        BinaryOp::Eq if is_element(right) => name(left).map(|label| vec![label]),
        BinaryOp::In if is_element(left) => match right.as_ref() {
            LogicalExpression::List(items) => items.iter().map(name).collect(),
            _ => None,
        },
        _ => None,
    }
}

/// The label of a `hasLabel(variable, 'Label')` check, or of its spelling
/// `'Label' IN labels(variable)`.
fn label_check<'a>(expression: &'a LogicalExpression, variable: &str) -> Option<&'a str> {
    match expression {
        LogicalExpression::FunctionCall {
            callee: Callee::Builtin(BuiltinFunction::HasLabel),
            args,
            ..
        } => match args.as_slice() {
            [
                LogicalExpression::Variable(checked),
                LogicalExpression::Literal(Value::String(label)),
            ] if checked == variable => Some(label.as_str()),
            _ => None,
        },
        LogicalExpression::Binary {
            left,
            op: BinaryOp::In,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (
                LogicalExpression::Literal(Value::String(label)),
                LogicalExpression::FunctionCall {
                    callee: Callee::Builtin(BuiltinFunction::Labels),
                    args,
                    ..
                },
            ) if matches!(args.as_slice(), [LogicalExpression::Variable(checked)] if checked == variable) => {
                Some(label.as_str())
            }
            _ => None,
        },
        _ => None,
    }
}

/// Adds the labels of `variable` that the conjuncts of `predicate` check, in
/// the order they are written.
fn label_checks<'a>(predicate: &'a LogicalExpression, variable: &str, out: &mut Vec<&'a str>) {
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            label_checks(left, variable, out);
            label_checks(right, variable, out);
        }
        other => out.extend(label_check(other, variable)),
    }
}

/// Turns the first conjunct of `predicate` that checks `label` on `variable`
/// into a check of `checked`; returns whether there was one.
fn replace_label_check(
    predicate: &mut LogicalExpression,
    variable: &str,
    label: &str,
    checked: &str,
) -> bool {
    if label_check(predicate, variable) == Some(label) {
        *predicate = LogicalExpression::builtin(
            BuiltinFunction::HasLabel,
            vec![
                LogicalExpression::Variable(variable.to_string()),
                LogicalExpression::Literal(Value::String(checked.into())),
            ],
        );
        return true;
    }
    match predicate {
        LogicalExpression::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            replace_label_check(left, variable, label, checked)
                || replace_label_check(right, variable, label, checked)
        }
        _ => false,
    }
}

/// Gives the node scan below the filters of `op` the label `label`; while
/// `pending`, the first of those filters that checks `label` checks `checked`
/// instead.
fn move_scan_label(
    op: &mut LogicalOperator,
    variable: &str,
    label: &str,
    checked: &str,
    pending: bool,
) {
    match op {
        LogicalOperator::Filter(filter) => {
            let pending =
                pending && !replace_label_check(&mut filter.predicate, variable, label, checked);
            move_scan_label(&mut filter.input, variable, label, checked, pending);
        }
        LogicalOperator::NodeScan(scan) => scan.label = Some(label.to_string()),
        _ => {}
    }
}

/// Whether `predicate` reads a property of `variable` that a vector or text
/// index on one of `labels` covers.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
pub(super) fn search_index_covers(
    predicate: &LogicalExpression,
    variable: &str,
    labels: [&str; 2],
    store: &dyn GraphStoreSearch,
) -> bool {
    let covered = |property: &str| {
        labels.iter().any(|label| {
            #[cfg(feature = "vector-index")]
            if store.has_vector_index(label, property) {
                return true;
            }
            #[cfg(feature = "text-index")]
            if store.has_text_index(label, property) {
                return true;
            }
            false
        })
    };
    reads_property(predicate, variable, &covered)
}

/// Whether `expression` reads a property of `variable` for which `wanted`
/// holds. Looks through the operators and function calls an index search is
/// made of (see `filter_hybrid.rs`), not into subqueries.
#[cfg(any(feature = "vector-index", feature = "text-index"))]
fn reads_property(
    expression: &LogicalExpression,
    variable: &str,
    wanted: &dyn Fn(&str) -> bool,
) -> bool {
    match expression {
        LogicalExpression::Property {
            variable: read,
            property,
        } => read == variable && wanted(property),
        LogicalExpression::Binary { left, right, .. } => {
            reads_property(left, variable, wanted) || reads_property(right, variable, wanted)
        }
        LogicalExpression::Unary { operand, .. } => reads_property(operand, variable, wanted),
        LogicalExpression::FunctionCall { args, .. } | LogicalExpression::List(args) => {
            args.iter().any(|arg| reads_property(arg, variable, wanted))
        }
        _ => false,
    }
}

impl super::Planner {
    /// The copy of `filter` whose node scan reads the label with the fewest
    /// nodes (see [`with_smallest_scan_label`]), when this query may choose
    /// (see [`may_choose_scan_label`]).
    pub(super) fn scan_smallest_label(&self, filter: &FilterOp) -> Option<FilterOp> {
        if !self.reads_the_current_store() {
            return None;
        }
        with_smallest_scan_label(filter, self.store.as_ref())
    }

    /// Whether this query reads the store as it is now (see
    /// [`may_choose_scan_label`]): label counts and property indexes describe
    /// that state, so a read of a past epoch outside a transaction uses
    /// neither.
    pub(super) fn reads_the_current_store(&self) -> bool {
        let current_epoch = self
            .transaction_manager
            .as_ref()
            .map(|manager| manager.current_epoch());
        may_choose_scan_label(
            self.viewing_epoch,
            self.transaction_id,
            current_epoch,
            self.store.answers_at_view(),
        )
    }

    /// Plans a node scan operator.
    pub(super) fn plan_node_scan(
        &self,
        scan: &NodeScanOp,
    ) -> Result<(Box<dyn Operator>, Vec<String>)> {
        let scan_op = if let Some(label) = &scan.label {
            ScanOperator::with_label(Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>, label)
        } else {
            ScanOperator::new(Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>)
        };

        // Apply MVCC context if available
        let scan_operator: Box<dyn Operator> = Box::new(
            scan_op
                .with_transaction_context(self.viewing_epoch, self.transaction_id)
                .with_read_counter(self.read_counter()),
        );

        // If there's an input, chain operators with a nested loop join (cross join)
        if let Some(input) = &scan.input {
            let (mut input_op, mut input_columns) = self.plan_operator(input)?;

            // If the scan variable already exists in the input (e.g., from a
            // correlated ParameterScan), skip the redundant scan and reuse the
            // bound value. This avoids a cross product in CALL { WITH var MATCH (var)... }.
            // After a write (`... CREATE (h)-[:R]->() WITH h MATCH (h)-[:R]->(q)`)
            // the whole input is read first, so that the pattern from the
            // bound node (its labels and properties, the expands and paths
            // from it) sees what every row wrote, as the scan below does;
            // an input a clause read whole already is not read again (see
            // `after_write`).
            if input_columns.contains(&scan.variable) {
                if super::after_write::writes_pending(input) {
                    input_op = Box::new(EagerOperator::new(input_op));
                }
                // If the second MATCH clause has a label constraint, enforce it
                // as a filter on the already-bound variable.
                if let Some(label) = &scan.label {
                    let variable_columns: HashMap<String, usize> = input_columns
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.clone(), i))
                        .collect();
                    let filter_expr = FilterExpression::FunctionCall {
                        function: grafeo_core::execution::operators::BuiltinFunction::HasLabel,
                        args: vec![
                            FilterExpression::Variable(scan.variable.clone()),
                            FilterExpression::Literal(Value::String(label.as_str().into())),
                        ],
                    };
                    let predicate = ExpressionPredicate::new(
                        filter_expr,
                        variable_columns,
                        Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
                    )
                    .with_transaction_context(self.viewing_epoch, self.transaction_id);
                    let filtered = Box::new(FilterOperator::new(input_op, Box::new(predicate)));
                    return Ok((filtered, input_columns));
                }
                return Ok((input_op, input_columns));
            }

            // Build output schema: input columns + scan column
            let mut output_schema: Vec<LogicalType> =
                input_columns.iter().map(|_| LogicalType::Any).collect();
            output_schema.push(LogicalType::Node);

            // Add scan column to input columns
            input_columns.push(scan.variable.clone());

            // Use nested loop join to combine input rows with scanned nodes
            let mut join_op = NestedLoopJoinOperator::new(
                input_op,
                scan_operator,
                None, // No join condition (cross join)
                PhysicalJoinType::Cross,
                output_schema,
            );
            // A scan after a write (`INSERT ... WITH ... MATCH`) sees the write.
            if input.has_mutations() {
                join_op = join_op.with_left_first();
            }
            let join_op = Box::new(join_op);

            Ok((join_op, input_columns))
        } else {
            let columns = vec![scan.variable.clone()];
            Ok((scan_operator, columns))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::plan::UnaryOp;
    use crate::transaction::TransactionManager;
    use grafeo_common::types::{EpochId, TransactionId};
    use grafeo_core::graph::lpg::LpgStore;

    /// A read at a past epoch outside a transaction may use label scans and
    /// indexes only on a store that answers them at a view; a transaction
    /// and a read of now may on any store.
    #[test]
    fn a_past_epoch_read_chooses_labels_only_on_a_store_that_answers_at_a_view() {
        let (past, now) = (EpochId::new(3), Some(EpochId::new(19)));
        assert!(!may_choose_scan_label(past, None, now, false));
        assert!(may_choose_scan_label(past, None, now, true));
        assert!(may_choose_scan_label(
            past,
            Some(TransactionId::new(7)),
            now,
            false
        ));
        assert!(may_choose_scan_label(EpochId::new(19), None, now, false));
        assert!(may_choose_scan_label(past, None, None, false));
        assert!(
            !grafeo_core::graph::GraphStore::answers_at_view(&LpgStore::new().unwrap()),
            "LpgStore answers its lookups of now"
        );
    }

    /// 19 `Big` nodes, three of them also `Small`, and three `Same` nodes.
    fn store() -> Arc<LpgStore> {
        let store = Arc::new(LpgStore::new().unwrap());
        for i in 0..19 {
            if i < 3 {
                store.create_node(&["Big", "Small"]);
            } else {
                store.create_node(&["Big"]);
            }
        }
        for _ in 0..3 {
            store.create_node(&["Same"]);
        }
        store
    }

    fn has_label(variable: &str, label: &str) -> LogicalExpression {
        LogicalExpression::FunctionCall {
            callee: crate::query::plan::Callee::named("hasLabel"),
            name: "hasLabel".to_string(),
            args: vec![
                LogicalExpression::Variable(variable.to_string()),
                LogicalExpression::Literal(Value::String(label.into())),
            ],
            distinct: false,
        }
    }

    fn property_is_three(variable: &str) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(LogicalExpression::Property {
                variable: variable.to_string(),
                property: "x".to_string(),
            }),
            op: BinaryOp::Eq,
            right: Box::new(LogicalExpression::Literal(Value::Int64(3))),
        }
    }

    fn and(left: LogicalExpression, right: LogicalExpression) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(left),
            op: BinaryOp::And,
            right: Box::new(right),
        }
    }

    fn scan(label: &str) -> LogicalOperator {
        LogicalOperator::NodeScan(NodeScanOp {
            variable: "n".to_string(),
            label: Some(label.to_string()),
            input: None,
        })
    }

    fn filter(predicate: LogicalExpression, input: LogicalOperator) -> FilterOp {
        FilterOp {
            predicate,
            input: Box::new(input),
            pushdown_hint: None,
        }
    }

    /// The plan text of a filter: its predicates and the label its scan reads.
    fn shape(filter: &FilterOp) -> String {
        LogicalOperator::Filter(filter.clone()).explain_tree()
    }

    #[test]
    fn the_smallest_required_label_becomes_the_scan_label() {
        let store = store();
        let written = filter(
            and(has_label("n", "Small"), property_is_three("n")),
            scan("Big"),
        );
        let reordered = with_smallest_scan_label(&written, store.as_ref())
            .expect("Small has fewer nodes than Big");
        let expected = filter(
            and(has_label("n", "Big"), property_is_three("n")),
            scan("Small"),
        );
        assert_eq!(shape(&reordered), shape(&expected));
    }

    #[test]
    fn a_check_in_a_lower_filter_moves_too() {
        let store = store();
        let written = filter(
            property_is_three("n"),
            LogicalOperator::Filter(filter(has_label("n", "Small"), scan("Big"))),
        );
        let reordered = with_smallest_scan_label(&written, store.as_ref())
            .expect("Small has fewer nodes than Big");
        let expected = filter(
            property_is_three("n"),
            LogicalOperator::Filter(filter(has_label("n", "Big"), scan("Small"))),
        );
        assert_eq!(shape(&reordered), shape(&expected));
    }

    #[test]
    fn checks_that_are_not_required_of_the_scanned_node_do_not_count() {
        let store = store();
        let other_variable = filter(has_label("m", "Small"), scan("Big"));
        let in_or = filter(
            LogicalExpression::Binary {
                left: Box::new(has_label("n", "Small")),
                op: BinaryOp::Or,
                right: Box::new(property_is_three("n")),
            },
            scan("Big"),
        );
        let negated = filter(
            LogicalExpression::Unary {
                op: UnaryOp::Not,
                operand: Box::new(has_label("n", "Small")),
            },
            scan("Big"),
        );
        let in_xor = filter(
            LogicalExpression::Binary {
                left: Box::new(has_label("n", "Small")),
                op: BinaryOp::Xor,
                right: Box::new(property_is_three("n")),
            },
            scan("Big"),
        );
        let compared = filter(
            LogicalExpression::Binary {
                left: Box::new(has_label("n", "Small")),
                op: BinaryOp::Eq,
                right: Box::new(LogicalExpression::Literal(Value::Bool(true))),
            },
            scan("Big"),
        );
        for written in [other_variable, in_or, negated, in_xor, compared] {
            assert!(
                with_smallest_scan_label(&written, store.as_ref()).is_none(),
                "{}",
                shape(&written)
            );
        }
    }

    #[test]
    fn the_scan_label_stays_when_it_is_the_smallest_or_as_small() {
        let store = store();
        // Small (3) against Big (19), and Same (3) against Small (3).
        for written in [
            filter(has_label("n", "Big"), scan("Small")),
            filter(has_label("n", "Small"), scan("Same")),
        ] {
            assert!(
                with_smallest_scan_label(&written, store.as_ref()).is_none(),
                "{}",
                shape(&written)
            );
        }
    }

    #[test]
    fn of_labels_with_as_many_nodes_the_first_written_is_scanned() {
        let store = store();
        let written = filter(
            and(has_label("n", "Same"), has_label("n", "Small")),
            scan("Big"),
        );
        let reordered = with_smallest_scan_label(&written, store.as_ref())
            .expect("Same and Small have fewer nodes than Big");
        let expected = filter(
            and(has_label("n", "Big"), has_label("n", "Small")),
            scan("Same"),
        );
        assert_eq!(shape(&reordered), shape(&expected));
    }

    #[test]
    fn a_label_no_node_has_is_scanned() {
        let store = store();
        let written = filter(has_label("n", "Missing"), scan("Big"));
        let reordered =
            with_smallest_scan_label(&written, store.as_ref()).expect("no node has Missing");
        let expected = filter(has_label("n", "Big"), scan("Missing"));
        assert_eq!(shape(&reordered), shape(&expected));
    }

    /// Outside a transaction, a past epoch keeps the label as written; the
    /// current epoch, or a transaction, scans the smallest label.
    #[test]
    fn a_read_of_a_past_epoch_keeps_the_scan_label() {
        let store = store();
        let manager = Arc::new(TransactionManager::new());
        manager.sync_epoch(EpochId::new(3));
        let written = filter(has_label("n", "Small"), scan("Big"));
        let planner = |epoch: u64, transaction: Option<TransactionId>| {
            super::super::Planner::with_context(
                Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
                None,
                Arc::clone(&manager),
                transaction,
                EpochId::new(epoch),
            )
        };
        assert!(planner(1, None).scan_smallest_label(&written).is_none());
        assert!(planner(3, None).scan_smallest_label(&written).is_some());
        assert!(
            planner(1, Some(TransactionId::new(19)))
                .scan_smallest_label(&written)
                .is_some()
        );
    }
}
