//! Node scan planning, and the choice of the label a node scan reads.

use super::{
    Arc, BinaryOp, EpochId, ExpressionPredicate, FilterExpression, FilterOp, FilterOperator,
    GraphStoreSearch, HashMap, LogicalExpression, LogicalOperator, LogicalType,
    NestedLoopJoinOperator, NodeScanOp, Operator, PhysicalJoinType, Result, ScanOperator,
    TransactionId, Value,
};

/// The copy of `filter` whose node scan reads the label with the fewest
/// nodes, or `None` when the scan reads that label already.
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
    let scan_label = scan.label.as_deref()?;
    let mut checks = Vec::new();
    for predicate in &predicates {
        label_checks(predicate, &scan.variable, &mut checks);
    }
    if checks.is_empty() {
        return None;
    }
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
    if smallest == scan_label {
        return None;
    }
    #[cfg(any(feature = "vector-index", feature = "text-index"))]
    if predicates.iter().any(|predicate| {
        search_index_covers(predicate, &scan.variable, [scan_label, smallest], store)
    }) {
        return None;
    }
    let variable = scan.variable.clone();
    let (smallest, scan_label) = (smallest.to_string(), scan_label.to_string());
    let mut reordered = filter.clone();
    let checked = replace_label_check(&mut reordered.predicate, &variable, &smallest, &scan_label);
    move_scan_label(
        &mut reordered.input,
        &variable,
        &smallest,
        &scan_label,
        !checked,
    );
    Some(reordered)
}

/// Whether a query that reads at `viewing_epoch`, in `transaction_id`, may
/// scan another of a pattern's labels than the one written
/// ([`with_smallest_scan_label`]). Not at a past epoch outside a transaction
/// (`current_epoch` is the transaction manager's, `None` without one): a label
/// scan finds the nodes that have the label now, while the check per row
/// reads the labels a node had at the epoch. The same rule decides whether a
/// query may look values up in a property index, which holds the values of
/// now.
pub(crate) fn may_choose_scan_label(
    viewing_epoch: EpochId,
    transaction_id: Option<TransactionId>,
    current_epoch: Option<EpochId>,
) -> bool {
    transaction_id.is_some() || current_epoch.is_none_or(|current| viewing_epoch >= current)
}

/// The label of a `hasLabel(variable, 'Label')` check.
fn label_check<'a>(expression: &'a LogicalExpression, variable: &str) -> Option<&'a str> {
    let LogicalExpression::FunctionCall { name, args, .. } = expression else {
        return None;
    };
    if !name.eq_ignore_ascii_case("hasLabel") {
        return None;
    }
    match args.as_slice() {
        [
            LogicalExpression::Variable(checked),
            LogicalExpression::Literal(Value::String(label)),
        ] if checked == variable => Some(label.as_str()),
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
        if let LogicalExpression::FunctionCall { args, .. } = predicate {
            args[1] = LogicalExpression::Literal(Value::String(checked.into()));
        }
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
        may_choose_scan_label(self.viewing_epoch, self.transaction_id, current_epoch)
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
        let scan_operator: Box<dyn Operator> =
            Box::new(scan_op.with_transaction_context(self.viewing_epoch, self.transaction_id));

        // If there's an input, chain operators with a nested loop join (cross join)
        if let Some(input) = &scan.input {
            let (input_op, mut input_columns) = self.plan_operator(input)?;

            // If the scan variable already exists in the input (e.g., from a
            // correlated ParameterScan), skip the redundant scan and reuse the
            // bound value. This avoids a cross product in CALL { WITH var MATCH (var)... }.
            if input_columns.contains(&scan.variable) {
                // If the second MATCH clause has a label constraint, enforce it
                // as a filter on the already-bound variable.
                if let Some(label) = &scan.label {
                    let variable_columns: HashMap<String, usize> = input_columns
                        .iter()
                        .enumerate()
                        .map(|(i, name)| (name.clone(), i))
                        .collect();
                    let filter_expr = FilterExpression::FunctionCall {
                        name: "hasLabel".to_string(),
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
