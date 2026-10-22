//! Value joins: a filter over a node scan that runs for each row of its
//! input, whose conjuncts equate a value of the scanned node with a value of
//! the row (`MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath =
//! f.path`), becomes a hash join on those values (#455). The scan runs once
//! instead of once per row, and each row meets the nodes whose values `=` may
//! find equal to its own; the filter above the join decides every row.

use std::cell::Cell;
use std::collections::HashSet;

use grafeo_common::types::LogicalType;
use grafeo_core::execution::operators::{
    ExpressionPredicate, FilterOperator, HashJoinOperator, JoinType as PhysicalJoinType, Operator,
    ProjectExpr, ProjectOperator, ScanOperator,
};
use grafeo_core::graph::GraphStoreSearch;

use super::seek::{CheckedScan, checked_scan, split_conjuncts, value_variables};
use super::{Arc, HashMap, Result};
use crate::query::optimizer::movable_variables;
use crate::query::plan::{BinaryOp, FilterOp, LogicalExpression, LogicalOperator};

/// A key of a value join: a conjunct `scanned = row` of its filter.
pub(crate) struct JoinKey<'a> {
    /// The value of the scanned node: reads it and no other variable.
    pub scanned: &'a LogicalExpression,
    /// The value of the input row: reads variables the input binds, and not
    /// the scanned node.
    pub row: &'a LogicalExpression,
}

/// A filter that runs as a value join, with the scan it replaces.
pub(crate) struct ValueJoin<'a> {
    /// The scan and the checks of the scanned node between it and the filter.
    pub checked: CheckedScan<'a>,
    /// The input the scan runs for: the rows the join probes with.
    pub input: &'a LogicalOperator,
    /// The keys, in the order the filter's conjuncts are written.
    pub keys: Vec<JoinKey<'a>>,
}

#[cfg(test)]
thread_local! {
    /// Off in the tests that compare a value join with the scan per row.
    static ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Whether value joins are planned: always, except in tests that turn them
/// off on their thread to compare with the scan per row.
fn enabled() -> bool {
    #[cfg(test)]
    {
        ENABLED.get()
    }
    #[cfg(not(test))]
    {
        true
    }
}

/// Whether the statement being planned writes, from its root: the operator
/// `plan_operator` plans first. While the root is planned the cell holds it,
/// and it is cleared again when the root is planned, so a planner that plans
/// another statement looks again.
pub(super) struct StatementScope<'a>(Option<&'a Cell<Option<bool>>>);

impl<'a> StatementScope<'a> {
    /// Records whether `op` writes when it is the root (the cell is empty);
    /// an operator below the root leaves the record as it is.
    pub(super) fn enter(writes: &'a Cell<Option<bool>>, op: &LogicalOperator) -> Self {
        if writes.get().is_some() {
            return Self(None);
        }
        writes.set(Some(op.has_mutations()));
        Self(Some(writes))
    }
}

impl Drop for StatementScope<'_> {
    fn drop(&mut self) {
        if let Some(writes) = self.0 {
            writes.set(None);
        }
    }
}

/// The value join `filter` runs as, if any. The filter sits on a node scan
/// with an input, directly or above checks of the scanned node alone (see
/// [`checked_scan`]), and has a key: a conjunct `a = b` (either way round)
/// where `a` reads the scanned node and no other variable, `b` reads
/// variables the input binds and not the scanned node, both are values a
/// seek may evaluate (see [`value_variables`]), and each reads a property:
/// the node itself and its `id()` are no keys. Not when the input writes
/// (the scan reads its whole input first, to see the writes; see
/// `plan_node_scan`), nor when the filter has a subquery or a function whose
/// value changes between calls: such a filter runs on each pair of the scan
/// per row. Nor when the statement writes anywhere (`statement_writes`): a
/// write above the join can change the keys the scan per row reads when the
/// later input rows arrive, which a hash join would have keyed once.
///
/// The planner tries the seek of the filter first (see `plan_filter`): an
/// indexed key or an ID keeps the lookup per row.
pub(crate) fn value_join(filter: &FilterOp, statement_writes: bool) -> Option<ValueJoin<'_>> {
    if statement_writes || !enabled() {
        return None;
    }
    let checked = checked_scan(filter)?;
    let input = checked.scan.input.as_deref()?;
    if input.has_mutations() || movable_variables(&filter.predicate).is_none() {
        return None;
    }
    // A subquery's input that binds the node already is checked, not scanned
    // (see `plan_node_scan`).
    let bound = input.bound_variables(None)?;
    let variable = &checked.scan.variable;
    if bound.contains(variable) {
        return None;
    }
    let mut conjuncts = Vec::new();
    split_conjuncts(&filter.predicate, &mut conjuncts);
    let keys: Vec<JoinKey<'_>> = conjuncts
        .into_iter()
        .filter_map(|conjunct| join_key(conjunct, variable, &bound))
        .collect();
    if keys.is_empty() {
        return None;
    }
    Some(ValueJoin {
        checked,
        input,
        keys,
    })
}

/// The key a conjunct gives the scan of `variable` over rows that bind
/// `bound` (see [`value_join`]).
fn join_key<'a>(
    conjunct: &'a LogicalExpression,
    variable: &str,
    bound: &HashSet<String>,
) -> Option<JoinKey<'a>> {
    let LogicalExpression::Binary {
        left,
        op: BinaryOp::Eq,
        right,
    } = conjunct
    else {
        return None;
    };
    let key = |scanned: &'a LogicalExpression, row: &'a LogicalExpression| {
        let reads_scanned = value_variables(scanned)?;
        let reads_row = value_variables(row)?;
        let keyed = reads_scanned.len() == 1
            && reads_scanned.contains(variable)
            && !reads_row.is_empty()
            && !reads_row.contains(variable)
            && reads_row.is_subset(bound)
            && reads_property(scanned)
            && reads_property(row);
        keyed.then_some(JoinKey { scanned, row })
    };
    key(left, right).or_else(|| key(right, left))
}

/// Whether `expr` reads a property, in the expressions [`value_variables`]
/// accepts.
fn reads_property(expr: &LogicalExpression) -> bool {
    match expr {
        LogicalExpression::Property { .. } => true,
        LogicalExpression::Binary { left, right, .. } => {
            reads_property(left) || reads_property(right)
        }
        LogicalExpression::Unary { operand, .. } => reads_property(operand),
        LogicalExpression::IndexAccess { base, index } => {
            reads_property(base) || reads_property(index)
        }
        LogicalExpression::MapAccess { base, .. } => reads_property(base),
        LogicalExpression::List(items) => items.iter().any(reads_property),
        LogicalExpression::Map(entries) => entries.iter().any(|(_, value)| reads_property(value)),
        LogicalExpression::FunctionCall { args, .. } => args.iter().any(reads_property),
        _ => false,
    }
}

impl super::Planner {
    /// Plans a filter that runs as a value join (see [`value_join`]): the
    /// nodes of the scan that pass the checks, each with the scanned values
    /// of the keys, are hashed on them; the input's rows, each with the row
    /// values of the keys, meet the nodes whose keys are equal (see
    /// [`HashJoinOperator::with_value_equality_keys`]), in the order of the
    /// scan. Without the key columns, the rows are those of the scan per row
    /// that may pass, in its order, and the filter on top decides each.
    pub(super) fn try_plan_filter_with_value_join(
        &self,
        filter: &FilterOp,
    ) -> Result<Option<(Box<dyn Operator>, Vec<String>)>> {
        let Some(ValueJoin {
            checked:
                CheckedScan {
                    scan,
                    operator,
                    checks,
                },
            input,
            keys,
        }) = value_join(filter, self.statement_writes.get().unwrap_or(true))
        else {
            return Ok(None);
        };

        let (probe, input_columns) = self.plan_operator(input)?;
        // PROFILE entries for the scan and the checks the join absorbs, in
        // the order the plan's tree lists them (children first).
        self.record_absorbed_scan_entry("HashJoin", operator);
        for (check, _) in &checks {
            self.record_absorbed_scan_entry("Filter", check);
        }
        // `value_join` leaves an input that binds the node to the scan.
        debug_assert!(
            !input_columns.contains(&scan.variable),
            "the input of a value join binds the scanned node"
        );

        let store = Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>;
        let scan_op = match &scan.label {
            Some(label) => ScanOperator::with_label(Arc::clone(&store), label),
            None => ScanOperator::new(Arc::clone(&store)),
        }
        .with_transaction_context(self.viewing_epoch, self.transaction_id);
        let scan_columns = vec![scan.variable.clone()];
        let mut build: Box<dyn Operator> = Box::new(scan_op);
        let checks = checks.iter().map(|(_, predicate)| (*predicate).clone());
        if let Some(check) = checks.reduce(and) {
            let predicate = self.value_join_predicate(check, &scan_columns)?;
            build = Box::new(FilterOperator::new(build, Box::new(predicate)));
        }
        let build =
            self.with_key_columns(build, &scan_columns, keys.iter().map(|key| key.scanned))?;
        let probe = self.with_key_columns(probe, &input_columns, keys.iter().map(|key| key.row))?;

        let width = input_columns.len();
        let key_count = keys.len();
        let mut schema = vec![LogicalType::Any; width + key_count];
        schema.push(LogicalType::Node);
        schema.extend(std::iter::repeat_n(LogicalType::Any, key_count));
        let join = HashJoinOperator::new(
            probe,
            build,
            (width..width + key_count).collect(),
            (1..=key_count).collect(),
            PhysicalJoinType::Inner,
            schema,
        )
        .with_value_equality_keys();
        // The columns of the scan per row: the input's, then the node.
        let kept = (0..width)
            .chain([width + key_count])
            .map(ProjectExpr::Column)
            .collect();
        let rows = ProjectOperator::new(Box::new(join), kept, vec![LogicalType::Any; width + 1]);

        let mut columns = input_columns;
        columns.push(scan.variable.clone());
        let predicate = self.value_join_predicate(filter.predicate.clone(), &columns)?;
        Ok(Some((
            Box::new(FilterOperator::new(Box::new(rows), Box::new(predicate))),
            columns,
        )))
    }

    /// `input` with a column per key value, appended to its `columns`, each
    /// evaluated as a filter evaluates it (the same store, transaction and
    /// session), so the join compares the values the filter compares.
    fn with_key_columns<'a>(
        &self,
        input: Box<dyn Operator>,
        columns: &[String],
        keys: impl Iterator<Item = &'a LogicalExpression>,
    ) -> Result<Box<dyn Operator>> {
        let variable_columns = variable_columns(columns);
        let mut projections: Vec<ProjectExpr> =
            (0..columns.len()).map(ProjectExpr::Column).collect();
        for key in keys {
            projections.push(ProjectExpr::Expression {
                expr: self.convert_expression(key)?,
                variable_columns: variable_columns.clone(),
            });
        }
        let types = vec![LogicalType::Any; projections.len()];
        Ok(Box::new(
            ProjectOperator::with_store(
                input,
                projections,
                types,
                Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
            )
            .with_transaction_context(self.viewing_epoch, self.transaction_id)
            .with_session_context(self.session_context.clone()),
        ))
    }

    /// The predicate of a filter over rows with `columns`.
    fn value_join_predicate(
        &self,
        predicate: LogicalExpression,
        columns: &[String],
    ) -> Result<ExpressionPredicate> {
        Ok(ExpressionPredicate::new(
            self.convert_expression(&predicate)?,
            variable_columns(columns),
            Arc::clone(&self.store) as Arc<dyn GraphStoreSearch>,
        )
        .with_transaction_context(self.viewing_epoch, self.transaction_id)
        .with_session_context(self.session_context.clone()))
    }
}

/// The column of each variable, as a filter over `columns` reads them.
fn variable_columns(columns: &[String]) -> HashMap<String, usize> {
    columns
        .iter()
        .enumerate()
        .map(|(i, name)| (name.clone(), i))
        .collect()
}

fn and(left: LogicalExpression, right: LogicalExpression) -> LogicalExpression {
    LogicalExpression::Binary {
        left: Box::new(left),
        op: BinaryOp::And,
        right: Box::new(right),
    }
}

#[cfg(all(test, feature = "gql", feature = "cypher"))]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::{Date, Duration, PropertyKey, Time, Value, ZonedDatetime};

    use super::{ENABLED, value_join};
    use crate::GrafeoDB;
    use crate::query::plan::{
        BinaryOp, CreateNodeOp, FilterOp, LogicalExpression, LogicalOperator, NodeScanOp,
    };

    fn property(variable: &str, name: &str) -> LogicalExpression {
        LogicalExpression::Property {
            variable: variable.to_string(),
            property: name.to_string(),
        }
    }

    fn binary(
        left: LogicalExpression,
        op: BinaryOp,
        right: LogicalExpression,
    ) -> LogicalExpression {
        LogicalExpression::Binary {
            left: Box::new(left),
            op,
            right: Box::new(right),
        }
    }

    /// `predicate` over a scan of `t:TypeDefinition` for each row of
    /// `input`, by default a scan of `f:File`.
    fn filter_over(predicate: LogicalExpression, input: Option<LogicalOperator>) -> FilterOp {
        let files = LogicalOperator::NodeScan(NodeScanOp {
            variable: "f".to_string(),
            label: Some("File".to_string()),
            input: None,
        });
        FilterOp {
            predicate,
            pushdown_hint: None,
            input: Box::new(LogicalOperator::NodeScan(NodeScanOp {
                variable: "t".to_string(),
                label: Some("TypeDefinition".to_string()),
                input: Some(Box::new(input.unwrap_or(files))),
            })),
        }
    }

    /// The keys a value join of `filter` has, as text, or `None`.
    fn keys(filter: &FilterOp, statement_writes: bool) -> Option<Vec<String>> {
        value_join(filter, statement_writes).map(|join| {
            join.keys
                .iter()
                .map(|key| format!("{:?} = {:?}", key.scanned, key.row))
                .collect()
        })
    }

    /// A key and the conditions that keep the scan per row: a statement
    /// that writes, a write in the input, a function whose value changes
    /// between calls beside the key, and keys that read no property.
    #[test]
    fn the_guards_of_a_value_join() {
        let key = binary(property("t", "k"), BinaryOp::Eq, property("f", "k"));
        let joined = filter_over(key.clone(), None);
        assert_eq!(
            keys(&joined, false),
            Some(vec![format!(
                "{:?} = {:?}",
                property("t", "k"),
                property("f", "k")
            )])
        );
        assert_eq!(keys(&joined, true), None, "a statement that writes");

        let random = LogicalExpression::FunctionCall {
            name: "rand".to_string(),
            args: Vec::new(),
            distinct: false,
        };
        let volatile = binary(
            key.clone(),
            BinaryOp::And,
            binary(
                random,
                BinaryOp::Lt,
                LogicalExpression::Literal(Value::Int64(2)),
            ),
        );
        assert_eq!(keys(&filter_over(volatile, None), false), None, "rand()");

        let writes = LogicalOperator::CreateNode(CreateNodeOp {
            variable: "f".to_string(),
            labels: vec!["File".to_string()],
            properties: Vec::new(),
            input: None,
        });
        assert_eq!(
            keys(&filter_over(key, Some(writes)), false),
            None,
            "a write in the input"
        );

        for (scanned, row) in [
            (
                LogicalExpression::Variable("t".to_string()),
                LogicalExpression::Variable("f".to_string()),
            ),
            (LogicalExpression::Id("t".to_string()), property("f", "k")),
            (property("t", "k"), LogicalExpression::Id("f".to_string())),
        ] {
            let filter = filter_over(binary(scanned, BinaryOp::Eq, row), None);
            assert_eq!(keys(&filter, false), None, "{:?}", filter.predicate);
        }
    }

    const LANGUAGES: [&str; 2] = ["gql", "cypher"];

    /// The rows of `query`, planned with value joins, or with the scan per
    /// row (the plan before them).
    fn rows(db: &GrafeoDB, language: &str, query: &str, joined: bool) -> Vec<Vec<Value>> {
        ENABLED.set(joined);
        let result = db.session().execute_language(query, language, None);
        ENABLED.set(true);
        result
            .unwrap_or_else(|error| panic!("{language}: {query}: {error}"))
            .rows()
            .to_vec()
    }

    /// Asserts that `query` runs as a hash join on `keys` (EXPLAIN says so)
    /// and returns the rows of the scan per row, in the same order; returns
    /// them.
    fn assert_same_rows(db: &GrafeoDB, language: &str, query: &str, keys: &str) -> Vec<Vec<Value>> {
        let plan = rows(db, language, &format!("EXPLAIN {query}"), true);
        let plan = plan[0][0].as_str().unwrap_or_default().to_string();
        assert!(
            plan.contains(&format!(" [hash join: {keys}]")),
            "{language}: {query}\n{plan}"
        );
        let scanned = rows(db, language, &format!("EXPLAIN {query}"), false);
        assert!(
            !scanned[0][0]
                .as_str()
                .unwrap_or_default()
                .contains("[hash join"),
            "{language}: {query} without value joins"
        );
        let joined = rows(db, language, query, true);
        assert_eq!(
            joined,
            rows(db, language, query, false),
            "{language}: {query}"
        );
        joined
    }

    fn list(items: Vec<Value>) -> Value {
        Value::List(items.into())
    }

    fn map(key: &str, value: Value) -> Value {
        Value::Map(Arc::new(BTreeMap::from([(PropertyKey::new(key), value)])))
    }

    fn time(text: &str) -> Value {
        Value::Time(Time::parse(text).expect("a time"))
    }

    fn zoned(text: &str) -> Value {
        Value::ZonedDatetime(ZonedDatetime::parse(text).expect("a zoned datetime"))
    }

    /// Property values that `=` compares in more than one way: integers and
    /// floats near zero, near 2 and past 2^53, strings that read as numbers,
    /// nulls in lists, maps, times and datetimes at one instant in different
    /// offsets, durations, bytes and vectors with a negative zero.
    fn values() -> Vec<Value> {
        let two_to_53 = 1_i64 << 53;
        let mut values: Vec<Value> = [0, 1, -1, 2, 5, 1000, two_to_53, two_to_53 + 1, i64::MAX]
            .map(Value::Int64)
            .into();
        values.extend(
            [
                1.0,
                0.0,
                -0.0,
                1e-17,
                -1e-17,
                0.5,
                2.0,
                1.999_999_999_999_999_8,
                0.999_999_999_999_999_9,
                -0.999_999_999_999_999_9,
                5.0,
                1000.0,
                9_007_199_254_740_992.0,
                f64::NAN,
                f64::INFINITY,
            ]
            .map(Value::Float64),
        );
        values.extend(
            [
                "5", "05", "+5", "5.0", "-0", "1e3", "NaN", "inf", "abc", "", " 5",
            ]
            .map(Value::from),
        );
        values.extend([
            Value::Bool(true),
            Value::Bool(false),
            list(vec![Value::Int64(1)]),
            list(vec![Value::Float64(1.0)]),
            list(vec![Value::from("1")]),
            list(vec![Value::Null]),
            list(vec![Value::Int64(1), Value::Null]),
            list(Vec::new()),
            map("a", Value::Int64(1)),
            map("a", Value::Float64(1.0)),
            Value::Date(Date::parse("2026-10-07").expect("a date")),
            Value::Date(Date::parse("2026-10-08").expect("a date")),
            time("14:00:00+01:00"),
            time("13:00:00Z"),
            time("13:00:00"),
            zoned("2026-10-07T14:00:00+01:00"),
            zoned("2026-10-07T13:00:00Z"),
            Value::Duration(Duration::new(0, 1, 0)),
            Value::Duration(Duration::new(0, 0, 86_400_000_000_000)),
            Value::Bytes(vec![1_u8, 2].into()),
            Value::Vector(vec![0.0_f32].into()),
            Value::Vector(vec![-0.0_f32].into()),
        ]);
        values
    }

    /// A file and a type definition per value, in turn, as `k` with their
    /// number `i` (and `j`, `i` modulo 3), then one of each without `k`.
    fn valued() -> (GrafeoDB, Vec<Value>) {
        let db = GrafeoDB::new_in_memory();
        let values = values();
        for (i, value) in (0_i64..).zip(&values) {
            for label in ["File", "TypeDefinition"] {
                db.create_node_with_props(
                    &[label],
                    [
                        ("i", Value::Int64(i)),
                        ("j", Value::Int64(i % 3)),
                        ("k", value.clone()),
                    ],
                )
                .unwrap();
            }
        }
        for label in ["File", "TypeDefinition"] {
            db.create_node_with_props(&[label], [("i", Value::Int64(-1)), ("j", Value::Int64(2))])
                .unwrap();
        }
        (db, values)
    }

    /// The number of the value `value` in `values`.
    fn number(values: &[Value], value: &Value) -> Value {
        let at = values
            .iter()
            .position(|candidate| format!("{candidate:?}") == format!("{value:?}"))
            .unwrap_or_else(|| panic!("{value:?} is one of the values"));
        Value::Int64(i64::try_from(at).unwrap())
    }

    /// Each pair of values, a property of each node, meets as `=` decides:
    /// the rows are those of the scan per row, in the same order, also for
    /// nodes without the property.
    #[test]
    fn every_pair_of_values_joins_as_the_scan_per_row_does() {
        let (db, values) = valued();
        let query = "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.k = f.k RETURN f.i, t.i";
        for language in LANGUAGES {
            let joined = assert_same_rows(&db, language, query, "t.k = f.k");
            // The pairs that make the keys more than exact are there, and
            // pairs `=` tells apart are not.
            let pair = |a: &Value, b: &Value| vec![number(&values, a), number(&values, b)];
            let two_to_53 = 1_i64 << 53;
            for (a, b) in [
                (Value::from("5"), Value::Int64(5)),
                (Value::from("+5"), Value::Float64(5.0)),
                (Value::from("1e3"), Value::Float64(1000.0)),
                (Value::Float64(1e-17), Value::Int64(0)),
                (Value::Float64(-0.0), Value::Float64(0.0)),
                (Value::Float64(0.999_999_999_999_999_9), Value::Int64(1)),
                (Value::Float64(0.999_999_999_999_999_9), Value::Float64(1.0)),
                (Value::Float64(-0.999_999_999_999_999_9), Value::Int64(-1)),
                (
                    Value::Int64(two_to_53 + 1),
                    Value::Float64(9_007_199_254_740_992.0),
                ),
                (list(vec![Value::Null]), list(vec![Value::Null])),
                (
                    list(vec![Value::from("1")]),
                    list(vec![Value::Float64(1.0)]),
                ),
                (map("a", Value::Float64(1.0)), map("a", Value::Int64(1))),
                (time("14:00:00+01:00"), time("13:00:00Z")),
                (
                    zoned("2026-10-07T13:00:00Z"),
                    zoned("2026-10-07T14:00:00+01:00"),
                ),
                (
                    Value::Vector(vec![-0.0_f32].into()),
                    Value::Vector(vec![0.0_f32].into()),
                ),
            ] {
                assert!(joined.contains(&pair(&a, &b)), "{language}: {a:?} = {b:?}");
            }
            for (a, b) in [
                (Value::from("5.0"), Value::Int64(5)),
                (Value::Float64(f64::NAN), Value::Float64(f64::NAN)),
                (Value::Float64(2.0), Value::Float64(1.999_999_999_999_999_8)),
                (time("13:00:00"), time("13:00:00Z")),
                (Value::Int64(two_to_53), Value::Int64(two_to_53 + 1)),
            ] {
                assert!(
                    !joined.contains(&pair(&a, &b)),
                    "{language}: {a:?} <> {b:?}"
                );
            }
            assert!(
                !joined.iter().any(|row| row.contains(&Value::Int64(-1))),
                "{language}: a missing `k` meets nothing"
            );
        }
    }

    /// Keys made of more than one conjunct or computed from the values, and
    /// conjuncts beside the keys, on the same pairs.
    #[test]
    fn more_keys_and_computed_keys_join_as_the_scan_per_row_does() {
        let (db, _) = valued();
        for (query, keys) in [
            (
                "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.k = f.k AND f.j = t.j \
                 RETURN f.i, t.i",
                "t.k = f.k, t.j = f.j",
            ),
            (
                "MATCH (f:File) MATCH (t:TypeDefinition) WHERE [t.k] = [f.k] RETURN f.i, t.i",
                "[t.k] = [f.k]",
            ),
            (
                "MATCH (f:File) MATCH (t:TypeDefinition) WHERE toString(t.k) = toString(f.k) \
                 RETURN f.i, t.i",
                "toString(t.k) = toString(f.k)",
            ),
            (
                "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.k = f.k AND t.i <> f.i \
                 RETURN f.i, t.i",
                "t.k = f.k",
            ),
            (
                "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.j + 1 = f.j \
                 AND (t.k = f.k OR t.i < 3) RETURN f.i, t.i",
                "t.j Add 1 = f.j",
            ),
        ] {
            for language in LANGUAGES {
                let joined = assert_same_rows(&db, language, query, keys);
                assert!(!joined.is_empty(), "{language}: {query}");
            }
        }
    }

    /// The query of #455 on a graph of a few hundred nodes: each `Model`
    /// scan is a hash join, and the rows are those of the scans per row.
    #[test]
    fn the_issue_query_joins_as_the_scans_per_row_do() {
        let db = GrafeoDB::new_in_memory();
        let names = ["Alix", "Gus", "Vincent", "Jules", "Mia"];
        let mut functions = Vec::new();
        for i in 0..60_i64 {
            let id = if i % 7 == 0 {
                Value::Int64(i)
            } else {
                Value::from(format!("f{i}").as_str())
            };
            functions.push(
                db.create_node_with_props(
                    &["Function"],
                    [
                        ("id", id),
                        ("name", Value::from(names[usize::try_from(i).unwrap() % 5])),
                        ("active", Value::Bool(i % 5 != 0)),
                    ],
                )
                .unwrap(),
            );
        }
        for (i, &from) in functions.iter().enumerate() {
            for step in [1, 3, 11] {
                db.create_edge(from, functions[(i * step + 1) % functions.len()], "CALLS")
                    .unwrap();
            }
        }
        for i in 0..90_i64 {
            let target = i % 70;
            let identifier = if target % 7 == 0 {
                Value::Float64(f64::from(i32::try_from(target).unwrap()))
            } else {
                Value::from(format!("f{target}").as_str())
            };
            db.create_node_with_props(
                &["Model"],
                [("source_identifier", identifier), ("n", Value::Int64(i))],
            )
            .unwrap();
        }
        let one_where = "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
             MATCH (model_src:Model), (model_tgt:Model) \
             WHERE graph_src.active = true AND graph_tgt.active = true \
               AND model_src.source_identifier = graph_src.id \
               AND model_tgt.source_identifier = graph_tgt.id \
             RETURN graph_src.id, graph_tgt.id, model_src.n, model_tgt.n";
        for (language, query) in [
            ("gql", one_where),
            ("cypher", one_where),
            (
                "cypher",
                "MATCH (graph_src)-[edge:CALLS]->(graph_tgt) \
                 WHERE graph_src.active = true AND graph_tgt.active = true \
                 MATCH (model_src:Model), (model_tgt:Model) \
                 WHERE model_src.source_identifier = graph_src.id \
                   AND model_tgt.source_identifier = graph_tgt.id \
                 RETURN graph_src.id, graph_tgt.id, model_src.n, model_tgt.n",
            ),
        ] {
            let joined = assert_same_rows(
                &db,
                language,
                query,
                "model_tgt.source_identifier = graph_tgt.id",
            );
            assert!(joined.len() > 100, "{language}: {} rows", joined.len());
            let plan = rows(&db, language, &format!("EXPLAIN {query}"), true);
            assert!(
                plan[0][0]
                    .as_str()
                    .unwrap_or_default()
                    .contains("[hash join: model_src.source_identifier = graph_src.id]"),
                "{language}"
            );
        }
    }

    /// In an open transaction, nodes created, labels and keys changed in it
    /// on both sides join as the scan per row joins them.
    #[test]
    fn a_transaction_joins_as_the_scan_per_row_does() {
        let (db, _) = valued();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        for change in [
            "CREATE (:File {i: 100, k: 5.0}), (:TypeDefinition {i: 100, k: '5'})",
            "MATCH (t:TypeDefinition) WHERE t.i % 4 = 1 REMOVE t:TypeDefinition",
            "MATCH (f:File) WHERE f.i % 5 = 2 SET f.k = 1000",
            "MATCH (f:File) WHERE f.i % 6 = 3 REMOVE f:File SET f:Draft",
            "MATCH (d:Draft) WHERE d.i % 12 = 3 SET d:TypeDefinition",
        ] {
            session.execute_cypher(change).unwrap();
        }
        let query = "MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.k = f.k RETURN f.i, t.i";
        for language in LANGUAGES {
            ENABLED.set(false);
            let scanned = session.execute_language(query, language, None).unwrap();
            ENABLED.set(true);
            let joined = session.execute_language(query, language, None).unwrap();
            assert_eq!(joined.rows(), scanned.rows(), "{language}");
            assert!(
                joined
                    .rows()
                    .contains(&vec![Value::Int64(100), Value::Int64(100)]),
                "{language}"
            );
        }
        session.rollback().unwrap();
    }
}
