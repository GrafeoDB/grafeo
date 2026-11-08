//! The condition of a SPARQL `OPTIONAL` whose `FILTER` holds an `EXISTS` or
//! `NOT EXISTS` (SPARQL 1.1 sec 18.2.2.6 and 18.6): decided per joined pair,
//! over the merged solution, so the subquery sees the variables bound before
//! the `OPTIONAL` as well as the optional side's.
//!
//! Where a run without the row answers (see [`super::exists_read`]), the
//! subquery's result for a pair depends only on the values of the subquery's
//! variables that the pair binds, so the subquery runs once, as it is, on the
//! first pair, and keeps its rows on those variables; each pair is then a
//! lookup (a semi or anti join over the pairs, which DuckDB calls a mark join).
//! A variable the pair leaves unbound matches any value, and so does one the
//! subquery's row leaves unbound. Otherwise it runs per pair
//! ([`ExistsPerRow`]).
//!
//! An `EXISTS` elsewhere in an expression is the same test over one row, its
//! answer a column the expression reads ([`ExistsMarkOperator`]).

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_common::utils::hash::{FxHashMap, FxHashSet};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{JoinCondition, Operator, OperatorError};

use super::exists_per_row::ExistsPerRow;
use super::term_ids::QueryTermIds;
use super::{chunk_with_column, intern_value};

/// One `EXISTS` (or `NOT EXISTS`) test of the condition.
pub(super) struct ExistsTest {
    /// Whether it is a `NOT EXISTS`.
    negated: bool,
    /// For each variable the subquery shares with the pair: its column in
    /// the left row and in the right row, either of which may be absent.
    keys: Vec<(Option<usize>, Option<usize>)>,
    /// The column of each of those variables in the subquery's rows.
    subquery_columns: Vec<usize>,
    /// The subquery, and its rows once it ran.
    state: parking_lot::Mutex<State>,
}

struct State {
    subquery: Box<dyn Operator>,
    /// The subquery's rows, `None` until it runs (again after a reset).
    matches: Option<Matches>,
}

/// The subquery's rows on the shared variables: those with every variable
/// bound in a set, the others in a list.
#[derive(Default)]
struct Matches {
    bound: FxHashSet<Vec<i64>>,
    partly_bound: Vec<Vec<Option<i64>>>,
}

impl Matches {
    /// Whether a row of the subquery is compatible with `key`: equal where
    /// both are bound.
    fn contains(&self, key: &[Option<i64>]) -> bool {
        let compatible = |row: &[Option<i64>]| {
            row.iter()
                .zip(key)
                .all(|(row, key)| row.is_none() || key.is_none() || row == key)
        };
        if let Some(bound) = key.iter().copied().collect::<Option<Vec<i64>>>() {
            if self.bound.contains(&bound) {
                return true;
            }
        } else if self.bound.iter().any(|row| {
            row.iter()
                .zip(key)
                .all(|(row, key)| key.is_none_or(|key| *row == key))
        }) {
            return true;
        }
        self.partly_bound.iter().any(|row| compatible(row))
    }
}

use super::id_at;

impl ExistsTest {
    /// The test of `subquery`, whose rows have the shared variables at
    /// `subquery_columns`; `keys` gives each one's columns in the pair.
    pub(super) fn new(
        subquery: Box<dyn Operator>,
        negated: bool,
        keys: Vec<(Option<usize>, Option<usize>)>,
        subquery_columns: Vec<usize>,
    ) -> Self {
        Self {
            negated,
            keys,
            subquery_columns,
            state: parking_lot::Mutex::new(State {
                subquery,
                matches: None,
            }),
        }
    }

    /// Runs the subquery again at the next test: a reset of the plan above
    /// may give it other rows.
    fn reset(&mut self) {
        let state = self.state.get_mut();
        state.subquery.reset();
        state.matches = None;
    }

    fn holds(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> Result<bool, OperatorError> {
        let key: Vec<Option<i64>> = self
            .keys
            .iter()
            .map(|(left, right)| {
                left.and_then(|column| id_at(left_chunk, column, left_row))
                    .or_else(|| right.and_then(|column| id_at(right_chunk, column, right_row)))
            })
            .collect();
        let mut state = self.state.lock();
        let state = &mut *state;
        if state.matches.is_none() {
            let mut matches = Matches::default();
            while let Some(chunk) = state.subquery.next()? {
                for row in chunk.selected_indices() {
                    let values: Vec<Option<i64>> = self
                        .subquery_columns
                        .iter()
                        .map(|&column| id_at(&chunk, column, row))
                        .collect();
                    match values.iter().copied().collect::<Option<Vec<i64>>>() {
                        Some(bound) => {
                            matches.bound.insert(bound);
                        }
                        None => matches.partly_bound.push(values),
                    }
                }
            }
            state.matches = Some(matches);
        }
        let matches = state.matches.as_ref().expect("the subquery ran above");
        Ok(matches.contains(&key) != self.negated)
    }
}

/// An `EXISTS` with conjuncts that read the row: the subquery runs once, its rows kept per key, and
/// a row holds when one of the rows of its key passes the conjuncts with the row's values.
pub(super) struct RowCheck {
    /// For each variable the subquery shares with the row: its column in each.
    keys: Vec<(usize, usize)>,
    /// The conjuncts, over the row followed by the subquery's row.
    check: Box<dyn JoinCondition>,
    subquery: Box<dyn Operator>,
    /// Whether the subquery ran (since the last reset), filling the fields below.
    ran: bool,
    chunks: Vec<DataChunk>,
    /// By key, each row of the subquery as its chunk and its row there.
    rows: FxHashMap<Vec<Option<i64>>, Vec<(usize, usize)>>,
    /// Whether a key of `rows` leaves a variable unbound, which a lookup would not find.
    partly_bound: bool,
}

impl RowCheck {
    pub(super) fn new(
        subquery: Box<dyn Operator>,
        keys: Vec<(usize, usize)>,
        check: Box<dyn JoinCondition>,
    ) -> Self {
        Self {
            keys,
            check,
            subquery,
            ran: false,
            chunks: Vec::new(),
            rows: FxHashMap::default(),
            partly_bound: false,
        }
    }

    /// Runs the subquery again at the next test: a reset of the plan above
    /// may give it other rows.
    fn reset(&mut self) {
        self.subquery.reset();
        self.ran = false;
        self.chunks.clear();
        self.rows.clear();
        self.partly_bound = false;
    }

    fn run(&mut self) -> Result<(), OperatorError> {
        self.ran = true;
        while let Some(chunk) = self.subquery.next()? {
            let at = self.chunks.len();
            for row in chunk.selected_indices() {
                let key: Vec<Option<i64>> = self
                    .keys
                    .iter()
                    .map(|&(_, column)| id_at(&chunk, column, row))
                    .collect();
                self.partly_bound |= key.iter().any(Option::is_none);
                self.rows.entry(key).or_default().push((at, row));
            }
            self.chunks.push(chunk);
        }
        Ok(())
    }

    fn passes(
        &self,
        input: &DataChunk,
        row: usize,
        rows: &[(usize, usize)],
    ) -> Result<bool, OperatorError> {
        for &(chunk, kept) in rows {
            if let Some(chunk) = self.chunks.get(chunk)
                && self.check.evaluate(input, row, chunk, kept)?
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn holds(&mut self, input: &DataChunk, row: usize) -> Result<bool, OperatorError> {
        if !self.ran {
            self.run()?;
        }
        let key: Vec<Option<i64>> = self
            .keys
            .iter()
            .map(|&(column, _)| id_at(input, column, row))
            .collect();
        if !self.partly_bound && key.iter().all(Option::is_some) {
            return match self.rows.get(&key) {
                Some(rows) => self.passes(input, row, rows),
                None => Ok(false),
            };
        }
        for (found, rows) in &self.rows {
            let compatible = found
                .iter()
                .zip(&key)
                .all(|(found, key)| found.is_none() || key.is_none() || found == key);
            if compatible && self.passes(input, row, rows)? {
                return Ok(true);
            }
        }
        Ok(false)
    }
}

/// How an [`ExistsMarkOperator`] answers per row.
pub(super) enum MarkTest {
    /// By a lookup among the subquery's rows.
    Lookup(ExistsTest),
    /// By a check over them.
    PerRow(RowCheck),
    /// By running the pattern for the row.
    EachRow(Box<ExistsPerRow>),
}

impl MarkTest {
    /// Forgets what the test learned from its subquery, for a plan run again.
    fn reset(&mut self) {
        match self {
            Self::Lookup(test) => test.reset(),
            Self::PerRow(test) => test.reset(),
            Self::EachRow(test) => test.reset(),
        }
    }
}

/// Every row of its input with a column more: whether an `EXISTS` holds for it, as a boolean term.
/// The expression that holds the `EXISTS` reads the column by a name no query can write, so nothing
/// else sees it.
pub(super) struct ExistsMarkOperator {
    child: Box<dyn Operator>,
    test: MarkTest,
    terms: Arc<QueryTermIds>,
}

impl ExistsMarkOperator {
    pub(super) fn new(child: Box<dyn Operator>, test: MarkTest, terms: Arc<QueryTermIds>) -> Self {
        Self { child, test, terms }
    }
}

impl Operator for ExistsMarkOperator {
    fn next(&mut self) -> Result<Option<DataChunk>, OperatorError> {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };
        let rows: Vec<usize> = input.selected_indices().collect();
        let no = intern_value(&self.terms, Some(Value::Bool(false)))?;
        let yes = intern_value(&self.terms, Some(Value::Bool(true)))?;
        let mut marks = Vec::with_capacity(rows.len());
        for &row in &rows {
            let holds = match &mut self.test {
                MarkTest::Lookup(test) => test.holds(&input, row, &input, row)?,
                MarkTest::PerRow(test) => test.holds(&input, row)?,
                MarkTest::EachRow(test) => test.holds(&input, row, &input, row)?,
            };
            marks.push(if holds { yes.clone() } else { no.clone() });
        }
        Ok(Some(chunk_with_column(&input, &rows, marks)))
    }

    fn reset(&mut self) {
        self.child.reset();
        self.test.reset();
    }

    fn name(&self) -> &'static str {
        "ExistsMark"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// How a test of a joined pair answers.
pub(super) enum PairTest {
    /// By a lookup among the subquery's rows.
    Lookup(ExistsTest),
    /// By running the pattern for the pair.
    EachPair {
        test: parking_lot::Mutex<Box<ExistsPerRow>>,
        negated: bool,
    },
}

impl PairTest {
    /// The test that runs `test` per pair.
    pub(super) fn each_pair(test: ExistsPerRow, negated: bool) -> Self {
        Self::EachPair {
            test: parking_lot::Mutex::new(Box::new(test)),
            negated,
        }
    }

    fn holds(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> Result<bool, OperatorError> {
        match self {
            Self::Lookup(test) => test.holds(left_chunk, left_row, right_chunk, right_row),
            Self::EachPair { test, negated } => {
                Ok(test
                    .lock()
                    .holds(left_chunk, left_row, right_chunk, right_row)?
                    != *negated)
            }
        }
    }
}

/// The condition: every test holds, then the other conjuncts, if any, which read the answers of
/// `marks` as columns after the right row's.
pub(super) struct ExistsPairCondition {
    tests: Vec<PairTest>,
    marks: Vec<PairTest>,
    rest: Option<Box<dyn JoinCondition>>,
    terms: Arc<QueryTermIds>,
}

impl ExistsPairCondition {
    pub(super) fn new(
        tests: Vec<PairTest>,
        marks: Vec<PairTest>,
        rest: Option<Box<dyn JoinCondition>>,
        terms: Arc<QueryTermIds>,
    ) -> Self {
        Self {
            tests,
            marks,
            rest,
            terms,
        }
    }

    /// The other conjuncts, over the right row with the marks after it.
    fn rest_with_marks(
        &self,
        rest: &dyn JoinCondition,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> Result<bool, OperatorError> {
        let mut marked: Option<DataChunk> = None;
        for mark in &self.marks {
            let holds = mark.holds(left_chunk, left_row, right_chunk, right_row)?;
            let answer = intern_value(&self.terms, Some(Value::Bool(holds)))?;
            marked = Some(match &marked {
                Some(chunk) => chunk_with_column(chunk, &[0], vec![answer]),
                None => chunk_with_column(right_chunk, &[right_row], vec![answer]),
            });
        }
        match &marked {
            Some(chunk) => rest.evaluate(left_chunk, left_row, chunk, 0),
            None => rest.evaluate(left_chunk, left_row, right_chunk, right_row),
        }
    }
}

impl JoinCondition for ExistsPairCondition {
    fn evaluate(
        &self,
        left_chunk: &DataChunk,
        left_row: usize,
        right_chunk: &DataChunk,
        right_row: usize,
    ) -> Result<bool, OperatorError> {
        if let Some(rest) = &self.rest
            && !self.rest_with_marks(rest.as_ref(), left_chunk, left_row, right_chunk, right_row)?
        {
            return Ok(false);
        }
        for test in &self.tests {
            if !test.holds(left_chunk, left_row, right_chunk, right_row)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use grafeo_common::types::LogicalType;

    /// A subquery of one column whose rows are whatever `ids` holds when it
    /// runs: a reset runs it again.
    struct Rows {
        ids: Arc<parking_lot::Mutex<Vec<i64>>>,
        done: bool,
    }

    impl Operator for Rows {
        fn next(&mut self) -> Result<Option<DataChunk>, OperatorError> {
            if self.done {
                return Ok(None);
            }
            self.done = true;
            Ok(Some(ids_chunk(&self.ids.lock())))
        }

        fn reset(&mut self) {
            self.done = false;
        }

        fn name(&self) -> &'static str {
            "Rows"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    fn ids_chunk(ids: &[i64]) -> DataChunk {
        let mut chunk = DataChunk::with_capacity(&[LogicalType::Int64], ids.len());
        for &id in ids {
            chunk
                .column_mut(0)
                .expect("one column")
                .push_value(Value::Int64(id));
        }
        chunk.set_count(ids.len());
        chunk
    }

    /// Every pair joins: the test is the key's lookup alone.
    struct Always;

    impl JoinCondition for Always {
        fn evaluate(
            &self,
            _left_chunk: &DataChunk,
            _left_row: usize,
            _right_chunk: &DataChunk,
            _right_row: usize,
        ) -> Result<bool, OperatorError> {
            Ok(true)
        }
    }

    /// Whether `test` holds for a row whose one shared variable is `id`.
    fn holds(test: &mut MarkTest, id: i64) -> bool {
        let row = ids_chunk(&[id]);
        match test {
            MarkTest::Lookup(test) => test.holds(&row, 0, &row, 0).unwrap(),
            MarkTest::PerRow(test) => test.holds(&row, 0).unwrap(),
            MarkTest::EachRow(_) => unreachable!("not built here"),
        }
    }

    /// A test keeps its subquery's rows between rows, and runs the subquery
    /// again after a reset: a plan run again may give it other rows.
    #[test]
    fn a_reset_runs_the_subquery_again() {
        for lookup in [true, false] {
            let ids = Arc::new(parking_lot::Mutex::new(vec![1]));
            let subquery = Box::new(Rows {
                ids: Arc::clone(&ids),
                done: false,
            });
            let mut test = if lookup {
                MarkTest::Lookup(ExistsTest::new(
                    subquery,
                    false,
                    vec![(Some(0), None)],
                    vec![0],
                ))
            } else {
                MarkTest::PerRow(RowCheck::new(subquery, vec![(0, 0)], Box::new(Always)))
            };
            assert!(holds(&mut test, 1), "lookup {lookup}: the first run");
            assert!(!holds(&mut test, 2), "lookup {lookup}: the first run");

            *ids.lock() = vec![2];
            assert!(!holds(&mut test, 2), "lookup {lookup}: kept until a reset");

            test.reset();
            assert!(holds(&mut test, 2), "lookup {lookup}: run again");
            assert!(
                !holds(&mut test, 1),
                "lookup {lookup}: the old rows are gone"
            );
        }
    }
}
