//! The condition of a SPARQL `OPTIONAL` whose `FILTER` holds an `EXISTS` or
//! `NOT EXISTS` (SPARQL 1.1 sec 18.2.2.6 and 18.6): decided per joined pair,
//! over the merged solution, so the subquery sees the variables bound before
//! the `OPTIONAL` as well as the optional side's.
//!
//! The subquery's result for a pair depends only on the values of the
//! subquery's variables that the pair binds, so the subquery runs once, as
//! it is, on the first pair, and keeps its rows on those variables; each
//! pair is then a lookup (a semi or anti join over the pairs, which DuckDB
//! calls a mark join). A variable the pair leaves unbound matches any value,
//! and so does one the subquery's row leaves unbound.

use grafeo_common::types::Value;
use grafeo_common::utils::hash::FxHashSet;
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{JoinCondition, Operator, OperatorError};

/// One `EXISTS` (or `NOT EXISTS`) test of the condition.
pub(super) struct ExistsTest {
    /// Whether it is a `NOT EXISTS`.
    negated: bool,
    /// For each variable the subquery shares with the pair: its column in
    /// the left row and in the right row, either of which may be absent.
    keys: Vec<(Option<usize>, Option<usize>)>,
    /// The column of each of those variables in the subquery's rows.
    subquery_columns: Vec<usize>,
    /// The subquery, until it ran, then its rows.
    state: parking_lot::Mutex<State>,
}

enum State {
    NotRun(Box<dyn Operator>),
    Ran(Matches),
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

/// The term id at `row` of `column`; `None` when it is unbound.
fn id_at(chunk: &DataChunk, column: usize, row: usize) -> Option<i64> {
    match chunk.column(column)?.get_value(row)? {
        Value::Int64(id) => Some(id),
        _ => None,
    }
}

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
            state: parking_lot::Mutex::new(State::NotRun(subquery)),
        }
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
        if let State::NotRun(subquery) = &mut *state {
            let mut matches = Matches::default();
            while let Some(chunk) = subquery.next()? {
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
            *state = State::Ran(matches);
        }
        let State::Ran(matches) = &*state else {
            unreachable!("the subquery ran above");
        };
        Ok(matches.contains(&key) != self.negated)
    }
}

/// The condition: every test holds, then the other conjuncts, if any.
pub(super) struct ExistsPairCondition {
    tests: Vec<ExistsTest>,
    rest: Option<Box<dyn JoinCondition>>,
}

impl ExistsPairCondition {
    pub(super) fn new(tests: Vec<ExistsTest>, rest: Option<Box<dyn JoinCondition>>) -> Self {
        Self { tests, rest }
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
            && !rest.evaluate(left_chunk, left_row, right_chunk, right_row)?
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
