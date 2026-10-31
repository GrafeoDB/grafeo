//! A join of a triple pattern with rows that bind its subject, predicate or
//! object (`VALUES ?p { ... } ?p v:name ?n`, LDBC IS1 on RDF): each row looks
//! its triples up through the store's indexes, instead of a scan of every
//! triple of the pattern joined with the rows after. Past a limit of rows
//! (see [`lookup_limit`]) the rows read so far and the rest go to the hash
//! join a plan without lookups runs.

use std::collections::VecDeque;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, TermId, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult};
use grafeo_core::graph::rdf::{Term, TriplePattern};
use smallvec::SmallVec;

use super::RdfTripleScanOperator;
use super::intern_value;
use super::term_ids::{QueryTermIds, id_of};

/// The most rows that look their triples up one by one.
pub(super) const LOOKUP_ROWS: usize = 1024;

/// How many rows look their triples up before the join hashes the scan
/// instead, for a pattern estimated to match `triples`: a lookup costs about
/// as much as hashing 8 triples of the scan (1.3 against 0.18 microseconds,
/// measured), so a lookup per row pays while the rows are fewer than an
/// eighth of the triples, and never more than [`LOOKUP_ROWS`].
pub(super) fn lookup_limit(triples: Option<f64>) -> usize {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "clamped to 1..=LOOKUP_ROWS before the cast"
    )]
    let limit = triples
        .map_or(LOOKUP_ROWS as f64, |triples| triples / 8.0)
        .clamp(1.0, LOOKUP_ROWS as f64) as usize;
    limit
}

/// What the planner decided: which positions of the pattern (0 subject, 1
/// predicate, 2 object) the rows bind, from which input columns, and where
/// the pattern's variables sit in the full scan's output.
pub(super) struct TripleLookupPlan {
    /// `(position, input column, column of the full scan)` of each bound
    /// position.
    pub(super) bound: Vec<(usize, usize, usize)>,
    /// The full scan's columns the rows do not bind, in order: the output
    /// adds them after the input's.
    pub(super) added: Vec<usize>,
}

/// What [`TripleLookupOperator`] is built from, boxed so the planner's
/// recursion over a chain of joins keeps a small stack frame.
pub(super) struct TripleLookupParts {
    pub(super) full: RdfTripleScanOperator,
    pub(super) lookup: RdfTripleScanOperator,
    pub(super) plan: TripleLookupPlan,
    /// The full scan's columns.
    pub(super) full_columns: Vec<String>,
    /// The names of the columns the join adds.
    pub(super) added_names: Vec<String>,
    /// How many rows look up before the join hashes the scan.
    pub(super) limit: usize,
}

/// The input's columns and types, and the estimates of the input and the
/// scan, for the hash join the operator hands over to.
pub(super) struct HashJoinInputs {
    pub(super) columns: Vec<String>,
    pub(super) types: Vec<LogicalType>,
    pub(super) cardinalities: Option<(f64, f64)>,
    /// Whether the join keeps a row no triple joins (an `OPTIONAL` pattern),
    /// with the pattern's variables unbound.
    pub(super) left: bool,
}

/// How the operator joins.
enum Mode {
    /// Reading the first rows, up to the limit.
    Reading,
    /// Each row looks its triples up.
    Lookup,
    /// The hash join of the rows with the scan.
    Hashed(Box<dyn Operator>),
}

/// The rows of `input`, each joined with the triples of its pattern that
/// hold its terms at the bound positions (see [`TripleLookupPlan`]). The
/// output is the input's columns, then the pattern's other variables.
pub(super) struct TripleLookupOperator {
    input: Option<Box<dyn Operator>>,
    full: Option<RdfTripleScanOperator>,
    lookup: RdfTripleScanOperator,
    plan: TripleLookupPlan,
    full_columns: Vec<String>,
    limit: usize,
    hash_join: HashJoinInputs,
    terms: Arc<QueryTermIds>,
    chunk_size: usize,
    mode: Mode,
    /// The chunks read before choosing, and how many rows they hold.
    buffered: Vec<DataChunk>,
    buffered_rows: usize,
    /// The output rows not handed out yet (lookups).
    pending: VecDeque<Vec<Value>>,
}

impl TripleLookupOperator {
    pub(super) fn new(
        input: Box<dyn Operator>,
        parts: TripleLookupParts,
        hash_join: HashJoinInputs,
        terms: Arc<QueryTermIds>,
        chunk_size: usize,
    ) -> Self {
        Self {
            input: Some(input),
            full: Some(parts.full),
            lookup: parts.lookup,
            plan: parts.plan,
            full_columns: parts.full_columns,
            limit: parts.limit,
            hash_join,
            terms,
            chunk_size: chunk_size.max(1),
            mode: Mode::Reading,
            buffered: Vec::new(),
            buffered_rows: 0,
            pending: VecDeque::new(),
        }
    }

    /// The ids of the terms of row `row` of `chunk` at the bound columns,
    /// `None` for a row with no term there (which no triple joins).
    fn key(
        &self,
        chunk: &DataChunk,
        row: usize,
    ) -> Result<Option<SmallVec<[TermId; 2]>>, OperatorError> {
        let mut key = SmallVec::new();
        for &(_, column, _) in &self.plan.bound {
            let value = chunk
                .column(column)
                .and_then(|c| c.get_value(row))
                .unwrap_or(Value::Null);
            // An id column holds ids; a value of another kind (a BIND) is
            // interned, as the scan's terms are.
            let id = match id_of(&value) {
                Some(id) => Some(id),
                None => id_of(&intern_value(&self.terms, Some(value))?),
            };
            let Some(id) = id else {
                return Ok(None);
            };
            key.push(id);
        }
        Ok(Some(key))
    }

    /// Looks up the triples of each row of `chunk`, queueing the joined
    /// rows.
    fn look_up(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        for row in chunk.selected_indices() {
            let Some(key) = self.key(chunk, row)? else {
                self.unmatched(chunk, row);
                continue;
            };
            let mut pattern: TriplePattern = self.lookup.pattern.clone();
            let mut known = true;
            for (&(position, _, _), &id) in self.plan.bound.iter().zip(&key) {
                let Some(entry) = self.terms.entry(id) else {
                    known = false;
                    break;
                };
                let term: Term = Term::clone(entry.term());
                match position {
                    0 => pattern.subject = Some(term),
                    1 => pattern.predicate = Some(term),
                    _ => pattern.object = Some(term),
                }
            }
            if !known {
                self.unmatched(chunk, row);
                continue;
            }
            let queued = self.pending.len();
            let mut scan = self.lookup.with_pattern(pattern);
            while let Some(found) = scan.next()? {
                for at in found.selected_indices() {
                    let mut out: Vec<Value> = (0..chunk.column_count())
                        .map(|column| {
                            chunk
                                .column(column)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null)
                        })
                        .collect();
                    out.extend((0..found.column_count()).map(|column| {
                        found
                            .column(column)
                            .and_then(|c| c.get_value(at))
                            .unwrap_or(Value::Null)
                    }));
                    self.pending.push_back(out);
                }
            }
            if self.pending.len() == queued {
                self.unmatched(chunk, row);
            }
        }
        Ok(())
    }

    /// Queues row `row` of `chunk`, which no triple joins, with the
    /// pattern's variables unbound, for a left join (an inner join drops it).
    fn unmatched(&mut self, chunk: &DataChunk, row: usize) {
        if !self.hash_join.left {
            return;
        }
        let mut out: Vec<Value> = (0..chunk.column_count())
            .map(|column| {
                chunk
                    .column(column)
                    .and_then(|c| c.get_value(row))
                    .unwrap_or(Value::Null)
            })
            .collect();
        out.extend(std::iter::repeat_n(Value::Null, self.plan.added.len()));
        self.pending.push_back(out);
    }

    /// Hands the rows read so far and the rest of the input to the hash
    /// join of a plan without lookups.
    fn hash(&mut self) -> Box<dyn Operator> {
        let input: Box<dyn Operator> = Box::new(Prepended {
            buffered: std::mem::take(&mut self.buffered).into(),
            rest: self.input.take().expect("the input until the join hashes"),
        });
        let full = self
            .full
            .take()
            .expect("the full scan until the join hashes");
        let full_types = vec![LogicalType::RdfTermId; self.full_columns.len()];
        let (join, _, _) = if self.hash_join.left {
            crate::query::planner::common::build_left_join(
                input,
                Box::new(full),
                &self.hash_join.columns,
                &self.full_columns,
                &self.hash_join.types,
                &full_types,
                |join| join,
            )
        } else {
            crate::query::planner::common::build_inner_join(
                input,
                Box::new(full),
                &self.hash_join.columns,
                &self.full_columns,
                &self.hash_join.types,
                &full_types,
                self.hash_join.cardinalities,
            )
        };
        join
    }

    /// The next chunk of looked-up rows.
    fn emit(&mut self) -> DataChunk {
        let count = self.pending.len().min(self.chunk_size);
        let mut schema = self.hash_join.types.clone();
        schema.extend(std::iter::repeat_n(
            LogicalType::RdfTermId,
            self.plan.added.len(),
        ));
        let mut out = DataChunk::with_capacity(&schema, count);
        for row in self.pending.drain(..count) {
            for (column, value) in row.into_iter().enumerate() {
                if let Some(target) = out.column_mut(column) {
                    target.push_value(value);
                }
            }
        }
        out.set_count(count);
        out
    }
}

impl Operator for TripleLookupOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            match &mut self.mode {
                Mode::Hashed(join) => return join.next(),
                Mode::Lookup => {
                    if !self.pending.is_empty() {
                        return Ok(Some(self.emit()));
                    }
                    let chunk = match self.buffered.pop() {
                        Some(chunk) => Some(chunk),
                        None => self
                            .input
                            .as_mut()
                            .map(|input| input.next())
                            .transpose()?
                            .flatten(),
                    };
                    let Some(chunk) = chunk else {
                        return Ok(None);
                    };
                    self.look_up(&chunk)?;
                }
                Mode::Reading => {
                    let chunk = self
                        .input
                        .as_mut()
                        .map(|input| input.next())
                        .transpose()?
                        .flatten();
                    let Some(chunk) = chunk else {
                        // The input ended within the limit: each row looks
                        // up, in the order it came.
                        self.buffered.reverse();
                        self.mode = Mode::Lookup;
                        continue;
                    };
                    self.buffered_rows += chunk.selected_indices().count();
                    self.buffered.push(chunk);
                    if self.buffered_rows > self.limit {
                        let join = self.hash();
                        self.mode = Mode::Hashed(join);
                    }
                }
            }
        }
    }

    fn reset(&mut self) {
        match &mut self.mode {
            Mode::Hashed(join) => join.reset(),
            Mode::Reading | Mode::Lookup => {
                if let Some(input) = self.input.as_mut() {
                    input.reset();
                }
                self.buffered.clear();
                self.buffered_rows = 0;
                self.pending.clear();
                self.mode = Mode::Reading;
            }
        }
    }

    fn name(&self) -> &'static str {
        "RdfTripleLookup"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// The chunks read before the join chose to hash, then the rest of the
/// input. A reset reads the input again from its start, which gives those
/// chunks again.
struct Prepended {
    buffered: VecDeque<DataChunk>,
    rest: Box<dyn Operator>,
}

impl Operator for Prepended {
    fn next(&mut self) -> OperatorResult {
        match self.buffered.pop_front() {
            Some(chunk) => Ok(Some(chunk)),
            None => self.rest.next(),
        }
    }

    fn reset(&mut self) {
        self.buffered.clear();
        self.rest.reset();
    }

    fn name(&self) -> &'static str {
        self.rest.name()
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}
