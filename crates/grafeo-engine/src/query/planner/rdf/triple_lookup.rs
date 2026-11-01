//! A join of triple patterns with rows that bind their subject, predicate or
//! object (`VALUES ?p { ... } ?p v:name ?n`, LDBC IS1 on RDF), or of a UNION
//! of such patterns (an alternative path `?s (v:KNOWS|^v:KNOWS) ?f` is one):
//! each row looks its triples up through the store's indexes, in each
//! pattern, instead of a scan of every triple of the patterns joined with the
//! rows after. Past a limit of rows (see [`lookup_limit`]) the rows read so
//! far and the rest go to the hash join a plan without lookups runs.

use std::collections::VecDeque;
use std::sync::Arc;

use grafeo_common::types::{LogicalType, Value};
use grafeo_core::execution::DataChunk;
use grafeo_core::execution::operators::{Operator, OperatorError, OperatorResult};
use grafeo_core::graph::rdf::{Term, TriplePattern};

use super::RdfTripleScanOperator;
use super::intern_value;
use super::term_ids::{QueryTermIds, id_of};

/// The most rows that look their triples up one by one.
pub(super) const LOOKUP_ROWS: usize = 1024;

/// How many rows look their triples up before the join hashes the scan
/// instead, for patterns estimated to match `triples` each: a lookup costs
/// about as much as hashing 8 triples of the scan (1.3 against 0.18
/// microseconds, measured), so a lookup per row and pattern pays while the
/// rows are fewer than an eighth of a pattern's triples, and never more than
/// [`LOOKUP_ROWS`].
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

/// One pattern a row looks its triples up in.
pub(super) struct PatternLookup {
    /// The pattern's scan, with the positions the rows bind left open.
    pub(super) scan: RdfTripleScanOperator,
    /// `(position, input column)` of each position the rows bind (0
    /// subject, 1 predicate, 2 object).
    pub(super) bound: Vec<(usize, usize)>,
    /// Where each column of `scan` goes among the columns the join adds.
    pub(super) targets: Vec<usize>,
}

/// What [`TripleLookupOperator`] is built from. Its scans sit behind a box
/// and a `Vec`, so the planner's recursion over a chain of joins keeps a
/// small stack frame.
pub(super) struct TripleLookupParts {
    /// What the join hashes past the limit: the pattern's full scan, or the
    /// UNION of the patterns.
    pub(super) full: Box<dyn Operator>,
    /// The columns of `full`, and their types.
    pub(super) full_columns: Vec<String>,
    pub(super) full_types: Vec<LogicalType>,
    /// The patterns, a UNION's in the order of its branches.
    pub(super) lookups: Vec<PatternLookup>,
    /// The names of the columns the join adds: those of `full` the rows do
    /// not have, in its order.
    pub(super) added_names: Vec<String>,
    /// How many rows look up before the join hashes `full`.
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

/// The rows of `input`, each joined with the triples of each of its patterns
/// that hold its terms at the bound positions (see [`PatternLookup`]). The
/// output is the input's columns, then the patterns' other variables, unbound
/// in the rows of a pattern without them.
pub(super) struct TripleLookupOperator {
    input: Option<Box<dyn Operator>>,
    full: Option<Box<dyn Operator>>,
    full_columns: Vec<String>,
    full_types: Vec<LogicalType>,
    lookups: Vec<PatternLookup>,
    /// How many columns the join adds.
    added: usize,
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
            full_columns: parts.full_columns,
            full_types: parts.full_types,
            lookups: parts.lookups,
            added: parts.added_names.len(),
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

    /// The pattern of `lookup` with the terms of row `row` of `chunk` at its
    /// bound positions, `None` for a row with no term there or a term no
    /// triple holds (which no triple of the pattern joins).
    fn pattern(
        &self,
        lookup: &PatternLookup,
        chunk: &DataChunk,
        row: usize,
    ) -> Result<Option<TriplePattern>, OperatorError> {
        let mut pattern = lookup.scan.pattern.clone();
        for &(position, column) in &lookup.bound {
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
            let Some(entry) = id.and_then(|id| self.terms.entry(id)) else {
                return Ok(None);
            };
            let term: Term = Term::clone(entry.term());
            match position {
                0 => pattern.subject = Some(term),
                1 => pattern.predicate = Some(term),
                _ => pattern.object = Some(term),
            }
        }
        Ok(Some(pattern))
    }

    /// Looks up the triples of each row of `chunk` in each pattern, queueing
    /// the joined rows.
    fn look_up(&mut self, chunk: &DataChunk) -> Result<(), OperatorError> {
        for row in chunk.selected_indices() {
            let values: Vec<Value> = (0..chunk.column_count())
                .map(|column| {
                    chunk
                        .column(column)
                        .and_then(|c| c.get_value(row))
                        .unwrap_or(Value::Null)
                })
                .collect();
            let queued = self.pending.len();
            for lookup in &self.lookups {
                let Some(pattern) = self.pattern(lookup, chunk, row)? else {
                    continue;
                };
                let mut scan = lookup.scan.with_pattern(pattern);
                while let Some(found) = scan.next()? {
                    for at in found.selected_indices() {
                        let mut out = values.clone();
                        out.resize(values.len() + self.added, Value::Null);
                        for (column, &target) in lookup.targets.iter().enumerate() {
                            out[values.len() + target] = found
                                .column(column)
                                .and_then(|c| c.get_value(at))
                                .unwrap_or(Value::Null);
                        }
                        self.pending.push_back(out);
                    }
                }
            }
            // A row no triple joins: a left join keeps it with the patterns'
            // variables unbound, an inner join drops it.
            if self.pending.len() == queued && self.hash_join.left {
                let mut out = values;
                out.resize(out.len() + self.added, Value::Null);
                self.pending.push_back(out);
            }
        }
        Ok(())
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
        let (join, _, _) = if self.hash_join.left {
            crate::query::planner::common::build_left_join(
                input,
                full,
                &self.hash_join.columns,
                &self.full_columns,
                &self.hash_join.types,
                &self.full_types,
                |join| join,
            )
        } else {
            crate::query::planner::common::build_inner_join(
                input,
                full,
                &self.hash_join.columns,
                &self.full_columns,
                &self.hash_join.types,
                &self.full_types,
                self.hash_join.cardinalities,
            )
        };
        join
    }

    /// The next chunk of looked-up rows.
    fn emit(&mut self) -> DataChunk {
        let count = self.pending.len().min(self.chunk_size);
        let mut schema = self.hash_join.types.clone();
        schema.extend(std::iter::repeat_n(LogicalType::RdfTermId, self.added));
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
