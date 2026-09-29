//! Node seek: the nodes an input row's key selects, found by ID or through a
//! property index instead of a scan (an index nested-loop join).

use std::sync::Arc;

use grafeo_common::types::{EpochId, LogicalType, NodeId, TransactionId, Value};

use super::{ExpressionPredicate, Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::graph::GraphStoreSearch;

/// What a seek key selects nodes by.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SeekKey {
    /// The node ID: `id(n) = key`.
    Id,
    /// A property value, through the property index: `n.property = key`.
    Property(String),
}

/// For each input row, finds the nodes its key selects and emits the row
/// joined with each of them.
///
/// The key is evaluated per row, so it may come from the row itself (an
/// `UNWIND` element, a variable of an earlier `MATCH`) as well as from a
/// constant. With `list` set the key is a list and each element is looked up
/// (`IN`). A NULL or NaN key selects nothing.
///
/// Only nodes visible to the query that carry the scan's label are emitted.
/// The seek produces candidates for the filter it replaces, which stays on top
/// and decides: a candidate whose value no longer matches (the index holds the
/// latest value) is dropped there, so results equal those of a scan. Equality
/// across types (`1 = 1.0`, `42 = '42'`) finds the canonical forms of the
/// key: the integer, the float and the decimal string of a number.
pub struct NodeSeekOperator {
    store: Arc<dyn GraphStoreSearch>,
    input: Box<dyn Operator>,
    key: SeekKey,
    /// Evaluates the key for an input row.
    key_expression: ExpressionPredicate,
    list: bool,
    label: Option<String>,
    viewing_epoch: Option<EpochId>,
    transaction_id: Option<TransactionId>,
    chunk_capacity: usize,
    /// The input chunk being joined, its next row, and the row whose
    /// matches are being emitted.
    current_input: Option<DataChunk>,
    next_row: usize,
    current_row: usize,
    matches: Vec<NodeId>,
    next_match: usize,
}

impl NodeSeekOperator {
    /// Creates a seek of `key` (evaluated by `key_expression` over the input
    /// rows) for nodes with `label`.
    #[must_use]
    pub fn new(
        store: Arc<dyn GraphStoreSearch>,
        input: Box<dyn Operator>,
        key: SeekKey,
        key_expression: ExpressionPredicate,
        label: Option<String>,
    ) -> Self {
        Self {
            store,
            input,
            key,
            key_expression,
            list: false,
            label,
            viewing_epoch: None,
            transaction_id: None,
            chunk_capacity: 2048,
            current_input: None,
            next_row: 0,
            current_row: 0,
            matches: Vec::new(),
            next_match: 0,
        }
    }

    /// Looks up each element of a list key (`n.p IN key`).
    #[must_use]
    pub fn with_list_key(mut self) -> Self {
        self.list = true;
        self
    }

    /// Sets the transaction context for MVCC visibility.
    #[must_use]
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// The visible nodes with the label that the key of `row` selects, in ID
    /// order.
    fn find(&self, chunk: &DataChunk, row: usize) -> Vec<NodeId> {
        let Some(key) = self.key_expression.eval_at(chunk, row) else {
            return Vec::new();
        };
        let keys = if self.list {
            match key {
                Value::List(items) => items.iter().cloned().collect(),
                _ => Vec::new(),
            }
        } else {
            vec![key]
        };
        let mut found = Vec::new();
        for key in &keys {
            match &self.key {
                SeekKey::Id => found.extend(node_id(key)),
                SeekKey::Property(property) => {
                    for probe in equal_forms(key) {
                        found.extend(self.store.find_nodes_by_property(property, &probe));
                    }
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found.retain(|&id| self.visible_with_label(id));
        found
    }

    fn visible_with_label(&self, id: NodeId) -> bool {
        let node = match (self.viewing_epoch, self.transaction_id) {
            (Some(epoch), Some(transaction_id)) => {
                self.store.get_node_versioned(id, epoch, transaction_id)
            }
            (Some(epoch), None) => self.store.get_node_at_epoch(id, epoch),
            (None, _) => self.store.get_node(id),
        };
        node.is_some_and(|node| {
            self.label
                .as_deref()
                .is_none_or(|label| node.has_label(label))
        })
    }

    /// Joins rows of `chunk`, from `next_row` on, with their matches until
    /// the output is full or the chunk is done.
    fn fill(&mut self, chunk: &DataChunk) -> Option<DataChunk> {
        let columns = chunk.column_count();
        let mut schema: Vec<LogicalType> = (0..columns)
            .map(|i| {
                chunk
                    .column(i)
                    .map_or(LogicalType::Any, |c| c.data_type().clone())
            })
            .collect();
        schema.push(LogicalType::Node);
        let mut output = DataChunk::with_capacity(&schema, self.chunk_capacity);
        let mut count = 0;
        while count < self.chunk_capacity {
            if self.next_match >= self.matches.len() {
                if self.next_row >= chunk.row_count() {
                    break;
                }
                self.current_row = self.next_row;
                self.next_row += 1;
                self.matches = self.find(chunk, self.current_row);
                self.next_match = 0;
                continue;
            }
            for column in 0..columns {
                if let (Some(source), Some(target)) =
                    (chunk.column(column), output.column_mut(column))
                {
                    source.copy_row_to(self.current_row, target);
                }
            }
            if let Some(target) = output.column_mut(columns) {
                target.push_node_id(self.matches[self.next_match]);
            }
            self.next_match += 1;
            count += 1;
        }
        (count > 0).then(|| {
            output.set_count(count);
            output
        })
    }
}

impl Operator for NodeSeekOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            let chunk = match self.current_input.take() {
                Some(chunk) => chunk,
                None => {
                    let Some(mut chunk) = self.input.next()? else {
                        return Ok(None);
                    };
                    chunk.flatten();
                    self.next_row = 0;
                    self.matches.clear();
                    self.next_match = 0;
                    chunk
                }
            };
            let output = self.fill(&chunk);
            let done = self.next_row >= chunk.row_count() && self.next_match >= self.matches.len();
            if !done {
                self.current_input = Some(chunk);
            }
            if output.is_some() {
                return Ok(output);
            }
        }
    }

    fn reset(&mut self) {
        self.input.reset();
        self.current_input = None;
        self.next_row = 0;
        self.current_row = 0;
        self.matches.clear();
        self.next_match = 0;
    }

    fn name(&self) -> &'static str {
        "NodeSeek"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// The node ID a key names: a non-negative integer, or a float equal to one.
fn node_id(key: &Value) -> Option<NodeId> {
    let id = match key {
        Value::Int64(id) => *id,
        Value::Float64(f) => integral(*f)?,
        _ => return None,
    };
    u64::try_from(id).ok().map(NodeId::new)
}

/// The values a key finds through the index: the key itself and the other
/// canonical forms that compare equal to it. NULL and NaN find nothing.
fn equal_forms(key: &Value) -> Vec<Value> {
    match key {
        Value::Null => Vec::new(),
        Value::Float64(f) if f.is_nan() => Vec::new(),
        Value::Int64(i) => vec![
            Value::Int64(*i),
            Value::Float64(*i as f64),
            Value::from(i.to_string()),
        ],
        Value::Float64(f) => {
            let mut forms = vec![Value::Float64(*f), Value::from(f.to_string())];
            if let Some(i) = integral(*f) {
                forms.push(Value::Int64(i));
            }
            forms
        }
        Value::String(s) => {
            let mut forms = vec![key.clone()];
            if let Ok(i) = s.parse::<i64>() {
                forms.push(Value::Int64(i));
            }
            if let Ok(f) = s.parse::<f64>()
                && !f.is_nan()
            {
                forms.push(Value::Float64(f));
            }
            forms
        }
        other => vec![other.clone()],
    }
}

/// The integer a float equals, if it is integral and within the `i64` range.
fn integral(f: f64) -> Option<i64> {
    // 2^63 as f64 is exact; every integral float in [-2^63, 2^63) fits i64.
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if f.fract() != 0.0 || !(-LIMIT..LIMIT).contains(&f) {
        return None;
    }
    #[allow(
        clippy::cast_possible_truncation,
        reason = "f is integral and within [-2^63, 2^63), checked above"
    )]
    Some(f as i64)
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::operators::{FilterExpression, SingleRowOperator};
    use crate::graph::lpg::LpgStore;
    use std::collections::HashMap;

    fn seek(store: &Arc<LpgStore>, key: SeekKey, value: Value, label: Option<&str>) -> Vec<NodeId> {
        let dyn_store: Arc<dyn GraphStoreSearch> = Arc::clone(store) as Arc<dyn GraphStoreSearch>;
        let key_expression = ExpressionPredicate::new(
            FilterExpression::Literal(value),
            HashMap::new(),
            Arc::clone(&dyn_store),
        );
        let mut op = NodeSeekOperator::new(
            dyn_store,
            Box::new(SingleRowOperator::new()),
            key,
            key_expression,
            label.map(ToString::to_string),
        );
        let mut found = Vec::new();
        while let Some(chunk) = op.next().unwrap() {
            let column = chunk.column(chunk.column_count() - 1).unwrap();
            found.extend((0..chunk.row_count()).filter_map(|row| column.get_node_id(row)));
        }
        found
    }

    #[test]
    fn seeks_by_id_and_label() {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        let city = store.create_node(&["City"]);
        let id = |node: NodeId| Value::Int64(node.as_u64().cast_signed());

        assert_eq!(seek(&store, SeekKey::Id, id(alix), None), [alix]);
        assert_eq!(seek(&store, SeekKey::Id, id(city), Some("Person")), []);
        assert_eq!(seek(&store, SeekKey::Id, Value::Int64(-1), None), []);
        assert_eq!(seek(&store, SeekKey::Id, Value::Null, None), []);
        store.delete_node(alix);
        assert_eq!(seek(&store, SeekKey::Id, id(alix), None), []);
    }

    #[test]
    fn seeks_the_equal_forms_of_a_property_value() {
        let store = Arc::new(LpgStore::new().unwrap());
        store.create_property_index("id");
        let as_int = store.create_node(&["Doc"]);
        store.set_node_property(as_int, "id", Value::Int64(42));
        let as_float = store.create_node(&["Doc"]);
        store.set_node_property(as_float, "id", Value::Float64(42.0));
        let as_string = store.create_node(&["Doc"]);
        store.set_node_property(as_string, "id", Value::from("42"));
        let other = store.create_node(&["Doc"]);
        store.set_node_property(other, "id", Value::Int64(7));

        let id = || SeekKey::Property("id".to_string());
        let all = vec![as_int, as_float, as_string];
        assert_eq!(seek(&store, id(), Value::Int64(42), None), all);
        assert_eq!(seek(&store, id(), Value::Float64(42.0), None), all);
        assert_eq!(seek(&store, id(), Value::from("42"), None), all);
        assert_eq!(seek(&store, id(), Value::Float64(f64::NAN), None), []);
        assert_eq!(seek(&store, id(), Value::Null, None), []);
    }
}
