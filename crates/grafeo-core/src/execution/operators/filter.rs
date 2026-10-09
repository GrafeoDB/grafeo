//! Filter operator for applying predicates.

use super::project::{
    field_type, holds_entities, item_type_at, list_type, reversed_type, slice_type, struct_type,
};
use super::{Operator, OperatorResult};
use crate::execution::{ChunkZoneHints, DataChunk, SelectionVector, ValueVector};
use crate::graph::Direction;
use crate::graph::GraphStoreSearch;
use crate::graph::lpg::{Edge, Node};
use grafeo_common::types::{
    EdgeId, EpochId, HashableValue, LogicalType, NodeId, PropertyKey, PropertyMap, TransactionId,
    Value,
};
#[cfg(feature = "regex")]
use regex::Regex;
#[cfg(all(feature = "regex-lite", not(feature = "regex")))]
use regex_lite::Regex;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// Extracts a required integer field from a temporal constructor map.
///
/// Returns `Some(value)` when the key is present and holds a valid integer
/// (or a finite float that can be truncated to i64). Returns `None` when
/// the key is absent. Rejects non-finite floats (NaN, Infinity) by
/// returning `Some(None)` through `map_int_checked`, which lets callers
/// distinguish "missing" from "invalid".
fn map_int(m: &BTreeMap<PropertyKey, Value>, key: &str) -> Option<i64> {
    match map_int_checked(m, key) {
        MapIntResult::Absent => None,
        MapIntResult::Valid(v) => Some(v),
        // Invalid floats (NaN, Infinity): treat as missing so callers
        // that use map_int for required fields will fail with None.
        MapIntResult::Invalid => None,
    }
}

/// Result of extracting an integer field, distinguishing absent from invalid.
enum MapIntResult {
    /// Key was not present in the map.
    Absent,
    /// Key was present with a valid integer value.
    Valid(i64),
    /// Key was present but the value is not convertible (NaN, Infinity, wrong type).
    Invalid,
}

/// Extracts an integer field with three-way result: absent, valid, or invalid.
fn map_int_checked(m: &BTreeMap<PropertyKey, Value>, key: &str) -> MapIntResult {
    match m.get(&PropertyKey::from(key)) {
        None => MapIntResult::Absent,
        Some(Value::Int64(v)) => MapIntResult::Valid(*v),
        Some(Value::Float64(f)) => {
            let f = *f;
            if f.is_nan() || f.is_infinite() || f > i64::MAX as f64 || f < i64::MIN as f64 {
                MapIntResult::Invalid
            } else {
                // reason: intentional truncation of finite float to integer for temporal field extraction
                #[allow(clippy::cast_possible_truncation)]
                MapIntResult::Valid(f as i64)
            }
        }
        Some(_) => MapIntResult::Invalid,
    }
}

/// Extracts an optional integer field from a temporal constructor map,
/// returning a default value when the key is absent. Returns `None` when
/// the key is present but holds an invalid value (NaN, Infinity).
fn map_int_or(m: &BTreeMap<PropertyKey, Value>, key: &str, default: i64) -> Option<i64> {
    match map_int_checked(m, key) {
        MapIntResult::Absent => Some(default),
        MapIntResult::Valid(v) => Some(v),
        MapIntResult::Invalid => None,
    }
}

/// A predicate for filtering rows.
pub trait Predicate: Send + Sync {
    /// Evaluates the predicate for a single row.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool;

    /// Returns `false` if zone map proves no rows in this chunk can match.
    ///
    /// This method enables chunk-level filtering optimization. When a chunk
    /// has zone map hints attached, the filter operator calls this method
    /// first. If it returns `false`, the entire chunk is skipped without
    /// evaluating any rows.
    ///
    /// The default implementation is conservative and returns `true` (might match).
    /// Predicates that support zone map checking should override this.
    fn might_match_chunk(&self, _hints: &ChunkZoneHints) -> bool {
        true
    }
}

/// A comparison operator.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompareOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
}

/// A simple comparison predicate.
#[cfg(test)]
pub(crate) struct ComparisonPredicate {
    /// Column index to compare.
    column: usize,
    /// Comparison operator.
    op: CompareOp,
    /// Value to compare against.
    value: Value,
}

#[cfg(test)]
impl ComparisonPredicate {
    /// Creates a new comparison predicate.
    pub(crate) fn new(column: usize, op: CompareOp, value: Value) -> Self {
        Self { column, op, value }
    }
}

#[cfg(test)]
impl Predicate for ComparisonPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        let Some(col) = chunk.column(self.column) else {
            return false;
        };

        let Some(cell_value) = col.get_value(row) else {
            return false;
        };

        match (&cell_value, &self.value) {
            (Value::Int64(a), Value::Int64(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::Float64(a), Value::Float64(b)) => match self.op {
                CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (Value::String(a), Value::String(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            // Cross-type Int64/Float64 coercion
            (Value::Int64(a), Value::Float64(b)) => {
                let a = *a as f64;
                match self.op {
                    CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                    CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                    CompareOp::Lt => a < *b,
                    CompareOp::Le => a <= *b,
                    CompareOp::Gt => a > *b,
                    CompareOp::Ge => a >= *b,
                }
            }
            (Value::Float64(a), Value::Int64(b)) => {
                let b = *b as f64;
                match self.op {
                    CompareOp::Eq => (a - b).abs() < f64::EPSILON,
                    CompareOp::Ne => (a - b).abs() >= f64::EPSILON,
                    CompareOp::Lt => *a < b,
                    CompareOp::Le => *a <= b,
                    CompareOp::Gt => *a > b,
                    CompareOp::Ge => *a >= b,
                }
            }
            (Value::Bool(a), Value::Bool(b)) => match self.op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                _ => false, // Ordering on booleans doesn't make sense
            },
            _ => false, // Type mismatch
        }
    }

    fn might_match_chunk(&self, hints: &ChunkZoneHints) -> bool {
        let Some(zone_map) = hints.column_hints.get(&self.column) else {
            return true; // No zone map for this column = conservative
        };

        match self.op {
            CompareOp::Eq => zone_map.might_contain_equal(&self.value),
            CompareOp::Ne => true, // Ne is always conservative (might have non-matching values)
            CompareOp::Lt => zone_map.might_contain_less_than(&self.value, false),
            CompareOp::Le => zone_map.might_contain_less_than(&self.value, true),
            CompareOp::Gt => zone_map.might_contain_greater_than(&self.value, false),
            CompareOp::Ge => zone_map.might_contain_greater_than(&self.value, true),
        }
    }
}

/// An expression-based predicate that evaluates logical expressions.
///
/// This predicate can evaluate complex expressions involving variables,
/// properties, and operators.
pub struct ExpressionPredicate {
    /// The expression to evaluate.
    expression: FilterExpression,
    /// Map from variable name to column index.
    variable_columns: HashMap<String, usize>,
    /// The graph store for property lookups.
    store: Arc<dyn GraphStoreSearch>,
    /// Transaction ID for MVCC-aware lookups.
    transaction_id: Option<TransactionId>,
    /// Viewing epoch for MVCC-aware lookups.
    viewing_epoch: Option<EpochId>,
    /// Session context for introspection functions (info, schema, current_schema, etc.).
    session_context: SessionContext,
    /// Compiled patterns of `=~` and `LIKE` (see [`PatternCache`]).
    patterns: Arc<PatternCache>,
}

/// Compiled patterns of `=~` and `LIKE`, by their final regex text, shared by
/// an evaluator and the list scopes it creates, so a pattern compiles once
/// instead of once per row. `None` marks a pattern that does not compile.
/// Empty in builds without a regex feature.
#[derive(Default)]
struct PatternCache(
    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    parking_lot::Mutex<HashMap<String, Option<Regex>>>,
);

#[cfg(any(feature = "regex", feature = "regex-lite"))]
impl PatternCache {
    /// Patterns kept at most; a query with more distinct ones (patterns
    /// taken from rows) starts over.
    const CAPACITY: usize = 256;

    /// Whether `text` matches `regex`, or `None` when `regex` does not compile.
    fn is_match(&self, regex: String, text: &str) -> Option<bool> {
        let mut patterns = self.0.lock();
        if patterns.len() >= Self::CAPACITY && !patterns.contains_key(&regex) {
            patterns.clear();
        }
        patterns
            .entry(regex)
            .or_insert_with_key(|regex| Regex::new(regex).ok())
            .as_ref()
            .map(|re| re.is_match(text))
    }
}

/// A lazily-computed, cloneable value.
///
/// The factory runs at most once (via `OnceLock`). Cloning is cheap because
/// both the lock and the factory are behind `Arc`.
#[derive(Clone)]
pub struct LazyValue {
    cell: Arc<std::sync::OnceLock<Value>>,
    factory: Arc<dyn Fn() -> Value + Send + Sync>,
}

impl LazyValue {
    /// Creates a new lazy value with the given factory.
    pub fn new(factory: impl Fn() -> Value + Send + Sync + 'static) -> Self {
        Self {
            cell: Arc::new(std::sync::OnceLock::new()),
            factory: Arc::new(factory),
        }
    }

    /// Returns the value, computing it on first access.
    pub fn get(&self) -> &Value {
        self.cell.get_or_init(|| (self.factory)())
    }
}

impl std::fmt::Debug for LazyValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.cell.get() {
            Some(v) => write!(f, "LazyValue({v:?})"),
            None => write!(f, "LazyValue(<not yet computed>)"),
        }
    }
}

impl Default for LazyValue {
    fn default() -> Self {
        Self::new(|| Value::Null)
    }
}

/// Session-level context passed to the filter evaluator for introspection functions.
///
/// Lightweight strings (`current_schema`, `current_graph`) are stored directly.
/// Expensive introspection maps (`db_info`, `schema_info`) are lazily computed
/// on first access, so queries that never call `info()` or `schema()` pay zero cost.
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    /// Current session schema name (for `CURRENT_SCHEMA`).
    pub current_schema: Option<String>,
    /// Current session graph name (for `CURRENT_GRAPH`).
    pub current_graph: Option<String>,
    /// Lazily-computed `info()` result.
    pub db_info: LazyValue,
    /// Lazily-computed `schema()` result.
    pub schema_info: LazyValue,
}

/// A filter expression that can be evaluated.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum FilterExpression {
    /// A literal value.
    Literal(Value),
    /// A variable reference (column index).
    Variable(String),
    /// Property access on a variable.
    Property {
        /// The variable name.
        variable: String,
        /// The property name.
        property: String,
    },
    /// Binary operation.
    Binary {
        /// Left operand.
        left: Box<FilterExpression>,
        /// Operator.
        op: BinaryFilterOp,
        /// Right operand.
        right: Box<FilterExpression>,
    },
    /// Unary operation.
    Unary {
        /// Operator.
        op: UnaryFilterOp,
        /// Operand.
        operand: Box<FilterExpression>,
    },
    /// Function call.
    FunctionCall {
        /// Function name (e.g., "id", "labels", "type", "size", "coalesce", "exists").
        name: String,
        /// Arguments.
        args: Vec<FilterExpression>,
    },
    /// List literal.
    List(Vec<FilterExpression>),
    /// Map literal (e.g., {name: 'Alix', age: 30}).
    Map(Vec<(String, FilterExpression)>),
    /// Index access (e.g., `list[0]`).
    IndexAccess {
        /// The base expression.
        base: Box<FilterExpression>,
        /// The index expression.
        index: Box<FilterExpression>,
    },
    /// Slice access (e.g., list[1..3]).
    SliceAccess {
        /// The base expression.
        base: Box<FilterExpression>,
        /// Start index (None means from beginning).
        start: Option<Box<FilterExpression>>,
        /// End index (None means to end).
        end: Option<Box<FilterExpression>>,
    },
    /// CASE expression.
    Case {
        /// Test expression (for simple CASE).
        operand: Option<Box<FilterExpression>>,
        /// WHEN clauses (condition, result).
        when_clauses: Vec<(FilterExpression, FilterExpression)>,
        /// ELSE clause.
        else_clause: Option<Box<FilterExpression>>,
    },
    /// Entity ID access.
    Id(String),
    /// Node labels access.
    Labels(String),
    /// Edge type access.
    Type(String),
    /// List comprehension: [x IN list WHERE predicate | expression]
    ListComprehension {
        /// Variable name for each element.
        variable: String,
        /// The source list expression.
        list_expr: Box<FilterExpression>,
        /// Optional filter predicate.
        filter_expr: Option<Box<FilterExpression>>,
        /// The mapping expression for each element.
        map_expr: Box<FilterExpression>,
    },
    /// List predicate: all/any/none/single(x IN list WHERE pred).
    ListPredicate {
        /// The kind of list predicate.
        kind: ListPredicateKind,
        /// The iteration variable name.
        variable: String,
        /// The source list expression.
        list_expr: Box<FilterExpression>,
        /// The predicate to test for each element.
        predicate: Box<FilterExpression>,
    },
    /// EXISTS subquery over one edge, or one variable-length edge, between
    /// `start_var` and `end_var` (fast path): whether the pattern matches for
    /// the row. A pattern variable the row binds must match what the row
    /// holds (a null matches nothing); see [`ExpressionPredicate`] for a row
    /// that binds the end but not the start, or neither.
    ExistsSubquery {
        /// The pattern's start node variable.
        start_var: String,
        /// The pattern's other end.
        end_var: String,
        /// The pattern's edge variable, if it has one.
        edge_var: Option<String>,
        /// Direction of edge traversal, from the start.
        direction: Direction,
        /// Edge type filter (empty = match all types, multiple = match any).
        edge_types: Vec<String>,
        /// Labels the other end must have (single-edge patterns only).
        end_labels: Option<Vec<String>>,
        /// `Some(1)` for a variable-length pattern, `None` for one edge.
        min_hops: Option<u32>,
        /// Maximum number of hops of a variable-length pattern (`None`:
        /// unbounded).
        max_hops: Option<u32>,
    },
    /// COUNT subquery over one edge between `start_var` and `end_var` (fast
    /// path): how many edges match for the row, with the same rules as
    /// [`ExistsSubquery`](Self::ExistsSubquery).
    CountSubquery {
        /// The pattern's start node variable.
        start_var: String,
        /// The pattern's other end.
        end_var: String,
        /// The pattern's edge variable, if it has one.
        edge_var: Option<String>,
        /// Direction of edge traversal, from the start.
        direction: Direction,
        /// Edge type filter (empty = match all types, multiple = match any).
        edge_types: Vec<String>,
        /// Labels the other end must have.
        end_labels: Option<Vec<String>>,
    },
    /// reduce() accumulator: `reduce(acc = init, x IN list | expr)`.
    Reduce {
        /// Accumulator variable name.
        accumulator: String,
        /// Initial value for the accumulator.
        initial: Box<FilterExpression>,
        /// Iteration variable name.
        variable: String,
        /// List to iterate over.
        list: Box<FilterExpression>,
        /// Body expression (references both accumulator and variable).
        expression: Box<FilterExpression>,
    },
}

/// The kind of list predicate function.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ListPredicateKind {
    /// all(x IN list WHERE pred): true if pred holds for every element.
    All,
    /// any(x IN list WHERE pred): true if pred holds for at least one element.
    Any,
    /// none(x IN list WHERE pred): true if pred holds for no element.
    None,
    /// single(x IN list WHERE pred): true if pred holds for exactly one element.
    Single,
}

/// Binary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum BinaryFilterOp {
    /// Equal.
    Eq,
    /// Not equal.
    Ne,
    /// Less than.
    Lt,
    /// Less than or equal.
    Le,
    /// Greater than.
    Gt,
    /// Greater than or equal.
    Ge,
    /// Logical AND.
    And,
    /// Logical OR.
    Or,
    /// Logical XOR.
    Xor,
    /// Addition.
    Add,
    /// Subtraction.
    Sub,
    /// Multiplication.
    Mul,
    /// Division.
    Div,
    /// Modulo.
    Mod,
    /// String starts with.
    StartsWith,
    /// String ends with.
    EndsWith,
    /// String contains.
    Contains,
    /// List membership.
    In,
    /// Regex match (=~).
    Regex,
    /// Power/exponentiation (^).
    Pow,
    /// SQL LIKE pattern matching (% = any chars, _ = single char).
    Like,
    /// String concatenation (||).
    Concat,
}

/// Unary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum UnaryFilterOp {
    /// Logical NOT.
    Not,
    /// IS NULL.
    IsNull,
    /// IS NOT NULL.
    IsNotNull,
    /// Numeric negation.
    Neg,
}

impl ExpressionPredicate {
    /// Creates a new expression predicate.
    pub fn new(
        expression: FilterExpression,
        variable_columns: HashMap<String, usize>,
        store: Arc<dyn GraphStoreSearch>,
    ) -> Self {
        Self {
            expression,
            variable_columns,
            store,
            transaction_id: None,
            viewing_epoch: None,
            session_context: SessionContext::default(),
            patterns: Arc::default(),
        }
    }

    /// Sets the transaction context for MVCC-aware property lookups.
    pub fn with_transaction_context(
        mut self,
        epoch: EpochId,
        transaction_id: Option<TransactionId>,
    ) -> Self {
        self.viewing_epoch = Some(epoch);
        self.transaction_id = transaction_id;
        self
    }

    /// Sets the session context for introspection functions.
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
    }

    /// Resolves a node using transaction-aware access when available.
    fn resolve_node(&self, node_id: NodeId) -> Option<Node> {
        if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.get_node_versioned(node_id, ep, tx)
        } else if let Some(ep) = self.viewing_epoch {
            self.store.get_node_at_epoch(node_id, ep)
        } else {
            self.store.get_node(node_id)
        }
    }

    /// Returns true if an edge (and its other endpoint) matches the type and label filters.
    ///
    /// Used by both `ExistsSubquery` and `CountSubquery` fast-path evaluation.
    fn edge_matches(
        &self,
        other_node_id: NodeId,
        edge_id: EdgeId,
        edge_types: &[String],
        end_labels: &Option<Vec<String>>,
    ) -> bool {
        self.edge_type_matches(edge_id, edge_types) && self.has_labels(other_node_id, end_labels)
    }

    /// Whether the edge has one of `edge_types` (any type when empty). The
    /// type is read as this query sees the edge, so a transaction finds the
    /// type of an edge it created.
    fn edge_type_matches(&self, edge_id: EdgeId, edge_types: &[String]) -> bool {
        if edge_types.is_empty() {
            return true;
        }
        let actual = if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.edge_type_versioned(edge_id, ep, tx)
        } else {
            self.store.edge_type(edge_id)
        };
        actual.is_some_and(|actual| {
            edge_types
                .iter()
                .any(|t| actual.as_str().eq_ignore_ascii_case(t.as_str()))
        })
    }

    /// The edges of `node` in `direction` that this query sees, with the
    /// node at their other end: not one created after the viewing epoch, by
    /// another transaction that has not committed, or deleted by this one
    /// (the checks the expand operators make).
    fn visible_edges_from(&self, node: NodeId, direction: Direction) -> Vec<(NodeId, EdgeId)> {
        let mut edges = self.store.edges_from(node, direction);
        if let Some(epoch) = self.viewing_epoch {
            edges.retain(|&(other, edge)| {
                if let Some(tx) = self.transaction_id {
                    self.store.is_edge_visible_versioned(edge, epoch, tx)
                        && self.store.is_node_visible_versioned(other, epoch, tx)
                } else {
                    self.store.is_edge_visible_at_epoch(edge, epoch)
                        && self.store.is_node_visible_at_epoch(other, epoch)
                }
            });
        }
        edges
    }

    /// Whether the node has every label of `labels` (any node when `None`).
    fn has_labels(&self, node_id: NodeId, labels: &Option<Vec<String>>) -> bool {
        labels.as_ref().is_none_or(|labels| {
            self.resolve_node(node_id)
                .is_some_and(|node| labels.iter().all(|label| node.has_label(label)))
        })
    }

    /// What the row binds the pattern variable `name` to, read with `read`.
    fn bound<T>(
        &self,
        name: &str,
        chunk: &DataChunk,
        row: usize,
        read: impl FnOnce(&ValueVector, usize) -> Option<T>,
    ) -> Bound<T> {
        match self
            .variable_columns
            .get(name)
            .and_then(|&index| chunk.column(index))
        {
            None => Bound::No,
            Some(column) => read(column, row).map_or(Bound::Null, Bound::To),
        }
    }

    /// The nodes this query sees.
    fn visible_nodes(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.store
            .node_ids()
            .into_iter()
            .filter(|&id| self.resolve_node(id).is_some())
    }

    /// How many matches a fast-path `EXISTS` or `COUNT` pattern has for the
    /// row, counting up to `limit` (`EXISTS` needs one).
    ///
    /// The subquery pattern shares a variable with the row when the row binds
    /// it: an end or edge the row binds must be that node or edge, and one the
    /// row holds as null matches nothing. A row that binds the edge is
    /// answered from that edge; one that binds the end but not the start has
    /// its edges found from the end; one that binds none of them, which makes
    /// the subquery the same for every row, from every node.
    fn subquery_matches(
        &self,
        pattern: &SubqueryPattern<'_>,
        chunk: &DataChunk,
        row: usize,
        limit: usize,
    ) -> usize {
        let (start, end) = match (
            self.bound(pattern.start, chunk, row, ValueVector::get_node_id),
            self.bound(pattern.end, chunk, row, ValueVector::get_node_id),
        ) {
            (Bound::Null, _) | (_, Bound::Null) => return 0,
            (start, end) => (start.node(), end.node()),
        };
        if let Hops::Path(max_hops) = pattern.hops {
            return usize::from(self.path_exists(pattern, chunk, row, start, end, max_hops));
        }
        match pattern.edge.map_or(Bound::No, |name| {
            self.bound(name, chunk, row, ValueVector::get_edge_id)
        }) {
            Bound::Null => return 0,
            Bound::To(edge) => {
                return self
                    .bound_edge_matches(edge, start, end, pattern)
                    .min(limit);
            }
            Bound::No => {}
        }
        match (start, end) {
            (Some(start), end) => self
                .visible_edges_from(start, pattern.direction)
                .into_iter()
                .filter(|&(other, id)| {
                    end.is_none_or(|end| end == other)
                        && self.edge_matches(other, id, pattern.edge_types, pattern.end_labels)
                })
                .take(limit)
                .count(),
            (None, Some(end)) => {
                if !self.has_labels(end, pattern.end_labels) {
                    return 0;
                }
                self.visible_edges_from(end, pattern.direction.reverse())
                    .into_iter()
                    .filter(|&(_, id)| self.edge_type_matches(id, pattern.edge_types))
                    .take(limit)
                    .count()
            }
            (None, None) => {
                let mut found = 0;
                for start in self.visible_nodes() {
                    found += self
                        .visible_edges_from(start, pattern.direction)
                        .into_iter()
                        .filter(|&(other, id)| {
                            self.edge_matches(other, id, pattern.edge_types, pattern.end_labels)
                        })
                        .take(limit - found)
                        .count();
                    if found == limit {
                        break;
                    }
                }
                found
            }
        }
    }

    /// How often the row's edge matches a one-edge pattern between `start`
    /// and `end` (the nodes the row binds, if any): once, or once each way
    /// for an undirected pattern with free ends, as walking the edges of
    /// every node would find it (a self-loop too).
    fn bound_edge_matches(
        &self,
        edge: EdgeId,
        start: Option<NodeId>,
        end: Option<NodeId>,
        pattern: &SubqueryPattern<'_>,
    ) -> usize {
        let Some(record) = self.resolve_edge(edge) else {
            return 0;
        };
        if !self.edge_type_matches(edge, pattern.edge_types) {
            return 0;
        }
        // The edge read from the start: forward from its source, backward
        // from its target.
        let forward = matches!(pattern.direction, Direction::Outgoing | Direction::Both)
            .then_some((record.src, record.dst));
        let backward = matches!(pattern.direction, Direction::Incoming | Direction::Both)
            .then_some((record.dst, record.src));
        [forward, backward]
            .into_iter()
            .flatten()
            .filter(|&(from, to)| {
                start.map_or_else(|| self.resolve_node(from).is_some(), |start| start == from)
                    && end.is_none_or(|end| end == to)
                    && self.has_labels(to, pattern.end_labels)
            })
            .count()
    }

    /// Whether a variable-length fast-path pattern (1 to `max_hops` edges,
    /// any number when `None`) matches for the row; `start` and `end` are the
    /// nodes the row binds. A path from an unbound end exists when its first
    /// edge does.
    fn path_exists(
        &self,
        pattern: &SubqueryPattern<'_>,
        chunk: &DataChunk,
        row: usize,
        start: Option<NodeId>,
        end: Option<NodeId>,
        max_hops: Option<u32>,
    ) -> bool {
        if let Some(name) = pattern.edge {
            match self.bound(name, chunk, row, edge_id_list) {
                Bound::Null => return false,
                Bound::To(edges) => {
                    return self.path_follows(&edges, start, end, pattern, max_hops);
                }
                Bound::No => {}
            }
        }
        let has_edge = |node: NodeId, direction: Direction| {
            self.visible_edges_from(node, direction)
                .into_iter()
                .any(|(_, id)| self.edge_type_matches(id, pattern.edge_types))
        };
        match (start, end) {
            (Some(start), Some(end)) => self.reaches(start, end, pattern, max_hops),
            (Some(start), None) => has_edge(start, pattern.direction),
            (None, Some(end)) => has_edge(end, pattern.direction.reverse()),
            (None, None) => self
                .visible_nodes()
                .any(|node| has_edge(node, pattern.direction)),
        }
    }

    /// Whether `to` is reached from `from` over 1 to `max_hops` edges of the
    /// pattern (any number when `None`).
    fn reaches(
        &self,
        from: NodeId,
        to: NodeId,
        pattern: &SubqueryPattern<'_>,
        max_hops: Option<u32>,
    ) -> bool {
        // `from` is not marked seen, so a cycle back to it counts.
        let mut seen = std::collections::HashSet::new();
        let mut frontier = vec![from];
        let mut hops = 0;
        while !frontier.is_empty() && max_hops.is_none_or(|max| hops < max) {
            hops += 1;
            let mut next = Vec::new();
            for node in frontier {
                for (other, id) in self.visible_edges_from(node, pattern.direction) {
                    if !self.edge_type_matches(id, pattern.edge_types) {
                        continue;
                    }
                    if other == to {
                        return true;
                    }
                    if seen.insert(other) {
                        next.push(other);
                    }
                }
            }
            frontier = next;
        }
        false
    }

    /// Whether `edges` are a path of the pattern, one after the other: 1 to
    /// `max_hops` edges of its types in its direction, from `start` and to
    /// `end` when the row binds them.
    fn path_follows(
        &self,
        edges: &[EdgeId],
        start: Option<NodeId>,
        end: Option<NodeId>,
        pattern: &SubqueryPattern<'_>,
        max_hops: Option<u32>,
    ) -> bool {
        if max_hops.is_some_and(|max| usize::try_from(max).is_ok_and(|max| edges.len() > max)) {
            return false;
        }
        let Some(first) = edges.first().and_then(|&id| self.resolve_edge(id)) else {
            return false;
        };
        let starts = match (start, pattern.direction) {
            (Some(start), _) => vec![start],
            (None, Direction::Outgoing) => vec![first.src],
            (None, Direction::Incoming) => vec![first.dst],
            (None, Direction::Both) => vec![first.src, first.dst],
        };
        starts.into_iter().any(|mut node| {
            for &id in edges {
                let Some(edge) = self.resolve_edge(id) else {
                    return false;
                };
                if !self.edge_type_matches(id, pattern.edge_types) {
                    return false;
                }
                node = match pattern.direction {
                    Direction::Outgoing if edge.src == node => edge.dst,
                    Direction::Incoming if edge.dst == node => edge.src,
                    Direction::Both if edge.src == node => edge.dst,
                    Direction::Both if edge.dst == node => edge.src,
                    _ => return false,
                };
            }
            end.is_none_or(|end| end == node)
        })
    }

    /// Resolves an edge using transaction-aware access when available.
    fn resolve_edge(&self, edge_id: grafeo_common::types::EdgeId) -> Option<Edge> {
        if let (Some(ep), Some(tx)) = (self.viewing_epoch, self.transaction_id) {
            self.store.get_edge_versioned(edge_id, ep, tx)
        } else if let Some(ep) = self.viewing_epoch {
            self.store.get_edge_at_epoch(edge_id, ep)
        } else {
            self.store.get_edge(edge_id)
        }
    }

    /// The node `expr` yields at `row`: the node of a node variable, or a
    /// node taken from a list, a map or a function (`head(ns)`, `x.msg`,
    /// `startNode(r)`, see [`shape`](Self::shape)).
    fn node_of(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Node> {
        let id = match expr {
            FilterExpression::Variable(var) => chunk
                .column(*self.variable_columns.get(var)?)?
                .get_node_id(row)?,
            _ => NodeId::new(self.entity_id(expr, &LogicalType::Node, chunk, row)?),
        };
        self.resolve_node(id)
    }

    /// The edge `expr` yields at `row`, as [`node_of`](Self::node_of) finds
    /// a node.
    fn edge_of(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Edge> {
        let id = match expr {
            FilterExpression::Variable(var) => chunk
                .column(*self.variable_columns.get(var)?)?
                .get_edge_id(row)?,
            _ => EdgeId::new(self.entity_id(expr, &LogicalType::Edge, chunk, row)?),
        };
        self.resolve_edge(id)
    }

    /// The ID `expr` yields at `row` when it yields a `kind` (a node or an
    /// edge) by its [`shape`](Self::shape).
    fn entity_id(
        &self,
        expr: &FilterExpression,
        kind: &LogicalType,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<u64> {
        if self.shape(expr, chunk, row) != *kind {
            return None;
        }
        match self.eval_expr(expr, chunk, row)? {
            Value::Int64(id) => u64::try_from(id).ok(),
            _ => None,
        }
    }

    /// Evaluates the expression for a specific row in a chunk, returning the result value.
    /// This is useful for evaluating expressions in contexts like RETURN clauses.
    pub fn eval_at(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    /// Evaluates the expression for a row, returning the result value.
    fn eval(&self, chunk: &DataChunk, row: usize) -> Option<Value> {
        self.eval_expr(&self.expression, chunk, row)
    }

    fn eval_expr(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> Option<Value> {
        match expr {
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Variable(name) => {
                let col_idx = *self.variable_columns.get(name)?;
                chunk.column(col_idx)?.get_value(row)
            }
            FilterExpression::Property { variable, property } => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                // Try as node first
                if let Some(node_id) = col.get_node_id(row)
                    && let Some(node) = self.resolve_node(node_id)
                {
                    return node.get_property(property).cloned();
                }
                // Try as edge if node lookup failed
                if let Some(edge_id) = col.get_edge_id(row)
                    && let Some(edge) = self.resolve_edge(edge_id)
                {
                    return edge.get_property(property).cloned();
                }
                match col.get_value(row)? {
                    // A key of a map value (e.g. from UNWIND with map elements)
                    Value::Map(map) => map.get(&PropertyKey::new(property)).cloned(),
                    // A component of a temporal value (`d.month`)
                    value => value.temporal_component(property),
                }
            }
            FilterExpression::Binary { left, op, right } => {
                // For IN operator, right side is a list that we evaluate specially
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_expr(left, chunk, row)?;
                    return self.eval_in_operator(&left_val, right, chunk, row);
                }
                // For logical operators (AND/OR/XOR), treat missing values as
                // NULL so three-valued logic works correctly. Without this,
                // `NULL OR true` would short-circuit to None instead of true.
                if matches!(
                    op,
                    BinaryFilterOp::And | BinaryFilterOp::Or | BinaryFilterOp::Xor
                ) {
                    let left_val = self.eval_expr(left, chunk, row).unwrap_or(Value::Null);
                    let right_val = self.eval_expr(right, chunk, row).unwrap_or(Value::Null);
                    return self.eval_binary_op(&left_val, *op, &right_val);
                }
                let left_val = self.eval_expr(left, chunk, row)?;
                let right_val = self.eval_expr(right, chunk, row)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_expr(operand, chunk, row);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::FunctionCall { name, args } => {
                self.eval_function(name, args, chunk, row)
            }
            // A list literal has one item per expression and a map literal one
            // entry per key: an expression without a value (a property of a
            // null entity, a property the entity does not have) is a null.
            FilterExpression::List(items) => {
                let values: Vec<Value> = items
                    .iter()
                    .map(|item| self.eval_expr(item, chunk, row).unwrap_or(Value::Null))
                    .collect();
                Some(Value::List(values.into()))
            }
            FilterExpression::Map(pairs) => {
                let mut map = BTreeMap::new();
                for (k, v) in pairs {
                    let val = self.eval_expr(v, chunk, row);
                    if k == "*" {
                        // AllProperties marker: flatten the inner map into the result
                        if let Some(Value::Map(inner)) = val {
                            map.extend(inner.iter().map(|(pk, pv)| (pk.clone(), pv.clone())));
                        }
                    } else {
                        map.insert(PropertyKey::new(k.as_str()), val.unwrap_or(Value::Null));
                    }
                }
                Some(Value::Map(Arc::new(map)))
            }
            FilterExpression::IndexAccess { base, index } => {
                let base_val = self.eval_expr(base, chunk, row)?;
                let index_val = self.eval_expr(index, chunk, row)?;
                match (&base_val, &index_val) {
                    (Value::List(items), Value::Int64(i)) => {
                        // reason: list/string lengths fit i64; index values are user-provided
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_possible_wrap,
                            clippy::cast_sign_loss
                        )]
                        let idx = if *i < 0 {
                            // Negative indexing from end
                            let len = items.len() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        items.get(idx).cloned()
                    }
                    (Value::String(s), Value::Int64(i)) => {
                        // reason: list/string lengths fit i64; index values are user-provided
                        #[allow(
                            clippy::cast_possible_truncation,
                            clippy::cast_possible_wrap,
                            clippy::cast_sign_loss
                        )]
                        let idx = if *i < 0 {
                            // From the end, in characters as the index counts
                            let len = s.chars().count() as i64;
                            (len + i) as usize
                        } else {
                            *i as usize
                        };
                        s.chars()
                            .nth(idx)
                            .map(|c| Value::String(c.to_string().into()))
                    }
                    (Value::Map(m), Value::String(key)) => {
                        let prop_key = PropertyKey::new(key.as_str());
                        m.get(&prop_key).cloned()
                    }
                    // A component of a temporal property (`n.born.month`).
                    (
                        Value::Date(_)
                        | Value::Time(_)
                        | Value::Timestamp(_)
                        | Value::ZonedDatetime(_)
                        | Value::Duration(_),
                        Value::String(key),
                    ) => base_val.temporal_component(key),
                    (_, Value::String(key)) => {
                        // Node/edge bracket access: n['name'] looks up a property
                        // via the store when the base variable refers to a node or edge.
                        if let FilterExpression::Variable(var) = base.as_ref()
                            && let Some(&col_idx) = self.variable_columns.get(var)
                            && let Some(col) = chunk.column(col_idx)
                        {
                            if let Some(node_id) = col.get_node_id(row)
                                && let Some(node) = self.resolve_node(node_id)
                            {
                                return node.get_property(key.as_str()).cloned();
                            }
                            if let Some(edge_id) = col.get_edge_id(row)
                                && let Some(edge) = self.resolve_edge(edge_id)
                            {
                                return edge.get_property(key.as_str()).cloned();
                            }
                        }
                        // A node or edge taken from a list, a map or a
                        // function (`rs[0].w`, `head(rs).w`, `x.msg.id`,
                        // `startNode(r).name`) is its ID.
                        if let Value::Int64(id) = &base_val
                            && let Ok(id) = u64::try_from(*id)
                        {
                            return match self.shape(base, chunk, row) {
                                LogicalType::Node => self
                                    .resolve_node(NodeId::new(id))
                                    .and_then(|node| node.get_property(key.as_str()).cloned()),
                                LogicalType::Edge => self
                                    .resolve_edge(EdgeId::new(id))
                                    .and_then(|edge| edge.get_property(key.as_str()).cloned()),
                                _ => None,
                            };
                        }
                        None
                    }
                    _ => None,
                }
            }
            FilterExpression::SliceAccess { base, start, end } => {
                let base_val = self.eval_expr(base, chunk, row)?;
                let start_idx = start
                    .as_ref()
                    .and_then(|s| self.eval_expr(s, chunk, row))
                    .and_then(|v| {
                        if let Value::Int64(i) = v {
                            // reason: slice index from user query, non-negative for valid slices
                            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                            Some(i as usize)
                        } else {
                            None
                        }
                    })
                    .unwrap_or(0);

                match &base_val {
                    Value::List(items) => {
                        let end_idx = end
                            .as_ref()
                            .and_then(|e| self.eval_expr(e, chunk, row))
                            .and_then(|v| {
                                if let Value::Int64(i) = v {
                                    // reason: slice end index from user query
                                    #[allow(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_sign_loss
                                    )]
                                    Some(i as usize)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(items.len());
                        let sliced: Vec<Value> = items
                            .get(start_idx..end_idx.min(items.len()))
                            .unwrap_or(&[])
                            .to_vec();
                        Some(Value::List(sliced.into()))
                    }
                    Value::String(s) => {
                        let chars: Vec<char> = s.chars().collect();
                        let end_idx = end
                            .as_ref()
                            .and_then(|e| self.eval_expr(e, chunk, row))
                            .and_then(|v| {
                                if let Value::Int64(i) = v {
                                    // reason: slice end index from user query
                                    #[allow(
                                        clippy::cast_possible_truncation,
                                        clippy::cast_sign_loss
                                    )]
                                    Some(i as usize)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(chars.len());
                        let sliced: String = chars
                            .get(start_idx..end_idx.min(chars.len()))
                            .unwrap_or(&[])
                            .iter()
                            .collect();
                        Some(Value::String(sliced.into()))
                    }
                    _ => None,
                }
            }
            FilterExpression::Case {
                operand,
                when_clauses,
                else_clause,
            } => self.eval_case(
                operand.as_deref(),
                when_clauses,
                else_clause.as_deref(),
                chunk,
                row,
            ),
            FilterExpression::Id(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                // Try as node first, then as edge
                if let Some(node_id) = col.get_node_id(row) {
                    // reason: entity IDs stored as i64 values, standard encoding
                    #[allow(clippy::cast_possible_wrap)]
                    Some(Value::Int64(node_id.0 as i64))
                } else {
                    col.get_edge_id(row)
                        // reason: entity IDs stored as i64 values, standard encoding
                        .map(|edge_id| {
                            // reason: entity IDs are sequential counters, well within i64::MAX
                            #[allow(clippy::cast_possible_wrap)]
                            let val = Value::Int64(edge_id.0 as i64);
                            val
                        })
                }
            }
            FilterExpression::Labels(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let node_id = col.get_node_id(row)?;
                let node = self.resolve_node(node_id)?;
                // Sort labels so sets with the same members always produce
                // the same list, regardless of internal storage order.
                let mut sorted: Vec<&arcstr::ArcStr> = node.labels.iter().collect();
                sorted.sort();
                let labels: Vec<Value> = sorted
                    .into_iter()
                    .map(|l| Value::String(l.clone()))
                    .collect();
                Some(Value::List(labels.into()))
            }
            FilterExpression::Type(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let edge_id = col.get_edge_id(row)?;
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::String(edge.edge_type.clone()))
            }
            FilterExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                let shape = self.shape(list_expr, chunk, row);
                let items = Self::list_items(self.eval_expr(list_expr, chunk, row)?)?;
                let mut scope = ItemScope::new(self, &[variable], chunk, row);
                let mut result = Vec::with_capacity(items.len());
                for (position, item) in items.iter().enumerate() {
                    scope.bind(0, item, &item_shape(&shape, position));
                    let passes = filter_expr
                        .as_ref()
                        .is_none_or(|filter| matches!(scope.eval(filter), Some(Value::Bool(true))));
                    if passes {
                        result.push(scope.eval(map_expr).unwrap_or(Value::Null));
                    }
                }
                Some(Value::List(result.into()))
            }
            FilterExpression::ListPredicate {
                kind,
                variable,
                list_expr,
                predicate,
            } => {
                let shape = self.shape(list_expr, chunk, row);
                let items = Self::list_items(self.eval_expr(list_expr, chunk, row)?)?;
                let mut scope = ItemScope::new(self, &[variable], chunk, row);
                let mut match_count = 0usize;
                for (position, item) in items.iter().enumerate() {
                    scope.bind(0, item, &item_shape(&shape, position));
                    if matches!(scope.eval(predicate), Some(Value::Bool(true))) {
                        match_count += 1;
                    }
                }
                let result = match kind {
                    ListPredicateKind::All => match_count == items.len(),
                    ListPredicateKind::Any => match_count > 0,
                    ListPredicateKind::None => match_count == 0,
                    ListPredicateKind::Single => match_count == 1,
                };
                Some(Value::Bool(result))
            }
            FilterExpression::ExistsSubquery {
                start_var,
                end_var,
                edge_var,
                direction,
                edge_types,
                end_labels,
                min_hops,
                max_hops,
            } => {
                let pattern = SubqueryPattern {
                    start: start_var,
                    end: end_var,
                    edge: edge_var.as_deref(),
                    direction: *direction,
                    edge_types,
                    end_labels,
                    hops: if min_hops.is_some() {
                        Hops::Path(*max_hops)
                    } else {
                        Hops::One
                    },
                };
                Some(Value::Bool(
                    self.subquery_matches(&pattern, chunk, row, 1) > 0,
                ))
            }
            FilterExpression::CountSubquery {
                start_var,
                end_var,
                edge_var,
                direction,
                edge_types,
                end_labels,
            } => {
                let pattern = SubqueryPattern {
                    start: start_var,
                    end: end_var,
                    edge: edge_var.as_deref(),
                    direction: *direction,
                    edge_types,
                    end_labels,
                    hops: Hops::One,
                };
                let count = self.subquery_matches(&pattern, chunk, row, usize::MAX);
                Some(Value::Int64(i64::try_from(count).unwrap_or(i64::MAX)))
            }
            FilterExpression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                let shape = self.shape(list, chunk, row);
                let mut acc = self.eval_expr(initial, chunk, row)?;
                let items = Self::list_items(self.eval_expr(list, chunk, row)?)?;
                let mut scope = ItemScope::new(self, &[accumulator, variable], chunk, row);
                for (position, item) in items.iter().enumerate() {
                    scope.bind(0, &acc, &LogicalType::Any);
                    scope.bind(1, item, &item_shape(&shape, position));
                    acc = scope.eval(expression)?;
                }
                Some(acc)
            }
        }
    }

    /// The items of a list a comprehension, list predicate or `reduce`
    /// iterates over (a vector counts as a list of floats), or `None` for any
    /// other value.
    fn list_items(value: Value) -> Option<Vec<Value>> {
        match value {
            Value::List(items) => Some(items.to_vec()),
            Value::Vector(vector) => Some(
                vector
                    .iter()
                    .map(|&f| Value::Float64(f64::from(f)))
                    .collect(),
            ),
            _ => None,
        }
    }

    /// The properties of the node or edge the variable's column holds at
    /// `row`, taken from the entity the store resolves (no copy). An edge
    /// column holds edges and a node column nodes; an untyped column holding
    /// raw IDs holds a node when one has the ID, otherwise an edge. `None` for
    /// a value that is neither.
    fn element_properties(
        &self,
        variable: &str,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<PropertyMap> {
        let column = chunk.column(*self.variable_columns.get(variable)?)?;
        let node = || Some(self.resolve_node(column.get_node_id(row)?)?.properties);
        let edge = || Some(self.resolve_edge(column.get_edge_id(row)?)?.properties);
        match column.data_type() {
            LogicalType::Edge => edge(),
            LogicalType::Node => node(),
            _ => node().or_else(edge),
        }
    }

    /// Where what `expr` yields at `row` holds nodes or edges, as ids, as
    /// the type that says it (see [`EntityValue`](super::EntityValue)):
    /// `Node` or `Edge` for one, a list of them, a map that holds them
    /// (`Struct`), a list literal of mixed kinds (`Tuple`) or a path. A
    /// column gives its type (the planner types a column by what it holds);
    /// `nodes(p)`, `relationships(p)`, `startNode(r)` and `endNode(r)` give
    /// theirs; list and map literals, keys, items, slices, `head`, `last`,
    /// `tail` and `reverse` follow what they are made of or read from. A
    /// plain value, and anything else, is `Any`.
    fn shape(&self, expr: &FilterExpression, chunk: &DataChunk, row: usize) -> LogicalType {
        let column = |name: &str| {
            self.variable_columns
                .get(name)
                .and_then(|&index| chunk.column(index))
        };
        match expr {
            FilterExpression::Variable(name) => {
                column(name).map_or(LogicalType::Any, |column| column.data_type().clone())
            }
            FilterExpression::Property { variable, property } => column(variable)
                .map_or(LogicalType::Any, |column| {
                    field_type(column.data_type(), property)
                }),
            FilterExpression::IndexAccess { base, index } => {
                let base = self.shape(base, chunk, row);
                if !holds_entities(&base) {
                    return LogicalType::Any;
                }
                match self.eval_expr(index, chunk, row) {
                    Some(Value::String(key)) => field_type(&base, &key),
                    Some(Value::Int64(position)) => item_type_at(&base, position),
                    _ => LogicalType::Any,
                }
            }
            FilterExpression::SliceAccess { base, start, end } => {
                let base = self.shape(base, chunk, row);
                let bound = |bound: &Option<Box<FilterExpression>>| match bound
                    .as_deref()
                    .and_then(|bound| self.eval_expr(bound, chunk, row))
                {
                    Some(Value::Int64(i)) => usize::try_from(i).ok(),
                    _ => None,
                };
                slice_type(base, bound(start).unwrap_or(0), bound(end))
            }
            FilterExpression::FunctionCall { name, args } => {
                let argument = || {
                    args.first()
                        .map_or(LogicalType::Any, |list| self.shape(list, chunk, row))
                };
                match name.to_lowercase().as_str() {
                    "nodes" => LogicalType::List(Box::new(LogicalType::Node)),
                    "edges" | "relationships" => LogicalType::List(Box::new(LogicalType::Edge)),
                    "startnode" | "start_node" | "endnode" | "end_node" => LogicalType::Node,
                    "head" => item_type_at(&argument(), 0),
                    "last" => item_type_at(&argument(), -1),
                    "tail" => slice_type(argument(), 1, None),
                    "reverse" => reversed_type(argument()),
                    _ => LogicalType::Any,
                }
            }
            FilterExpression::List(items) => list_type(
                items
                    .iter()
                    .map(|item| self.shape(item, chunk, row))
                    .collect(),
            ),
            FilterExpression::Map(pairs) => struct_type(
                pairs
                    .iter()
                    .filter(|(key, _)| key != "*")
                    .map(|(key, value)| (key.clone(), self.shape(value, chunk, row)))
                    .collect(),
            ),
            _ => LogicalType::Any,
        }
    }

    /// Evaluates a binary operator with ISO three-valued logic.
    ///
    /// NULL propagation: `NULL = x`, `x = NULL`, `NULL <> x` all yield
    /// `Value::Null` (UNKNOWN), not `Value::Bool`. AND/OR/XOR follow the
    /// standard truth tables where FALSE AND UNKNOWN = FALSE, etc.
    ///
    /// For structural equality (DISTINCT, GROUP BY), use [`values_equal`]
    /// directly, which treats NULL == NULL as true.
    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            // Three-valued logic for AND/OR/XOR (ISO/IEC 39075 Section 21)
            BinaryFilterOp::And => match (left.as_bool(), right.as_bool()) {
                (Some(false), _) | (_, Some(false)) => Some(Value::Bool(false)),
                (Some(true), Some(true)) => Some(Value::Bool(true)),
                _ => Some(Value::Null), // UNKNOWN
            },
            BinaryFilterOp::Or => match (left.as_bool(), right.as_bool()) {
                (Some(true), _) | (_, Some(true)) => Some(Value::Bool(true)),
                (Some(false), Some(false)) => Some(Value::Bool(false)),
                _ => Some(Value::Null), // UNKNOWN
            },
            BinaryFilterOp::Xor => match (left.as_bool(), right.as_bool()) {
                (Some(l), Some(r)) => Some(Value::Bool(l ^ r)),
                _ => Some(Value::Null), // UNKNOWN
            },
            // NULL = anything or anything = NULL is UNKNOWN (three-valued logic).
            // values_equal is preserved for structural equality (DISTINCT, GROUP BY).
            BinaryFilterOp::Eq => {
                if left.is_null() || right.is_null() {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(Self::values_equal(left, right)))
                }
            }
            BinaryFilterOp::Ne => {
                if left.is_null() || right.is_null() {
                    Some(Value::Null)
                } else {
                    Some(Value::Bool(!Self::values_equal(left, right)))
                }
            }
            BinaryFilterOp::Lt => self.compare_values(left, right).map(|c| Value::Bool(c < 0)),
            BinaryFilterOp::Le => self
                .compare_values(left, right)
                .map(|c| Value::Bool(c <= 0)),
            BinaryFilterOp::Gt => self.compare_values(left, right).map(|c| Value::Bool(c > 0)),
            BinaryFilterOp::Ge => self
                .compare_values(left, right)
                .map(|c| Value::Bool(c >= 0)),
            // Arithmetic operators
            BinaryFilterOp::Add => {
                // String concatenation: string + string, or string + other
                match (left, right) {
                    (Value::String(a), Value::String(b)) => {
                        let mut s = String::with_capacity(a.len() + b.len());
                        s.push_str(a);
                        s.push_str(b);
                        Some(Value::String(s.into()))
                    }
                    (Value::String(a), other) => {
                        let b = match other {
                            Value::Int64(i) => i.to_string(),
                            Value::Float64(f) => f.to_string(),
                            Value::Bool(b) => b.to_string(),
                            Value::Null => return Some(Value::Null),
                            _ => return None,
                        };
                        let mut s = String::with_capacity(a.len() + b.len());
                        s.push_str(a);
                        s.push_str(&b);
                        Some(Value::String(s.into()))
                    }
                    // Temporal addition
                    (Value::Date(d), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Date(d)) => {
                        Some(Value::Date(d.add_duration(dur)))
                    }
                    (Value::Time(t), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Time(t)) => {
                        Some(Value::Time(t.add_duration(dur)))
                    }
                    (Value::Timestamp(ts), Value::Duration(dur))
                    | (Value::Duration(dur), Value::Timestamp(ts)) => {
                        Some(Value::Timestamp(ts.add_duration(dur)))
                    }
                    (Value::Duration(a), Value::Duration(b)) => Some(Value::Duration(a.add(*b))),
                    (Value::List(a), Value::List(b)) => {
                        let mut combined = Vec::with_capacity(a.len() + b.len());
                        combined.extend_from_slice(a);
                        combined.extend_from_slice(b);
                        Some(Value::List(combined.into()))
                    }
                    _ => self.eval_arithmetic(left, right, i64::checked_add, |a, b| a + b),
                }
            }
            BinaryFilterOp::Sub => match (left, right) {
                // Temporal subtraction
                (Value::Date(a), Value::Duration(dur)) => Some(Value::Date(a.sub_duration(dur))),
                (Value::Time(a), Value::Duration(dur)) => {
                    Some(Value::Time(a.add_duration(&dur.neg())))
                }
                (Value::Timestamp(a), Value::Duration(dur)) => {
                    Some(Value::Timestamp(a.add_duration(&dur.neg())))
                }
                (Value::Date(a), Value::Date(b)) => {
                    let days = a.as_days() as i64 - b.as_days() as i64;
                    Some(Value::Duration(grafeo_common::types::Duration::from_days(
                        days,
                    )))
                }
                (Value::Time(a), Value::Time(b)) => {
                    // reason: time-of-day nanos (max ~86.4 trillion) fit i64
                    #[allow(clippy::cast_possible_wrap)]
                    let nanos = a.as_nanos() as i64 - b.as_nanos() as i64;
                    Some(Value::Duration(grafeo_common::types::Duration::from_nanos(
                        nanos,
                    )))
                }
                (Value::Timestamp(a), Value::Timestamp(b)) => {
                    let micros = a.duration_since(*b);
                    Some(Value::Duration(grafeo_common::types::Duration::from_nanos(
                        micros * 1000,
                    )))
                }
                (Value::Duration(a), Value::Duration(b)) => Some(Value::Duration(a.sub(*b))),
                _ => self.eval_arithmetic(left, right, i64::checked_sub, |a, b| a - b),
            },
            BinaryFilterOp::Mul => match (left, right) {
                (Value::Duration(d), Value::Int64(n)) | (Value::Int64(n), Value::Duration(d)) => {
                    Some(Value::Duration(d.mul(*n)))
                }
                _ => self.eval_arithmetic(left, right, i64::checked_mul, |a, b| a * b),
            },
            BinaryFilterOp::Div => match (left, right) {
                (Value::Duration(d), Value::Int64(n)) if *n != 0 => {
                    Some(Value::Duration(d.div(*n)))
                }
                _ => self.eval_arithmetic(left, right, i64::checked_div, |a, b| a / b),
            },
            BinaryFilterOp::Mod => self.eval_modulo(left, right),
            // String operators
            BinaryFilterOp::StartsWith => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.starts_with(r)))
            }
            BinaryFilterOp::EndsWith => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.ends_with(r)))
            }
            BinaryFilterOp::Contains => {
                let l = left.as_str()?;
                let r = right.as_str()?;
                Some(Value::Bool(l.contains(r)))
            }
            // IN is handled separately
            BinaryFilterOp::In => None,
            // Regex match (=~): the pattern must match the whole string, as in
            // openCypher (Gremlin's partial `regex()` is widened by its
            // translator).
            BinaryFilterOp::Regex => {
                #[cfg(any(feature = "regex", feature = "regex-lite"))]
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => self
                        .patterns
                        .is_match(format!("^(?:{pattern})$"), s)
                        .map(Value::Bool),
                    _ => None,
                }
                #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
                {
                    let _ = (left, right);
                    None
                }
            }
            // Power/exponentiation (^)
            BinaryFilterOp::Pow => {
                match (left, right) {
                    (Value::Int64(base), Value::Int64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp as f64)))
                    }
                    (Value::Float64(base), Value::Float64(exp)) => {
                        Some(Value::Float64(base.powf(*exp)))
                    }
                    (Value::Int64(base), Value::Float64(exp)) => {
                        Some(Value::Float64((*base as f64).powf(*exp)))
                    }
                    (Value::Float64(base), Value::Int64(exp)) => {
                        Some(Value::Float64(base.powf(*exp as f64)))
                    }
                    _ => None, // Type mismatch
                }
            }
            // SQL LIKE pattern matching
            BinaryFilterOp::Like => {
                #[cfg(any(feature = "regex", feature = "regex-lite"))]
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        let mut regex_pattern = String::with_capacity(pattern.len() + 4);
                        regex_pattern.push('^');
                        let mut chars = pattern.chars().peekable();
                        while let Some(ch) = chars.next() {
                            match ch {
                                '%' => regex_pattern.push_str(".*"),
                                '_' => regex_pattern.push('.'),
                                '\\' => {
                                    if let Some(next) = chars.next() {
                                        regex_escape_char(next, &mut regex_pattern);
                                    }
                                }
                                _ => regex_escape_char(ch, &mut regex_pattern),
                            }
                        }
                        regex_pattern.push('$');
                        self.patterns.is_match(regex_pattern, s).map(Value::Bool)
                    }
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
                #[cfg(not(any(feature = "regex", feature = "regex-lite")))]
                {
                    let _ = (left, right);
                    None
                }
            }
            // String concatenation (||)
            BinaryFilterOp::Concat => match (left, right) {
                (Value::String(a), Value::String(b)) => {
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::String(a), other) => {
                    let b = value_to_string(other)?;
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(a);
                    s.push_str(&b);
                    Some(Value::String(s.into()))
                }
                (other, Value::String(b)) => {
                    let a = value_to_string(other)?;
                    let mut s = String::with_capacity(a.len() + b.len());
                    s.push_str(&a);
                    s.push_str(b);
                    Some(Value::String(s.into()))
                }
                (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                _ => None,
            },
        }
    }

    fn eval_arithmetic<F1, F2>(
        &self,
        left: &Value,
        right: &Value,
        int_op: F1,
        float_op: F2,
    ) -> Option<Value>
    where
        F1: Fn(i64, i64) -> Option<i64>,
        F2: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => int_op(*a, *b).map(Value::Int64),
            (Value::Float64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a, *b))),
            (Value::Int64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a as f64, *b))),
            (Value::Float64(a), Value::Int64(b)) => Some(Value::Float64(float_op(*a, *b as f64))),
            _ => None,
        }
    }

    fn eval_modulo(&self, left: &Value, right: &Value) -> Option<Value> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) if *b != 0 => a.checked_rem(*b).map(Value::Int64),
            (Value::Float64(a), Value::Float64(b)) if *b != 0.0 => Some(Value::Float64(a % b)),
            (Value::Int64(a), Value::Float64(b)) if *b != 0.0 => {
                Some(Value::Float64(*a as f64 % b))
            }
            (Value::Float64(a), Value::Int64(b)) if *b != 0 => Some(Value::Float64(a % *b as f64)),
            _ => None,
        }
    }

    /// Evaluates `left IN right` with three-valued NULL semantics.
    ///
    /// - `NULL IN [...]` yields UNKNOWN.
    /// - If no element matches but the list contains NULLs, yields UNKNOWN.
    /// - Otherwise yields `true` on first match, `false` if none match.
    fn eval_in_operator(
        &self,
        left: &Value,
        right: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        let right_val = self.eval_expr(right, chunk, row)?;
        match right_val {
            Value::List(items) => {
                // Three-valued IN: NULL IN (...) is UNKNOWN
                if left.is_null() {
                    return Some(Value::Null);
                }
                let mut has_null = false;
                for item in items.iter() {
                    if item.is_null() {
                        has_null = true;
                    } else if Self::values_equal(left, item) {
                        return Some(Value::Bool(true));
                    }
                }
                if has_null {
                    Some(Value::Null) // no match but NULLs present: UNKNOWN
                } else {
                    Some(Value::Bool(false))
                }
            }
            _ => None,
        }
    }

    fn eval_function(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        let name_lower = name.to_lowercase();
        let name = name_lower.as_str();
        self.eval_graph_element_fn(name, args, chunk, row)
            .or_else(|| self.eval_type_fn(name, args, chunk, row))
            .or_else(|| self.eval_collection_fn(name, args, chunk, row))
            .or_else(|| self.eval_string_fn(name, args, chunk, row))
            .or_else(|| self.eval_numeric_fn(name, args, chunk, row))
            .or_else(|| self.eval_temporal_fn(name, args, chunk, row))
            .or_else(|| self.eval_path_fn(name, args, chunk, row))
            .or_else(|| self.eval_vector_fn(name, args, chunk, row))
            .or_else(|| self.eval_text_fn(name, args, chunk, row))
            .or_else(|| self.eval_session_fn(name, args, chunk, row))
    }

    fn eval_graph_element_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "id" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        // reason: entity IDs stored as i64, standard encoding
                        #[allow(clippy::cast_possible_wrap)]
                        return Some(Value::Int64(node_id.0 as i64));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        // reason: entity IDs stored as i64, standard encoding
                        #[allow(clippy::cast_possible_wrap)]
                        return Some(Value::Int64(edge_id.0 as i64));
                    }
                }
                None
            }
            "element_id" | "elementid" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    // Resolve ambiguity between node/edge by verifying against the
                    // store. VectorData::Generic stores raw Int64 values that both
                    // get_node_id and get_edge_id accept, so we must check which
                    // entity actually exists.
                    if let Some(edge_id) = col.get_edge_id(row)
                        && self.resolve_edge(edge_id).is_some()
                    {
                        return Some(Value::String(format!("e:{}", edge_id.0).into()));
                    }
                    if let Some(node_id) = col.get_node_id(row)
                        && self.resolve_node(node_id).is_some()
                    {
                        return Some(Value::String(format!("n:{}", node_id.0).into()));
                    }
                }
                None
            }
            "labels" => {
                if args.len() != 1 {
                    return None;
                }
                let node = self.node_of(&args[0], chunk, row)?;
                let mut sorted: Vec<&arcstr::ArcStr> = node.labels.iter().collect();
                sorted.sort();
                let labels: Vec<Value> = sorted
                    .into_iter()
                    .map(|l| Value::String(l.clone()))
                    .collect();
                Some(Value::List(labels.into()))
            }
            "type" => {
                if args.len() != 1 {
                    return None;
                }
                let edge = self.edge_of(&args[0], chunk, row)?;
                Some(Value::String(edge.edge_type.clone()))
            }
            // startNode(edge) and endNode(edge): the edge's source and
            // destination node, as the node's ID in a node column (RETURN
            // gives the node, and a property read reads the node's).
            "startnode" | "start_node" => {
                if args.len() != 1 {
                    return None;
                }
                let edge = self.edge_of(&args[0], chunk, row)?;
                i64::try_from(edge.src.as_u64()).ok().map(Value::Int64)
            }
            "endnode" | "end_node" => {
                if args.len() != 1 {
                    return None;
                }
                let edge = self.edge_of(&args[0], chunk, row)?;
                i64::try_from(edge.dst.as_u64()).ok().map(Value::Int64)
            }
            "property_exists" => {
                // property_exists(entity, key) - checks if a property key exists on an entity
                if args.len() != 2 {
                    return None;
                }
                let Value::String(key) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                // Try node first, then edge
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(nid) = col.get_node_id(row)
                        && let Some(node) = self.resolve_node(nid)
                    {
                        let exists = node
                            .properties
                            .iter()
                            .any(|(k, _)| k.as_str() == key.as_str());
                        return Some(Value::Bool(exists));
                    }
                    if let Some(eid) = col.get_edge_id(row)
                        && let Some(edge) = self.resolve_edge(eid)
                    {
                        let exists = edge
                            .properties
                            .iter()
                            .any(|(k, _)| k.as_str() == key.as_str());
                        return Some(Value::Bool(exists));
                    }
                }
                Some(Value::Bool(false))
            }
            "haslabel" => {
                // hasLabel(node, label) - checks if a node has a specific label
                if args.len() != 2 {
                    return None;
                }
                // First arg is the node variable
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                // Second arg is the label to check
                let Value::String(label) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                // Check if the node has this label
                let node = self.resolve_node(node_id)?;
                let has_label = node.labels.iter().any(|l| l.as_str() == label.as_str());
                Some(Value::Bool(has_label))
            }
            "issource" => {
                // isSource(node, edge) - checks if node is the source of edge
                if args.len() != 2 {
                    return None;
                }
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                let edge_id = if let FilterExpression::Variable(var) = &args[1] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_edge_id(row)?
                } else {
                    return None;
                };
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::Bool(edge.src == node_id))
            }
            "isdestination" => {
                // isDestination(node, edge) - checks if node is the destination of edge
                if args.len() != 2 {
                    return None;
                }
                let node_id = if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_node_id(row)?
                } else {
                    return None;
                };
                let edge_id = if let FilterExpression::Variable(var) = &args[1] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    col.get_edge_id(row)?
                } else {
                    return None;
                };
                let edge = self.resolve_edge(edge_id)?;
                Some(Value::Bool(edge.dst == node_id))
            }
            "isdirected" => {
                // isDirected(edge) - checks if an edge is directed (always true in LPG)
                if args.len() != 1 {
                    return None;
                }
                // In LPG, all edges are directed
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    // If the column contains an edge ID, it's directed
                    if col.get_edge_id(row).is_some() {
                        return Some(Value::Bool(true));
                    }
                }
                Some(Value::Bool(false))
            }
            _ => None,
        }
    }

    fn eval_type_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "tostring" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let s = match &val {
                    Value::String(s) => s.to_string(),
                    Value::Int64(i) => i.to_string(),
                    Value::Float64(f) => f.to_string(),
                    Value::Bool(b) => b.to_string(),
                    Value::Null => return Some(Value::Null),
                    // ISO 8601 for temporal values, `[1, 2]` for lists.
                    _ => val.to_string(),
                };
                Some(Value::String(s.into()))
            }
            "tointeger" | "toint" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    // reason: toInteger() intentionally truncates float to int per GQL spec
                    #[allow(clippy::cast_possible_truncation)]
                    Value::Float64(f) => Some(Value::Int64(f as i64)),
                    Value::Bool(b) => Some(Value::Int64(i64::from(b))),
                    Value::String(s) => s.parse::<i64>().ok().map(Value::Int64),
                    _ => None,
                }
            }
            "tofloat" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64(i as f64)),
                    Value::Float64(f) => Some(Value::Float64(f)),
                    Value::String(s) => s.parse::<f64>().ok().map(Value::Float64),
                    _ => None,
                }
            }
            "toboolean" | "tobool" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Bool(b) => Some(Value::Bool(b)),
                    Value::String(s) => match s.to_lowercase().as_str() {
                        "true" => Some(Value::Bool(true)),
                        "false" => Some(Value::Bool(false)),
                        _ => None,
                    },
                    _ => None,
                }
            }
            "tolist" => {
                // toList(value) - wraps a scalar in a single-element list, or returns list as-is
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(_) => Some(val),
                    Value::Null => Some(Value::Null),
                    other => Some(Value::List(vec![other].into())),
                }
            }
            "totypedlist" => {
                // toTypedList(value, element_type) - coerces value to a list with typed elements
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let Value::String(elem_type) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                if matches!(val, Value::Null) {
                    return Some(Value::Null);
                }
                // Wrap scalar in a list first
                let items = match val {
                    Value::List(items) => items.to_vec(),
                    other => vec![other],
                };
                // Coerce each element to the target type
                let coerced: Option<Vec<Value>> = items
                    .into_iter()
                    .map(|v| Self::coerce_to_type(v, &elem_type))
                    .collect();
                coerced.map(|v| Value::List(v.into()))
            }
            "istyped" => {
                // isTyped(value, type_name) - checks if a value has a specific GQL type
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let Value::String(type_name) = self.eval_expr(&args[1], chunk, row)? else {
                    return None;
                };
                let matches = match type_name.to_uppercase().as_str() {
                    "BOOLEAN" | "BOOL" => matches!(val, Value::Bool(_)),
                    "INTEGER" | "INT" | "INT64" => matches!(val, Value::Int64(_)),
                    "FLOAT" | "FLOAT64" | "DOUBLE" => matches!(val, Value::Float64(_)),
                    "STRING" => matches!(val, Value::String(_)),
                    "LIST" => matches!(val, Value::List(_)),
                    "MAP" | "RECORD" => matches!(val, Value::Map(_)),
                    "NULL" => matches!(val, Value::Null),
                    "DATE" => matches!(val, Value::Date(_)),
                    "TIME" => matches!(val, Value::Time(_)),
                    "DATETIME" | "TIMESTAMP" => matches!(val, Value::Timestamp(_)),
                    "DURATION" => matches!(val, Value::Duration(_)),
                    "PATH" => matches!(val, Value::Path { .. }),
                    "NODE" | "EDGE" | "GRAPH" => false,
                    s if s.starts_with("LIST<") && s.ends_with('>') => {
                        let elem_type = &s[5..s.len() - 1];
                        match &val {
                            Value::List(items) => {
                                items.iter().all(|v| Self::value_matches_type(v, elem_type))
                            }
                            _ => false,
                        }
                    }
                    _ => false,
                };
                Some(Value::Bool(matches))
            }
            _ => None,
        }
    }

    fn eval_collection_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "size" | "length" | "cardinality" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    // reason: collection lengths fit i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::List(items) => Some(Value::Int64(items.len() as i64)),
                    // A string's size is its number of characters, as
                    // `char_length` counts them (openCypher `size()`), not
                    // its UTF-8 bytes (`octet_length`).
                    // reason: string lengths fit i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::String(s) => Some(Value::Int64(s.chars().count() as i64)),
                    // reason: path lengths fit i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::Path { edges, .. } => Some(Value::Int64(edges.len() as i64)),
                    _ => None,
                }
            }
            "coalesce" => {
                for arg in args {
                    if let Some(val) = self.eval_expr(arg, chunk, row)
                        && !matches!(val, Value::Null)
                    {
                        return Some(val);
                    }
                }
                Some(Value::Null)
            }
            "exists" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row);
                Some(Value::Bool(
                    val.is_some() && !matches!(val, Some(Value::Null)),
                ))
            }
            "keys" => {
                if args.len() != 1 {
                    return None;
                }
                // keys(n) or keys(r) on a node or edge variable: the property
                // keys from the store, sorted (as `properties` and map keys
                // are), so their order does not depend on how they are stored
                if let FilterExpression::Variable(var) = &args[0]
                    && let Some(properties) = self.element_properties(var, chunk, row)
                {
                    let mut keys: Vec<Value> = properties
                        .into_iter()
                        .map(|(k, _)| Value::String(k.as_str().into()))
                        .collect();
                    keys.sort_by(|a, b| a.as_str().cmp(&b.as_str()));
                    return Some(Value::List(keys.into()));
                }
                // keys(map) on a map value
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Map(map) => {
                        let keys: Vec<Value> = map
                            .keys()
                            .map(|k| Value::String(k.as_str().into()))
                            .collect();
                        Some(Value::List(keys.into()))
                    }
                    _ => None,
                }
            }
            "properties" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let map: std::collections::BTreeMap<PropertyKey, Value> = self
                        .element_properties(var, chunk, row)?
                        .into_iter()
                        .collect();
                    return Some(Value::Map(Arc::new(map)));
                }
                None
            }
            // property_values(n) returns all property values of a node or edge as a flat list.
            // Used by Gremlin values() with no keys.
            "property_values" => {
                if args.len() != 1 {
                    return None;
                }
                // In key order, so they line up with keys(n) (a store keeps
                // properties in no particular order).
                if let FilterExpression::Variable(var) = &args[0] {
                    let mut properties: Vec<(PropertyKey, Value)> = self
                        .element_properties(var, chunk, row)?
                        .into_iter()
                        .collect();
                    properties.sort_by(|(a, _), (b, _)| a.as_str().cmp(b.as_str()));
                    let values: Vec<Value> = properties.into_iter().map(|(_, v)| v).collect();
                    return Some(Value::List(values.into()));
                }
                None
            }
            "head" => {
                // head(list) - returns the first element of a list
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => items.first().cloned(),
                    _ => None,
                }
            }
            "tail" => {
                // tail(list) - returns all elements except the first
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        if items.is_empty() {
                            Some(Value::List(vec![].into()))
                        } else {
                            Some(Value::List(items[1..].to_vec().into()))
                        }
                    }
                    _ => None,
                }
            }
            "last" => {
                // last(list) - returns the last element of a list
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => items.last().cloned(),
                    _ => None,
                }
            }
            "reverse" => {
                // reverse(list) - returns the list in reverse order
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        let reversed: Vec<Value> = items.iter().rev().cloned().collect();
                        Some(Value::List(reversed.into()))
                    }
                    Value::String(s) => {
                        let reversed: String = s.chars().rev().collect();
                        Some(Value::String(reversed.into()))
                    }
                    _ => None,
                }
            }
            // vector(list) - converts a list of numbers to a Vector
            "vector" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => {
                        let floats: Vec<f32> = items
                            .iter()
                            .filter_map(|v| match v {
                                // reason: vector() intentionally converts f64 to f32 for storage
                                #[allow(clippy::cast_possible_truncation)]
                                Value::Float64(f) => Some(*f as f32),
                                Value::Int64(i) => Some(*i as f32),
                                _ => None,
                            })
                            .collect();
                        if floats.len() == items.len() {
                            Some(Value::Vector(floats.into()))
                        } else {
                            None
                        }
                    }
                    Value::Vector(v) => Some(Value::Vector(v)),
                    _ => None,
                }
            }
            "all_different" => {
                // ALL_DIFFERENT - ISO GQL predicate (G113)
                // Two calling conventions:
                //   all_different(list)          - check list elements are distinct
                //   ALL_DIFFERENT(var1, var2, ..) - check graph elements are distinct
                if args.is_empty() {
                    return Some(Value::Bool(true));
                }
                if args.len() == 1 {
                    // Single-argument: treat as list check
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::List(items) => {
                            let mut seen = std::collections::HashSet::new();
                            let all_diff = items.iter().all(|item| {
                                let key = format!("{item:?}");
                                seen.insert(key)
                            });
                            Some(Value::Bool(all_diff))
                        }
                        _ => Some(Value::Bool(true)),
                    };
                }
                // Multi-argument: compare element IDs
                let mut ids: Vec<u64> = Vec::with_capacity(args.len());
                for arg in args {
                    let FilterExpression::Variable(var) = arg else {
                        return None;
                    };
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(nid) = col.get_node_id(row) {
                        ids.push(nid.0);
                    } else {
                        ids.push(col.get_edge_id(row)?.0);
                    }
                }
                let length = ids.len();
                ids.sort_unstable();
                ids.dedup();
                Some(Value::Bool(ids.len() == length))
            }
            "same" => {
                // SAME - ISO GQL predicate (G114)
                // Two calling conventions:
                //   same(list)          - check list elements are equal
                //   SAME(var1, var2, ..) - check graph elements are identical
                if args.is_empty() {
                    return Some(Value::Bool(true));
                }
                if args.len() == 1 {
                    // Single-argument: treat as list check
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::List(items) => {
                            let all_same = if items.is_empty() {
                                true
                            } else {
                                items.iter().all(|item| item == &items[0])
                            };
                            Some(Value::Bool(all_same))
                        }
                        _ => Some(Value::Bool(true)),
                    };
                }
                // Multi-argument: compare element IDs
                let mut first_id: Option<u64> = None;
                for arg in args {
                    let FilterExpression::Variable(var) = arg else {
                        return None;
                    };
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    let current_id = if let Some(nid) = col.get_node_id(row) {
                        nid.0
                    } else {
                        col.get_edge_id(row)?.0
                    };
                    match first_id {
                        None => first_id = Some(current_id),
                        Some(fid) if fid != current_id => return Some(Value::Bool(false)),
                        _ => {}
                    }
                }
                Some(Value::Bool(true))
            }
            "range" => {
                if args.len() < 2 || args.len() > 3 {
                    return None;
                }
                let start = self.eval_expr(&args[0], chunk, row)?;
                let stop = self.eval_expr(&args[1], chunk, row)?;
                let Value::Int64(start_val) = start else {
                    return None;
                };
                let Value::Int64(end_val) = stop else {
                    return None;
                };
                let step = if args.len() == 3 {
                    let s = self.eval_expr(&args[2], chunk, row)?;
                    let Value::Int64(sv) = s else {
                        return None;
                    };
                    if sv == 0 {
                        return None;
                    }
                    sv
                } else {
                    1
                };
                let mut result = Vec::new();
                let mut current = start_val;
                if step > 0 {
                    while current <= end_val {
                        result.push(Value::Int64(current));
                        current += step;
                    }
                } else {
                    while current >= end_val {
                        result.push(Value::Int64(current));
                        current += step;
                    }
                }
                Some(Value::List(result.into()))
            }
            "string_join" => {
                // string_join(list, separator) - join list elements with separator
                if args.len() != 2 {
                    return None;
                }
                let list_val = self.eval_expr(&args[0], chunk, row)?;
                let sep_val = self.eval_expr(&args[1], chunk, row)?;
                match (list_val, sep_val) {
                    (Value::List(items), Value::String(sep)) => {
                        let sep_str: &str = &sep;
                        let joined: String = items
                            .iter()
                            .filter_map(|v| match v {
                                Value::String(s) => Some(s.to_string()),
                                Value::Int64(i) => Some(i.to_string()),
                                Value::Float64(f) => Some(f.to_string()),
                                Value::Bool(b) => Some(b.to_string()),
                                Value::Null => None,
                                other => Some(format!("{other}")),
                            })
                            .collect::<Vec<String>>()
                            .join(sep_str);
                        Some(Value::String(joined.into()))
                    }
                    (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_string_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "trim" => {
                if args.len() == 1 {
                    // Simple trim(string) - trim whitespace
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    return match val {
                        Value::String(s) => Some(Value::String(s.trim().to_string().into())),
                        _ => None,
                    };
                }
                if args.len() == 3 {
                    // Extended trim(string, chars, mode)
                    // mode: 0=both, 1=leading, 2=trailing
                    let val = self.eval_expr(&args[0], chunk, row)?;
                    let chars_val = self.eval_expr(&args[1], chunk, row)?;
                    let mode_val = self.eval_expr(&args[2], chunk, row)?;
                    let Value::String(s) = val else { return None };
                    let Value::String(chars) = chars_val else {
                        return None;
                    };
                    let Value::Int64(mode) = mode_val else {
                        return None;
                    };
                    let char_set: Vec<char> = chars.chars().collect();
                    let result = match mode {
                        0 => s.trim_matches(|c| char_set.contains(&c)).to_string(),
                        1 => s.trim_start_matches(|c| char_set.contains(&c)).to_string(),
                        2 => s.trim_end_matches(|c| char_set.contains(&c)).to_string(),
                        _ => return None,
                    };
                    return Some(Value::String(result.into()));
                }
                None
            }
            "ltrim" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.trim_start().to_string().into())),
                    _ => None,
                }
            }
            "rtrim" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.trim_end().to_string().into())),
                    _ => None,
                }
            }
            "replace" => {
                if args.len() != 3 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let search = self.eval_expr(&args[1], chunk, row)?;
                let replacement = self.eval_expr(&args[2], chunk, row)?;
                match (&val, &search, &replacement) {
                    (Value::String(s), Value::String(from), Value::String(to)) => {
                        Some(Value::String(s.replace(from.as_str(), to.as_str()).into()))
                    }
                    _ => None,
                }
            }
            "substring" => {
                if args.len() < 2 || args.len() > 3 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let start = self.eval_expr(&args[1], chunk, row)?;
                let Value::String(s) = val else {
                    return None;
                };
                let Value::Int64(start_idx) = start else {
                    return None;
                };
                // reason: clamped to >= 0 by max(0), safe to cast to usize
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let start_idx = start_idx.max(0) as usize;
                if args.len() == 3 {
                    let length = self.eval_expr(&args[2], chunk, row)?;
                    let Value::Int64(len) = length else {
                        return None;
                    };
                    // reason: clamped to >= 0 by max(0), safe to cast to usize
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let len = len.max(0) as usize;
                    let chars: String = s.chars().skip(start_idx).take(len).collect();
                    Some(Value::String(chars.into()))
                } else {
                    let chars: String = s.chars().skip(start_idx).collect();
                    Some(Value::String(chars.into()))
                }
            }
            "split" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let delim = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &delim) {
                    (Value::String(s), Value::String(d)) => {
                        let parts: Vec<Value> = s
                            .split(d.as_str())
                            .map(|p| Value::String(p.to_string().into()))
                            .collect();
                        Some(Value::List(parts.into()))
                    }
                    _ => None,
                }
            }
            "toupper" | "upper" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.to_uppercase().into())),
                    _ => None,
                }
            }
            "tolower" | "lower" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s.to_lowercase().into())),
                    _ => None,
                }
            }
            "char_length" | "charlength" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    // reason: char count fits i64 for practical string sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::String(s) => Some(Value::Int64(s.chars().count() as i64)),
                    _ => None,
                }
            }
            "left" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let len = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &len) {
                    (Value::String(s), Value::Int64(n)) => {
                        // reason: clamped to >= 0 by max(0), safe to cast to usize
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let n = (*n).max(0) as usize;
                        let result: String = s.chars().take(n).collect();
                        Some(Value::String(result.into()))
                    }
                    _ => None,
                }
            }
            "right" => {
                if args.len() != 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let len = self.eval_expr(&args[1], chunk, row)?;
                match (&val, &len) {
                    (Value::String(s), Value::Int64(n)) => {
                        // reason: clamped to >= 0 by max(0), safe to cast to usize
                        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                        let n = (*n).max(0) as usize;
                        let char_count = s.chars().count();
                        let skip = char_count.saturating_sub(n);
                        let result: String = s.chars().skip(skip).collect();
                        Some(Value::String(result.into()))
                    }
                    _ => None,
                }
            }
            "octet_length" | "byte_length" => {
                // octet_length(string) - byte length
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    // reason: string byte length fits i64 for practical sizes
                    #[allow(clippy::cast_possible_wrap)]
                    Value::String(s) => Some(Value::Int64(s.len() as i64)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "normalize" => {
                // normalize(string) - returns the string as-is (NFC normalization).
                // Rust strings are valid UTF-8; full NFC normalization requires the
                // unicode-normalization crate, which is deferred to Phase 2.
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => Some(Value::String(s)),
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "isnormalized" => {
                // IS [NFC|NFD|NFKC|NFKD] NORMALIZED - check Unicode normalization form.
                // Args: (string_value) or (string_value, form_name)
                // Default form is NFC when not specified.
                if args.is_empty() || args.len() > 2 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                let form = if args.len() == 2 {
                    match self.eval_expr(&args[1], chunk, row)? {
                        Value::String(s) => s.to_uppercase(),
                        _ => return None,
                    }
                } else {
                    "NFC".to_string()
                };
                match val {
                    Value::String(ref s) => {
                        use unicode_normalization::UnicodeNormalization;
                        let normalized = match form.as_str() {
                            "NFC" => s.nfc().collect::<String>() == s.as_ref(),
                            "NFD" => s.nfd().collect::<String>() == s.as_ref(),
                            "NFKC" => s.nfkc().collect::<String>() == s.as_ref(),
                            "NFKD" => s.nfkd().collect::<String>() == s.as_ref(),
                            _ => return None,
                        };
                        Some(Value::Bool(normalized))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_numeric_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "abs" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i.abs())),
                    Value::Float64(f) => Some(Value::Float64(f.abs())),
                    _ => None,
                }
            }
            "ceil" | "ceiling" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Float64(f.ceil())),
                    _ => None,
                }
            }
            "floor" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Float64(f.floor())),
                    _ => None,
                }
            }
            "round" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Float64(f.round())),
                    _ => None,
                }
            }
            "sqrt" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).sqrt())),
                    Value::Float64(f) => Some(Value::Float64(f.sqrt())),
                    _ => None,
                }
            }
            "sign" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i.signum())),
                    Value::Float64(f) => {
                        if f > 0.0 {
                            Some(Value::Int64(1))
                        } else if f < 0.0 {
                            Some(Value::Int64(-1))
                        } else {
                            Some(Value::Int64(0))
                        }
                    }
                    _ => None,
                }
            }
            "power" | "pow" => {
                if args.len() != 2 {
                    return None;
                }
                let base_val = self.eval_expr(&args[0], chunk, row)?;
                let exp_val = self.eval_expr(&args[1], chunk, row)?;
                let base = match base_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                let exponent = match exp_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                Some(Value::Float64(base.powf(exponent)))
            }
            "rand" | "random" => {
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                row.hash(&mut hasher);
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()?
                    .as_nanos()
                    .hash(&mut hasher);
                let hash = hasher.finish();
                let random = (hash as f64) / (u64::MAX as f64);
                Some(Value::Float64(random))
            }
            "log" | "ln" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).ln())),
                    Value::Float64(f) => Some(Value::Float64(f.ln())),
                    _ => None,
                }
            }
            "log10" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).log10())),
                    Value::Float64(f) => Some(Value::Float64(f.log10())),
                    _ => None,
                }
            }
            "log2" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).log2())),
                    Value::Float64(f) => Some(Value::Float64(f.log2())),
                    _ => None,
                }
            }
            "exp" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).exp())),
                    Value::Float64(f) => Some(Value::Float64(f.exp())),
                    _ => None,
                }
            }
            "e" => Some(Value::Float64(std::f64::consts::E)),
            "pi" => Some(Value::Float64(std::f64::consts::PI)),
            "sin" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).sin())),
                    Value::Float64(f) => Some(Value::Float64(f.sin())),
                    _ => None,
                }
            }
            "cos" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).cos())),
                    Value::Float64(f) => Some(Value::Float64(f.cos())),
                    _ => None,
                }
            }
            "tan" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).tan())),
                    Value::Float64(f) => Some(Value::Float64(f.tan())),
                    _ => None,
                }
            }
            "asin" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).asin())),
                    Value::Float64(f) => Some(Value::Float64(f.asin())),
                    _ => None,
                }
            }
            "acos" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).acos())),
                    Value::Float64(f) => Some(Value::Float64(f.acos())),
                    _ => None,
                }
            }
            "atan" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).atan())),
                    Value::Float64(f) => Some(Value::Float64(f.atan())),
                    _ => None,
                }
            }
            "atan2" => {
                if args.len() != 2 {
                    return None;
                }
                let y_val = self.eval_expr(&args[0], chunk, row)?;
                let x_val = self.eval_expr(&args[1], chunk, row)?;
                let y = match y_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                let x = match x_val {
                    Value::Int64(i) => i as f64,
                    Value::Float64(f) => f,
                    _ => return None,
                };
                Some(Value::Float64(y.atan2(x)))
            }
            "degrees" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).to_degrees())),
                    Value::Float64(f) => Some(Value::Float64(f.to_degrees())),
                    _ => None,
                }
            }
            "radians" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Float64((i as f64).to_radians())),
                    Value::Float64(f) => Some(Value::Float64(f.to_radians())),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_temporal_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "date" | "todate" => {
                if args.is_empty() {
                    return Some(Value::Date(grafeo_common::types::Date::today()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => grafeo_common::types::Date::parse(&s).map(Value::Date),
                    Value::Timestamp(ts) => Some(Value::Date(ts.to_date())),
                    Value::Date(_) => Some(val),
                    Value::Map(m) => {
                        let year = i32::try_from(map_int(&m, "year")?).ok()?;
                        let month = u32::try_from(map_int_or(&m, "month", 1)?).ok()?;
                        let day = u32::try_from(map_int_or(&m, "day", 1)?).ok()?;
                        grafeo_common::types::Date::from_ymd(year, month, day).map(Value::Date)
                    }
                    _ => None,
                }
            }
            "time" | "totime" | "local_time" => {
                if args.is_empty() {
                    return Some(Value::Time(grafeo_common::types::Time::now()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => grafeo_common::types::Time::parse(&s).map(Value::Time),
                    Value::Timestamp(ts) => Some(Value::Time(ts.to_time())),
                    Value::Time(_) => Some(val),
                    Value::Map(m) => {
                        let hour = u32::try_from(map_int_or(&m, "hour", 0)?).ok()?;
                        let minute = u32::try_from(map_int_or(&m, "minute", 0)?).ok()?;
                        let second = u32::try_from(map_int_or(&m, "second", 0)?).ok()?;
                        let nanosecond = u32::try_from(map_int_or(&m, "nanosecond", 0)?).ok()?;
                        grafeo_common::types::Time::from_hms_nano(hour, minute, second, nanosecond)
                            .map(Value::Time)
                    }
                    _ => None,
                }
            }
            "datetime" | "localdatetime" | "local_datetime" | "todatetime" => {
                if args.is_empty() {
                    return Some(Value::Timestamp(grafeo_common::types::Timestamp::now()));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        // Parse ISO datetime: try Date first, then full timestamp
                        if let Some(d) = grafeo_common::types::Date::parse(&s) {
                            return Some(Value::Timestamp(d.to_timestamp()));
                        }
                        // Try full ISO format: YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM]
                        if let Some(pos) = s.find('T') {
                            let date_part = &s[..pos];
                            let time_part = &s[pos + 1..];
                            if let (Some(d), Some(t)) = (
                                grafeo_common::types::Date::parse(date_part),
                                grafeo_common::types::Time::parse(time_part),
                            ) {
                                return Some(Value::Timestamp(
                                    grafeo_common::types::Timestamp::from_date_time(d, t),
                                ));
                            }
                        }
                        None
                    }
                    Value::Timestamp(_) => Some(val),
                    // `{epochMillis: n}` and `{epochSeconds: n, nanosecond: m}`:
                    // the instant that long after 1970-01-01T00:00:00Z.
                    Value::Map(m) if m.contains_key(&PropertyKey::from("epochMillis")) => {
                        let micros = map_int(&m, "epochMillis")?.checked_mul(1_000)?;
                        Some(Value::Timestamp(
                            grafeo_common::types::Timestamp::from_micros(micros),
                        ))
                    }
                    Value::Map(m) if m.contains_key(&PropertyKey::from("epochSeconds")) => {
                        let nanosecond = map_int_or(&m, "nanosecond", 0)?;
                        let micros = map_int(&m, "epochSeconds")?
                            .checked_mul(1_000_000)?
                            .checked_add(nanosecond.div_euclid(1_000))?;
                        Some(Value::Timestamp(
                            grafeo_common::types::Timestamp::from_micros(micros),
                        ))
                    }
                    Value::Map(m) => {
                        let year = i32::try_from(map_int(&m, "year")?).ok()?;
                        let month = u32::try_from(map_int_or(&m, "month", 1)?).ok()?;
                        let day = u32::try_from(map_int_or(&m, "day", 1)?).ok()?;
                        let hour = u32::try_from(map_int_or(&m, "hour", 0)?).ok()?;
                        let minute = u32::try_from(map_int_or(&m, "minute", 0)?).ok()?;
                        let second = u32::try_from(map_int_or(&m, "second", 0)?).ok()?;
                        let nanosecond = u32::try_from(map_int_or(&m, "nanosecond", 0)?).ok()?;
                        let date = grafeo_common::types::Date::from_ymd(year, month, day)?;
                        let time = grafeo_common::types::Time::from_hms_nano(
                            hour, minute, second, nanosecond,
                        )?;
                        Some(Value::Timestamp(
                            grafeo_common::types::Timestamp::from_date_time(date, time),
                        ))
                    }
                    _ => None,
                }
            }
            "duration" | "toduration" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        grafeo_common::types::Duration::parse(&s).map(Value::Duration)
                    }
                    Value::Duration(_) => Some(val),
                    Value::Map(m) => {
                        let years = map_int_or(&m, "years", 0)?;
                        let months = map_int_or(&m, "months", 0)?;
                        let weeks = map_int_or(&m, "weeks", 0)?;
                        let days = map_int_or(&m, "days", 0)?;
                        let hours = map_int_or(&m, "hours", 0)?;
                        let minutes = map_int_or(&m, "minutes", 0)?;
                        let seconds = map_int_or(&m, "seconds", 0)?;
                        let nanoseconds = map_int_or(&m, "nanoseconds", 0)?;
                        let total_months = years * 12 + months;
                        let total_days = weeks * 7 + days;
                        let total_nanos = hours * 3_600_000_000_000
                            + minutes * 60_000_000_000
                            + seconds * 1_000_000_000
                            + nanoseconds;
                        Some(Value::Duration(grafeo_common::types::Duration::new(
                            total_months,
                            total_days,
                            total_nanos,
                        )))
                    }
                    _ => None,
                }
            }
            "tozoneddatetime" | "zoneddatetime" | "zoned_datetime" => {
                if args.is_empty() {
                    return Some(Value::ZonedDatetime(
                        grafeo_common::types::ZonedDatetime::from_timestamp_offset(
                            grafeo_common::types::Timestamp::now(),
                            0,
                        ),
                    ));
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::String(s) => {
                        grafeo_common::types::ZonedDatetime::parse(&s).map(Value::ZonedDatetime)
                    }
                    Value::Timestamp(ts) => Some(Value::ZonedDatetime(
                        grafeo_common::types::ZonedDatetime::from_timestamp_offset(ts, 0),
                    )),
                    Value::ZonedDatetime(_) => Some(val),
                    _ => None,
                }
            }
            "tozonedtime" | "zonedtime" => {
                let val = self.eval_expr(args.first()?, chunk, row)?;
                match val {
                    Value::String(s) => {
                        let t = grafeo_common::types::Time::parse(&s)?;
                        if t.offset_seconds().is_some() {
                            Some(Value::Time(t))
                        } else {
                            None
                        }
                    }
                    Value::Time(t) if t.offset_seconds().is_some() => Some(val),
                    _ => None,
                }
            }
            "current_date" | "currentdate" => {
                Some(Value::Date(grafeo_common::types::Date::today()))
            }
            "current_time" | "currenttime" => Some(Value::Time(grafeo_common::types::Time::now())),
            "now" | "current_timestamp" | "currenttimestamp" => {
                Some(Value::Timestamp(grafeo_common::types::Timestamp::now()))
            }
            "timestamp" => Some(Value::Int64(
                grafeo_common::types::Timestamp::now().as_millis(),
            )),
            // The component functions read what the property form (`d.year`)
            // reads.
            "year" | "month" | "day" | "hour" | "minute" | "second" => self
                .eval_expr(args.first()?, chunk, row)?
                .temporal_component(name),
            "date_trunc" | "truncate" => {
                if args.len() < 2 {
                    return None;
                }
                let unit = match self.eval_expr(&args[0], chunk, row)? {
                    Value::String(s) => s.to_lowercase(),
                    _ => return None,
                };
                let val = self.eval_expr(&args[1], chunk, row)?;
                match val {
                    Value::Date(d) => Some(Value::Date(d.truncate(&unit)?)),
                    Value::Time(t) => Some(Value::Time(t.truncate(&unit)?)),
                    Value::Timestamp(ts) => Some(Value::Timestamp(ts.truncate(&unit)?)),
                    Value::ZonedDatetime(zdt) => Some(Value::ZonedDatetime(zdt.truncate(&unit)?)),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn eval_path_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "path" => {
                if args.len() == 2 {
                    // path(nodes_list, edges_list) - construct from component lists
                    let nodes_val = self.eval_expr(&args[0], chunk, row)?;
                    let edges_val = self.eval_expr(&args[1], chunk, row)?;
                    match (&nodes_val, &edges_val) {
                        (Value::Null, _) | (_, Value::Null) => Some(Value::Null),
                        (Value::List(nodes), Value::List(edges)) => {
                            if nodes.is_empty() || edges.len() != nodes.len() - 1 {
                                return None;
                            }
                            Some(Value::Path {
                                nodes: Arc::from(nodes.as_ref()),
                                edges: Arc::from(edges.as_ref()),
                            })
                        }
                        _ => None,
                    }
                } else {
                    // path(node1, edge1, node2, ...) - alternating nodes and edges
                    if args.is_empty() || args.len().is_multiple_of(2) {
                        return None;
                    }
                    let mut nodes = Vec::with_capacity(args.len() / 2 + 1);
                    let mut edges = Vec::with_capacity(args.len() / 2);
                    for (i, arg) in args.iter().enumerate() {
                        let val = self.eval_expr(arg, chunk, row)?;
                        if i % 2 == 0 {
                            nodes.push(val);
                        } else {
                            edges.push(val);
                        }
                    }
                    Some(Value::Path {
                        nodes: nodes.into(),
                        edges: edges.into(),
                    })
                }
            }
            "nodes" => {
                // nodes(path) - extracts nodes from a path value
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        // Resolve Int64 node IDs to property maps for property access
                        let resolved: Vec<Value> = nodes
                            .iter()
                            .map(|n| {
                                if let Value::Int64(id) = n {
                                    // reason: ID encoding: i64 <-> u64 round-trip
                                    #[allow(clippy::cast_sign_loss)]
                                    let node_id = NodeId(*id as u64);
                                    if let Some(node) = self.resolve_node(node_id) {
                                        let mut map = BTreeMap::new();
                                        // reason: entity IDs stored as i64, standard encoding
                                        #[allow(clippy::cast_possible_wrap)]
                                        map.insert(
                                            PropertyKey::new("_id"),
                                            Value::Int64(node.id.as_u64() as i64),
                                        );
                                        let labels: Vec<Value> = node
                                            .labels
                                            .iter()
                                            .map(|l| Value::String(l.clone()))
                                            .collect();
                                        map.insert(
                                            PropertyKey::new("_labels"),
                                            Value::List(labels.into()),
                                        );
                                        for (key, value) in &node.properties {
                                            map.insert(key.clone(), value.clone());
                                        }
                                        Value::Map(Arc::new(map))
                                    } else {
                                        n.clone()
                                    }
                                } else {
                                    n.clone()
                                }
                            })
                            .collect();
                        Some(Value::List(resolved.into()))
                    }
                    Value::Map(map) => map.get(&PropertyKey::from("nodes")).cloned(),
                    Value::List(items) => {
                        // Legacy: alternating node, edge, node, edge, ...
                        let nodes: Vec<Value> = items.iter().step_by(2).cloned().collect();
                        Some(Value::List(nodes.into()))
                    }
                    _ => None,
                }
            }
            "edges" | "relationships" => {
                // edges(path) / relationships(path) - extracts edges from a path value
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { edges, .. } => Some(Value::List(edges)),
                    Value::Map(map) => map.get(&PropertyKey::from("edges")).cloned(),
                    Value::List(items) => {
                        // Legacy: alternating node, edge, node, edge, ...
                        let edges: Vec<Value> = items.iter().skip(1).step_by(2).cloned().collect();
                        Some(Value::List(edges.into()))
                    }
                    _ => None,
                }
            }
            "isacyclic" => {
                // isAcyclic(path) - true if no node appears more than once
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        let mut seen = std::collections::HashSet::new();
                        let acyclic = nodes
                            .iter()
                            .all(|n| seen.insert(HashableValue::new(n.clone())));
                        Some(Value::Bool(acyclic))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "issimple" => {
                // isSimple(path) - true if no node repeats except possibly first == last
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { nodes, .. } => {
                        if nodes.is_empty() {
                            return Some(Value::Bool(true));
                        }
                        let mut seen = std::collections::HashSet::new();
                        let simple = nodes.iter().enumerate().all(|(i, n)| {
                            let hv = HashableValue::new(n.clone());
                            if !seen.insert(hv) {
                                // Duplicate allowed only if last node == first node
                                i == nodes.len() - 1 && n == &nodes[0]
                            } else {
                                true
                            }
                        });
                        Some(Value::Bool(simple))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            "istrail" => {
                // isTrail(path) - true if no edge repeats
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Path { edges, .. } => {
                        let mut seen = std::collections::HashSet::new();
                        let trail = edges
                            .iter()
                            .all(|e| seen.insert(HashableValue::new(e.clone())));
                        Some(Value::Bool(trail))
                    }
                    Value::Null => Some(Value::Null),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Coerces a `Value` to a borrowed or owned float slice for vector math.
    ///
    /// Handles both `Value::Vector` (native) and `Value::List` (GQL inline literal
    /// form `[0.9, 0.1, 0.0]` which the parser translates to a list of Float64/Int64).
    fn coerce_to_float_vec(val: &Value) -> Option<std::borrow::Cow<'_, [f32]>> {
        match val {
            Value::Vector(v) => Some(std::borrow::Cow::Borrowed(v.as_ref())),
            Value::List(list) => {
                let mut vec = Vec::with_capacity(list.len());
                for item in list.iter() {
                    match item {
                        // GQL numeric literals are Float64; f64→f32 precision loss is intentional
                        // since all HNSW indexes store f32 components.
                        #[allow(clippy::cast_possible_truncation)]
                        Value::Float64(f) => vec.push(*f as f32),
                        Value::Int64(i) => vec.push(*i as f32),
                        _ => return None,
                    }
                }
                Some(std::borrow::Cow::Owned(vec))
            }
            _ => None,
        }
    }

    fn eval_vector_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "cosine_similarity" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::cosine_similarity(&a, &b) as f64,
                ))
            }
            "cosine_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                // cosine_distance = 1 - cosine_similarity, range [0, 2]
                Some(Value::Float64(
                    1.0 - crate::index::vector::cosine_similarity(&a, &b) as f64,
                ))
            }
            "euclidean_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::euclidean_distance(&a, &b) as f64,
                ))
            }
            "dot_product" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::dot_product(&a, &b) as f64
                ))
            }
            "manhattan_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = Self::coerce_to_float_vec(&a_val)?;
                let b = Self::coerce_to_float_vec(&b_val)?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::manhattan_distance(&a, &b) as f64,
                ))
            }
            _ => None,
        }
    }

    #[cfg(feature = "text-index")]
    fn eval_text_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "text_score" | "text_match" => {
                if args.len() != 2 {
                    return None;
                }

                // First arg must be a property access (e.g., n.body)
                let FilterExpression::Property { variable, property } = &args[0] else {
                    return None;
                };

                // Get node_id from the chunk
                let col_idx = *self.variable_columns.get(variable.as_str())?;
                let col = chunk.column(col_idx)?;
                let node_id = col.get_node_id(row)?;

                // Second arg: query string
                let query_val = self.eval_expr(&args[1], chunk, row)?;
                let Value::String(query_str) = &query_val else {
                    return None;
                };

                // Get the node's labels and try each for a text index match
                let node = self.resolve_node(node_id)?;
                let score = node
                    .labels
                    .iter()
                    .find_map(|label| self.store.score_text(node_id, label, property, query_str))?;

                if name == "text_match" {
                    Some(Value::Bool(score > 0.0))
                } else {
                    Some(Value::Float64(score))
                }
            }
            _ => None,
        }
    }

    #[cfg(not(feature = "text-index"))]
    fn eval_text_fn(
        &self,
        _name: &str,
        _args: &[FilterExpression],
        _chunk: &DataChunk,
        _row: usize,
    ) -> Option<Value> {
        None
    }

    fn eval_session_fn(
        &self,
        name: &str,
        args: &[FilterExpression],
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        match name {
            "session_user" => {
                // session_user() - returns the current session user
                // For embedded databases, returns a default user string
                Some(Value::String("default".into()))
            }
            // ISO/IEC 39075 Section 17.1 / Section 21: session schema/graph references
            "current_schema" => Some(self.session_context.current_schema.as_ref().map_or_else(
                || Value::String("default".into()),
                |s| Value::String(s.clone().into()),
            )),
            "current_graph" => Some(self.session_context.current_graph.as_ref().map_or_else(
                || Value::String("default".into()),
                |g| Value::String(g.clone().into()),
            )),
            "home_schema" | "home_graph" => {
                // Home schema/graph: not configurable yet, returns null
                Some(Value::Null)
            }
            // Grafeo extension: info() returns database metadata as a map
            "info" => Some(self.session_context.db_info.get().clone()),
            // Grafeo extension: schema() returns schema metadata as a map
            "schema" => Some(self.session_context.schema_info.get().clone()),
            "nullif" => {
                // NULLIF(expr1, expr2) - returns NULL if expr1 = expr2, else expr1
                if args.len() != 2 {
                    return None;
                }
                let val1 = self.eval_expr(&args[0], chunk, row)?;
                let val2 = self.eval_expr(&args[1], chunk, row)?;
                // Three-valued: NULLIF(NULL, x) = NULL; NULLIF(x, NULL) = x
                if val1.is_null() || val2.is_null() {
                    Some(val1)
                } else if Self::values_equal(&val1, &val2) {
                    Some(Value::Null)
                } else {
                    Some(val1)
                }
            }
            _ => None,
        }
    }

    fn eval_case(
        &self,
        operand: Option<&FilterExpression>,
        when_clauses: &[(FilterExpression, FilterExpression)],
        else_clause: Option<&FilterExpression>,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        if let Some(test_expr) = operand {
            // Simple CASE: CASE expr WHEN val1 THEN res1 ...
            // Use unwrap_or(Null) so a NULL test expression falls through to ELSE
            // rather than short-circuiting the entire CASE (NULL != anything).
            let test_val = self.eval_expr(test_expr, chunk, row).unwrap_or(Value::Null);
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row).unwrap_or(Value::Null);
                // Three-valued logic: NULL never matches anything in simple CASE
                if !test_val.is_null()
                    && !when_val.is_null()
                    && Self::values_equal(&test_val, &when_val)
                {
                    return self.eval_expr(then_expr, chunk, row);
                }
            }
        } else {
            // Searched CASE: CASE WHEN cond1 THEN res1 ...
            // Use unwrap_or(Null) so a NULL/UNKNOWN condition falls through
            // to the next WHEN or ELSE (three-valued logic: only TRUE matches).
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row).unwrap_or(Value::Null);
                if when_val.as_bool() == Some(true) {
                    return self.eval_expr(then_expr, chunk, row);
                }
            }
        }
        // No match - return ELSE or NULL
        if let Some(else_expr) = else_clause {
            self.eval_expr(else_expr, chunk, row)
        } else {
            Some(Value::Null)
        }
    }

    fn eval_unary_op(&self, op: UnaryFilterOp, val: Option<Value>) -> Option<Value> {
        match op {
            UnaryFilterOp::Not => {
                let v = val?.as_bool()?;
                Some(Value::Bool(!v))
            }
            UnaryFilterOp::IsNull => Some(Value::Bool(
                val.is_none() || matches!(val, Some(Value::Null)),
            )),
            UnaryFilterOp::IsNotNull => Some(Value::Bool(
                val.is_some() && !matches!(val, Some(Value::Null)),
            )),
            UnaryFilterOp::Neg => match val? {
                Value::Int64(i) => i.checked_neg().map(Value::Int64),
                Value::Float64(f) => Some(Value::Float64(-f)),
                _ => None,
            },
        }
    }

    /// Structural equality for DISTINCT, GROUP BY, and list/map comparison.
    ///
    /// Treats NULL == NULL as `true` (grouping semantics). For SQL/GQL
    /// comparison operators, use [`eval_binary_op`] which returns UNKNOWN
    /// when either operand is NULL.
    fn values_equal(left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
                (*a as f64 - b).abs() < f64::EPSILON
            }
            // RDF stores numeric literals as strings; allow cross-type equality
            (Value::String(s), Value::Int64(i)) | (Value::Int64(i), Value::String(s)) => {
                s.parse::<i64>().is_ok_and(|n| n == *i)
            }
            (Value::String(s), Value::Float64(f)) | (Value::Float64(f), Value::String(s)) => {
                s.parse::<f64>().is_ok_and(|n| (n - f).abs() < f64::EPSILON)
            }
            (Value::List(a), Value::List(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|(x, y)| Self::values_equal(x, y))
            }
            (Value::Map(a), Value::Map(b)) => {
                a.len() == b.len()
                    && a.iter()
                        .zip(b.iter())
                        .all(|((k1, v1), (k2, v2))| k1 == k2 && Self::values_equal(v1, v2))
            }
            (
                Value::Path {
                    nodes: n1,
                    edges: e1,
                },
                Value::Path {
                    nodes: n2,
                    edges: e2,
                },
            ) => {
                n1.len() == n2.len()
                    && e1.len() == e2.len()
                    && n1
                        .iter()
                        .zip(n2.iter())
                        .all(|(a, b)| Self::values_equal(a, b))
                    && e1
                        .iter()
                        .zip(e2.iter())
                        .all(|(a, b)| Self::values_equal(a, b))
            }
            // Temporal values, bytes, vectors and the rest: the same variant
            // with the same value (zoned datetimes compare by instant). This
            // used to be `false`, so `date('2024-01-01') = date('2024-01-01')`
            // was false.
            _ => left == right,
        }
    }

    /// Checks if a value matches a GQL type name (used by IS TYPED).
    fn value_matches_type(val: &Value, type_name: &str) -> bool {
        match type_name {
            "BOOLEAN" | "BOOL" => matches!(val, Value::Bool(_)),
            "INTEGER" | "INT" | "INT64" => matches!(val, Value::Int64(_)),
            "FLOAT" | "FLOAT64" | "DOUBLE" => matches!(val, Value::Float64(_)),
            "STRING" => matches!(val, Value::String(_)),
            "LIST" => matches!(val, Value::List(_)),
            "MAP" | "RECORD" => matches!(val, Value::Map(_)),
            "NULL" => matches!(val, Value::Null),
            "DATE" => matches!(val, Value::Date(_)),
            "TIME" => matches!(val, Value::Time(_)),
            "DATETIME" | "TIMESTAMP" => matches!(val, Value::Timestamp(_)),
            "DURATION" => matches!(val, Value::Duration(_)),
            "PATH" => matches!(val, Value::Path { .. }),
            "NODE" | "EDGE" | "GRAPH" => false, // element refs not stored as values
            _ => false,
        }
    }

    /// Coerces a value to a target GQL type. Returns None if coercion is impossible.
    fn coerce_to_type(val: Value, type_name: &str) -> Option<Value> {
        match type_name.to_uppercase().as_str() {
            "INTEGER" | "INT" | "INT64" => match val {
                Value::Int64(_) => Some(val),
                // reason: type coercion intentionally truncates float to int
                #[allow(clippy::cast_possible_truncation)]
                Value::Float64(f) => Some(Value::Int64(f as i64)),
                Value::String(ref s) => s.parse::<i64>().ok().map(Value::Int64),
                Value::Bool(b) => Some(Value::Int64(i64::from(b))),
                _ => None,
            },
            "FLOAT" | "FLOAT64" | "DOUBLE" => match val {
                Value::Float64(_) => Some(val),
                Value::Int64(i) => Some(Value::Float64(i as f64)),
                Value::String(ref s) => s.parse::<f64>().ok().map(Value::Float64),
                _ => None,
            },
            "STRING" => match val {
                Value::String(_) => Some(val),
                other => Some(Value::String(other.to_string().into())),
            },
            "BOOLEAN" | "BOOL" => match val {
                Value::Bool(_) => Some(val),
                Value::String(ref s) => match s.to_lowercase().as_str() {
                    "true" => Some(Value::Bool(true)),
                    "false" => Some(Value::Bool(false)),
                    _ => None,
                },
                _ => None,
            },
            _ => {
                // For unknown types, keep value as-is if it already matches
                if Self::value_matches_type(&val, &type_name.to_uppercase()) {
                    Some(val)
                } else {
                    None
                }
            }
        }
    }

    fn compare_values(&self, left: &Value, right: &Value) -> Option<i32> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => Some(a.cmp(b) as i32),
            (Value::Float64(a), Value::Float64(b)) => {
                if a < b {
                    Some(-1)
                } else if a > b {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (Value::String(a), Value::String(b)) => Some(a.cmp(b) as i32),
            (Value::Int64(a), Value::Float64(b)) => (*a as f64).partial_cmp(b).map(|o| o as i32),
            (Value::Float64(a), Value::Int64(b)) => a.partial_cmp(&(*b as f64)).map(|o| o as i32),
            // RDF stores numeric literals as strings; allow cross-type comparison
            (Value::String(s), Value::Int64(i)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| n.partial_cmp(&(*i as f64)).map(|o| o as i32)),
            (Value::Int64(i), Value::String(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| (*i as f64).partial_cmp(&n).map(|o| o as i32)),
            (Value::String(s), Value::Float64(f)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| n.partial_cmp(f).map(|o| o as i32)),
            (Value::Float64(f), Value::String(s)) => s
                .parse::<f64>()
                .ok()
                .and_then(|n| f.partial_cmp(&n).map(|o| o as i32)),
            // Temporal comparisons
            (Value::Timestamp(a), Value::Timestamp(b)) => Some(a.cmp(b) as i32),
            (Value::Date(a), Value::Date(b)) => Some(a.cmp(b) as i32),
            (Value::Time(a), Value::Time(b)) => Some(a.cmp(b) as i32),
            _ => None,
        }
    }
}

/// A fast-path `EXISTS` or `COUNT` pattern (see
/// [`FilterExpression::ExistsSubquery`]).
struct SubqueryPattern<'a> {
    start: &'a str,
    end: &'a str,
    edge: Option<&'a str>,
    direction: Direction,
    edge_types: &'a [String],
    end_labels: &'a Option<Vec<String>>,
    hops: Hops,
}

/// How many edges a subquery pattern has.
#[derive(Clone, Copy)]
enum Hops {
    /// One edge.
    One,
    /// A variable-length edge: 1 to this many (`None`: any number).
    Path(Option<u32>),
}

/// What a row binds a variable of a subquery pattern to.
enum Bound<T> {
    /// The row has no such variable: the pattern binds it.
    No,
    /// The row holds null there, or not a node or edge: nothing matches it.
    Null,
    /// The row holds this node or edge.
    To(T),
}

impl Bound<NodeId> {
    /// The node, when the row binds one.
    fn node(self) -> Option<NodeId> {
        match self {
            Self::To(node) => Some(node),
            Self::No | Self::Null => None,
        }
    }
}

/// The edges of a path's edge list in `column`, as a variable-length expand
/// writes it.
fn edge_id_list(column: &ValueVector, row: usize) -> Option<Vec<EdgeId>> {
    let Value::List(items) = column.get_value(row)? else {
        return None;
    };
    items
        .iter()
        .map(|item| {
            item.as_int64()
                .and_then(|id| u64::try_from(id).ok())
                .map(EdgeId::new)
        })
        .collect()
}

/// What item `position` of a list of type `list` holds (see
/// `ExpressionPredicate::shape`), for the variable a comprehension, list
/// predicate or `reduce` binds to it: `edges(p)`, `relationships(p)` and
/// `nodes(p)` hold entity ids, and an item bound as an edge or a node is
/// read like an edge or node variable of the row.
fn item_shape(list: &LogicalType, position: usize) -> LogicalType {
    i64::try_from(position).map_or(LogicalType::Any, |position| item_type_at(list, position))
}

/// The row a list comprehension, list predicate or `reduce` is evaluated for,
/// with a column of its own for each variable it binds, read by the normal
/// evaluator: every expression that works in RETURN works on the items, an
/// edge or node item sits in an edge or node column (so `type(e)`, `id(n)`
/// and `labels(n)` work on it), and an inner expression that binds a name
/// again gets its own column, which shadows the outer one.
struct ItemScope {
    evaluator: ExpressionPredicate,
    chunk: DataChunk,
    /// The column of the first bound variable; the others follow it.
    first: usize,
}

impl ItemScope {
    fn new(
        outer: &ExpressionPredicate,
        variables: &[&String],
        chunk: &DataChunk,
        row: usize,
    ) -> Self {
        let mut columns: Vec<ValueVector> = chunk
            .columns()
            .iter()
            .map(|column| {
                let mut copy = ValueVector::with_capacity(column.logical_type(), 1);
                column.copy_row_to(row, &mut copy);
                copy
            })
            .collect();
        let first = columns.len();
        let mut variable_columns = outer.variable_columns.clone();
        for (offset, variable) in variables.iter().enumerate() {
            variable_columns.insert((*variable).clone(), first + offset);
            let mut column = ValueVector::with_capacity(LogicalType::Any, 1);
            column.push_value(Value::Null);
            columns.push(column);
        }
        Self {
            evaluator: ExpressionPredicate {
                expression: FilterExpression::Literal(Value::Null),
                variable_columns,
                store: Arc::clone(&outer.store),
                transaction_id: outer.transaction_id,
                viewing_epoch: outer.viewing_epoch,
                session_context: outer.session_context.clone(),
                patterns: Arc::clone(&outer.patterns),
            },
            chunk: DataChunk::new(columns),
            first,
        }
    }

    /// Binds the `index`-th variable given to [`new`](Self::new) to `value`,
    /// which holds what `shape` says (a node, an edge, a map with nodes in
    /// it, ...; `Any` for a plain value).
    fn bind(&mut self, index: usize, value: &Value, shape: &LogicalType) {
        let entity = match value {
            Value::Int64(id) => u64::try_from(*id).ok(),
            Value::Map(map) => match map.get(&PropertyKey::new("_id")) {
                Some(Value::Int64(id)) => u64::try_from(*id).ok(),
                _ => None,
            },
            _ => None,
        };
        let column = match (shape, entity) {
            (LogicalType::Edge, Some(id)) => {
                let mut column = ValueVector::with_capacity(LogicalType::Edge, 1);
                column.push_edge_id(EdgeId::new(id));
                column
            }
            (LogicalType::Node, Some(id)) => {
                let mut column = ValueVector::with_capacity(LogicalType::Node, 1);
                column.push_node_id(NodeId::new(id));
                column
            }
            // A value with nodes or edges inside keeps its type, so a key or
            // item read from it is one (`[x IN xs | x.msg.name]`).
            (shape, _) if holds_entities(shape) && !shape.is_graph_element() => {
                let mut column = ValueVector::with_capacity(shape.clone(), 1);
                column.push_value(value.clone());
                column
            }
            _ => {
                let mut column = ValueVector::with_capacity(LogicalType::Any, 1);
                column.push_value(value.clone());
                column
            }
        };
        if let Some(slot) = self.chunk.column_mut(self.first + index) {
            *slot = column;
        }
    }

    fn eval(&self, expr: &FilterExpression) -> Option<Value> {
        self.evaluator.eval_expr(expr, &self.chunk, 0)
    }
}

impl Predicate for ExpressionPredicate {
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool {
        match self.eval(chunk, row) {
            Some(Value::Bool(b)) => b,
            _ => false,
        }
    }
}

/// A filter operator that applies a predicate to filter rows.
pub struct FilterOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Predicate to apply.
    predicate: Box<dyn Predicate>,
}

impl FilterOperator {
    /// Creates a new filter operator.
    pub fn new(child: Box<dyn Operator>, predicate: Box<dyn Predicate>) -> Self {
        Self { child, predicate }
    }

    /// Decomposes this operator into its child and predicate for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Box<dyn Predicate>) {
        (self.child, self.predicate)
    }
}

impl Operator for FilterOperator {
    fn next(&mut self) -> OperatorResult {
        loop {
            // Get next chunk from child
            let Some(mut chunk) = self.child.next()? else {
                return Ok(None);
            };

            // Zone map check: skip entire chunk if no rows can match
            if let Some(hints) = chunk.zone_hints()
                && !self.predicate.might_match_chunk(hints)
            {
                continue; // Skip entire chunk - zone map proves no matches
            }

            // Apply predicate to create selection vector, respecting any
            // existing selection from child operators (stacked filters).
            let selection = if let Some(existing) = chunk.selection() {
                let mut sel = SelectionVector::new_empty();
                for pos in 0..existing.len() {
                    if let Some(row) = existing.get(pos)
                        && self.predicate.evaluate(&chunk, row)
                    {
                        sel.push(row);
                    }
                }
                sel
            } else {
                let count = chunk.total_row_count();
                SelectionVector::from_predicate(count, |row| self.predicate.evaluate(&chunk, row))
            };

            // If nothing passes, skip to next chunk
            if selection.is_empty() {
                continue;
            }

            chunk.set_selection(selection);
            return Ok(Some(chunk));
        }
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Filter"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Escapes a character for use in a regex pattern.
#[cfg(any(feature = "regex", feature = "regex-lite"))]
fn regex_escape_char(ch: char, out: &mut String) {
    if ".+*?^${}()|[]\\".contains(ch) {
        out.push('\\');
    }
    out.push(ch);
}

/// Converts a `Value` to its string representation for concatenation.
fn value_to_string(val: &Value) -> Option<String> {
    match val {
        Value::Int64(i) => Some(i.to_string()),
        Value::Float64(f) => Some(f.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        Value::String(s) => Some(s.to_string()),
        Value::Null => None,
        _ => Some(format!("{val}")),
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use grafeo_common::types::LogicalType;

    struct MockScanOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl Operator for MockScanOperator {
        fn next(&mut self) -> OperatorResult {
            if self.position < self.chunks.len() {
                let chunk = std::mem::replace(&mut self.chunks[self.position], DataChunk::empty());
                self.position += 1;
                Ok(Some(chunk))
            } else {
                Ok(None)
            }
        }

        fn reset(&mut self) {
            self.position = 0;
        }

        fn name(&self) -> &'static str {
            "MockScan"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    #[test]
    fn test_filter_comparison() {
        // Create a chunk with values [10, 20, 30, 40, 50]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 1..=5 {
            builder.column_mut(0).unwrap().push_int64(i * 10);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 25
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap().unwrap();
        // Should have 30, 40, 50 (3 values)
        assert_eq!(result.row_count(), 3);
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_regex_operator() {
        use crate::graph::lpg::LpgStore;

        // Create a store and expression predicate to test regex
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create predicate to test "Smith" =~ ".*Smith$" (should match)
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "John Smith".into(),
                ))),
                op: BinaryFilterOp::Regex,
                right: Box::new(FilterExpression::Literal(Value::String(".*Smith$".into()))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Should match
        assert!(predicate.evaluate(&chunk, 0));

        // Test non-matching pattern
        let predicate_no_match = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("John Doe".into()))),
                op: BinaryFilterOp::Regex,
                right: Box::new(FilterExpression::Literal(Value::String(".*Smith$".into()))),
            },
            variable_columns,
            store,
        );

        // Should not match
        assert!(!predicate_no_match.evaluate(&chunk, 0));
    }

    #[test]
    fn test_pow_operator() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Create predicate to test 2^3 = 8.0
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(2))),
                    op: BinaryFilterOp::Pow,
                    right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(8.0))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );

        // 2^3 should equal 8.0
        assert!(predicate.evaluate(&chunk, 0));

        // Test with floats: 2.5^2.0 = 6.25
        let predicate_float = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
                    op: BinaryFilterOp::Pow,
                    right: Box::new(FilterExpression::Literal(Value::Float64(2.0))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(6.25))),
            },
            variable_columns,
            store,
        );

        assert!(predicate_float.evaluate(&chunk, 0));
    }

    #[test]
    fn test_map_expression() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Create map {name: 'Alix', age: 30}
        let predicate = ExpressionPredicate::new(
            FilterExpression::Map(vec![
                (
                    "name".to_string(),
                    FilterExpression::Literal(Value::String("Alix".into())),
                ),
                (
                    "age".to_string(),
                    FilterExpression::Literal(Value::Int64(30)),
                ),
            ]),
            variable_columns,
            store,
        );

        // Evaluate the map expression
        let result = predicate.eval(&chunk, 0);
        assert!(result.is_some());

        if let Some(Value::Map(m)) = result {
            assert_eq!(
                m.get(&PropertyKey::new("name")),
                Some(&Value::String("Alix".into()))
            );
            assert_eq!(m.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Map value");
        }
    }

    #[test]
    fn test_index_access_list() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test [1, 2, 3][1] = 2
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::IndexAccess {
                    base: Box::new(FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])),
                    index: Box::new(FilterExpression::Literal(Value::Int64(1))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(2))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );

        assert!(predicate.evaluate(&chunk, 0));

        // Test negative indexing: [1, 2, 3][-1] = 3
        let predicate_neg = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::IndexAccess {
                    base: Box::new(FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])),
                    index: Box::new(FilterExpression::Literal(Value::Int64(-1))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            variable_columns,
            store,
        );

        assert!(predicate_neg.evaluate(&chunk, 0));
    }

    #[test]
    fn test_slice_access() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test [1, 2, 3, 4, 5][1..3] should return [2, 3]
        let predicate = ExpressionPredicate::new(
            FilterExpression::SliceAccess {
                base: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                    FilterExpression::Literal(Value::Int64(4)),
                    FilterExpression::Literal(Value::Int64(5)),
                ])),
                start: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
                end: Some(Box::new(FilterExpression::Literal(Value::Int64(3)))),
            },
            variable_columns,
            store,
        );

        let result = predicate.eval(&chunk, 0);
        assert!(result.is_some());

        if let Some(Value::List(items)) = result {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], Value::Int64(2));
            assert_eq!(items[1], Value::Int64(3));
        } else {
            panic!("Expected List value");
        }
    }

    #[test]
    fn test_might_match_chunk_no_hints() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));
        let hints = ChunkZoneHints::default();

        // With no zone map for the column, should return true (conservative)
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_equality_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // 50 is within [10, 100], should return true
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_equality_no_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(200));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // 200 is outside [10, 100], should return false
        assert!(!predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_greater_than_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // max=100 > 50, so some values might be > 50
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_greater_than_no_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(200));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // max=100 < 200, so no values can be > 200
        assert!(!predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_less_than_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // min=10 < 50, so some values might be < 50
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_less_than_no_match() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(5));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(100), 0, 10),
        );

        // min=10 > 5, so no values can be < 5
        assert!(!predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_might_match_chunk_not_equal_always_conservative() {
        let predicate = ComparisonPredicate::new(0, CompareOp::Ne, Value::Int64(50));

        let mut hints = ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(50), Value::Int64(50), 0, 10),
        );

        // Even if min=max=50, Ne is conservative and returns true
        assert!(predicate.might_match_chunk(&hints));
    }

    #[test]
    fn test_comparison_string() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        builder.column_mut(0).unwrap().push_string("banana");
        builder.advance_row();
        let chunk = builder.finish();

        // Test string equality
        let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::String("banana".into()));
        assert!(pred_eq.evaluate(&chunk, 0));

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::String("apple".into()));
        assert!(pred_ne.evaluate(&chunk, 0));

        // Test string ordering
        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::String("cherry".into()));
        assert!(pred_lt.evaluate(&chunk, 0)); // "banana" < "cherry"

        let pred_gt = ComparisonPredicate::new(0, CompareOp::Gt, Value::String("apple".into()));
        assert!(pred_gt.evaluate(&chunk, 0)); // "banana" > "apple"
    }

    #[test]
    fn test_comparison_float64() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Float64]);
        builder
            .column_mut(0)
            .unwrap()
            .push_float64(std::f64::consts::PI);
        builder.advance_row();
        let chunk = builder.finish();

        // Test float equality (within epsilon)
        let pred_eq =
            ComparisonPredicate::new(0, CompareOp::Eq, Value::Float64(std::f64::consts::PI));
        assert!(pred_eq.evaluate(&chunk, 0));

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Float64(2.71));
        assert!(pred_ne.evaluate(&chunk, 0));

        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Float64(4.0));
        assert!(pred_lt.evaluate(&chunk, 0));

        let pred_ge =
            ComparisonPredicate::new(0, CompareOp::Ge, Value::Float64(std::f64::consts::PI));
        assert!(pred_ge.evaluate(&chunk, 0));
    }

    #[test]
    fn test_comparison_bool() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Bool]);
        builder.column_mut(0).unwrap().push_bool(true);
        builder.advance_row();
        let chunk = builder.finish();

        let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::Bool(true));
        assert!(pred_eq.evaluate(&chunk, 0));

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Bool(false));
        assert!(pred_ne.evaluate(&chunk, 0));

        // Ordering on booleans returns false
        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Bool(false));
        assert!(!pred_lt.evaluate(&chunk, 0));
    }

    #[test]
    fn test_unary_operators() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test NOT
        let pred_not = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::Not,
                operand: Box::new(FilterExpression::Literal(Value::Bool(false))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_not.evaluate(&chunk, 0));

        // Test IS NULL
        let pred_is_null = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::IsNull,
                operand: Box::new(FilterExpression::Literal(Value::Null)),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_is_null.evaluate(&chunk, 0));

        // Test IS NOT NULL
        let pred_is_not_null = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::IsNotNull,
                operand: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_is_not_null.evaluate(&chunk, 0));

        // Test negation
        let pred_neg = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Unary {
                    op: UnaryFilterOp::Neg,
                    operand: Box::new(FilterExpression::Literal(Value::Int64(5))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(-5))),
            },
            variable_columns,
            store,
        );
        assert!(pred_neg.evaluate(&chunk, 0));
    }

    #[test]
    fn test_arithmetic_operators() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test Add: 2 + 3 = 5
        let pred_add = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(2))),
                    op: BinaryFilterOp::Add,
                    right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_add.evaluate(&chunk, 0));

        // Test Sub: 10 - 4 = 6
        let pred_sub = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                    op: BinaryFilterOp::Sub,
                    right: Box::new(FilterExpression::Literal(Value::Int64(4))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(6))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_sub.evaluate(&chunk, 0));

        // Test Mul: 3 * 4 = 12
        let pred_mul = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(3))),
                    op: BinaryFilterOp::Mul,
                    right: Box::new(FilterExpression::Literal(Value::Int64(4))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(12))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_mul.evaluate(&chunk, 0));

        // Test Div: 20 / 4 = 5
        let pred_div = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(20))),
                    op: BinaryFilterOp::Div,
                    right: Box::new(FilterExpression::Literal(Value::Int64(4))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_div.evaluate(&chunk, 0));

        // Test Mod: 17 % 5 = 2
        let pred_mod = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(17))),
                    op: BinaryFilterOp::Mod,
                    right: Box::new(FilterExpression::Literal(Value::Int64(5))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(2))),
            },
            variable_columns,
            store,
        );
        assert!(pred_mod.evaluate(&chunk, 0));
    }

    #[test]
    fn test_string_operators() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test STARTS WITH
        let pred_starts = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "hello world".into(),
                ))),
                op: BinaryFilterOp::StartsWith,
                right: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_starts.evaluate(&chunk, 0));

        // Test ENDS WITH
        let pred_ends = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "hello world".into(),
                ))),
                op: BinaryFilterOp::EndsWith,
                right: Box::new(FilterExpression::Literal(Value::String("world".into()))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_ends.evaluate(&chunk, 0));

        // Test CONTAINS
        let pred_contains = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String(
                    "hello world".into(),
                ))),
                op: BinaryFilterOp::Contains,
                right: Box::new(FilterExpression::Literal(Value::String("lo wo".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_contains.evaluate(&chunk, 0));
    }

    #[test]
    fn test_in_operator() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test 3 IN [1, 2, 3, 4, 5]
        let pred_in = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(3))),
                op: BinaryFilterOp::In,
                right: Box::new(FilterExpression::List(vec![
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::Int64(3)),
                    FilterExpression::Literal(Value::Int64(4)),
                    FilterExpression::Literal(Value::Int64(5)),
                ])),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_in.evaluate(&chunk, 0));

        // Test 10 NOT IN [1, 2, 3]
        let pred_not_in = ExpressionPredicate::new(
            FilterExpression::Unary {
                op: UnaryFilterOp::Not,
                operand: Box::new(FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                    op: BinaryFilterOp::In,
                    right: Box::new(FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])),
                }),
            },
            variable_columns,
            store,
        );
        assert!(pred_not_in.evaluate(&chunk, 0));
    }

    #[test]
    fn test_logical_operators() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test AND: true AND true = true
        let pred_and = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Bool(true))),
                op: BinaryFilterOp::And,
                right: Box::new(FilterExpression::Literal(Value::Bool(true))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_and.evaluate(&chunk, 0));

        // Test OR: false OR true = true
        let pred_or = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Bool(false))),
                op: BinaryFilterOp::Or,
                right: Box::new(FilterExpression::Literal(Value::Bool(true))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_or.evaluate(&chunk, 0));

        // Test XOR: true XOR false = true
        let pred_xor = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Bool(true))),
                op: BinaryFilterOp::Xor,
                right: Box::new(FilterExpression::Literal(Value::Bool(false))),
            },
            variable_columns,
            store,
        );
        assert!(pred_xor.evaluate(&chunk, 0));
    }

    #[test]
    fn test_case_expression_simple() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test simple CASE: CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END = 'two'
        let pred_case = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Case {
                    operand: Some(Box::new(FilterExpression::Literal(Value::Int64(2)))),
                    when_clauses: vec![
                        (
                            FilterExpression::Literal(Value::Int64(1)),
                            FilterExpression::Literal(Value::String("one".into())),
                        ),
                        (
                            FilterExpression::Literal(Value::Int64(2)),
                            FilterExpression::Literal(Value::String("two".into())),
                        ),
                    ],
                    else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                        "other".into(),
                    )))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::String("two".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_case.evaluate(&chunk, 0));
    }

    #[test]
    fn test_case_expression_searched() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test searched CASE: CASE WHEN 5 > 3 THEN 'yes' ELSE 'no' END = 'yes'
        let pred_case = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Case {
                    operand: None,
                    when_clauses: vec![(
                        FilterExpression::Binary {
                            left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                            op: BinaryFilterOp::Gt,
                            right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                        },
                        FilterExpression::Literal(Value::String("yes".into())),
                    )],
                    else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                        "no".into(),
                    )))),
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::String("yes".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_case.evaluate(&chunk, 0));
    }

    #[test]
    fn test_list_functions() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test head([1, 2, 3]) = 1
        let pred_head = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "head".to_string(),
                    args: vec![FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(1))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_head.evaluate(&chunk, 0));

        // Test last([1, 2, 3]) = 3
        let pred_last = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "last".to_string(),
                    args: vec![FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_last.evaluate(&chunk, 0));

        // Test size([1, 2, 3]) = 3
        let pred_size = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "size".to_string(),
                    args: vec![FilterExpression::List(vec![
                        FilterExpression::Literal(Value::Int64(1)),
                        FilterExpression::Literal(Value::Int64(2)),
                        FilterExpression::Literal(Value::Int64(3)),
                    ])],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(3))),
            },
            variable_columns,
            store,
        );
        assert!(pred_size.evaluate(&chunk, 0));
    }

    #[test]
    fn test_type_conversion_functions() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test toInteger("42") = 42
        let pred_to_int = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "toInteger".to_string(),
                    args: vec![FilterExpression::Literal(Value::String("42".into()))],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_to_int.evaluate(&chunk, 0));

        // Test toFloat(42) = 42.0
        let pred_to_float = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "toFloat".to_string(),
                    args: vec![FilterExpression::Literal(Value::Int64(42))],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(42.0))),
            },
            variable_columns.clone(),
            Arc::clone(&store),
        );
        assert!(pred_to_float.evaluate(&chunk, 0));

        // Test toBoolean("true") = true
        let pred_to_bool = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "toBoolean".to_string(),
                    args: vec![FilterExpression::Literal(Value::String("true".into()))],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Bool(true))),
            },
            variable_columns,
            store,
        );
        assert!(pred_to_bool.evaluate(&chunk, 0));
    }

    #[test]
    fn test_coalesce_function() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test coalesce(null, null, 'default') = 'default'
        let pred_coalesce = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "coalesce".to_string(),
                    args: vec![
                        FilterExpression::Literal(Value::Null),
                        FilterExpression::Literal(Value::Null),
                        FilterExpression::Literal(Value::String("default".into())),
                    ],
                }),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::String("default".into()))),
            },
            variable_columns,
            store,
        );
        assert!(pred_coalesce.evaluate(&chunk, 0));
    }

    #[test]
    fn test_filter_empty_result() {
        // Create a chunk with values that won't match the predicate
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 1..=5 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 100 (none will match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(100));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // Should return None since nothing matches
        let result = filter.next().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_filter_operator_reset() {
        // Test that reset() calls child.reset()
        // Since MockScanOperator doesn't preserve chunks after reading,
        // we test that reset is called by checking position resets
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(50);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(50));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // First iteration
        let result = filter.next().unwrap();
        assert!(result.is_some());
        let result = filter.next().unwrap();
        assert!(result.is_none());

        // Note: MockScanOperator replaces chunks with empty ones when read,
        // so reset doesn't restore the data. This test verifies reset() is called.
        filter.reset();
        // After reset, position is 0 but chunk is empty
        let result = filter.next().unwrap();
        // Empty chunk produces no matches, returns None
        assert!(result.is_none());
    }

    #[test]
    fn test_mixed_type_comparison_int_float() {
        let store: Arc<dyn GraphStoreSearch> =
            Arc::new(crate::graph::lpg::LpgStore::new().unwrap());
        let variable_columns = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Test 5 == 5.0 (mixed int/float comparison)
        let pred_mixed = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(5.0))),
            },
            variable_columns,
            store,
        );
        assert!(pred_mixed.evaluate(&chunk, 0));
    }

    #[test]
    fn test_zone_map_allows_matching_chunk() {
        // Test that a chunk with zone hints indicating potential matches is evaluated
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 10..=20 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }
        let mut chunk = builder.finish();

        // Set zone hints: min=10, max=20
        let mut hints = crate::execution::chunk::ChunkZoneHints::default();
        hints.column_hints.insert(
            0,
            crate::index::ZoneMapEntry::with_min_max(Value::Int64(10), Value::Int64(20), 0, 11),
        );
        chunk.set_zone_hints(hints);

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 15 (some will match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(15));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // Should return matching rows
        let result = filter.next().unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();

        // Should have rows 16, 17, 18, 19, 20 (5 rows)
        assert_eq!(chunk.row_count(), 5);
    }

    #[test]
    fn test_filter_with_all_rows_matching() {
        // All values in chunk match the predicate
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for i in 100..=110 {
            builder.column_mut(0).unwrap().push_int64(i);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 50 (all will match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(50));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();

        // All 11 rows should be returned
        assert_eq!(chunk.row_count(), 11);
    }

    #[test]
    fn test_filter_with_sparse_data() {
        // Test filtering with sparse matching data
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        // Create values where only some match: 1, 10, 2, 20, 3, 30
        for &v in &[1i64, 10, 2, 20, 3, 30] {
            builder.column_mut(0).unwrap().push_int64(v);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Filter for values > 5 (only 10, 20, 30 should match)
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(5));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        let result = filter.next().unwrap();
        assert!(result.is_some());
        let chunk = result.unwrap();

        // Only 10, 20, 30 should match (3 rows)
        assert_eq!(chunk.row_count(), 3);
    }

    #[test]
    fn test_predicate_on_wrong_column_returns_empty() {
        // When the predicate references a column index that's out of bounds
        // or the column type is incompatible
        let mut builder = DataChunkBuilder::new(&[LogicalType::String]);
        builder.column_mut(0).unwrap().push_string("hello");
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Predicate on column 5 (doesn't exist)
        let predicate = ComparisonPredicate::new(5, CompareOp::Eq, Value::Int64(42));
        let mut filter = FilterOperator::new(Box::new(mock_scan), Box::new(predicate));

        // Should handle gracefully (either error or empty result)
        let result = filter.next();
        // The behavior depends on implementation - just verify no panic
        let _ = result;
    }

    #[test]
    fn test_expression_predicate_with_labels_function() {
        use crate::graph::GraphStoreMut;

        // Test the labels() function in predicates
        let store: Arc<dyn GraphStoreMut> = Arc::new(crate::graph::lpg::LpgStore::new().unwrap());

        // Create a node with a label
        let node_id = store.create_node(&["Person", "Employee"]);

        // Build a chunk with the node
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(node_id);
        builder.advance_row();
        let chunk = builder.finish();

        // Map column 0 to variable "n"
        let mut variable_columns = HashMap::new();
        variable_columns.insert("n".to_string(), 0);

        // Test: 'Person' IN labels(n)
        let pred = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("Person".into()))),
                op: BinaryFilterOp::In,
                right: Box::new(FilterExpression::FunctionCall {
                    name: "labels".to_string(),
                    args: vec![FilterExpression::Variable("n".to_string())],
                }),
            },
            variable_columns,
            store.clone() as Arc<dyn GraphStoreSearch>,
        );

        assert!(pred.evaluate(&chunk, 0));
    }

    #[test]
    fn test_comparison_with_boundary_values() {
        // Test comparisons at exact boundary values
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(i64::MAX);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(i64::MIN);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(0);
        builder.advance_row();
        let chunk = builder.finish();

        // Test >= 0
        let pred_ge = ComparisonPredicate::new(0, CompareOp::Ge, Value::Int64(0));
        assert!(pred_ge.evaluate(&chunk, 0)); // i64::MAX >= 0
        assert!(!pred_ge.evaluate(&chunk, 1)); // i64::MIN >= 0 is false
        assert!(pred_ge.evaluate(&chunk, 2)); // 0 >= 0

        // Test <= 0
        let pred_le = ComparisonPredicate::new(0, CompareOp::Le, Value::Int64(0));
        assert!(!pred_le.evaluate(&chunk, 0)); // i64::MAX <= 0 is false
        assert!(pred_le.evaluate(&chunk, 1)); // i64::MIN <= 0
        assert!(pred_le.evaluate(&chunk, 2)); // 0 <= 0
    }

    // ── Cross-type equality (String ↔ numeric) ──────────────────────────

    /// Regression test: RDF stores numeric literals as strings, so filters
    /// like `FILTER(?age = 30)` compare `Value::String("30")` with
    /// `Value::Int64(30)`.  The `values_equal` path must coerce.
    #[test]
    fn test_cross_type_string_int_equality() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // String "42" == Int64(42)
        let pred = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("42".into()))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred.evaluate(&chunk, 0));

        // String "42" != Int64(99)
        let pred_ne = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("42".into()))),
                op: BinaryFilterOp::Ne,
                right: Box::new(FilterExpression::Literal(Value::Int64(99))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_ne.evaluate(&chunk, 0));

        // Non-numeric string should NOT equal any integer
        let pred_bad = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Int64(42))),
            },
            vc,
            store,
        );
        assert!(!pred_bad.evaluate(&chunk, 0));
    }

    /// String ↔ Float64 equality: "7.25" == Float64(7.25)
    #[test]
    fn test_cross_type_string_float_equality() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("7.25".into()))),
                op: BinaryFilterOp::Eq,
                right: Box::new(FilterExpression::Literal(Value::Float64(7.25))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred.evaluate(&chunk, 0));

        // "7.25" != 2.5
        let pred_ne = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
                op: BinaryFilterOp::Ne,
                right: Box::new(FilterExpression::Literal(Value::String("7.25".into()))),
            },
            vc,
            store,
        );
        assert!(pred_ne.evaluate(&chunk, 0));
    }

    // ── Cross-type ordering (String ↔ numeric) ──────────────────────────

    /// Regression test: String-encoded numbers must support range comparisons
    /// so that `FILTER(?age > 25)` works when `?age` is stored as "30".
    #[test]
    fn test_cross_type_string_numeric_ordering() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // "30" > Int64(25)
        let pred_gt = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("30".into()))),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(25))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_gt.evaluate(&chunk, 0));

        // Int64(10) < "20.5" (cross Float64 path)
        let pred_lt = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Int64(10))),
                op: BinaryFilterOp::Lt,
                right: Box::new(FilterExpression::Literal(Value::String("20.5".into()))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_lt.evaluate(&chunk, 0));

        // "2.5" <= Float64(2.5)
        let pred_le = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::String("2.5".into()))),
                op: BinaryFilterOp::Le,
                right: Box::new(FilterExpression::Literal(Value::Float64(2.5))),
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert!(pred_le.evaluate(&chunk, 0));

        // Float64(100.0) >= "99.9"
        let pred_ge = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::Literal(Value::Float64(100.0))),
                op: BinaryFilterOp::Ge,
                right: Box::new(FilterExpression::Literal(Value::String("99.9".into()))),
            },
            vc,
            store,
        );
        assert!(pred_ge.evaluate(&chunk, 0));
    }

    // ── Stacked filter (selection vector preservation) ───────────────────

    /// Regression test: when two FilterOperators are stacked (child filter →
    /// parent filter), the parent must respect the child's selection vector
    /// instead of re-evaluating all physical rows.
    #[test]
    fn test_stacked_filters_respect_selection_vector() {
        // Chunk: ages = [20, 35, 45, 25, 50]
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        for age in [20, 35, 45, 25, 50] {
            builder.column_mut(0).unwrap().push_int64(age);
            builder.advance_row();
        }
        let chunk = builder.finish();

        let scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // First filter: age > 25 → rows 1(35), 2(45), 4(50)
        let pred1 = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(25));
        let filter1 = FilterOperator::new(Box::new(scan), Box::new(pred1));

        // Second (stacked) filter: age < 50 → should intersect → rows 1(35), 2(45)
        let pred2 = ComparisonPredicate::new(0, CompareOp::Lt, Value::Int64(50));
        let mut filter2 = FilterOperator::new(Box::new(filter1), Box::new(pred2));

        let result = filter2.next().unwrap().unwrap();
        assert_eq!(
            result.row_count(),
            2,
            "stacked filter should yield 2 rows (35, 45)"
        );

        // Verify it's exhausted
        assert!(filter2.next().unwrap().is_none());
    }

    // === eval_binary_op: Arithmetic Tests ===

    /// Helper: creates an `ExpressionPredicate` wrapping a literal expression,
    /// evaluates it against an empty chunk, and returns the result `Value`.
    fn eval_literal_expr(expr: FilterExpression) -> Option<Value> {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();
        pred.eval_at(&chunk, 0)
    }

    fn binary(left: Value, op: BinaryFilterOp, right: Value) -> FilterExpression {
        FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(left)),
            op,
            right: Box::new(FilterExpression::Literal(right)),
        }
    }

    fn unary(op: UnaryFilterOp, operand: FilterExpression) -> FilterExpression {
        FilterExpression::Unary {
            op,
            operand: Box::new(operand),
        }
    }

    #[test]
    fn test_eval_binary_addition_int() {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Add,
            Value::Int64(20),
        ));
        assert_eq!(result, Some(Value::Int64(30)));
    }

    #[test]
    fn test_eval_binary_subtraction_int() {
        let result = eval_literal_expr(binary(
            Value::Int64(50),
            BinaryFilterOp::Sub,
            Value::Int64(18),
        ));
        assert_eq!(result, Some(Value::Int64(32)));
    }

    #[test]
    fn test_eval_binary_multiplication_int() {
        let result = eval_literal_expr(binary(
            Value::Int64(7),
            BinaryFilterOp::Mul,
            Value::Int64(6),
        ));
        assert_eq!(result, Some(Value::Int64(42)));
    }

    #[test]
    fn test_eval_binary_division_int() {
        let result = eval_literal_expr(binary(
            Value::Int64(100),
            BinaryFilterOp::Div,
            Value::Int64(4),
        ));
        assert_eq!(result, Some(Value::Int64(25)));
    }

    #[test]
    fn test_eval_binary_modulo_int() {
        let result = eval_literal_expr(binary(
            Value::Int64(17),
            BinaryFilterOp::Mod,
            Value::Int64(5),
        ));
        assert_eq!(result, Some(Value::Int64(2)));
    }

    // === eval_binary_op: Comparisons ===

    #[test]
    fn test_eval_comparison_lt() {
        let result =
            eval_literal_expr(binary(Value::Int64(3), BinaryFilterOp::Lt, Value::Int64(5)));
        assert_eq!(result, Some(Value::Bool(true)));

        let result =
            eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Lt, Value::Int64(3)));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_comparison_gt() {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Gt,
            Value::Int64(5),
        ));
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_comparison_eq() {
        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Eq,
            Value::Int64(42),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Eq,
            Value::Int64(43),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_comparison_ne() {
        let result = eval_literal_expr(binary(
            Value::String("hello".into()),
            BinaryFilterOp::Ne,
            Value::String("world".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("same".into()),
            BinaryFilterOp::Ne,
            Value::String("same".into()),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_comparison_le_ge() {
        // <=
        let result =
            eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Le, Value::Int64(5)));
        assert_eq!(result, Some(Value::Bool(true)));

        let result =
            eval_literal_expr(binary(Value::Int64(6), BinaryFilterOp::Le, Value::Int64(5)));
        assert_eq!(result, Some(Value::Bool(false)));

        // >=
        let result =
            eval_literal_expr(binary(Value::Int64(5), BinaryFilterOp::Ge, Value::Int64(5)));
        assert_eq!(result, Some(Value::Bool(true)));

        let result =
            eval_literal_expr(binary(Value::Int64(4), BinaryFilterOp::Ge, Value::Int64(5)));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    // === eval_binary_op: Logical Operators ===

    #[test]
    fn test_eval_logical_and() {
        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::And,
            Value::Bool(true),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::And,
            Value::Bool(false),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_logical_or() {
        let result = eval_literal_expr(binary(
            Value::Bool(false),
            BinaryFilterOp::Or,
            Value::Bool(true),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Bool(false),
            BinaryFilterOp::Or,
            Value::Bool(false),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_logical_xor() {
        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::Xor,
            Value::Bool(false),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::Bool(true),
            BinaryFilterOp::Xor,
            Value::Bool(true),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    // === Type Coercion: Int + Float Arithmetic ===

    #[test]
    fn test_eval_type_coercion_int_plus_float() {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Add,
            Value::Float64(2.5),
        ));
        assert_eq!(result, Some(Value::Float64(12.5)));
    }

    #[test]
    fn test_eval_type_coercion_float_minus_int() {
        let result = eval_literal_expr(binary(
            Value::Float64(10.0),
            BinaryFilterOp::Sub,
            Value::Int64(3),
        ));
        assert_eq!(result, Some(Value::Float64(7.0)));
    }

    #[test]
    fn test_eval_type_coercion_int_mul_float() {
        let result = eval_literal_expr(binary(
            Value::Int64(4),
            BinaryFilterOp::Mul,
            Value::Float64(2.5),
        ));
        assert_eq!(result, Some(Value::Float64(10.0)));
    }

    #[test]
    fn test_eval_type_coercion_int_eq_float() {
        // Int 42 should equal Float 42.0
        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Eq,
            Value::Float64(42.0),
        ));
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_type_coercion_int_lt_float() {
        let result = eval_literal_expr(binary(
            Value::Int64(3),
            BinaryFilterOp::Lt,
            Value::Float64(3.5),
        ));
        assert_eq!(result, Some(Value::Bool(true)));
    }

    // === String Comparison ===

    #[test]
    fn test_eval_string_comparison() {
        let result = eval_literal_expr(binary(
            Value::String("apple".into()),
            BinaryFilterOp::Lt,
            Value::String("banana".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("zebra".into()),
            BinaryFilterOp::Gt,
            Value::String("apple".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_string_concatenation() {
        let result = eval_literal_expr(binary(
            Value::String("Hello".into()),
            BinaryFilterOp::Add,
            Value::String(" World".into()),
        ));
        assert_eq!(result, Some(Value::String("Hello World".into())));
    }

    // === IS NULL / IS NOT NULL ===

    #[test]
    fn test_eval_is_null() {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNull,
            FilterExpression::Literal(Value::Null),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNull,
            FilterExpression::Literal(Value::Int64(42)),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_is_not_null() {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNotNull,
            FilterExpression::Literal(Value::Int64(42)),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::IsNotNull,
            FilterExpression::Literal(Value::Null),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_is_null_on_missing_variable() {
        // Accessing a non-existent variable should produce None,
        // which IS NULL treats as true
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let expr = FilterExpression::Unary {
            op: UnaryFilterOp::IsNull,
            operand: Box::new(FilterExpression::Variable("missing_var".to_string())),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();
        let result = pred.eval_at(&chunk, 0);
        assert_eq!(result, Some(Value::Bool(true)));
    }

    // === STARTS WITH / ENDS WITH / CONTAINS ===

    #[test]
    fn test_eval_starts_with() {
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::StartsWith,
            Value::String("hello".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::StartsWith,
            Value::String("world".into()),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_ends_with() {
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::EndsWith,
            Value::String("world".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::EndsWith,
            Value::String("hello".into()),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_contains() {
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Contains,
            Value::String("lo wo".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Contains,
            Value::String("xyz".into()),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    // === List Operations: IN Operator ===

    #[test]
    fn test_eval_in_operator() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // 2 IN [1, 2, 3] should be true
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(2))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), Arc::clone(&store));
        let result = pred.eval_at(&chunk, 0);
        assert_eq!(result, Some(Value::Bool(true)));

        // 5 IN [1, 2, 3] should be false
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::Int64(5))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), Arc::clone(&store));
        let result = pred.eval_at(&chunk, 0);
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_eval_in_operator_strings() {
        use crate::graph::lpg::LpgStore;

        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // "banana" IN ["apple", "banana", "cherry"]
        let expr = FilterExpression::Binary {
            left: Box::new(FilterExpression::Literal(Value::String("banana".into()))),
            op: BinaryFilterOp::In,
            right: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::String("apple".into())),
                FilterExpression::Literal(Value::String("banana".into())),
                FilterExpression::Literal(Value::String("cherry".into())),
            ])),
        };
        let pred = ExpressionPredicate::new(expr, HashMap::new(), store);
        let result = pred.eval_at(&chunk, 0);
        assert_eq!(result, Some(Value::Bool(true)));
    }

    // === List Index Access ===

    #[test]
    fn test_eval_list_index_access() {
        // [10, 20, 30][2] = 30
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(10)),
                FilterExpression::Literal(Value::Int64(20)),
                FilterExpression::Literal(Value::Int64(30)),
            ])),
            index: Box::new(FilterExpression::Literal(Value::Int64(2))),
        });
        assert_eq!(result, Some(Value::Int64(30)));
    }

    #[test]
    fn test_eval_list_negative_index() {
        // [10, 20, 30][-2] = 20
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(10)),
                FilterExpression::Literal(Value::Int64(20)),
                FilterExpression::Literal(Value::Int64(30)),
            ])),
            index: Box::new(FilterExpression::Literal(Value::Int64(-2))),
        });
        assert_eq!(result, Some(Value::Int64(20)));
    }

    // === CASE / NULLIF Pattern ===

    #[test]
    fn test_eval_case_simple() {
        // CASE WHEN true THEN 'yes' ELSE 'no' END
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Literal(Value::Bool(true)),
                FilterExpression::Literal(Value::String("yes".into())),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                "no".into(),
            )))),
        });
        assert_eq!(result, Some(Value::String("yes".into())));
    }

    #[test]
    fn test_eval_case_falls_to_else() {
        // CASE WHEN false THEN 'yes' ELSE 'no' END
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Literal(Value::Bool(false)),
                FilterExpression::Literal(Value::String("yes".into())),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                "no".into(),
            )))),
        });
        assert_eq!(result, Some(Value::String("no".into())));
    }

    #[test]
    fn test_eval_case_no_else_returns_null() {
        // CASE WHEN false THEN 'yes' END (no ELSE, so NULL)
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Literal(Value::Bool(false)),
                FilterExpression::Literal(Value::String("yes".into())),
            )],
            else_clause: None,
        });
        assert_eq!(result, Some(Value::Null));
    }

    #[test]
    fn test_eval_nullif_via_case() {
        // NULLIF(a, b) is equivalent to: CASE WHEN a = b THEN NULL ELSE a END
        // Test NULLIF(5, 5) => NULL
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(Value::Int64(5))),
                },
                FilterExpression::Literal(Value::Null),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::Int64(5)))),
        });
        assert_eq!(result, Some(Value::Null));

        // NULLIF(5, 3) => 5
        let result = eval_literal_expr(FilterExpression::Case {
            operand: None,
            when_clauses: vec![(
                FilterExpression::Binary {
                    left: Box::new(FilterExpression::Literal(Value::Int64(5))),
                    op: BinaryFilterOp::Eq,
                    right: Box::new(FilterExpression::Literal(Value::Int64(3))),
                },
                FilterExpression::Literal(Value::Null),
            )],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::Int64(5)))),
        });
        assert_eq!(result, Some(Value::Int64(5)));
    }

    #[test]
    fn test_eval_simple_case_with_operand() {
        // CASE 2 WHEN 1 THEN 'one' WHEN 2 THEN 'two' ELSE 'other' END
        let result = eval_literal_expr(FilterExpression::Case {
            operand: Some(Box::new(FilterExpression::Literal(Value::Int64(2)))),
            when_clauses: vec![
                (
                    FilterExpression::Literal(Value::Int64(1)),
                    FilterExpression::Literal(Value::String("one".into())),
                ),
                (
                    FilterExpression::Literal(Value::Int64(2)),
                    FilterExpression::Literal(Value::String("two".into())),
                ),
            ],
            else_clause: Some(Box::new(FilterExpression::Literal(Value::String(
                "other".into(),
            )))),
        });
        assert_eq!(result, Some(Value::String("two".into())));
    }

    // === Unary Operators ===

    #[test]
    fn test_eval_unary_not() {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::Not,
            FilterExpression::Literal(Value::Bool(true)),
        ));
        assert_eq!(result, Some(Value::Bool(false)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::Not,
            FilterExpression::Literal(Value::Bool(false)),
        ));
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_unary_neg() {
        let result = eval_literal_expr(unary(
            UnaryFilterOp::Neg,
            FilterExpression::Literal(Value::Int64(42)),
        ));
        assert_eq!(result, Some(Value::Int64(-42)));

        let result = eval_literal_expr(unary(
            UnaryFilterOp::Neg,
            FilterExpression::Literal(Value::Float64(7.25)),
        ));
        assert_eq!(result, Some(Value::Float64(-7.25)));
    }

    // === Reduce Expression Evaluation ===

    #[test]
    fn test_eval_reduce_sum() {
        // reduce(acc = 0, x IN [1, 2, 3] | acc + x) = 6
        let result = eval_literal_expr(FilterExpression::Reduce {
            accumulator: "acc".to_string(),
            initial: Box::new(FilterExpression::Literal(Value::Int64(0))),
            variable: "x".to_string(),
            list: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            expression: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("acc".to_string())),
                op: BinaryFilterOp::Add,
                right: Box::new(FilterExpression::Variable("x".to_string())),
            }),
        });
        assert_eq!(result, Some(Value::Int64(6)));
    }

    #[test]
    fn test_eval_reduce_product() {
        // reduce(acc = 1, x IN [2, 3, 4] | acc * x) = 24
        let result = eval_literal_expr(FilterExpression::Reduce {
            accumulator: "acc".to_string(),
            initial: Box::new(FilterExpression::Literal(Value::Int64(1))),
            variable: "x".to_string(),
            list: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
                FilterExpression::Literal(Value::Int64(4)),
            ])),
            expression: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("acc".to_string())),
                op: BinaryFilterOp::Mul,
                right: Box::new(FilterExpression::Variable("x".to_string())),
            }),
        });
        assert_eq!(result, Some(Value::Int64(24)));
    }

    // === List Comprehension ===

    #[test]
    fn test_eval_list_comprehension_with_filter() {
        // [x IN [1, 2, 3, 4, 5] WHERE x > 2 | x * 10]
        // Should produce [30, 40, 50]
        let result = eval_literal_expr(FilterExpression::ListComprehension {
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
                FilterExpression::Literal(Value::Int64(4)),
                FilterExpression::Literal(Value::Int64(5)),
            ])),
            filter_expr: Some(Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(2))),
            })),
            map_expr: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Mul,
                right: Box::new(FilterExpression::Literal(Value::Int64(10))),
            }),
        });

        if let Some(Value::List(items)) = result {
            assert_eq!(items.len(), 3);
            assert_eq!(items[0], Value::Int64(30));
            assert_eq!(items[1], Value::Int64(40));
            assert_eq!(items[2], Value::Int64(50));
        } else {
            panic!("Expected List, got {:?}", result);
        }
    }

    // === List Predicate (any/all/none/single) ===

    #[test]
    fn test_eval_list_predicate_any() {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::Any,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(5)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
        });
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_list_predicate_all() {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::All,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(10)),
                FilterExpression::Literal(Value::Int64(20)),
                FilterExpression::Literal(Value::Int64(30)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(5))),
            }),
        });
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_list_predicate_none() {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::None,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(10))),
            }),
        });
        assert_eq!(result, Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_list_predicate_single() {
        let result = eval_literal_expr(FilterExpression::ListPredicate {
            kind: ListPredicateKind::Single,
            variable: "x".to_string(),
            list_expr: Box::new(FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(5)),
                FilterExpression::Literal(Value::Int64(3)),
            ])),
            predicate: Box::new(FilterExpression::Binary {
                left: Box::new(FilterExpression::Variable("x".to_string())),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Int64(4))),
            }),
        });
        // Only x=5 satisfies x > 4, so exactly one
        assert_eq!(result, Some(Value::Bool(true)));
    }

    // === Map key access via index ===

    #[test]
    fn test_eval_map_key_access() {
        // {name: 'Alix'}['name'] = 'Alix'
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::Map(vec![(
                "name".to_string(),
                FilterExpression::Literal(Value::String("Alix".into())),
            )])),
            index: Box::new(FilterExpression::Literal(Value::String("name".into()))),
        });
        assert_eq!(result, Some(Value::String("Alix".into())));
    }

    // === LIKE operator tests (require regex for pattern conversion) ===

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_eval_like_wildcard() {
        // 'hello world' LIKE 'hello%'
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Like,
            Value::String("hello%".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        // 'hello world' LIKE '%world'
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Like,
            Value::String("%world".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        // 'hello world' LIKE '%llo%'
        let result = eval_literal_expr(binary(
            Value::String("hello world".into()),
            BinaryFilterOp::Like,
            Value::String("%llo%".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        // 'hello' LIKE 'world%'
        let result = eval_literal_expr(binary(
            Value::String("hello".into()),
            BinaryFilterOp::Like,
            Value::String("world%".into()),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_eval_like_single_char() {
        // 'cat' LIKE 'c_t'
        let result = eval_literal_expr(binary(
            Value::String("cat".into()),
            BinaryFilterOp::Like,
            Value::String("c_t".into()),
        ));
        assert_eq!(result, Some(Value::Bool(true)));

        // 'cart' LIKE 'c_t'
        let result = eval_literal_expr(binary(
            Value::String("cart".into()),
            BinaryFilterOp::Like,
            Value::String("c_t".into()),
        ));
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[cfg(any(feature = "regex", feature = "regex-lite"))]
    #[test]
    fn test_eval_like_null() {
        // NULL LIKE '%' -> NULL
        let result = eval_literal_expr(binary(
            Value::Null,
            BinaryFilterOp::Like,
            Value::String("%".into()),
        ));
        assert_eq!(result, Some(Value::Null));
    }

    // === Concat operator (||) tests ===

    #[test]
    fn test_eval_concat_strings() {
        let result = eval_literal_expr(binary(
            Value::String("hello".into()),
            BinaryFilterOp::Concat,
            Value::String(" world".into()),
        ));
        assert_eq!(result, Some(Value::String("hello world".into())));
    }

    #[test]
    fn test_eval_concat_string_with_int() {
        let result = eval_literal_expr(binary(
            Value::String("count: ".into()),
            BinaryFilterOp::Concat,
            Value::Int64(42),
        ));
        assert_eq!(result, Some(Value::String("count: 42".into())));
    }

    #[test]
    fn test_eval_concat_int_with_string() {
        let result = eval_literal_expr(binary(
            Value::Int64(42),
            BinaryFilterOp::Concat,
            Value::String(" items".into()),
        ));
        assert_eq!(result, Some(Value::String("42 items".into())));
    }

    #[test]
    fn test_eval_concat_null() {
        // Null || Null -> Null (hits the null arm)
        let result = eval_literal_expr(binary(Value::Null, BinaryFilterOp::Concat, Value::Null));
        assert_eq!(result, Some(Value::Null));
    }

    // === Modulo operator tests ===

    #[test]
    fn test_eval_modulo_float() {
        let result = eval_literal_expr(binary(
            Value::Float64(10.5),
            BinaryFilterOp::Mod,
            Value::Float64(3.0),
        ));
        if let Some(Value::Float64(v)) = result {
            assert!((v - 1.5).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_eval_modulo_mixed() {
        // int % float
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Mod,
            Value::Float64(3.0),
        ));
        if let Some(Value::Float64(v)) = result {
            assert!((v - 1.0).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }

        // float % int
        let result = eval_literal_expr(binary(
            Value::Float64(10.0),
            BinaryFilterOp::Mod,
            Value::Int64(3),
        ));
        if let Some(Value::Float64(v)) = result {
            assert!((v - 1.0).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }
    }

    #[test]
    fn test_eval_modulo_by_zero() {
        let result = eval_literal_expr(binary(
            Value::Int64(10),
            BinaryFilterOp::Mod,
            Value::Int64(0),
        ));
        assert_eq!(result, None);

        let result = eval_literal_expr(binary(
            Value::Float64(10.0),
            BinaryFilterOp::Mod,
            Value::Float64(0.0),
        ));
        assert_eq!(result, None);
    }

    // === String addition with type coercion ===

    #[test]
    fn test_eval_string_add_int() {
        let result = eval_literal_expr(binary(
            Value::String("val:".into()),
            BinaryFilterOp::Add,
            Value::Int64(42),
        ));
        assert_eq!(result, Some(Value::String("val:42".into())));
    }

    #[test]
    fn test_eval_string_add_bool() {
        let result = eval_literal_expr(binary(
            Value::String("is:".into()),
            BinaryFilterOp::Add,
            Value::Bool(true),
        ));
        assert_eq!(result, Some(Value::String("is:true".into())));
    }

    #[test]
    fn test_eval_string_add_null() {
        let result = eval_literal_expr(binary(
            Value::String("val:".into()),
            BinaryFilterOp::Add,
            Value::Null,
        ));
        assert_eq!(result, Some(Value::Null));
    }

    // === Slice access tests ===

    #[test]
    fn test_eval_string_slice() {
        // "hello"[1..3] = "el"
        let result = eval_literal_expr(FilterExpression::SliceAccess {
            base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            start: Some(Box::new(FilterExpression::Literal(Value::Int64(1)))),
            end: Some(Box::new(FilterExpression::Literal(Value::Int64(3)))),
        });
        assert_eq!(result, Some(Value::String("el".into())));
    }

    #[test]
    fn test_eval_string_index_access() {
        // "hello"[1] = "e"
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            index: Box::new(FilterExpression::Literal(Value::Int64(1))),
        });
        assert_eq!(result, Some(Value::String("e".into())));
    }

    #[test]
    fn test_eval_string_negative_index() {
        // "hello"[-1] = "o"
        let result = eval_literal_expr(FilterExpression::IndexAccess {
            base: Box::new(FilterExpression::Literal(Value::String("hello".into()))),
            index: Box::new(FilterExpression::Literal(Value::Int64(-1))),
        });
        assert_eq!(result, Some(Value::String("o".into())));
    }

    // === Function tests for uncovered branches ===

    #[test]
    fn test_eval_tostring_types() {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Bool -> String
        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![FilterExpression::Literal(Value::Bool(true))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::String("true".into())));

        // Float -> String
        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![FilterExpression::Literal(Value::Float64(2.72))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::String("2.72".into())));

        // Null -> Null
        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toString".to_string(),
                args: vec![FilterExpression::Literal(Value::Null)],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Null));
    }

    #[test]
    fn test_eval_toboolean() {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::String("true".into()))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Bool(true)));

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::String("false".into()))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Bool(false)));

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toBoolean".to_string(),
                args: vec![FilterExpression::Literal(Value::Bool(true))],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Bool(true)));
    }

    #[test]
    fn test_eval_tofloat() {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toFloat".to_string(),
                args: vec![FilterExpression::Literal(Value::String("2.72".into()))],
            },
            vc.clone(),
            Arc::clone(&store),
        );
        if let Some(Value::Float64(v)) = pred.eval_at(&chunk, 0) {
            assert!((v - 2.72).abs() < 0.001);
        } else {
            panic!("Expected Float64");
        }

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toFloat".to_string(),
                args: vec![FilterExpression::Literal(Value::Int64(42))],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Float64(42.0)));
    }

    #[test]
    fn test_eval_tointeger_from_float() {
        use crate::graph::lpg::LpgStore;
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let vc = HashMap::new();
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        let pred = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "toInteger".to_string(),
                args: vec![FilterExpression::Literal(Value::Float64(3.7))],
            },
            vc,
            store,
        );
        assert_eq!(pred.eval_at(&chunk, 0), Some(Value::Int64(3)));
    }

    #[test]
    fn test_eval_reverse_list() {
        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "reverse".to_string(),
            args: vec![FilterExpression::List(vec![
                FilterExpression::Literal(Value::Int64(1)),
                FilterExpression::Literal(Value::Int64(2)),
                FilterExpression::Literal(Value::Int64(3)),
            ])],
        });
        assert_eq!(
            result,
            Some(Value::List(
                vec![Value::Int64(3), Value::Int64(2), Value::Int64(1)].into()
            ))
        );
    }

    #[test]
    fn test_eval_reverse_string() {
        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "reverse".to_string(),
            args: vec![FilterExpression::Literal(Value::String("abc".into()))],
        });
        assert_eq!(result, Some(Value::String("cba".into())));
    }

    #[test]
    fn test_eval_exists_function() {
        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "exists".to_string(),
            args: vec![FilterExpression::Literal(Value::Int64(42))],
        });
        assert_eq!(result, Some(Value::Bool(true)));

        let result = eval_literal_expr(FilterExpression::FunctionCall {
            name: "exists".to_string(),
            args: vec![FilterExpression::Literal(Value::Null)],
        });
        assert_eq!(result, Some(Value::Bool(false)));
    }

    #[test]
    fn test_filter_into_any() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let predicate = ComparisonPredicate::new(0, CompareOp::Eq, Value::Int64(1));
        let op = FilterOperator::new(Box::new(mock), Box::new(predicate));
        let any = Box::new(op).into_any();
        assert!(any.downcast::<FilterOperator>().is_ok());
    }

    #[test]
    fn test_filter_into_parts() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let predicate = ComparisonPredicate::new(0, CompareOp::Gt, Value::Int64(5));
        let op = FilterOperator::new(Box::new(mock), Box::new(predicate));
        let (mut child, _predicate) = op.into_parts();
        assert!(child.next().unwrap().is_none());
    }
}

#[cfg(all(test, feature = "text-index", feature = "lpg"))]
mod text_fn_tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::graph::GraphStoreSearch;
    use crate::graph::lpg::LpgStore;
    use crate::index::text::{BM25Config, InvertedIndex};
    use grafeo_common::types::LogicalType;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn setup_store_with_text_index() -> (
        Arc<LpgStore>,
        grafeo_common::types::NodeId,
        grafeo_common::types::NodeId,
    ) {
        let store = Arc::new(LpgStore::new().unwrap());

        // Create nodes with text properties
        let n1 = store.create_node(&["Article"]);
        store.set_node_property(
            n1,
            "body",
            Value::String("rust graph database engine".into()),
        );
        let n2 = store.create_node(&["Article"]);
        store.set_node_property(n2, "body", Value::String("python web framework".into()));

        // Create and populate text index
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(n1, "rust graph database engine");
        index.insert(n2, "python web framework");
        store.add_text_index("Article", "body", Arc::new(RwLock::new(index)));

        (store, n1, n2)
    }

    #[test]
    fn test_text_score_function() {
        let (store, n1, n2) = setup_store_with_text_index();

        // Build a chunk with two rows: n1 in row 0, n2 in row 1
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(n1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_node_id(n2);
        builder.advance_row();
        let chunk = builder.finish();

        let mut variable_columns = HashMap::new();
        variable_columns.insert("doc".to_string(), 0);

        // text_score(doc.body, "rust database") > 0.0
        let predicate = ExpressionPredicate::new(
            FilterExpression::Binary {
                left: Box::new(FilterExpression::FunctionCall {
                    name: "text_score".to_string(),
                    args: vec![
                        FilterExpression::Property {
                            variable: "doc".to_string(),
                            property: "body".to_string(),
                        },
                        FilterExpression::Literal(Value::String("rust database".into())),
                    ],
                }),
                op: BinaryFilterOp::Gt,
                right: Box::new(FilterExpression::Literal(Value::Float64(0.0))),
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );

        // n1 matches "rust database" — should pass
        assert!(
            predicate.evaluate(&chunk, 0),
            "n1 should score > 0 for 'rust database'"
        );
        // n2 does not match — should fail
        assert!(
            !predicate.evaluate(&chunk, 1),
            "n2 should score 0 for 'rust database'"
        );
    }

    /// `keys()`, `properties()` and `property_values()` read the entity the
    /// column holds: an edge column reads the edge even when a node has the
    /// same ID, and an untyped column holding a raw ID reads the edge when no
    /// node has that ID (it used to stop at the missing node).
    /// `property_values(n)` lists the values in key order, as `keys(n)` lists
    /// the keys.
    #[test]
    fn property_values_line_up_with_keys() {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        // Set in an order other than the key order.
        store.set_node_property(alix, "name", Value::from("Alix"));
        store.set_node_property(alix, "age", Value::Int64(30));
        let eval = |function: &str| {
            let mut column = ValueVector::with_capacity(LogicalType::Node, 1);
            column.push_node_id(alix);
            ExpressionPredicate::new(
                FilterExpression::FunctionCall {
                    name: function.to_string(),
                    args: vec![FilterExpression::Variable("n".to_string())],
                },
                HashMap::from([("n".to_string(), 0)]),
                Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            )
            .eval(&DataChunk::new(vec![column]), 0)
        };
        assert_eq!(
            eval("keys"),
            Some(Value::List(
                vec![Value::from("age"), Value::from("name")].into()
            ))
        );
        assert_eq!(
            eval("property_values"),
            Some(Value::List(
                vec![Value::Int64(30), Value::from("Alix")].into()
            ))
        );
    }

    #[test]
    fn element_functions_read_the_entity_the_column_holds() {
        let store = Arc::new(LpgStore::new().unwrap());
        let alix = store.create_node(&["Person"]);
        store.set_node_property(alix, "name", Value::from("Alix"));
        let gus = store.create_node(&["Person"]);
        // Edge 0 has the ID of node 0 (Alix); edge 2 has no node of its ID.
        let first = store.create_edge(alix, gus, "KNOWS");
        store.set_edge_property(first, "since", Value::Int64(2010));
        store.create_edge(gus, alix, "KNOWS");
        let third = store.create_edge(alix, alix, "KNOWS");
        store.set_edge_property(third, "w", Value::Int64(3));

        let eval = |column: ValueVector, function: &str| {
            let predicate = ExpressionPredicate::new(
                FilterExpression::FunctionCall {
                    name: function.to_string(),
                    args: vec![FilterExpression::Variable("r".to_string())],
                },
                HashMap::from([("r".to_string(), 0)]),
                Arc::clone(&store) as Arc<dyn GraphStoreSearch>,
            );
            predicate.eval(&DataChunk::new(vec![column]), 0)
        };
        let keys = |names: &[&str]| {
            Some(Value::List(
                names
                    .iter()
                    .map(|name| Value::from(*name))
                    .collect::<Vec<_>>()
                    .into(),
            ))
        };

        let mut typed = ValueVector::with_capacity(LogicalType::Edge, 1);
        typed.push_edge_id(first);
        assert_eq!(eval(typed.clone(), "keys"), keys(&["since"]));
        assert_eq!(
            eval(typed.clone(), "property_values"),
            Some(Value::List(vec![Value::Int64(2010)].into()))
        );
        let Some(Value::Map(map)) = eval(typed, "properties") else {
            panic!("expected a map");
        };
        assert_eq!(
            map.get(&PropertyKey::new("since")),
            Some(&Value::Int64(2010))
        );
        assert_eq!(map.len(), 1);

        let mut untyped = ValueVector::with_capacity(LogicalType::Any, 1);
        untyped.push_value(Value::Int64(i64::try_from(third.as_u64()).unwrap()));
        assert_eq!(eval(untyped, "keys"), keys(&["w"]));

        let mut node = ValueVector::with_capacity(LogicalType::Node, 1);
        node.push_node_id(alix);
        assert_eq!(eval(node, "keys"), keys(&["name"]));
    }

    #[test]
    fn test_text_match_function() {
        let (store, n1, n2) = setup_store_with_text_index();

        // Build a chunk with two rows
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(n1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_node_id(n2);
        builder.advance_row();
        let chunk = builder.finish();

        let mut variable_columns = HashMap::new();
        variable_columns.insert("doc".to_string(), 0);

        // text_match(doc.body, "rust") = true/false
        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "text_match".to_string(),
                args: vec![
                    FilterExpression::Property {
                        variable: "doc".to_string(),
                        property: "body".to_string(),
                    },
                    FilterExpression::Literal(Value::String("rust".into())),
                ],
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );

        // n1 contains "rust" — text_match should return Bool(true) → evaluates to true
        assert!(predicate.evaluate(&chunk, 0), "n1 should match 'rust'");
        // n2 does not contain "rust" — text_match should return Bool(false)
        assert!(!predicate.evaluate(&chunk, 1), "n2 should not match 'rust'");
    }

    #[test]
    fn test_text_score_wrong_arg_count_returns_none() {
        let store = Arc::new(LpgStore::new().unwrap());
        let builder = DataChunkBuilder::new(&[LogicalType::Node]);
        let chunk = builder.finish();
        let variable_columns = HashMap::new();

        // text_score with wrong number of args should return None (evaluate = false)
        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "text_score".to_string(),
                args: vec![FilterExpression::Literal(Value::String("only_one".into()))],
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );
        assert!(!predicate.evaluate(&chunk, 0));
    }

    #[test]
    fn test_text_score_no_index_returns_none() {
        // Node has a label but no text index for that label+property
        let store = Arc::new(LpgStore::new().unwrap());
        let n1 = store.create_node(&["Article"]);
        store.set_node_property(n1, "body", Value::String("rust graph database".into()));
        // No text index added

        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(n1);
        builder.advance_row();
        let chunk = builder.finish();

        let mut variable_columns = HashMap::new();
        variable_columns.insert("doc".to_string(), 0);

        let predicate = ExpressionPredicate::new(
            FilterExpression::FunctionCall {
                name: "text_score".to_string(),
                args: vec![
                    FilterExpression::Property {
                        variable: "doc".to_string(),
                        property: "body".to_string(),
                    },
                    FilterExpression::Literal(Value::String("rust".into())),
                ],
            },
            variable_columns,
            store as Arc<dyn GraphStoreSearch>,
        );
        // No index → score_text returns None → eval returns None → evaluate returns false
        assert!(!predicate.evaluate(&chunk, 0));
    }
}
