//! Filter operator for applying predicates.

use super::{Operator, OperatorResult};
use crate::execution::{ChunkZoneHints, DataChunk, SelectionVector};
use crate::graph::Direction;
use crate::graph::lpg::LpgStore;
use grafeo_common::types::{PropertyKey, Value};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

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
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompareOp {
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
#[allow(dead_code)]
pub struct ComparisonPredicate {
    /// Column index to compare.
    column: usize,
    /// Comparison operator.
    op: CompareOp,
    /// Value to compare against.
    value: Value,
}

impl ComparisonPredicate {
    /// Creates a new comparison predicate.
    #[allow(dead_code)]
    pub fn new(column: usize, op: CompareOp, value: Value) -> Self {
        Self { column, op, value }
    }
}

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
    store: Arc<LpgStore>,
}

/// A filter expression that can be evaluated.
#[derive(Debug, Clone)]
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
    /// Map literal (e.g., {name: 'Alice', age: 30}).
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
    /// EXISTS subquery - evaluates inner plan and returns true if results exist.
    ExistsSubquery {
        /// The start node variable from outer query.
        start_var: String,
        /// Direction of edge traversal.
        direction: Direction,
        /// Optional edge type filter.
        edge_type: Option<String>,
        /// Optional end node labels filter.
        end_labels: Option<Vec<String>>,
        /// Minimum number of hops (for variable-length patterns).
        min_hops: Option<u32>,
        /// Maximum number of hops (for variable-length patterns).
        max_hops: Option<u32>,
    },
}

/// Binary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
}

/// Unary operators for filter expressions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
        store: Arc<LpgStore>,
    ) -> Self {
        Self {
            expression,
            variable_columns,
            store,
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
                    && let Some(node) = self.store.get_node(node_id)
                {
                    return node.get_property(property).cloned();
                }
                // Try as edge if node lookup failed
                if let Some(edge_id) = col.get_edge_id(row)
                    && let Some(edge) = self.store.get_edge(edge_id)
                {
                    return edge.get_property(property).cloned();
                }
                None
            }
            FilterExpression::Binary { left, op, right } => {
                // For IN operator, right side is a list that we evaluate specially
                if *op == BinaryFilterOp::In {
                    let left_val = self.eval_expr(left, chunk, row)?;
                    return self.eval_in_operator(&left_val, right, chunk, row);
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
            FilterExpression::List(items) => {
                let values: Vec<Value> = items
                    .iter()
                    .filter_map(|item| self.eval_expr(item, chunk, row))
                    .collect();
                Some(Value::List(values.into()))
            }
            FilterExpression::Map(pairs) => {
                let map: BTreeMap<PropertyKey, Value> = pairs
                    .iter()
                    .filter_map(|(k, v)| {
                        self.eval_expr(v, chunk, row)
                            .map(|val| (PropertyKey::new(k.clone()), val))
                    })
                    .collect();
                Some(Value::Map(Arc::new(map)))
            }
            FilterExpression::IndexAccess { base, index } => {
                let base_val = self.eval_expr(base, chunk, row)?;
                let index_val = self.eval_expr(index, chunk, row)?;
                match (&base_val, &index_val) {
                    (Value::List(items), Value::Int64(i)) => {
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
                        let idx = if *i < 0 {
                            let len = s.len() as i64;
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
                    Some(Value::Int64(node_id.0 as i64))
                } else if let Some(edge_id) = col.get_edge_id(row) {
                    Some(Value::Int64(edge_id.0 as i64))
                } else {
                    None
                }
            }
            FilterExpression::Labels(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let node_id = col.get_node_id(row)?;
                let node = self.store.get_node(node_id)?;
                let labels: Vec<Value> = node
                    .labels
                    .iter()
                    .map(|l| Value::String(l.clone()))
                    .collect();
                Some(Value::List(labels.into()))
            }
            FilterExpression::Type(variable) => {
                let col_idx = *self.variable_columns.get(variable)?;
                let col = chunk.column(col_idx)?;
                let edge_id = col.get_edge_id(row)?;
                let edge = self.store.get_edge(edge_id)?;
                Some(Value::String(edge.edge_type.clone()))
            }
            FilterExpression::ListComprehension {
                variable,
                list_expr,
                filter_expr,
                map_expr,
            } => {
                // Evaluate the source list
                let list_val = self.eval_expr(list_expr, chunk, row)?;
                let Value::List(items) = list_val else {
                    return None; // Not a list
                };

                // Build the result list by iterating over source items
                let mut result = Vec::new();
                for item in items.iter() {
                    // Create a temporary context with the iteration variable bound
                    // For now, we'll do a simplified version that works for literals
                    // A full implementation would need to create a sub-evaluator

                    // Check filter predicate if present
                    let passes_filter = if let Some(filter) = filter_expr {
                        // Simplified: evaluate filter with item as context
                        // This works for simple cases like x > 5
                        matches!(
                            self.eval_comprehension_expr(filter, item, variable),
                            Some(Value::Bool(true))
                        )
                    } else {
                        true
                    };

                    if passes_filter {
                        // Apply the mapping expression
                        if let Some(mapped) = self.eval_comprehension_expr(map_expr, item, variable)
                        {
                            result.push(mapped);
                        }
                    }
                }

                Some(Value::List(result.into()))
            }
            FilterExpression::ExistsSubquery {
                start_var,
                direction,
                edge_type,
                ..
            } => {
                // Get the start node ID from the current row
                let col_idx = *self.variable_columns.get(start_var)?;
                let col = chunk.column(col_idx)?;
                let start_node_id = col.get_node_id(row)?;

                // Check if any matching edges exist
                let exists =
                    self.store
                        .edges_from(start_node_id, *direction)
                        .any(|(_, edge_id)| {
                            // Check edge type if specified
                            if let Some(required_type) = edge_type {
                                if let Some(actual_type) = self.store.edge_type(edge_id) {
                                    actual_type.as_str() == required_type.as_str()
                                } else {
                                    false
                                }
                            } else {
                                true
                            }
                        });

                Some(Value::Bool(exists))
            }
        }
    }

    /// Evaluates an expression in the context of a list comprehension.
    /// The `item` is the current iteration value bound to `variable`.
    fn eval_comprehension_expr(
        &self,
        expr: &FilterExpression,
        item: &Value,
        variable: &str,
    ) -> Option<Value> {
        match expr {
            FilterExpression::Variable(name) if name == variable => Some(item.clone()),
            FilterExpression::Literal(v) => Some(v.clone()),
            FilterExpression::Binary { left, op, right } => {
                let left_val = self.eval_comprehension_expr(left, item, variable)?;
                let right_val = self.eval_comprehension_expr(right, item, variable)?;
                self.eval_binary_op(&left_val, *op, &right_val)
            }
            FilterExpression::Unary { op, operand } => {
                let val = self.eval_comprehension_expr(operand, item, variable);
                self.eval_unary_op(*op, val)
            }
            FilterExpression::Property {
                variable: var,
                property,
            } if var == variable => {
                // Property access on the iteration variable
                if let Value::Map(m) = item {
                    let key = PropertyKey::new(property.clone());
                    m.get(&key).cloned()
                } else {
                    None
                }
            }
            // For other expression types, return None (unsupported in comprehension)
            _ => None,
        }
    }

    fn eval_binary_op(&self, left: &Value, op: BinaryFilterOp, right: &Value) -> Option<Value> {
        match op {
            BinaryFilterOp::And => {
                let l = left.as_bool()?;
                let r = right.as_bool()?;
                Some(Value::Bool(l && r))
            }
            BinaryFilterOp::Or => {
                let l = left.as_bool()?;
                let r = right.as_bool()?;
                Some(Value::Bool(l || r))
            }
            BinaryFilterOp::Xor => {
                let l = left.as_bool()?;
                let r = right.as_bool()?;
                Some(Value::Bool(l ^ r))
            }
            BinaryFilterOp::Eq => Some(Value::Bool(self.values_equal(left, right))),
            BinaryFilterOp::Ne => Some(Value::Bool(!self.values_equal(left, right))),
            BinaryFilterOp::Lt => self.compare_values(left, right).map(|c| Value::Bool(c < 0)),
            BinaryFilterOp::Le => self
                .compare_values(left, right)
                .map(|c| Value::Bool(c <= 0)),
            BinaryFilterOp::Gt => self.compare_values(left, right).map(|c| Value::Bool(c > 0)),
            BinaryFilterOp::Ge => self
                .compare_values(left, right)
                .map(|c| Value::Bool(c >= 0)),
            // Arithmetic operators
            BinaryFilterOp::Add => self.eval_arithmetic(left, right, |a, b| a + b, |a, b| a + b),
            BinaryFilterOp::Sub => self.eval_arithmetic(left, right, |a, b| a - b, |a, b| a - b),
            BinaryFilterOp::Mul => self.eval_arithmetic(left, right, |a, b| a * b, |a, b| a * b),
            BinaryFilterOp::Div => self.eval_arithmetic(left, right, |a, b| a / b, |a, b| a / b),
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
            // Regex match (=~)
            BinaryFilterOp::Regex => {
                match (left, right) {
                    (Value::String(s), Value::String(pattern)) => {
                        // Compile the regex pattern and match against the string
                        match Regex::new(pattern) {
                            Ok(re) => Some(Value::Bool(re.is_match(s))),
                            Err(_) => None, // Invalid regex pattern
                        }
                    }
                    _ => None, // Type mismatch - regex requires strings
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
        F1: Fn(i64, i64) -> i64,
        F2: Fn(f64, f64) -> f64,
    {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) => Some(Value::Int64(int_op(*a, *b))),
            (Value::Float64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a, *b))),
            (Value::Int64(a), Value::Float64(b)) => Some(Value::Float64(float_op(*a as f64, *b))),
            (Value::Float64(a), Value::Int64(b)) => Some(Value::Float64(float_op(*a, *b as f64))),
            _ => None,
        }
    }

    fn eval_modulo(&self, left: &Value, right: &Value) -> Option<Value> {
        match (left, right) {
            (Value::Int64(a), Value::Int64(b)) if *b != 0 => Some(Value::Int64(a % b)),
            (Value::Float64(a), Value::Float64(b)) if *b != 0.0 => Some(Value::Float64(a % b)),
            (Value::Int64(a), Value::Float64(b)) if *b != 0.0 => {
                Some(Value::Float64(*a as f64 % b))
            }
            (Value::Float64(a), Value::Int64(b)) if *b != 0 => Some(Value::Float64(a % *b as f64)),
            _ => None,
        }
    }

    fn eval_in_operator(
        &self,
        left: &Value,
        right: &FilterExpression,
        chunk: &DataChunk,
        row: usize,
    ) -> Option<Value> {
        // Evaluate the right side - it should be a list
        let right_val = self.eval_expr(right, chunk, row)?;
        match right_val {
            Value::List(items) => {
                let found = items.iter().any(|item| self.values_equal(left, item));
                Some(Value::Bool(found))
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
        match name.to_lowercase().as_str() {
            "id" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    if let Some(node_id) = col.get_node_id(row) {
                        return Some(Value::Int64(node_id.0 as i64));
                    } else if let Some(edge_id) = col.get_edge_id(row) {
                        return Some(Value::Int64(edge_id.0 as i64));
                    }
                }
                None
            }
            "labels" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    let node_id = col.get_node_id(row)?;
                    let node = self.store.get_node(node_id)?;
                    let labels: Vec<Value> = node
                        .labels
                        .iter()
                        .map(|l| Value::String(l.clone()))
                        .collect();
                    return Some(Value::List(labels.into()));
                }
                None
            }
            "type" => {
                if args.len() != 1 {
                    return None;
                }
                if let FilterExpression::Variable(var) = &args[0] {
                    let col_idx = *self.variable_columns.get(var)?;
                    let col = chunk.column(col_idx)?;
                    let edge_id = col.get_edge_id(row)?;
                    let edge = self.store.get_edge(edge_id)?;
                    return Some(Value::String(edge.edge_type.clone()));
                }
                None
            }
            "size" | "length" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::List(items) => Some(Value::Int64(items.len() as i64)),
                    Value::String(s) => Some(Value::Int64(s.len() as i64)),
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
            "tostring" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                Some(Value::String(format!("{:?}", val).into()))
            }
            "tointeger" | "toint" => {
                if args.len() != 1 {
                    return None;
                }
                let val = self.eval_expr(&args[0], chunk, row)?;
                match val {
                    Value::Int64(i) => Some(Value::Int64(i)),
                    Value::Float64(f) => Some(Value::Int64(f as i64)),
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
                let node = self.store.get_node(node_id)?;
                let has_label = node.labels.iter().any(|l| l.as_str() == label.as_str());
                Some(Value::Bool(has_label))
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
            // Vector distance / similarity functions (SIMD-accelerated)
            "cosine_similarity" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::cosine_similarity(a, b) as f64,
                ))
            }
            "euclidean_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::euclidean_distance(a, b) as f64,
                ))
            }
            "dot_product" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::dot_product(a, b) as f64
                ))
            }
            "manhattan_distance" => {
                if args.len() != 2 {
                    return None;
                }
                let a_val = self.eval_expr(&args[0], chunk, row)?;
                let b_val = self.eval_expr(&args[1], chunk, row)?;
                let a = a_val.as_vector()?;
                let b = b_val.as_vector()?;
                if a.len() != b.len() {
                    return None;
                }
                Some(Value::Float64(
                    crate::index::vector::manhattan_distance(a, b) as f64,
                ))
            }
            _ => None, // Unknown function
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
            let test_val = self.eval_expr(test_expr, chunk, row)?;
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row)?;
                if self.values_equal(&test_val, &when_val) {
                    return self.eval_expr(then_expr, chunk, row);
                }
            }
        } else {
            // Searched CASE: CASE WHEN cond1 THEN res1 ...
            for (when_expr, then_expr) in when_clauses {
                let when_val = self.eval_expr(when_expr, chunk, row)?;
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
                Value::Int64(i) => Some(Value::Int64(-i)),
                Value::Float64(f) => Some(Value::Float64(-f)),
                _ => None,
            },
        }
    }

    fn values_equal(&self, left: &Value, right: &Value) -> bool {
        match (left, right) {
            (Value::Null, Value::Null) => true,
            (Value::Bool(a), Value::Bool(b)) => a == b,
            (Value::Int64(a), Value::Int64(b)) => a == b,
            (Value::Float64(a), Value::Float64(b)) => (a - b).abs() < f64::EPSILON,
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Int64(a), Value::Float64(b)) | (Value::Float64(b), Value::Int64(a)) => {
                (*a as f64 - b).abs() < f64::EPSILON
            }
            _ => false,
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
            (Value::Int64(a), Value::Float64(b)) => {
                let af = *a as f64;
                if af < *b {
                    Some(-1)
                } else if af > *b {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            (Value::Float64(a), Value::Int64(b)) => {
                let bf = *b as f64;
                if *a < bf {
                    Some(-1)
                } else if *a > bf {
                    Some(1)
                } else {
                    Some(0)
                }
            }
            _ => None,
        }
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

            // Apply predicate to create selection vector
            let count = chunk.total_row_count();
            let selection =
                SelectionVector::from_predicate(count, |row| self.predicate.evaluate(&chunk, row));

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
}

#[cfg(test)]
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

    #[test]
    fn test_regex_operator() {
        use crate::graph::lpg::LpgStore;

        // Create a store and expression predicate to test regex
        let store = Arc::new(LpgStore::new());
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

        let store = Arc::new(LpgStore::new());
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

        let store = Arc::new(LpgStore::new());
        let variable_columns = HashMap::new();

        // Create a minimal chunk for evaluation
        let builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        let chunk = builder.finish();

        // Create map {name: 'Alice', age: 30}
        let predicate = ExpressionPredicate::new(
            FilterExpression::Map(vec![
                (
                    "name".to_string(),
                    FilterExpression::Literal(Value::String("Alice".into())),
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
                Some(&Value::String("Alice".into()))
            );
            assert_eq!(m.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Map value");
        }
    }

    #[test]
    fn test_index_access_list() {
        use crate::graph::lpg::LpgStore;

        let store = Arc::new(LpgStore::new());
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

        let store = Arc::new(LpgStore::new());
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
        builder.column_mut(0).unwrap().push_float64(3.14);
        builder.advance_row();
        let chunk = builder.finish();

        // Test float equality (within epsilon)
        let pred_eq = ComparisonPredicate::new(0, CompareOp::Eq, Value::Float64(3.14));
        assert!(pred_eq.evaluate(&chunk, 0));

        let pred_ne = ComparisonPredicate::new(0, CompareOp::Ne, Value::Float64(2.71));
        assert!(pred_ne.evaluate(&chunk, 0));

        let pred_lt = ComparisonPredicate::new(0, CompareOp::Lt, Value::Float64(4.0));
        assert!(pred_lt.evaluate(&chunk, 0));

        let pred_ge = ComparisonPredicate::new(0, CompareOp::Ge, Value::Float64(3.14));
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        let store = Arc::new(crate::graph::lpg::LpgStore::new());
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
        // Test the labels() function in predicates
        let store = Arc::new(crate::graph::lpg::LpgStore::new());

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
            Arc::clone(&store),
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
}
