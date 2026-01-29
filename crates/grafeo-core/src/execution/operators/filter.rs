//! Filter operator for applying predicates.

use super::{Operator, OperatorResult};
use crate::execution::{DataChunk, SelectionVector};
use crate::graph::Direction;
use crate::graph::lpg::LpgStore;
use grafeo_common::types::{PropertyKey, Value};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A predicate for filtering rows.
pub trait Predicate: Send + Sync {
    /// Evaluates the predicate for a row.
    fn evaluate(&self, chunk: &DataChunk, row: usize) -> bool;
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
        let col = match chunk.column(self.column) {
            Some(c) => c,
            None => return false,
        };

        let cell_value = match col.get_value(row) {
            Some(v) => v,
            None => return false,
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
                if let Some(node_id) = col.get_node_id(row) {
                    if let Some(node) = self.store.get_node(node_id) {
                        return node.get_property(property).cloned();
                    }
                }
                // Try as edge if node lookup failed
                if let Some(edge_id) = col.get_edge_id(row) {
                    if let Some(edge) = self.store.get_edge(edge_id) {
                        return edge.get_property(property).cloned();
                    }
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
                        let prop_key = PropertyKey::new(key.as_ref());
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
                let items = match list_val {
                    Value::List(items) => items,
                    _ => return None, // Not a list
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
                        match self.eval_comprehension_expr(filter, item, variable) {
                            Some(Value::Bool(true)) => true,
                            _ => false,
                        }
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
                                    actual_type.as_ref() == required_type.as_str()
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
                    if let Some(val) = self.eval_expr(arg, chunk, row) {
                        if !matches!(val, Value::Null) {
                            return Some(val);
                        }
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
                let label = match self.eval_expr(&args[1], chunk, row)? {
                    Value::String(s) => s,
                    _ => return None,
                };
                // Check if the node has this label
                let node = self.store.get_node(node_id)?;
                let has_label = node.labels.iter().any(|l| l.as_ref() == label.as_ref());
                Some(Value::Bool(has_label))
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
        // Get next chunk from child
        let mut chunk = match self.child.next()? {
            Some(c) => c,
            None => return Ok(None),
        };

        // Apply predicate to create selection vector
        let count = chunk.total_row_count();
        let selection =
            SelectionVector::from_predicate(count, |row| self.predicate.evaluate(&chunk, row));

        // If nothing passes, skip to next chunk
        if selection.is_empty() {
            return self.next();
        }

        chunk.set_selection(selection);
        Ok(Some(chunk))
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
}
