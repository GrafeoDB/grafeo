//! Project operator for selecting and transforming columns.

use super::filter::{ExpressionPredicate, FilterExpression, SessionContext};
use super::{Operator, OperatorError, OperatorResult};
use crate::execution::{DataChunk, ValueVector};
use crate::graph::GraphStore;
use crate::graph::lpg::{Edge, Node};
use grafeo_common::types::{EdgeId, LogicalType, NodeId, PropertyKey, ReadView, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

/// A projection expression.
#[non_exhaustive]
pub enum ProjectExpr {
    /// Reference to an input column.
    Column(usize),
    /// A constant value.
    Constant(Value),
    /// Property access on a node/edge column.
    PropertyAccess {
        /// The column containing the node or edge ID.
        column: usize,
        /// The property name to access.
        property: String,
    },
    /// Edge type accessor (for type(r) function).
    EdgeType {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Full expression evaluation (for CASE WHEN, etc.).
    Expression {
        /// The filter expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
    },
    /// Resolve a node ID column to a full node map with metadata and properties.
    NodeResolve {
        /// The column containing the node ID.
        column: usize,
    },
    /// Resolve an edge ID column to a full edge map with metadata and properties.
    EdgeResolve {
        /// The column containing the edge ID.
        column: usize,
    },
    /// Evaluates an expression whose value is a node, an edge, a list of them
    /// (as ids, like `relationships(p)` and the variable of a variable-length
    /// edge pattern) or a value that holds them (a map, a list literal, a
    /// path), and returns node and edge maps in their place, as
    /// [`NodeResolve`](Self::NodeResolve) and [`EdgeResolve`](Self::EdgeResolve)
    /// do for a column.
    Entities {
        /// The expression to evaluate.
        expr: FilterExpression,
        /// Variable name to column index mapping.
        variable_columns: HashMap<String, usize>,
        /// What the value holds.
        kind: EntityValue,
    },
    /// Returns the first non-null value from two columns (used for RIGHT/FULL join dedup).
    Coalesce {
        /// Primary column index.
        first: usize,
        /// Fallback column index.
        second: usize,
    },
}

/// What a [`ProjectExpr::Entities`] value holds: where in it nodes and edges
/// sit, as ids.
///
/// A list, a map or a path holds a node or an edge as its id, so what the
/// value holds is known from the query, not from the value: the planner
/// tracks it per column, and a column's type says it to the operators that
/// read the column (`LogicalType::Node`, `LogicalType::List(Node)`, a
/// `LogicalType::Struct` for a map, see [`EntityValue::of_type`]).
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntityValue {
    /// One node id.
    Node,
    /// One edge id.
    Edge,
    /// A list of node ids.
    Nodes,
    /// A list of edge ids.
    Edges,
    /// A value that holds nodes or edges deeper inside, where its type says:
    /// a path (`LogicalType::Path`: node ids, then edge ids), a map
    /// (`LogicalType::Struct`, by key), a list with one type per item
    /// (`LogicalType::Tuple`, a list literal such as `[n, 1]`) or a list of
    /// such values (`LogicalType::List`). Made by [`EntityValue::of_type`].
    Nested(LogicalType),
}

impl EntityValue {
    /// What a value of type `value_type` holds, or `None` when it holds no
    /// node or edge.
    #[must_use]
    pub fn of_type(value_type: &LogicalType) -> Option<Self> {
        match value_type {
            LogicalType::Node => Some(Self::Node),
            LogicalType::Edge => Some(Self::Edge),
            LogicalType::List(item) if **item == LogicalType::Node => Some(Self::Nodes),
            LogicalType::List(item) if **item == LogicalType::Edge => Some(Self::Edges),
            other if holds_entities(other) => Some(Self::Nested(other.clone())),
            _ => None,
        }
    }

    /// The type of a column that holds this.
    #[must_use]
    pub fn logical_type(&self) -> LogicalType {
        match self {
            Self::Node => LogicalType::Node,
            Self::Edge => LogicalType::Edge,
            Self::Nodes => LogicalType::List(Box::new(LogicalType::Node)),
            Self::Edges => LogicalType::List(Box::new(LogicalType::Edge)),
            Self::Nested(value_type) => value_type.clone(),
        }
    }

    /// What one item of a list that holds this holds (`head`, `last`, `[i]`
    /// with an unknown `i`, UNWIND): the items' common kind, `None` when the
    /// items of a list literal differ or hold no node or edge.
    #[must_use]
    pub fn item(&self) -> Option<Self> {
        Self::of_type(&item_type(&self.logical_type()))
    }

    /// What item `index` of a list that holds this holds (a negative index
    /// counts from the end).
    #[must_use]
    pub fn item_at(&self, index: i64) -> Option<Self> {
        Self::of_type(&item_type_at(&self.logical_type(), index))
    }

    /// What the value under `key` of a map that holds this holds.
    #[must_use]
    pub fn field(&self, key: &str) -> Option<Self> {
        Self::of_type(&field_type(&self.logical_type(), key))
    }

    /// What the items `start..end` of a list that holds this hold (a slice,
    /// `tail`; `end` `None` is the end): a list holds the same, a list
    /// literal what its items in that range hold. `None` for a value that is
    /// not a list.
    #[must_use]
    pub fn slice(&self, start: usize, end: Option<usize>) -> Option<Self> {
        Self::of_type(&slice_type(self.logical_type(), start, end))
    }

    /// What a list that holds this holds with its items in reverse order
    /// (`reverse`). `None` for a value that is not a list.
    #[must_use]
    pub fn reversed(&self) -> Option<Self> {
        Self::of_type(&reversed_type(self.logical_type()))
    }

    /// What a list of values that hold this holds (`collect`).
    #[must_use]
    pub fn list(&self) -> Self {
        match self {
            Self::Node => Self::Nodes,
            Self::Edge => Self::Edges,
            other => Self::Nested(LogicalType::List(Box::new(other.logical_type()))),
        }
    }

    /// What a list literal holds whose items hold `items` (`None` for a
    /// plain value), or `None` when no item holds a node or edge.
    #[must_use]
    pub fn list_of(items: &[Option<Self>]) -> Option<Self> {
        Self::of_type(&list_type(items.iter().map(Self::type_of).collect()))
    }

    /// What a map literal holds whose keys hold `fields` (`None` for a plain
    /// value), or `None` when no key holds a node or edge.
    #[must_use]
    pub fn map_of(fields: &[(String, Option<Self>)]) -> Option<Self> {
        Self::of_type(&struct_type(
            fields
                .iter()
                .map(|(key, kind)| (key.clone(), Self::type_of(kind)))
                .collect(),
        ))
    }

    /// The type of a value that holds `kind`: `Any` for a plain value.
    fn type_of(kind: &Option<Self>) -> LogicalType {
        kind.as_ref().map_or(LogicalType::Any, Self::logical_type)
    }
}

/// Whether a value of type `value_type` holds a node or an edge: a node, an
/// edge or a path, or a list, map or struct with one inside.
pub(crate) fn holds_entities(value_type: &LogicalType) -> bool {
    match value_type {
        LogicalType::Node | LogicalType::Edge | LogicalType::Path => true,
        LogicalType::List(item) => holds_entities(item),
        LogicalType::Map { value, .. } => holds_entities(value),
        LogicalType::Tuple(items) => items.iter().any(holds_entities),
        LogicalType::Struct(fields) => fields.iter().any(|(_, field)| holds_entities(field)),
        _ => false,
    }
}

/// The type of one item of a list of type `list`, whatever its position:
/// the item type of a list, the type all items of a tuple share, `Any`
/// otherwise.
pub(crate) fn item_type(list: &LogicalType) -> LogicalType {
    match list {
        LogicalType::List(item) => (**item).clone(),
        LogicalType::Tuple(items) => match items.split_first() {
            Some((first, rest)) if rest.iter().all(|item| item == first) => first.clone(),
            _ => LogicalType::Any,
        },
        _ => LogicalType::Any,
    }
}

/// The type of item `index` of a list of type `list` (a negative index
/// counts from the end).
pub(crate) fn item_type_at(list: &LogicalType, index: i64) -> LogicalType {
    match list {
        LogicalType::Tuple(items) => {
            let position = if index < 0 {
                i64::try_from(items.len())
                    .ok()
                    .and_then(|len| usize::try_from(len + index).ok())
            } else {
                usize::try_from(index).ok()
            };
            position
                .and_then(|position| items.get(position))
                .cloned()
                .unwrap_or(LogicalType::Any)
        }
        other => item_type(other),
    }
}

/// The type of the items `start..end` of a list of type `list` (a slice,
/// `tail`; `end` `None` is the end): a list keeps its type, a tuple keeps
/// the types of the items in that range. `Any` for a type that is not a
/// list.
pub(crate) fn slice_type(list: LogicalType, start: usize, end: Option<usize>) -> LogicalType {
    match list {
        LogicalType::Tuple(items) => {
            let end = end.unwrap_or(items.len()).min(items.len());
            let start = start.min(end);
            list_type(items[start..end].to_vec())
        }
        list @ LogicalType::List(_) => list,
        _ => LogicalType::Any,
    }
}

/// The type of a list of type `list` with its items in reverse order
/// (`reverse`). `Any` for a type that is not a list.
pub(crate) fn reversed_type(list: LogicalType) -> LogicalType {
    match list {
        LogicalType::Tuple(mut items) => {
            items.reverse();
            LogicalType::Tuple(items)
        }
        list @ LogicalType::List(_) => list,
        _ => LogicalType::Any,
    }
}

/// The type of the value under `key` of a map of type `map`.
pub(crate) fn field_type(map: &LogicalType, key: &str) -> LogicalType {
    match map {
        LogicalType::Struct(fields) => fields
            .iter()
            .find(|(name, _)| name == key)
            .map_or(LogicalType::Any, |(_, field)| field.clone()),
        LogicalType::Map { value, .. } => (**value).clone(),
        _ => LogicalType::Any,
    }
}

/// The type of a list literal whose items have the types `items`: a list of
/// their type when they agree, a tuple when they differ, `Any` when none
/// holds a node or edge.
pub(crate) fn list_type(items: Vec<LogicalType>) -> LogicalType {
    if !items.iter().any(holds_entities) {
        return LogicalType::Any;
    }
    match items.split_first() {
        Some((first, rest)) if rest.iter().all(|item| item == first) => {
            LogicalType::List(Box::new(first.clone()))
        }
        _ => LogicalType::Tuple(
            items
                .into_iter()
                .map(|item| {
                    if holds_entities(&item) {
                        item
                    } else {
                        LogicalType::Any
                    }
                })
                .collect(),
        ),
    }
}

/// The type of a map literal whose keys have the types `fields`: a struct
/// of them, `Any` when none holds a node or edge.
pub(crate) fn struct_type(fields: Vec<(String, LogicalType)>) -> LogicalType {
    if !fields.iter().any(|(_, field)| holds_entities(field)) {
        return LogicalType::Any;
    }
    LogicalType::Struct(
        fields
            .into_iter()
            .map(|(key, field)| {
                if holds_entities(&field) {
                    (key, field)
                } else {
                    (key, LogicalType::Any)
                }
            })
            .collect(),
    )
}

/// A project operator that selects and transforms columns.
pub struct ProjectOperator {
    /// Child operator to read from.
    child: Box<dyn Operator>,
    /// Projection expressions.
    projections: Vec<ProjectExpr>,
    /// Output column types.
    output_types: Vec<LogicalType>,
    /// Optional store for property access, with the view its reads answer
    /// at.
    store: Option<(Arc<dyn GraphStore>, ReadView)>,
    /// Session context for introspection functions in expression evaluation.
    session_context: SessionContext,
}

impl ProjectOperator {
    /// Creates a new project operator.
    ///
    /// # Panics
    ///
    /// Panics if `projections` and `output_types` have different lengths.
    pub fn new(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: None,
            session_context: SessionContext::default(),
        }
    }

    /// Creates a project operator whose property lookups read `store` at
    /// `view`.
    ///
    /// # Panics
    ///
    /// Panics if `projections` and `output_types` have different lengths.
    pub fn with_store(
        child: Box<dyn Operator>,
        projections: Vec<ProjectExpr>,
        output_types: Vec<LogicalType>,
        store: Arc<dyn GraphStore>,
        view: ReadView,
    ) -> Self {
        assert_eq!(projections.len(), output_types.len());
        Self {
            child,
            projections,
            output_types,
            store: Some((store, view)),
            session_context: SessionContext::default(),
        }
    }

    /// The store and the view its lookups read at, required by `what`.
    fn reader(&self, what: &str) -> Result<(&Arc<dyn GraphStore>, ReadView), OperatorError> {
        self.store
            .as_ref()
            .map(|(store, view)| (store, *view))
            .ok_or_else(|| OperatorError::Internal(format!("Store required for {what}")))
    }

    /// Sets the session context for introspection functions.
    pub fn with_session_context(mut self, context: SessionContext) -> Self {
        self.session_context = context;
        self
    }

    /// Decomposes this operator into its child and projections for push-based conversion.
    pub fn into_parts(self) -> (Box<dyn Operator>, Vec<ProjectExpr>, Vec<LogicalType>) {
        (self.child, self.projections, self.output_types)
    }

    /// Creates a project operator that selects specific columns.
    pub fn select_columns(
        child: Box<dyn Operator>,
        columns: Vec<usize>,
        types: Vec<LogicalType>,
    ) -> Self {
        let projections = columns.into_iter().map(ProjectExpr::Column).collect();
        Self::new(child, projections, types)
    }
}

impl Operator for ProjectOperator {
    fn next(&mut self) -> OperatorResult {
        // Get next chunk from child
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // Create output chunk
        let mut output = DataChunk::with_capacity(&self.output_types, input.row_count());

        // Evaluate each projection
        for (i, proj) in self.projections.iter().enumerate() {
            match proj {
                ProjectExpr::Column(col_idx) => {
                    // Copy column from input to output
                    let input_col = input.column(*col_idx).ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("Column {col_idx}"))
                    })?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    // A copy the planner declared `Any` keeps the input's
                    // type: node and edge IDs stay nodes and edges (in an
                    // `Any` column an ID reads the properties of whichever
                    // entity has it), and every value is copied as it is.
                    if self.output_types[i] == LogicalType::Any {
                        *output_col = ValueVector::with_capacity(
                            input_col.data_type().clone(),
                            input.row_count(),
                        );
                    }

                    // Copy selected rows
                    for row in input.selected_indices() {
                        if let Some(value) = input_col.get_value(row) {
                            output_col.push_value(value);
                        }
                    }
                }
                ProjectExpr::Constant(value) => {
                    // Push constant for each row
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");
                    for _ in input.selected_indices() {
                        output_col.push_value(value.clone());
                    }
                }
                ProjectExpr::PropertyAccess { column, property } => {
                    // Access property from node/edge in the specified column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let (store, view) = self.reader("property access")?;

                    // Extract property for each row.
                    // For typed columns (VectorData::NodeId / EdgeId) there is
                    // no ambiguity. For Generic/Any columns (e.g. after a hash
                    // join), both get_node_id and get_edge_id can succeed on the
                    // same Int64 value, so we verify against the store to resolve
                    // the entity type.
                    let prop_key = PropertyKey::new(property);
                    for row in input.selected_indices() {
                        // One value of the entity, read without building it.
                        let value = if let Some(node_id) = input_col.get_node_id(row) {
                            if let Some(prop) = store.node_property(node_id, property, view)? {
                                prop
                            } else if let Some(edge_id) = input_col.get_edge_id(row) {
                                // Node lookup failed: the ID may belong to an
                                // edge (common with Generic columns after joins).
                                store
                                    .edge_property(edge_id, property, view)?
                                    .unwrap_or(Value::Null)
                            } else {
                                Value::Null
                            }
                        } else if let Some(edge_id) = input_col.get_edge_id(row) {
                            store
                                .edge_property(edge_id, property, view)?
                                .unwrap_or(Value::Null)
                        } else {
                            match input_col.get_value(row) {
                                Some(Value::Map(map)) => {
                                    map.get(&prop_key).cloned().unwrap_or(Value::Null)
                                }
                                // A component of a temporal value (`d.month`)
                                Some(value) => {
                                    value.temporal_component(property).unwrap_or(Value::Null)
                                }
                                None => Value::Null,
                            }
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::EdgeType { column } => {
                    // Get edge type string from an edge column
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let (store, view) = self.reader("edge type access")?;

                    for row in input.selected_indices() {
                        let value = if let Some(edge_id) = input_col.get_edge_id(row) {
                            store
                                .edge_type(edge_id, view)?
                                .map_or(Value::Null, Value::String)
                        } else {
                            Value::Null
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Expression {
                    expr,
                    variable_columns,
                } => {
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let (store, view) = self.reader("expression evaluation")?;

                    // Use the ExpressionPredicate for expression evaluation
                    let evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                        view,
                    )
                    .with_session_context(self.session_context.clone());

                    // A node or edge column holds ids: it takes a node or edge
                    // map (the items of `nodes(p)`) by its id, and anything
                    // that names no entity as null, never as entity 0.
                    let entities = matches!(
                        output_col.data_type(),
                        LogicalType::Node | LogicalType::Edge
                    );
                    for row in input.selected_indices() {
                        let value = evaluator.eval_at(&input, row)?.unwrap_or(Value::Null);
                        let value = if entities { entity_id(value) } else { value };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::NodeResolve { column } | ProjectExpr::EdgeResolve { column } => {
                    let input_col = input
                        .column(*column)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {column}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let (store, view) = self.reader("entity resolution")?;

                    // The planner says by name whether the column holds nodes
                    // or edges; a column typed by its rows says it per chunk,
                    // which wins: the branches of a set operation may bind one
                    // name to nodes in one branch and to edges in another.
                    let edges = match input_col.data_type() {
                        LogicalType::Edge => true,
                        LogicalType::Node => false,
                        _ => matches!(proj, ProjectExpr::EdgeResolve { .. }),
                    };
                    for row in input.selected_indices() {
                        let value = if edges {
                            match input_col.get_edge_id(row) {
                                Some(edge_id) => store
                                    .edge(edge_id, view)?
                                    .map_or(Value::Null, |e| edge_to_map(&e)),
                                None => Value::Null,
                            }
                        } else {
                            match input_col.get_node_id(row) {
                                Some(node_id) => store
                                    .node(node_id, view)?
                                    .map_or(Value::Null, |n| node_to_map(&n)),
                                None => Value::Null,
                            }
                        };
                        output_col.push_value(value);
                    }
                }
                ProjectExpr::Entities {
                    expr,
                    variable_columns,
                    kind,
                } => {
                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    let (store, view) = self.reader("expression evaluation")?;

                    let evaluator = ExpressionPredicate::new(
                        expr.clone(),
                        variable_columns.clone(),
                        Arc::clone(store),
                        view,
                    )
                    .with_session_context(self.session_context.clone());

                    let shape = kind.logical_type();
                    for row in input.selected_indices() {
                        let value = evaluator.eval_at(&input, row)?.unwrap_or(Value::Null);
                        output_col.push_value(resolve_entities(
                            store.as_ref(),
                            view,
                            &value,
                            &shape,
                        )?);
                    }
                }
                ProjectExpr::Coalesce { first, second } => {
                    let first_col = input
                        .column(*first)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {first}")))?;
                    let second_col = input
                        .column(*second)
                        .ok_or_else(|| OperatorError::ColumnNotFound(format!("Column {second}")))?;

                    let output_col = output
                        .column_mut(i)
                        .expect("column exists: index matches projection schema");

                    for row in input.selected_indices() {
                        let value = match first_col.get_value(row) {
                            Some(Value::Null) | None => {
                                second_col.get_value(row).unwrap_or(Value::Null)
                            }
                            Some(v) => v,
                        };
                        output_col.push_value(value);
                    }
                }
            }
        }

        output.set_count(input.row_count());
        Ok(Some(output))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "Project"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

/// Replaces the entity ids in `value` with node or edge maps, where its
/// type `shape` (an [`EntityValue`]'s type) says they are: the value
/// itself, the items of a list, the keys of a map, the nodes and edges of a
/// path. Ids of entities `view` cannot see become null; values that are
/// already maps (or null) are kept.
fn resolve_entities(
    store: &dyn GraphStore,
    view: ReadView,
    value: &Value,
    shape: &LogicalType,
) -> Result<Value, OperatorError> {
    let each = |values: &[Value], item: &dyn Fn(usize) -> Option<LogicalType>| {
        values
            .iter()
            .enumerate()
            .map(|(i, value)| match item(i) {
                Some(item) => resolve_entities(store, view, value, &item),
                None => Ok(value.clone()),
            })
            .collect::<Result<Vec<_>, OperatorError>>()
    };
    Ok(match (shape, value) {
        (LogicalType::Node, Value::Int64(id)) => match u64::try_from(*id) {
            Ok(id) => store
                .node(NodeId::new(id), view)?
                .map_or(Value::Null, |n| node_to_map(&n)),
            Err(_) => Value::Null,
        },
        (LogicalType::Edge, Value::Int64(id)) => match u64::try_from(*id) {
            Ok(id) => store
                .edge(EdgeId::new(id), view)?
                .map_or(Value::Null, |e| edge_to_map(&e)),
            Err(_) => Value::Null,
        },
        (LogicalType::List(item), Value::List(items)) => {
            Value::List(each(items, &|_| Some((**item).clone()))?.into())
        }
        (LogicalType::Tuple(types), Value::List(items)) => {
            Value::List(each(items, &|i| types.get(i).cloned())?.into())
        }
        (LogicalType::Struct(_) | LogicalType::Map { .. }, Value::Map(map)) => {
            let mut resolved = BTreeMap::new();
            for (key, value) in map.iter() {
                let field = field_type(shape, key.as_str());
                resolved.insert(key.clone(), resolve_entities(store, view, value, &field)?);
            }
            Value::Map(Arc::new(resolved))
        }
        (LogicalType::Path, Value::Path { nodes, edges }) => Value::Path {
            nodes: each(nodes, &|_| Some(LogicalType::Node))?.into(),
            edges: each(edges, &|_| Some(LogicalType::Edge))?.into(),
        },
        _ => value.clone(),
    })
}

/// Converts a [`Node`] to a `Value::Map` with metadata and properties.
///
/// The map contains all node properties at the top level, `_id` (integer)
/// and `_labels` (list of strings): those two always hold the node's id and
/// labels, also when a property has the name (#254), which the map then
/// leaves out.
fn node_to_map(node: &Node) -> Value {
    let mut map: BTreeMap<PropertyKey, Value> = node
        .properties
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let node_id_i64 = node.id.as_u64() as i64;
    map.insert(PropertyKey::new("_id"), Value::Int64(node_id_i64));
    let labels: Vec<Value> = node
        .labels
        .iter()
        .map(|l| Value::String(l.clone()))
        .collect();
    map.insert(PropertyKey::new("_labels"), Value::List(labels.into()));
    Value::Map(Arc::new(map))
}

/// Converts an [`Edge`] to a `Value::Map` with metadata and properties.
///
/// The map contains all edge properties at the top level, `_id`, `_type`,
/// `_source` and `_target`: those always hold the edge's identity, also when
/// a property has the name (#254), which the map then leaves out.
fn edge_to_map(edge: &Edge) -> Value {
    let mut map: BTreeMap<PropertyKey, Value> = edge
        .properties
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let edge_id_i64 = edge.id.as_u64() as i64;
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let src_id_i64 = edge.src.as_u64() as i64;
    // reason: entity IDs stored as i64, standard encoding
    #[allow(clippy::cast_possible_wrap)]
    let dst_id_i64 = edge.dst.as_u64() as i64;
    map.insert(PropertyKey::new("_id"), Value::Int64(edge_id_i64));
    map.insert(
        PropertyKey::new("_type"),
        Value::String(edge.edge_type.clone()),
    );
    map.insert(PropertyKey::new("_source"), Value::Int64(src_id_i64));
    map.insert(PropertyKey::new("_target"), Value::Int64(dst_id_i64));
    Value::Map(Arc::new(map))
}

/// The id a value gives a node or edge column: an id, the `_id` of an entity
/// map, or null.
fn entity_id(value: Value) -> Value {
    match value {
        Value::Int64(_) | Value::Null => value,
        Value::Map(map) => match map.get(&PropertyKey::new("_id")) {
            Some(id @ Value::Int64(_)) => id.clone(),
            _ => Value::Null,
        },
        _ => Value::Null,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::graph::rowgroup::RowGroupStore;
    use grafeo_common::types::Value;

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
    fn test_project_select_columns() {
        // Create input with 3 columns: [int, string, int]
        let mut builder =
            DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String, LogicalType::Int64]);

        builder.column_mut(0).unwrap().push_int64(1);
        builder.column_mut(1).unwrap().push_string("hello");
        builder.column_mut(2).unwrap().push_int64(100);
        builder.advance_row();

        builder.column_mut(0).unwrap().push_int64(2);
        builder.column_mut(1).unwrap().push_string("world");
        builder.column_mut(2).unwrap().push_int64(200);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project to select columns 2 and 0 (reordering)
        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![2, 0],
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.row_count(), 2);

        // Check values are reordered
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(100));
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(1));
    }

    /// A copy declared `Any` keeps the input column's type, so node IDs stay
    /// node IDs (in an `Any` column an ID reads whichever entity has it); a
    /// declared type still applies.
    #[test]
    fn a_copy_declared_any_keeps_the_input_type() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Int64]);
        for id in 1..=3_u64 {
            builder
                .column_mut(0)
                .unwrap()
                .push_node_id(grafeo_common::types::NodeId::new(id));
            builder
                .column_mut(1)
                .unwrap()
                .push_int64(i64::try_from(id).unwrap() * 10);
            builder.advance_row();
        }
        let mock_scan = MockScanOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Column(1),
                ProjectExpr::Column(1),
            ],
            vec![LogicalType::Any, LogicalType::Any, LogicalType::Int32],
        );

        let result = project.next().unwrap().unwrap();
        let ids = result.column(0).unwrap();
        assert_eq!(ids.data_type(), &LogicalType::Node);
        assert_eq!(
            ids.get_node_id(2),
            Some(grafeo_common::types::NodeId::new(3))
        );
        assert_eq!(result.column(1).unwrap().data_type(), &LogicalType::Int64);
        assert_eq!(result.column(1).unwrap().get_int64(0), Some(10));
        // A copy declared with a type of its own keeps the declared type.
        assert_eq!(result.column(2).unwrap().data_type(), &LogicalType::Int32);
        assert_eq!(result.column(2).unwrap().get_int64(2), Some(30));
    }

    #[test]
    fn test_project_constant() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        builder.column_mut(0).unwrap().push_int64(2);
        builder.advance_row();

        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Project with a constant
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::String("constant".into())),
            ],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();

        assert_eq!(result.column_count(), 2);
        assert_eq!(result.column(1).unwrap().get_string(0), Some("constant"));
        assert_eq!(result.column(1).unwrap().get_string(1), Some("constant"));
    }

    #[test]
    fn test_project_empty_input() {
        let mock_scan = MockScanOperator {
            chunks: vec![],
            position: 0,
        };

        let mut project =
            ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);

        assert!(project.next().unwrap().is_none());
    }

    #[test]
    fn test_project_column_not_found() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        // Reference column index 5 which doesn't exist
        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![ProjectExpr::Column(5)],
            vec![LogicalType::Int64],
        );

        let result = project.next();
        assert!(result.is_err(), "Should fail with ColumnNotFound");
    }

    #[test]
    fn test_project_multiple_constants() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::new(
            Box::new(mock_scan),
            vec![
                ProjectExpr::Constant(Value::Int64(42)),
                ProjectExpr::Constant(Value::String("fixed".into())),
                ProjectExpr::Constant(Value::Bool(true)),
            ],
            vec![LogicalType::Int64, LogicalType::String, LogicalType::Bool],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column_count(), 3);
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(42));
        assert_eq!(result.column(1).unwrap().get_string(0), Some("fixed"));
        assert_eq!(
            result.column(2).unwrap().get_value(0),
            Some(Value::Bool(true))
        );
    }

    #[test]
    fn test_project_identity() {
        // Select all columns in original order
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::String]);
        builder.column_mut(0).unwrap().push_int64(10);
        builder.column_mut(1).unwrap().push_string("test");
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::select_columns(
            Box::new(mock_scan),
            vec![0, 1],
            vec![LogicalType::Int64, LogicalType::String],
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_int64(0), Some(10));
        assert_eq!(result.column(1).unwrap().get_string(0), Some("test"));
    }

    #[test]
    fn test_project_name() {
        let mock_scan = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let project =
            ProjectOperator::select_columns(Box::new(mock_scan), vec![0], vec![LogicalType::Int64]);
        assert_eq!(project.name(), "Project");
    }

    /// A node or edge column takes an entity map by its `_id`; a value that
    /// names no entity is null there, never entity 0.
    #[test]
    fn test_project_entity_values_into_entity_columns() {
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(1);
        builder.advance_row();
        let mock_scan = MockScanOperator {
            chunks: vec![builder.finish()],
            position: 0,
        };
        let map = |entries: &[(&str, Value)]| {
            Value::Map(Arc::new(
                entries
                    .iter()
                    .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
                    .collect(),
            ))
        };
        let expression = |value: Value| ProjectExpr::Expression {
            expr: FilterExpression::Literal(value),
            variable_columns: HashMap::new(),
        };
        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![
                expression(map(&[("_id", Value::Int64(7))])),
                expression(map(&[("name", Value::from("Alix"))])),
                expression(Value::from("Alix")),
            ],
            vec![LogicalType::Node, LogicalType::Node, LogicalType::Edge],
            Arc::new(RowGroupStore::new()),
            ReadView::latest(),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(
            result.column(0).unwrap().get_node_id(0),
            Some(NodeId::new(7))
        );
        assert_eq!(result.column(1).unwrap().get_node_id(0), None);
        assert_eq!(result.column(2).unwrap().get_edge_id(0), None);
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_node_resolve() {
        // Create a store with a test node
        let store = RowGroupStore::new();
        let node_id = store.create_node(&["Person"]);
        store.set_node_property(node_id, "name", Value::String("Alix".into()));
        store.set_node_property(node_id, "age", Value::Int64(30));

        // Create input chunk with a NodeId column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder.column_mut(0).unwrap().push_node_id(node_id);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::NodeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
            ReadView::latest(),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column_count(), 1);

        let value = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::Map(map) = value {
            assert_eq!(
                map.get(&PropertyKey::new("_id")),
                Some(&Value::Int64(node_id.as_u64() as i64))
            );
            assert!(map.get(&PropertyKey::new("_labels")).is_some());
            assert_eq!(
                map.get(&PropertyKey::new("name")),
                Some(&Value::String("Alix".into()))
            );
            assert_eq!(map.get(&PropertyKey::new("age")), Some(&Value::Int64(30)));
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn test_project_edge_resolve() {
        let store = RowGroupStore::new();
        let src = store.create_node(&["Person"]);
        let dst = store.create_node(&["Company"]);
        let edge_id = store.create_edge(src, dst, "WORKS_AT");
        store.set_edge_property(edge_id, "since", Value::Int64(2020));

        // Create input chunk with an EdgeId column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Edge]);
        builder.column_mut(0).unwrap().push_edge_id(edge_id);
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::EdgeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
            ReadView::latest(),
        );

        let result = project.next().unwrap().unwrap();
        let value = result.column(0).unwrap().get_value(0).unwrap();
        if let Value::Map(map) = value {
            assert_eq!(
                map.get(&PropertyKey::new("_id")),
                Some(&Value::Int64(edge_id.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_type")),
                Some(&Value::String("WORKS_AT".into()))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_source")),
                Some(&Value::Int64(src.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("_target")),
                Some(&Value::Int64(dst.as_u64() as i64))
            );
            assert_eq!(
                map.get(&PropertyKey::new("since")),
                Some(&Value::Int64(2020))
            );
        } else {
            panic!("Expected Value::Map, got {:?}", value);
        }
    }

    #[test]
    fn test_project_resolve_missing_entity() {
        use grafeo_common::types::NodeId;

        let store = RowGroupStore::new();

        // Create input chunk with a NodeId that doesn't exist in the store
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node]);
        builder
            .column_mut(0)
            .unwrap()
            .push_node_id(NodeId::new(999));
        builder.advance_row();
        let chunk = builder.finish();

        let mock_scan = MockScanOperator {
            chunks: vec![chunk],
            position: 0,
        };

        let mut project = ProjectOperator::with_store(
            Box::new(mock_scan),
            vec![ProjectExpr::NodeResolve { column: 0 }],
            vec![LogicalType::Any],
            Arc::new(store),
            ReadView::latest(),
        );

        let result = project.next().unwrap().unwrap();
        assert_eq!(result.column(0).unwrap().get_value(0), Some(Value::Null));
    }

    #[test]
    fn test_project_into_any() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let op = ProjectOperator::select_columns(Box::new(mock), vec![0], vec![LogicalType::Int64]);
        let any = Box::new(op).into_any();
        assert!(any.downcast::<ProjectOperator>().is_ok());
    }

    #[test]
    fn test_project_into_parts() {
        let mock = MockScanOperator {
            chunks: vec![],
            position: 0,
        };
        let op = ProjectOperator::new(
            Box::new(mock),
            vec![
                ProjectExpr::Column(0),
                ProjectExpr::Constant(Value::Int64(1)),
            ],
            vec![LogicalType::Int64, LogicalType::Int64],
        );
        let (child, projections, output_types) = op.into_parts();
        assert_eq!(projections.len(), 2);
        assert_eq!(output_types.len(), 2);
        // Verify child is still functional
        let mut child = child;
        assert!(child.next().unwrap().is_none());
    }

    /// A property named like a structural key never replaces the node's or
    /// edge's identity in its map (#254); other properties, those that start
    /// with `_` too, keep their values.
    #[test]
    fn entity_maps_keep_their_identity() {
        let map = |value: Value| match value {
            Value::Map(map) => map,
            other => panic!("{other:?}"),
        };
        let mut node = crate::graph::lpg::Node::new(NodeId::new(3));
        node.add_label("Person");
        node.set_property("_id", Value::from("x"));
        node.set_property("_labels", Value::from("y"));
        node.set_property("_created", Value::Int64(2020));
        node.set_property("id", Value::Int64(7));
        let node = map(node_to_map(&node));
        assert_eq!(node.get(&PropertyKey::new("_id")), Some(&Value::Int64(3)));
        assert_eq!(
            node.get(&PropertyKey::new("_labels")),
            Some(&Value::List(vec![Value::from("Person")].into()))
        );
        assert_eq!(
            node.get(&PropertyKey::new("_created")),
            Some(&Value::Int64(2020))
        );
        assert_eq!(node.get(&PropertyKey::new("id")), Some(&Value::Int64(7)));

        let mut edge =
            crate::graph::lpg::Edge::new(EdgeId::new(5), NodeId::new(3), NodeId::new(4), "KNOWS");
        for key in ["_id", "_type", "_source", "_target"] {
            edge.set_property(key, Value::from("user"));
        }
        edge.set_property("source", Value::from("jdt"));
        edge.set_property("_note", Value::from("met in Paris"));
        let edge = map(edge_to_map(&edge));
        assert_eq!(edge.get(&PropertyKey::new("_id")), Some(&Value::Int64(5)));
        assert_eq!(
            edge.get(&PropertyKey::new("_type")),
            Some(&Value::from("KNOWS"))
        );
        assert_eq!(
            edge.get(&PropertyKey::new("_source")),
            Some(&Value::Int64(3))
        );
        assert_eq!(
            edge.get(&PropertyKey::new("_target")),
            Some(&Value::Int64(4))
        );
        assert_eq!(
            edge.get(&PropertyKey::new("source")),
            Some(&Value::from("jdt"))
        );
        assert_eq!(
            edge.get(&PropertyKey::new("_note")),
            Some(&Value::from("met in Paris"))
        );
    }
}
