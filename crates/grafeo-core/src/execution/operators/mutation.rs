//! Mutation operators for creating and deleting graph elements.
//!
//! These operators modify the graph structure:
//! - `CreateNodeOperator`: Creates new nodes
//! - `CreateEdgeOperator`: Creates new edges
//! - `DeleteNodeOperator`: Deletes nodes
//! - `DeleteEdgeOperator`: Deletes edges

use std::sync::Arc;

use grafeo_common::types::{EdgeId, EpochId, LogicalType, NodeId, TxId, Value};

use super::{Operator, OperatorError, OperatorResult};
use crate::execution::chunk::DataChunkBuilder;
use crate::graph::lpg::LpgStore;

/// Operator that creates new nodes.
///
/// For each input row, creates a new node with the specified labels
/// and properties, then outputs the row with the new node.
pub struct CreateNodeOperator {
    /// The graph store to modify.
    store: Arc<LpgStore>,
    /// Input operator.
    input: Option<Box<dyn Operator>>,
    /// Labels for the new nodes.
    labels: Vec<String>,
    /// Properties to set (name -> column index or constant value).
    properties: Vec<(String, PropertySource)>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Column index for the created node variable.
    output_column: usize,
    /// Whether this operator has been executed (for no-input case).
    executed: bool,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning.
    tx_id: Option<TxId>,
}

/// Source for a property value.
#[derive(Debug, Clone)]
pub enum PropertySource {
    /// Get value from an input column.
    Column(usize),
    /// Use a constant value.
    Constant(Value),
}

impl CreateNodeOperator {
    /// Creates a new node creation operator.
    ///
    /// # Arguments
    /// * `store` - The graph store to modify.
    /// * `input` - Optional input operator (None for standalone CREATE).
    /// * `labels` - Labels to assign to created nodes.
    /// * `properties` - Properties to set on created nodes.
    /// * `output_schema` - Schema of the output.
    /// * `output_column` - Column index where the created node ID goes.
    pub fn new(
        store: Arc<LpgStore>,
        input: Option<Box<dyn Operator>>,
        labels: Vec<String>,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
        output_column: usize,
    ) -> Self {
        Self {
            store,
            input,
            labels,
            properties,
            output_schema,
            output_column,
            executed: false,
            viewing_epoch: None,
            tx_id: None,
        }
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_tx_context(mut self, epoch: EpochId, tx_id: Option<TxId>) -> Self {
        self.viewing_epoch = Some(epoch);
        self.tx_id = tx_id;
        self
    }
}

impl Operator for CreateNodeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned creation
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.tx_id.unwrap_or(TxId::SYSTEM);

        if let Some(ref mut input) = self.input {
            // For each input row, create a node
            if let Some(chunk) = input.next()? {
                let mut builder =
                    DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

                for row in chunk.selected_indices() {
                    // Create the node with MVCC versioning
                    let label_refs: Vec<&str> = self.labels.iter().map(String::as_str).collect();
                    let node_id = self.store.create_node_versioned(&label_refs, epoch, tx);

                    // Set properties
                    for (prop_name, source) in &self.properties {
                        let value = match source {
                            PropertySource::Column(col_idx) => chunk
                                .column(*col_idx)
                                .and_then(|c| c.get_value(row))
                                .unwrap_or(Value::Null),
                            PropertySource::Constant(v) => v.clone(),
                        };
                        self.store.set_node_property(node_id, prop_name, value);
                    }

                    // Copy input columns to output
                    for col_idx in 0..chunk.column_count() {
                        if col_idx < self.output_column
                            && let (Some(src), Some(dst)) =
                                (chunk.column(col_idx), builder.column_mut(col_idx))
                        {
                            if let Some(val) = src.get_value(row) {
                                dst.push_value(val);
                            } else {
                                dst.push_value(Value::Null);
                            }
                        }
                    }

                    // Add the new node ID
                    if let Some(dst) = builder.column_mut(self.output_column) {
                        dst.push_value(Value::Int64(node_id.0 as i64));
                    }

                    builder.advance_row();
                }

                return Ok(Some(builder.finish()));
            }
            Ok(None)
        } else {
            // No input - create a single node
            if self.executed {
                return Ok(None);
            }
            self.executed = true;

            // Create the node with MVCC versioning
            let label_refs: Vec<&str> = self.labels.iter().map(String::as_str).collect();
            let node_id = self.store.create_node_versioned(&label_refs, epoch, tx);

            // Set properties from constants only
            for (prop_name, source) in &self.properties {
                if let PropertySource::Constant(value) = source {
                    self.store
                        .set_node_property(node_id, prop_name, value.clone());
                }
            }

            // Build output chunk with just the node ID
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
            if let Some(dst) = builder.column_mut(self.output_column) {
                dst.push_value(Value::Int64(node_id.0 as i64));
            }
            builder.advance_row();

            Ok(Some(builder.finish()))
        }
    }

    fn reset(&mut self) {
        if let Some(ref mut input) = self.input {
            input.reset();
        }
        self.executed = false;
    }

    fn name(&self) -> &'static str {
        "CreateNode"
    }
}

/// Operator that creates new edges.
pub struct CreateEdgeOperator {
    /// The graph store to modify.
    store: Arc<LpgStore>,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the source node.
    from_column: usize,
    /// Column index for the target node.
    to_column: usize,
    /// Edge type.
    edge_type: String,
    /// Properties to set.
    properties: Vec<(String, PropertySource)>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Column index for the created edge variable (if any).
    output_column: Option<usize>,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning.
    tx_id: Option<TxId>,
}

impl CreateEdgeOperator {
    /// Creates a new edge creation operator.
    ///
    /// Use builder methods to set additional options:
    /// - [`with_properties`](Self::with_properties) - set edge properties
    /// - [`with_output_column`](Self::with_output_column) - output the created edge ID
    /// - [`with_tx_context`](Self::with_tx_context) - set transaction context
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        from_column: usize,
        to_column: usize,
        edge_type: String,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            from_column,
            to_column,
            edge_type,
            properties: Vec::new(),
            output_schema,
            output_column: None,
            viewing_epoch: None,
            tx_id: None,
        }
    }

    /// Sets the properties to assign to created edges.
    pub fn with_properties(mut self, properties: Vec<(String, PropertySource)>) -> Self {
        self.properties = properties;
        self
    }

    /// Sets the output column for the created edge ID.
    pub fn with_output_column(mut self, column: usize) -> Self {
        self.output_column = Some(column);
        self
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_tx_context(mut self, epoch: EpochId, tx_id: Option<TxId>) -> Self {
        self.viewing_epoch = Some(epoch);
        self.tx_id = tx_id;
        self
    }
}

impl Operator for CreateEdgeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned creation
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());
        let tx = self.tx_id.unwrap_or(TxId::SYSTEM);

        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                // Get source and target node IDs
                let from_id = chunk
                    .column(self.from_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("from column {}", self.from_column))
                    })?;

                let to_id = chunk
                    .column(self.to_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("to column {}", self.to_column))
                    })?;

                // Extract node IDs
                let from_node_id = match from_id {
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{from_id:?}"),
                        });
                    }
                };

                let to_node_id = match to_id {
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{to_id:?}"),
                        });
                    }
                };

                // Create the edge with MVCC versioning
                let edge_id = self.store.create_edge_versioned(
                    from_node_id,
                    to_node_id,
                    &self.edge_type,
                    epoch,
                    tx,
                );

                // Set properties
                for (prop_name, source) in &self.properties {
                    let value = match source {
                        PropertySource::Column(col_idx) => chunk
                            .column(*col_idx)
                            .and_then(|c| c.get_value(row))
                            .unwrap_or(Value::Null),
                        PropertySource::Constant(v) => v.clone(),
                    };
                    self.store.set_edge_property(edge_id, prop_name, value);
                }

                // Copy input columns
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }

                // Add edge ID if requested
                if let Some(out_col) = self.output_column
                    && let Some(dst) = builder.column_mut(out_col)
                {
                    dst.push_value(Value::Int64(edge_id.0 as i64));
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "CreateEdge"
    }
}

/// Operator that deletes nodes.
pub struct DeleteNodeOperator {
    /// The graph store to modify.
    store: Arc<LpgStore>,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the node to delete.
    node_column: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Whether to detach (delete connected edges) before deleting.
    detach: bool,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning (reserved for future use).
    #[allow(dead_code)]
    tx_id: Option<TxId>,
}

impl DeleteNodeOperator {
    /// Creates a new node deletion operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        node_column: usize,
        output_schema: Vec<LogicalType>,
        detach: bool,
    ) -> Self {
        Self {
            store,
            input,
            node_column,
            output_schema,
            detach,
            viewing_epoch: None,
            tx_id: None,
        }
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_tx_context(mut self, epoch: EpochId, tx_id: Option<TxId>) -> Self {
        self.viewing_epoch = Some(epoch);
        self.tx_id = tx_id;
        self
    }
}

impl Operator for DeleteNodeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned deletion
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());

        if let Some(chunk) = self.input.next()? {
            let mut deleted_count = 0;

            for row in chunk.selected_indices() {
                let node_val = chunk
                    .column(self.node_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("node column {}", self.node_column))
                    })?;

                let node_id = match node_val {
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{node_val:?}"),
                        });
                    }
                };

                if self.detach {
                    // Delete all connected edges first
                    // Note: Edge deletion will use epoch internally
                    self.store.delete_node_edges(node_id);
                }

                // Delete the node with MVCC versioning
                if self.store.delete_node_at_epoch(node_id, epoch) {
                    deleted_count += 1;
                }
            }

            // Return a chunk with the delete count
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
            if let Some(dst) = builder.column_mut(0) {
                dst.push_value(Value::Int64(deleted_count));
            }
            builder.advance_row();

            return Ok(Some(builder.finish()));
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "DeleteNode"
    }
}

/// Operator that deletes edges.
pub struct DeleteEdgeOperator {
    /// The graph store to modify.
    store: Arc<LpgStore>,
    /// Input operator.
    input: Box<dyn Operator>,
    /// Column index for the edge to delete.
    edge_column: usize,
    /// Output schema.
    output_schema: Vec<LogicalType>,
    /// Epoch for MVCC versioning.
    viewing_epoch: Option<EpochId>,
    /// Transaction ID for MVCC versioning (reserved for future use).
    #[allow(dead_code)]
    tx_id: Option<TxId>,
}

impl DeleteEdgeOperator {
    /// Creates a new edge deletion operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        edge_column: usize,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            edge_column,
            output_schema,
            viewing_epoch: None,
            tx_id: None,
        }
    }

    /// Sets the transaction context for MVCC versioning.
    pub fn with_tx_context(mut self, epoch: EpochId, tx_id: Option<TxId>) -> Self {
        self.viewing_epoch = Some(epoch);
        self.tx_id = tx_id;
        self
    }
}

impl Operator for DeleteEdgeOperator {
    fn next(&mut self) -> OperatorResult {
        // Get transaction context for versioned deletion
        let epoch = self
            .viewing_epoch
            .unwrap_or_else(|| self.store.current_epoch());

        if let Some(chunk) = self.input.next()? {
            let mut deleted_count = 0;

            for row in chunk.selected_indices() {
                let edge_val = chunk
                    .column(self.edge_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("edge column {}", self.edge_column))
                    })?;

                let edge_id = match edge_val {
                    Value::Int64(id) => EdgeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (edge ID)".to_string(),
                            found: format!("{edge_val:?}"),
                        });
                    }
                };

                // Delete the edge with MVCC versioning
                if self.store.delete_edge_at_epoch(edge_id, epoch) {
                    deleted_count += 1;
                }
            }

            // Return a chunk with the delete count
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
            if let Some(dst) = builder.column_mut(0) {
                dst.push_value(Value::Int64(deleted_count));
            }
            builder.advance_row();

            return Ok(Some(builder.finish()));
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "DeleteEdge"
    }
}

/// Operator that adds labels to nodes.
pub struct AddLabelOperator {
    /// The graph store.
    store: Arc<LpgStore>,
    /// Child operator providing nodes.
    input: Box<dyn Operator>,
    /// Column index containing node IDs.
    node_column: usize,
    /// Labels to add.
    labels: Vec<String>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
}

impl AddLabelOperator {
    /// Creates a new add label operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        node_column: usize,
        labels: Vec<String>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            node_column,
            labels,
            output_schema,
        }
    }
}

impl Operator for AddLabelOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(chunk) = self.input.next()? {
            let mut updated_count = 0;

            for row in chunk.selected_indices() {
                let node_val = chunk
                    .column(self.node_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("node column {}", self.node_column))
                    })?;

                let node_id = match node_val {
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{node_val:?}"),
                        });
                    }
                };

                // Add all labels
                for label in &self.labels {
                    if self.store.add_label(node_id, label) {
                        updated_count += 1;
                    }
                }
            }

            // Return a chunk with the update count
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
            if let Some(dst) = builder.column_mut(0) {
                dst.push_value(Value::Int64(updated_count));
            }
            builder.advance_row();

            return Ok(Some(builder.finish()));
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "AddLabel"
    }
}

/// Operator that removes labels from nodes.
pub struct RemoveLabelOperator {
    /// The graph store.
    store: Arc<LpgStore>,
    /// Child operator providing nodes.
    input: Box<dyn Operator>,
    /// Column index containing node IDs.
    node_column: usize,
    /// Labels to remove.
    labels: Vec<String>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
}

impl RemoveLabelOperator {
    /// Creates a new remove label operator.
    pub fn new(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        node_column: usize,
        labels: Vec<String>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            node_column,
            labels,
            output_schema,
        }
    }
}

impl Operator for RemoveLabelOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(chunk) = self.input.next()? {
            let mut updated_count = 0;

            for row in chunk.selected_indices() {
                let node_val = chunk
                    .column(self.node_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!("node column {}", self.node_column))
                    })?;

                let node_id = match node_val {
                    Value::Int64(id) => NodeId(id as u64),
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (node ID)".to_string(),
                            found: format!("{node_val:?}"),
                        });
                    }
                };

                // Remove all labels
                for label in &self.labels {
                    if self.store.remove_label(node_id, label) {
                        updated_count += 1;
                    }
                }
            }

            // Return a chunk with the update count
            let mut builder = DataChunkBuilder::with_capacity(&self.output_schema, 1);
            if let Some(dst) = builder.column_mut(0) {
                dst.push_value(Value::Int64(updated_count));
            }
            builder.advance_row();

            return Ok(Some(builder.finish()));
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "RemoveLabel"
    }
}

/// Operator that sets properties on nodes or edges.
///
/// This operator reads node/edge IDs from a column and sets the
/// specified properties on each entity.
pub struct SetPropertyOperator {
    /// The graph store.
    store: Arc<LpgStore>,
    /// Child operator providing entities.
    input: Box<dyn Operator>,
    /// Column index containing entity IDs (node or edge).
    entity_column: usize,
    /// Whether the entity is an edge (false = node).
    is_edge: bool,
    /// Properties to set (name -> source).
    properties: Vec<(String, PropertySource)>,
    /// Output schema.
    output_schema: Vec<LogicalType>,
}

impl SetPropertyOperator {
    /// Creates a new set property operator for nodes.
    pub fn new_for_node(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        node_column: usize,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            entity_column: node_column,
            is_edge: false,
            properties,
            output_schema,
        }
    }

    /// Creates a new set property operator for edges.
    pub fn new_for_edge(
        store: Arc<LpgStore>,
        input: Box<dyn Operator>,
        edge_column: usize,
        properties: Vec<(String, PropertySource)>,
        output_schema: Vec<LogicalType>,
    ) -> Self {
        Self {
            store,
            input,
            entity_column: edge_column,
            is_edge: true,
            properties,
            output_schema,
        }
    }
}

impl Operator for SetPropertyOperator {
    fn next(&mut self) -> OperatorResult {
        if let Some(chunk) = self.input.next()? {
            let mut builder =
                DataChunkBuilder::with_capacity(&self.output_schema, chunk.row_count());

            for row in chunk.selected_indices() {
                let entity_val = chunk
                    .column(self.entity_column)
                    .and_then(|c| c.get_value(row))
                    .ok_or_else(|| {
                        OperatorError::ColumnNotFound(format!(
                            "entity column {}",
                            self.entity_column
                        ))
                    })?;

                let entity_id = match entity_val {
                    Value::Int64(id) => id as u64,
                    _ => {
                        return Err(OperatorError::TypeMismatch {
                            expected: "Int64 (entity ID)".to_string(),
                            found: format!("{entity_val:?}"),
                        });
                    }
                };

                // Set all properties
                for (prop_name, source) in &self.properties {
                    let value = match source {
                        PropertySource::Column(col_idx) => chunk
                            .column(*col_idx)
                            .and_then(|c| c.get_value(row))
                            .unwrap_or(Value::Null),
                        PropertySource::Constant(v) => v.clone(),
                    };

                    if self.is_edge {
                        self.store
                            .set_edge_property(EdgeId(entity_id), prop_name, value);
                    } else {
                        self.store
                            .set_node_property(NodeId(entity_id), prop_name, value);
                    }
                }

                // Copy input columns to output
                for col_idx in 0..chunk.column_count() {
                    if let (Some(src), Some(dst)) =
                        (chunk.column(col_idx), builder.column_mut(col_idx))
                    {
                        if let Some(val) = src.get_value(row) {
                            dst.push_value(val);
                        } else {
                            dst.push_value(Value::Null);
                        }
                    }
                }

                builder.advance_row();
            }

            return Ok(Some(builder.finish()));
        }
        Ok(None)
    }

    fn reset(&mut self) {
        self.input.reset();
    }

    fn name(&self) -> &'static str {
        "SetProperty"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::DataChunk;
    use crate::execution::chunk::DataChunkBuilder;

    fn create_test_store() -> Arc<LpgStore> {
        Arc::new(LpgStore::new())
    }

    #[test]
    fn test_create_node_standalone() {
        let store = create_test_store();

        let mut op = CreateNodeOperator::new(
            Arc::clone(&store),
            None,
            vec!["Person".to_string()],
            vec![(
                "name".to_string(),
                PropertySource::Constant(Value::String("Alice".into())),
            )],
            vec![LogicalType::Int64],
            0,
        );

        // First call should create a node
        let chunk = op.next().unwrap().unwrap();
        assert_eq!(chunk.row_count(), 1);

        // Second call should return None
        assert!(op.next().unwrap().is_none());

        // Verify node was created
        assert_eq!(store.node_count(), 1);
    }

    #[test]
    fn test_create_edge() {
        let store = create_test_store();

        // Create two nodes first
        let node1 = store.create_node(&["Person"]);
        let node2 = store.create_node(&["Person"]);

        // Create input chunk with node IDs
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64, LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(node1.0 as i64);
        builder.column_mut(1).unwrap().push_int64(node2.0 as i64);
        builder.advance_row();
        let input_chunk = builder.finish();

        // Mock input operator
        struct MockInput {
            chunk: Option<DataChunk>,
        }
        impl Operator for MockInput {
            fn next(&mut self) -> OperatorResult {
                Ok(self.chunk.take())
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "MockInput"
            }
        }

        let mut op = CreateEdgeOperator::new(
            Arc::clone(&store),
            Box::new(MockInput {
                chunk: Some(input_chunk),
            }),
            0, // from column
            1, // to column
            "KNOWS".to_string(),
            vec![LogicalType::Int64, LogicalType::Int64],
        );

        // Execute
        let _chunk = op.next().unwrap().unwrap();

        // Verify edge was created
        assert_eq!(store.edge_count(), 1);
    }

    #[test]
    fn test_delete_node() {
        let store = create_test_store();

        // Create a node
        let node_id = store.create_node(&["Person"]);
        assert_eq!(store.node_count(), 1);

        // Create input chunk with the node ID
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(node_id.0 as i64);
        builder.advance_row();
        let input_chunk = builder.finish();

        struct MockInput {
            chunk: Option<DataChunk>,
        }
        impl Operator for MockInput {
            fn next(&mut self) -> OperatorResult {
                Ok(self.chunk.take())
            }
            fn reset(&mut self) {}
            fn name(&self) -> &'static str {
                "MockInput"
            }
        }

        let mut op = DeleteNodeOperator::new(
            Arc::clone(&store),
            Box::new(MockInput {
                chunk: Some(input_chunk),
            }),
            0,
            vec![LogicalType::Int64],
            false,
        );

        // Execute
        let chunk = op.next().unwrap().unwrap();

        // Verify deletion
        let deleted = chunk.column(0).unwrap().get_int64(0).unwrap();
        assert_eq!(deleted, 1);
        assert_eq!(store.node_count(), 0);
    }
}
