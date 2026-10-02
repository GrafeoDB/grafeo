//! Horizontal aggregation operator for per-row aggregation over list-valued columns.
//!
//! Used for GE09 (ISO GQL horizontal aggregation): aggregating over group-list
//! variables from variable-length path patterns. For each input row, reads a list
//! of entity IDs, accesses a property on each entity, computes the aggregate, and
//! appends the scalar result as a new column.

use std::sync::Arc;

use grafeo_common::types::{EdgeId, LogicalType, NodeId, PropertyKey, Value};

use super::accumulator::AggregateFunction;
use super::aggregate::AggregateState;
use super::{Operator, OperatorResult};
use crate::execution::DataChunk;
use crate::execution::vector::ValueVector;
use crate::graph::traits::GraphStoreSearch;

/// Whether the horizontal aggregate operates on edges or nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum EntityKind {
    /// Aggregate over edges in a path.
    Edge,
    /// Aggregate over nodes in a path.
    Node,
}

/// Per-row aggregation over a list-valued column.
///
/// For each input row:
/// 1. Reads a `Value::List` from `list_column_idx` (entity IDs from a path)
/// 2. For each entity ID, looks up `property` via the graph store
/// 3. Feeds property values through an `AggregateState`
/// 4. Finalizes and appends the result as a new column
pub struct HorizontalAggregateOperator {
    /// Child operator.
    child: Box<dyn Operator>,
    /// Column index containing the list of entity IDs.
    list_column_idx: usize,
    /// Whether entities are edges or nodes.
    entity_kind: EntityKind,
    /// The aggregate function to compute per row.
    function: AggregateFunction,
    /// Property name to access on each entity.
    property: String,
    /// Graph store for property lookups.
    store: Arc<dyn GraphStoreSearch>,
    /// Number of input columns (to know where to append the result).
    input_column_count: usize,
}

impl HorizontalAggregateOperator {
    /// Creates a new horizontal aggregate operator.
    pub fn new(
        child: Box<dyn Operator>,
        list_column_idx: usize,
        entity_kind: EntityKind,
        function: AggregateFunction,
        property: String,
        store: Arc<dyn GraphStoreSearch>,
        input_column_count: usize,
    ) -> Self {
        Self {
            child,
            list_column_idx,
            entity_kind,
            function,
            property,
            store,
            input_column_count,
        }
    }

    /// Looks up a property value for an entity ID.
    fn get_property_value(&self, entity_value: &Value) -> Option<Value> {
        let prop_key = PropertyKey::new(&self.property);
        match self.entity_kind {
            EntityKind::Edge => {
                let id = match entity_value {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(i) => EdgeId(*i as u64),
                    _ => return None,
                };
                self.store.get_edge_property(id, &prop_key)
            }
            EntityKind::Node => {
                let id = match entity_value {
                    // reason: ID encoding: i64 <-> u64 round-trip
                    #[allow(clippy::cast_sign_loss)]
                    Value::Int64(i) => NodeId(*i as u64),
                    _ => return None,
                };
                self.store.get_node_property(id, &prop_key)
            }
        }
    }
}

impl Operator for HorizontalAggregateOperator {
    fn next(&mut self) -> OperatorResult {
        let Some(input) = self.child.next()? else {
            return Ok(None);
        };

        // Build output columns: copy input columns, in their types (a node or
        // edge stays one), + one new aggregate result column
        let mut output_columns: Vec<ValueVector> = (0..self.input_column_count)
            .map(|col_idx| {
                let column_type = input
                    .column(col_idx)
                    .map_or(LogicalType::Any, |column| column.data_type().clone());
                ValueVector::with_capacity(column_type, input.row_count())
            })
            .collect();
        let mut result_column = ValueVector::with_capacity(LogicalType::Float64, input.row_count());

        // Collect selected indices since the iterator can only be consumed once
        let rows: Vec<usize> = input.selected_indices().collect();

        for row in rows {
            // Copy all input columns to output
            for col_idx in 0..self.input_column_count {
                let value = input
                    .column(col_idx)
                    .and_then(|c| c.get_value(row))
                    .unwrap_or(Value::Null);
                output_columns[col_idx].push_value(value);
            }

            // Read the list column and compute horizontal aggregate
            let agg_result = if let Some(Value::List(list)) = input
                .column(self.list_column_idx)
                .and_then(|c| c.get_value(row))
            {
                let mut state = AggregateState::new(self.function, false, None, None);
                for entity_val in list.iter() {
                    let prop_val = self.get_property_value(entity_val);
                    if prop_val.is_some() && !matches!(prop_val, Some(Value::Null)) {
                        state.update(prop_val);
                    }
                }
                state.finalize()
            } else {
                Value::Null
            };

            result_column.push_value(agg_result);
        }

        output_columns.push(result_column);
        Ok(Some(DataChunk::new(output_columns)))
    }

    fn reset(&mut self) {
        self.child.reset();
    }

    fn name(&self) -> &'static str {
        "HorizontalAggregate"
    }

    fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
        self
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::execution::chunk::DataChunkBuilder;
    use crate::graph::lpg::LpgStore;

    struct MockOperator {
        chunks: Vec<DataChunk>,
        position: usize,
    }

    impl MockOperator {
        fn new(chunks: Vec<DataChunk>) -> Self {
            Self {
                chunks,
                position: 0,
            }
        }
    }

    impl Operator for MockOperator {
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
            "Mock"
        }

        fn into_any(self: Box<Self>) -> Box<dyn std::any::Any + Send> {
            self
        }
    }

    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn setup_store_with_edges() -> (Arc<dyn GraphStoreSearch>, Vec<Value>) {
        let store = LpgStore::new().unwrap();
        let n1 = store.create_node(&[]);
        let n2 = store.create_node(&[]);

        let e1 = store.create_edge(n1, n2, "ROAD");
        let e2 = store.create_edge(n1, n2, "ROAD");
        let e3 = store.create_edge(n1, n2, "ROAD");

        store.set_edge_property(e1, "weight", Value::Float64(1.5));
        store.set_edge_property(e2, "weight", Value::Float64(2.5));
        store.set_edge_property(e3, "weight", Value::Float64(3.0));

        let edge_ids: Vec<Value> = vec![
            Value::Int64(e1.0 as i64),
            Value::Int64(e2.0 as i64),
            Value::Int64(e3.0 as i64),
        ];
        (Arc::new(store), edge_ids)
    }

    // reason: test IDs are small sequential counters
    #[allow(clippy::cast_possible_wrap)]
    fn setup_store_with_nodes() -> (Arc<dyn GraphStoreSearch>, Vec<Value>) {
        let store = LpgStore::new().unwrap();
        // Use Float64 properties since the result column is Float64-typed
        // (Int64 values from SumInt finalize would be silently dropped by the Float64 column)
        let n1 = store.create_node_with_props(&["City"], [("pop", Value::Float64(100.0))]);
        let n2 = store.create_node_with_props(&["City"], [("pop", Value::Float64(200.0))]);
        let n3 = store.create_node_with_props(&["City"], [("pop", Value::Float64(300.0))]);

        let node_ids: Vec<Value> = vec![
            Value::Int64(n1.0 as i64),
            Value::Int64(n2.0 as i64),
            Value::Int64(n3.0 as i64),
        ];
        (Arc::new(store), node_ids)
    }

    /// The copied input columns keep their types: a node stays a node.
    #[test]
    fn copied_columns_keep_their_types() {
        let (store, edge_ids) = setup_store_with_edges();
        let mut builder = DataChunkBuilder::new(&[LogicalType::Node, LogicalType::Any]);
        builder.column_mut(0).unwrap().push_node_id(NodeId::new(5));
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::List(edge_ids.into()));
        builder.advance_row();
        let mut op = HorizontalAggregateOperator::new(
            Box::new(MockOperator::new(vec![builder.finish()])),
            1,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            2,
        );

        let result = op.next().unwrap().unwrap();
        assert_eq!(result.column_types()[0], LogicalType::Node);
        assert_eq!(
            result.column(0).unwrap().get_node_id(0).unwrap().as_u64(),
            5
        );
        assert!(result.column(0).unwrap().get_edge_id(0).is_none());
    }

    #[test]
    fn test_horizontal_sum_over_edges() {
        let (store, edge_ids) = setup_store_with_edges();

        // Build a chunk with one row: column 0 = some label, column 1 = list of edge IDs
        let mut builder = DataChunkBuilder::new(&[LogicalType::String, LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::String("path1".into()));
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::List(edge_ids.into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            1, // list_column_idx
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            2, // input_column_count
        );

        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sum of weights: 1.5 + 2.5 + 3.0 = 7.0
        let agg_val = result.column(2).unwrap().get_float64(0).unwrap();
        assert!((agg_val - 7.0).abs() < 0.001);

        // Should be done
        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_horizontal_sum_over_nodes() {
        let (store, node_ids) = setup_store_with_nodes();

        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(node_ids.into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Node,
            AggregateFunction::Sum,
            "pop".to_string(),
            store,
            1,
        );

        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Sum of node populations: 100.0 + 200.0 + 300.0 = 600.0
        let agg_val = result.column(1).unwrap().get_float64(0).unwrap();
        assert!((agg_val - 600.0).abs() < 0.001);
    }

    #[test]
    fn test_horizontal_avg_over_edges() {
        let (store, edge_ids) = setup_store_with_edges();

        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(edge_ids.into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Avg,
            "weight".to_string(),
            store,
            1,
        );

        let result = op.next().unwrap().unwrap();
        // Avg: (1.5 + 2.5 + 3.0) / 3 = 2.333...
        let agg_val = result.column(1).unwrap().get_float64(0).unwrap();
        assert!((agg_val - 7.0 / 3.0).abs() < 0.001);
    }

    #[test]
    fn test_horizontal_min_max_over_edges() {
        let (store, edge_ids) = setup_store_with_edges();

        // Test MIN
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(edge_ids.clone().into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Min,
            "weight".to_string(),
            Arc::clone(&store),
            1,
        );

        let result = op.next().unwrap().unwrap();
        let min_val = result.column(1).unwrap().get_float64(0).unwrap();
        assert!((min_val - 1.5).abs() < 0.001);

        // Test MAX
        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(edge_ids.into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Max,
            "weight".to_string(),
            store,
            1,
        );

        let result = op.next().unwrap().unwrap();
        let max_val = result.column(1).unwrap().get_float64(0).unwrap();
        assert!((max_val - 3.0).abs() < 0.001);
    }

    #[test]
    fn test_horizontal_empty_list_returns_null() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());

        let mut builder = DataChunkBuilder::new(&[LogicalType::Any]);
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(vec![].into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            1,
        );

        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);
        // Empty list sum should finalize to NULL (ISO/IEC 39075 Section 20.9)
        let agg_val = result.column(1).unwrap().get_value(0);
        assert!(
            matches!(agg_val, Some(Value::Null)),
            "Expected Null, got {agg_val:?}"
        );
    }

    #[test]
    fn test_horizontal_non_list_column_returns_null() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());

        // Put a non-list value in the list column
        let mut builder = DataChunkBuilder::new(&[LogicalType::Int64]);
        builder.column_mut(0).unwrap().push_int64(42);
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            1,
        );

        let result = op.next().unwrap().unwrap();
        // Non-list value should produce Null
        let agg_val = result.column(1).unwrap().get_value(0);
        assert_eq!(agg_val, Some(Value::Null));
    }

    #[test]
    fn test_horizontal_multiple_rows() {
        let (store, edge_ids) = setup_store_with_edges();

        let mut builder = DataChunkBuilder::new(&[LogicalType::String, LogicalType::Any]);
        // Row 0: all three edges
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::String("path_all".into()));
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::List(edge_ids.clone().into()));
        builder.advance_row();
        // Row 1: only first edge
        builder
            .column_mut(0)
            .unwrap()
            .push_value(Value::String("path_one".into()));
        builder
            .column_mut(1)
            .unwrap()
            .push_value(Value::List(vec![edge_ids[0].clone()].into()));
        builder.advance_row();
        let chunk = builder.finish();

        let mock = MockOperator::new(vec![chunk]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            1,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            2,
        );

        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 2);

        // Row 0: sum = 7.0
        let val0 = result.column(2).unwrap().get_float64(0).unwrap();
        assert!((val0 - 7.0).abs() < 0.001);
        // Row 1: sum = 1.5
        let val1 = result.column(2).unwrap().get_float64(1).unwrap();
        assert!((val1 - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_horizontal_reset() {
        let (store, edge_ids) = setup_store_with_edges();

        // Build two identical chunks so after reset the second is still available
        let mut builder1 = DataChunkBuilder::new(&[LogicalType::Any]);
        builder1
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(edge_ids.clone().into()));
        builder1.advance_row();

        let mut builder2 = DataChunkBuilder::new(&[LogicalType::Any]);
        builder2
            .column_mut(0)
            .unwrap()
            .push_value(Value::List(edge_ids.into()));
        builder2.advance_row();

        let mock = MockOperator::new(vec![builder1.finish(), builder2.finish()]);
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            1,
        );

        // First chunk
        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);

        // Second chunk
        let result = op.next().unwrap().unwrap();
        assert_eq!(result.row_count(), 1);

        // Done
        assert!(op.next().unwrap().is_none());

        // After reset, position goes back to 0 but chunks are consumed
        // This verifies reset() propagates to the child
        op.reset();
    }

    #[test]
    fn test_horizontal_name() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let mock = MockOperator::new(vec![]);
        let op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            1,
        );
        assert_eq!(op.name(), "HorizontalAggregate");
    }

    #[test]
    fn test_horizontal_child_returns_none() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let mock = MockOperator::new(vec![]); // No chunks
        let mut op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Edge,
            AggregateFunction::Sum,
            "weight".to_string(),
            store,
            1,
        );

        assert!(op.next().unwrap().is_none());
    }

    #[test]
    fn test_horizontal_aggregate_into_any() {
        let store: Arc<dyn GraphStoreSearch> = Arc::new(LpgStore::new().unwrap());
        let mock = MockOperator::new(vec![]);
        let op = HorizontalAggregateOperator::new(
            Box::new(mock),
            0,
            EntityKind::Node,
            AggregateFunction::Count,
            "name".to_string(),
            store,
            1,
        );
        let any = Box::new(op).into_any();
        assert!(any.downcast::<HorizontalAggregateOperator>().is_ok());
    }
}
