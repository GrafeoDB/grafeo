//! Exact spill-aware vector reads for GrafeoDB.
//!
//! Provides [`GrafeoDB::read_indexed_node_vector`], which returns an owned copy
//! of the indexed vector for a `(label, property, NodeId)` using the same
//! spill-aware accessor as ANN search. This is the authority when ForceDisk
//! has drained the inline property column.

use grafeo_common::types::NodeId;
use grafeo_common::utils::error::Result;
use grafeo_core::index::vector::VectorAccessor;

/// Outcome of an exact indexed vector read by `(label, property, NodeId)`.
///
/// Normal control-flow outcomes (missing node/vector, unregistered index) are
/// machine-typed variants. Dimension corruption remains a typed [`Result::Err`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum IndexedVectorRead {
    /// Indexed vector present for a visible node under the requested label.
    Found(Vec<f32>),
    /// Node absent, lacks the label, or has no vector for the property.
    Absent,
    /// No vector index is registered for the label/property pair.
    IndexNotRegistered,
}

impl super::GrafeoDB {
    /// Reads the exact indexed vector for `node_id` under `(label, property)`.
    ///
    /// Uses the same committed, spill-aware accessor as
    /// [`vector_search`](Self::vector_search). Returns an owned vector copy that
    /// remains valid after the source node is purged. Does not perform
    /// approximate nearest-neighbor search or mutate the graph.
    ///
    /// # Returns
    ///
    /// - [`IndexedVectorRead::Found`] when the node is visible under `label`
    ///   and an indexed vector of the registered width is available (inline or
    ///   spilled)
    /// - [`IndexedVectorRead::Absent`] when the node is absent, lacks `label`,
    ///   or has no vector for `property` in the index-backed storage
    /// - [`IndexedVectorRead::IndexNotRegistered`] when no vector index exists
    ///   for `label`/`property`
    ///
    /// # Errors
    ///
    /// Returns [`grafeo_common::utils::error::Error::InvalidValue`] if a
    /// recovered vector's width does not match the index dimensions.
    pub fn read_indexed_node_vector(
        &self,
        label: &str,
        property: &str,
        node_id: NodeId,
    ) -> Result<IndexedVectorRead> {
        let Some(index) = self.lpg_store().get_vector_index(label, property) else {
            return Ok(IndexedVectorRead::IndexNotRegistered);
        };
        let expected_dims = index.config().dimensions;

        // Index registration is label-scoped; do not return a cross-label
        // property hit from the shared property column or a sibling spill file.
        let has_label = self
            .lpg_store()
            .get_node(node_id)
            .is_some_and(|node| node.labels.iter().any(|l| l.as_str() == label));
        if !has_label {
            return Ok(IndexedVectorRead::Absent);
        }

        let accessor = self.make_vector_accessor(label, property);
        let Some(vector) = accessor.get_vector(node_id) else {
            return Ok(IndexedVectorRead::Absent);
        };
        if vector.len() != expected_dims {
            return Err(grafeo_common::utils::error::Error::InvalidValue(format!(
                "Indexed vector for :{label}({property}) node {} has width {}, expected {expected_dims}",
                node_id.as_u64(),
                vector.len()
            )));
        }
        Ok(IndexedVectorRead::Found(vector.as_ref().to_vec()))
    }
}
