//! Vector accessor trait for reading vectors by node ID.
//!
//! This module provides the [`VectorAccessor`] trait, which decouples vector
//! storage from vector indexing. The HNSW index is topology-only (neighbor
//! lists only, no stored vectors) and reads vectors through this trait from
//! [`PropertyStorage`], the single source of truth, halving memory usage
//! for vector workloads.
//!
//! # Example
//!
//! ```
//! use grafeo_core::index::vector::VectorAccessor;
//! use grafeo_common::types::NodeId;
//! use std::sync::Arc;
//!
//! // Closure-based accessor for tests
//! let accessor = |id: NodeId| -> Option<Arc<[f32]>> {
//!     Some(vec![1.0, 2.0, 3.0].into())
//! };
//! assert!(accessor.get_vector(NodeId::new(1)).is_some());
//! ```

use std::sync::Arc;

use grafeo_common::types::{NodeId, PropertyKey, Value};

use crate::graph::GraphStore;

/// Trait for reading vectors by node ID.
///
/// HNSW is topology-only: vectors live in property storage, not in
/// HNSW nodes. This trait provides the bridge for reading them.
pub trait VectorAccessor: Send + Sync {
    /// Returns the vector associated with the given node ID, if it exists.
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>>;
}

/// Reads vectors from a graph store's property storage for a given property key.
///
/// This is the primary accessor used by the engine when performing vector
/// operations. It reads directly from the property store, avoiding any
/// duplication.
pub struct PropertyVectorAccessor<'a> {
    store: &'a dyn GraphStore,
    property: PropertyKey,
}

impl<'a> PropertyVectorAccessor<'a> {
    /// Creates a new accessor for the given store and property key.
    #[must_use]
    pub fn new(store: &'a dyn GraphStore, property: impl Into<PropertyKey>) -> Self {
        Self {
            store,
            property: property.into(),
        }
    }
}

impl VectorAccessor for PropertyVectorAccessor<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        match self.store.get_node_property(id, &self.property) {
            Some(Value::Vector(v)) => Some(v),
            _ => None,
        }
    }
}

/// Reads vectors from mutable property storage first, falling back to a
/// spill-backed store for vectors that have not changed since the last spill.
///
/// Created by the engine when a vector index has been spilled to disk. New
/// inserts and updates remain in the property-store delta until the next
/// checkpoint/spill cycle, so that delta is authoritative over mmap contents.
pub struct SpillableVectorAccessor<'a> {
    store: &'a dyn GraphStore,
    property: PropertyKey,
    spill_storage: Arc<dyn super::storage::VectorStorage>,
}

impl<'a> SpillableVectorAccessor<'a> {
    /// Creates a new accessor that checks `spill_storage` first, then falls
    /// back to the property store.
    #[must_use]
    pub fn new(
        store: &'a dyn GraphStore,
        property: impl Into<PropertyKey>,
        spill_storage: Arc<dyn super::storage::VectorStorage>,
    ) -> Self {
        Self {
            store,
            property: property.into(),
            spill_storage,
        }
    }
}

impl VectorAccessor for SpillableVectorAccessor<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        // Mutable post-spill writes are authoritative over the mmap snapshot.
        if let Some(Value::Vector(vector)) = self.store.get_node_property(id, &self.property) {
            return Some(vector);
        }
        self.spill_storage.get(id)
    }
}

/// An accessor that dispatches to either a property store or a spill-backed store.
///
/// This enum avoids dynamic dispatch (`Box<dyn VectorAccessor>`) so it can be
/// passed to `HnswIndex::search(&impl VectorAccessor)` without requiring `Sized`
/// workarounds.
#[non_exhaustive]
pub enum VectorAccessorKind<'a> {
    /// Direct property store lookup (default, no spill).
    Property(PropertyVectorAccessor<'a>),
    /// Spill-backed: checks MmapStorage first, falls back to property store.
    Spilled(SpillableVectorAccessor<'a>),
}

impl VectorAccessor for VectorAccessorKind<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        match self {
            Self::Property(a) => a.get_vector(id),
            Self::Spilled(a) => a.get_vector(id),
        }
    }
}

/// Blanket implementation for closures, useful in tests.
impl<F> VectorAccessor for F
where
    F: Fn(NodeId) -> Option<Arc<[f32]>> + Send + Sync,
{
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        self(id)
    }
}

#[cfg(all(test, feature = "lpg"))]
mod tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

    #[test]
    fn test_closure_accessor() {
        let vectors: std::collections::HashMap<NodeId, Arc<[f32]>> = [
            (NodeId::new(1), Arc::from(vec![1.0_f32, 0.0, 0.0])),
            (NodeId::new(2), Arc::from(vec![0.0_f32, 1.0, 0.0])),
        ]
        .into_iter()
        .collect();

        let accessor = move |id: NodeId| -> Option<Arc<[f32]>> { vectors.get(&id).cloned() };

        assert!(accessor.get_vector(NodeId::new(1)).is_some());
        assert_eq!(accessor.get_vector(NodeId::new(1)).unwrap().len(), 3);
        assert!(accessor.get_vector(NodeId::new(3)).is_none());
    }

    #[test]
    fn test_property_vector_accessor() {
        let store = LpgStore::new().unwrap();
        let id = store.create_node(&["Test"]);
        let vec_data: Arc<[f32]> = vec![1.0, 2.0, 3.0].into();
        store.set_node_property(id, "embedding", Value::Vector(vec_data.clone()));

        let accessor = PropertyVectorAccessor::new(&store, "embedding");
        let result = accessor.get_vector(id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), vec_data.as_ref());

        // Non-existent node
        assert!(accessor.get_vector(NodeId::new(999)).is_none());

        // Wrong property type
        store.set_node_property(id, "name", Value::from("hello"));
        let name_accessor = PropertyVectorAccessor::new(&store, "name");
        assert!(name_accessor.get_vector(id).is_none());
    }
}

#[cfg(all(test, feature = "lpg", feature = "vector-index"))]
mod spill_tests {
    use super::*;
    use crate::graph::lpg::LpgStore;
    use crate::index::vector::storage::{RamStorage, VectorStorage};

    #[test]
    fn spill_accessor_returns_vector_from_spill_storage() {
        let store = LpgStore::new().unwrap();
        let alix_id = store.create_node(&["Person"]);
        let spill_vec: Vec<f32> = vec![0.1, 0.2, 0.3];
        let spill = Arc::new(RamStorage::new(3));
        spill.insert(alix_id, &spill_vec).unwrap();

        let accessor = SpillableVectorAccessor::new(&store as &dyn GraphStore, "embedding", spill);
        let result = accessor.get_vector(alix_id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), spill_vec.as_slice());
    }

    #[test]
    fn spill_accessor_falls_back_to_property_store() {
        let store = LpgStore::new().unwrap();
        let gus_id = store.create_node(&["Person"]);
        let prop_vec: Arc<[f32]> = vec![0.4, 0.5, 0.6].into();
        store.set_node_property(gus_id, "embedding", Value::Vector(prop_vec.clone()));

        let spill: Arc<dyn VectorStorage> = Arc::new(RamStorage::new(3));
        let accessor = SpillableVectorAccessor::new(&store as &dyn GraphStore, "embedding", spill);
        let result = accessor.get_vector(gus_id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), prop_vec.as_ref());
    }

    #[test]
    fn spill_accessor_prefers_mutable_property_delta() {
        let store = LpgStore::new().unwrap();
        let vincent_id = store.create_node(&["Person"]);
        let prop_vec: Arc<[f32]> = vec![1.0, 0.0, 0.0].into();
        store.set_node_property(vincent_id, "embedding", Value::Vector(prop_vec.clone()));

        let spill_vec: Vec<f32> = vec![0.0, 1.0, 0.0];
        let spill = Arc::new(RamStorage::new(3));
        spill.insert(vincent_id, &spill_vec).unwrap();

        let accessor = SpillableVectorAccessor::new(&store as &dyn GraphStore, "embedding", spill);
        let result = accessor.get_vector(vincent_id).unwrap();
        assert_eq!(result.as_ref(), prop_vec.as_ref());
    }

    #[test]
    fn spill_accessor_returns_none_when_missing() {
        let store = LpgStore::new().unwrap();
        let spill: Arc<dyn VectorStorage> = Arc::new(RamStorage::new(3));
        let accessor = SpillableVectorAccessor::new(&store as &dyn GraphStore, "embedding", spill);
        assert!(accessor.get_vector(NodeId::new(999)).is_none());
    }

    #[test]
    fn accessor_kind_property_dispatches() {
        let store = LpgStore::new().unwrap();
        let jules_id = store.create_node(&["Person"]);
        let vec_data: Arc<[f32]> = vec![0.7, 0.8, 0.9].into();
        store.set_node_property(jules_id, "embedding", Value::Vector(vec_data.clone()));

        let accessor =
            VectorAccessorKind::Property(PropertyVectorAccessor::new(&store, "embedding"));
        let result = accessor.get_vector(jules_id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), vec_data.as_ref());
        assert!(accessor.get_vector(NodeId::new(999)).is_none());
    }

    #[test]
    fn accessor_kind_spilled_dispatches() {
        let store = LpgStore::new().unwrap();
        let mia_id = store.create_node(&["Person"]);
        let spill_vec: Vec<f32> = vec![0.3, 0.6, 0.9];
        let spill = Arc::new(RamStorage::new(3));
        spill.insert(mia_id, &spill_vec).unwrap();

        let accessor = VectorAccessorKind::Spilled(SpillableVectorAccessor::new(
            &store as &dyn GraphStore,
            "embedding",
            spill,
        ));
        let result = accessor.get_vector(mia_id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), spill_vec.as_slice());
    }

    #[test]
    fn accessor_kind_spilled_uses_fallback() {
        let store = LpgStore::new().unwrap();
        let butch_id = store.create_node(&["Person"]);
        let prop_vec: Arc<[f32]> = vec![0.2, 0.4, 0.6].into();
        store.set_node_property(butch_id, "embedding", Value::Vector(prop_vec.clone()));

        let spill: Arc<dyn VectorStorage> = Arc::new(RamStorage::new(3));
        let accessor = VectorAccessorKind::Spilled(SpillableVectorAccessor::new(
            &store as &dyn GraphStore,
            "embedding",
            spill,
        ));
        let result = accessor.get_vector(butch_id);
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_ref(), prop_vec.as_ref());
    }
}
