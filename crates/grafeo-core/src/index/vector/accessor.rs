//! Vector accessor trait for reading vectors by node ID.
//!
//! This module provides the [`VectorAccessor`] trait, which decouples vector
//! storage from vector indexing. The HNSW index is topology-only (neighbor
//! lists only, no stored vectors) and reads vectors through this trait from
//! the property store, the single source of truth (spilled columns
//! included), halving memory usage for vector workloads.
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

    /// Calls `f` with the vector of the given node and returns whether there
    /// was one. Search computes every distance through this, so an accessor
    /// should lend the vector without copying it (a spilled vector is read
    /// in place); the default goes through [`get_vector`](Self::get_vector).
    fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> bool {
        match self.get_vector(id) {
            Some(vector) => {
                f(&vector);
                true
            }
            None => false,
        }
    }
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

    fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> bool {
        self.store.with_node_vector(id, &self.property, f)
    }
}

/// The accessor the engine passes to vector operations.
///
/// An enum rather than `Box<dyn VectorAccessor>`, so it can be passed to
/// `HnswIndex::search(&impl VectorAccessor)` without `Sized` workarounds. A
/// spilled column needs no variant of its own: the property store reads
/// through it.
#[non_exhaustive]
pub enum VectorAccessorKind<'a> {
    /// Direct property store lookup.
    Property(PropertyVectorAccessor<'a>),
}

impl VectorAccessor for VectorAccessorKind<'_> {
    fn get_vector(&self, id: NodeId) -> Option<Arc<[f32]>> {
        match self {
            Self::Property(a) => a.get_vector(id),
        }
    }

    fn with_vector(&self, id: NodeId, f: &mut dyn FnMut(&[f32])) -> bool {
        match self {
            Self::Property(a) => a.with_vector(id, f),
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

#[cfg(all(test, feature = "lpg", not(feature = "temporal")))]
mod borrowed_read_tests {
    use super::*;
    use crate::graph::lpg::LpgStore;

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

    /// A search over a spilled column reads every vector in place: the
    /// backing never copies one out (ruling (d) of #594).
    #[cfg(feature = "vector-index")]
    #[test]
    fn search_reads_spilled_vectors_without_copying_them() {
        use crate::graph::lpg::test_backing::MemoryBacking;
        use crate::index::vector::{DistanceMetric, HnswConfig, HnswIndex};
        use std::sync::atomic::Ordering;

        let store = LpgStore::new().unwrap();
        let key = PropertyKey::new("embedding");
        let vectors = [[3.0, 19.0], [19.0, 88.0], [88.0, 3.0], [3.19, 19.88]];
        let ids: Vec<NodeId> = vectors
            .iter()
            .map(|vector| {
                store.create_node_with_props(
                    &["Item"],
                    [("embedding", Value::Vector(vector.to_vec().into()))],
                )
            })
            .collect();
        let accessor = PropertyVectorAccessor::new(&store, "embedding");
        let index = HnswIndex::new(HnswConfig::new(2, DistanceMetric::Euclidean));
        for (id, vector) in ids.iter().zip(vectors) {
            index.insert(*id, &vector, &accessor);
        }
        let snapshot = store.node_property_column_entries(&key).unwrap();
        let backing = MemoryBacking::of(&snapshot);
        assert!(store.spill_node_property_column(&key, backing.clone(), &snapshot));

        let hits = index.search(&[3.0, 19.0], 2, &accessor);
        assert_eq!(hits.first().map(|hit| hit.0), Some(ids[0]));
        assert_eq!(
            backing.copies.load(Ordering::Relaxed),
            0,
            "a vector was copied"
        );
    }

    /// The borrowed read hands out the stored vector, and nothing for a
    /// node without one or a value that is not a vector.
    #[test]
    fn with_vector_lends_the_stored_vector() {
        let store = LpgStore::new().unwrap();
        let mia = store.create_node(&["Person"]);
        let butch = store.create_node(&["Person"]);
        store.set_node_property(mia, "embedding", Value::Vector(vec![3.0, 19.0].into()));
        store.set_node_property(butch, "embedding", Value::from("not a vector"));
        let accessor =
            VectorAccessorKind::Property(PropertyVectorAccessor::new(&store, "embedding"));

        let mut seen = Vec::new();
        assert!(accessor.with_vector(mia, &mut |v| seen.extend_from_slice(v)));
        assert_eq!(seen, vec![3.0, 19.0]);
        assert!(!accessor.with_vector(butch, &mut |_| panic!("not a vector")));
        assert!(!accessor.with_vector(NodeId::new(999), &mut |_| panic!("no node")));
    }
}
