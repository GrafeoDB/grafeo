//! Spill-aware vector accessor construction for GrafeoDB.
//!
//! Shared by ANN search and exact indexed vector reads so both paths use the
//! same committed inline/ForceDisk accessor seam.

#[cfg(feature = "vector-index")]
impl super::GrafeoDB {
    /// Creates a vector accessor for the given label/property, using spilled
    /// MmapStorage if the index has been spilled to disk.
    #[cfg(all(feature = "mmap", not(feature = "temporal")))]
    pub(super) fn make_vector_accessor<'a>(
        &'a self,
        label: &str,
        property: &str,
    ) -> grafeo_core::index::vector::VectorAccessorKind<'a> {
        let key = format!("{label}:{property}");
        if let Some(ref spill_map) = self.vector_spill_storages {
            let map = spill_map.read();
            if let Some(storage) = map.get(&key) {
                return grafeo_core::index::vector::VectorAccessorKind::Spilled(
                    grafeo_core::index::vector::SpillableVectorAccessor::new(
                        self.graph_store_ref(),
                        property,
                        std::sync::Arc::clone(storage)
                            as std::sync::Arc<dyn grafeo_core::index::vector::VectorStorage>,
                    ),
                );
            }
        }
        grafeo_core::index::vector::VectorAccessorKind::Property(
            grafeo_core::index::vector::PropertyVectorAccessor::new(
                self.graph_store_ref(),
                property,
            ),
        )
    }

    /// Creates a vector accessor (no spill support when mmap or temporal unavailable).
    #[cfg(not(all(feature = "mmap", not(feature = "temporal"))))]
    pub(super) fn make_vector_accessor<'a>(
        &'a self,
        _label: &str,
        property: &str,
    ) -> grafeo_core::index::vector::VectorAccessorKind<'a> {
        grafeo_core::index::vector::VectorAccessorKind::Property(
            grafeo_core::index::vector::PropertyVectorAccessor::new(
                self.graph_store_ref(),
                property,
            ),
        )
    }
}
