//! Shared helpers for `exact_vector_read` integration tests.

use grafeo_common::storage::{SectionType, TierOverride};
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};

pub fn force_disk_config(db_path: &std::path::Path, spill_path: &std::path::Path) -> Config {
    Config::persistent(db_path)
        .with_spill_path(spill_path)
        .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk)
}

pub fn make_embedding(seed: u64, dim: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 7 + i as u64) % 100) as f32 / 100.0)
        .collect()
}

pub fn seed_indexed_node(db: &GrafeoDB, label: &str, property: &str, vector: &[f32]) -> NodeId {
    let id = db.create_node(&[label]).unwrap();
    db.set_node_property(id, property, Value::Vector(vector.to_vec().into()))
        .unwrap();
    db.create_vector_index(label, property, Some(vector.len()), None, None, None, None)
        .unwrap();
    id
}

/// Returns the embedding from the inline node property column, if present.
///
/// After ForceDisk spill this is expected to be `None` even when the
/// spill-aware exact read API still returns the vector.
pub fn inline_embedding(db: &GrafeoDB, node: NodeId, property: &str) -> Option<Vec<f32>> {
    let prop_key = PropertyKey::new(property);
    match db.get_node(node)?.properties.get(&prop_key)? {
        Value::Vector(v) => Some(v.as_ref().to_vec()),
        _ => None,
    }
}
