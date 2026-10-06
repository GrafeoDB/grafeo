//! Text Index section serializer for the `.grafeo` container format.
//!
//! Serializes BM25 inverted indexes (postings lists, document lengths)
//! for all text indexes. Persisting avoids rebuilding from LPG properties
//! on database open.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use grafeo_common::storage::section::{Section, SectionType};
use grafeo_common::types::NodeId;
use grafeo_common::utils::error::{Error, Result};

use super::InvertedIndex;

/// Current text index section format version.
const TEXT_SECTION_VERSION: u8 = 1;

// ── Snapshot types ──────────────────────────────────────────────────

#[derive(Serialize, Deserialize)]
struct TextIndexSnapshot {
    version: u8,
    indexes: Vec<SingleIndexSnapshot>,
}

#[derive(Serialize, Deserialize)]
struct SingleIndexSnapshot {
    /// Index key: "label:property"
    key: String,
    /// BM25 parameters
    k1: f64,
    b: f64,
    /// Postings: term -> vec of (node_id, term_freq)
    postings: Vec<(String, Vec<(NodeId, u32)>)>,
    /// Document lengths: node_id -> token count
    doc_lengths: Vec<(NodeId, u32)>,
    /// Sum of all document lengths
    total_length: u64,
}

// ── Section implementation ──────────────────────────────────────────

/// Text Index section for the `.grafeo` container.
pub struct TextIndexSection {
    indexes: Vec<(String, Arc<RwLock<InvertedIndex>>)>,
    dirty: AtomicBool,
}

impl TextIndexSection {
    /// Create a new Text Index section from the current indexes.
    pub fn new(indexes: Vec<(String, Arc<RwLock<InvertedIndex>>)>) -> Self {
        Self {
            indexes,
            dirty: AtomicBool::new(false),
        }
    }

    /// Mark this section as dirty.
    pub fn mark_dirty(&self) {
        self.dirty.store(true, Ordering::Release);
    }
}

impl Section for TextIndexSection {
    fn section_type(&self) -> SectionType {
        SectionType::TextIndex
    }

    fn version(&self) -> u8 {
        TEXT_SECTION_VERSION
    }

    fn serialize(&self) -> Result<Vec<u8>> {
        let indexes: Vec<SingleIndexSnapshot> = self
            .indexes
            .iter()
            .map(|(key, index_lock)| {
                let index = index_lock.read();
                let config = index.config();
                let (postings, doc_lengths, total_length) = index.snapshot();

                SingleIndexSnapshot {
                    key: key.clone(),
                    k1: config.k1,
                    b: config.b,
                    postings,
                    doc_lengths,
                    total_length,
                }
            })
            .collect();

        let snapshot = TextIndexSnapshot {
            version: TEXT_SECTION_VERSION,
            indexes,
        };

        let config = bincode::config::standard();
        bincode::serde::encode_to_vec(&snapshot, config)
            .map_err(|e| Error::Internal(format!("Text Index section serialization failed: {e}")))
    }

    fn deserialize(&mut self, data: &[u8]) -> Result<()> {
        let config = bincode::config::standard();
        let (snapshot, _): (TextIndexSnapshot, _) = bincode::serde::decode_from_slice(data, config)
            .map_err(|e| {
                Error::Serialization(format!("Text Index section deserialization failed: {e}"))
            })?;

        for idx_snap in snapshot.indexes {
            if let Some((_, index_lock)) = self.indexes.iter().find(|(k, _)| *k == idx_snap.key) {
                let mut index = index_lock.write();
                index.set_config(super::BM25Config {
                    k1: idx_snap.k1,
                    b: idx_snap.b,
                });
                index.restore(
                    idx_snap.postings,
                    idx_snap.doc_lengths,
                    idx_snap.total_length,
                );
            }
        }

        Ok(())
    }

    fn is_dirty(&self) -> bool {
        self.dirty.load(Ordering::Acquire)
    }

    fn mark_clean(&self) {
        self.dirty.store(false, Ordering::Release);
    }

    fn memory_usage(&self) -> usize {
        self.indexes
            .iter()
            .map(|(_, idx)| idx.read().heap_memory_bytes())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::text::BM25Config;

    #[test]
    fn text_section_round_trip() {
        let mut index = InvertedIndex::new(BM25Config::default());
        index.insert(NodeId::new(1), "rust graph database");
        index.insert(NodeId::new(2), "python web framework");
        index.insert(NodeId::new(3), "rust systems programming");

        let index_arc = Arc::new(RwLock::new(index));
        let section = TextIndexSection::new(vec![(
            "Item:description".to_string(),
            Arc::clone(&index_arc),
        )]);

        let bytes = section.serialize().expect("serialize should succeed");
        assert!(!bytes.is_empty(), "bytes is empty");

        // Restore into a fresh index
        let fresh = InvertedIndex::new(BM25Config::default());
        let fresh_arc = Arc::new(RwLock::new(fresh));
        let mut section2 =
            TextIndexSection::new(vec![("Item:description".to_string(), fresh_arc.clone())]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");

        assert_eq!(fresh_arc.read().len(), 3);
        // 8 unique terms: rust, graph, database, python, web, framework, systems, programming
        assert!(fresh_arc.read().term_count() > 0);
    }

    #[test]
    fn text_section_empty() {
        let section = TextIndexSection::new(vec![]);
        let bytes = section.serialize().expect("serialize should succeed");

        let mut section2 = TextIndexSection::new(vec![]);
        section2
            .deserialize(&bytes)
            .expect("deserialize should succeed");
    }

    #[test]
    fn text_section_type() {
        let section = TextIndexSection::new(vec![]);
        assert_eq!(section.section_type(), SectionType::TextIndex);
        assert_eq!(section.version(), TEXT_SECTION_VERSION);
    }

    #[test]
    fn text_section_dirty_tracking() {
        let section = TextIndexSection::new(vec![]);
        assert!(!section.is_dirty());
        section.mark_dirty();
        assert!(section.is_dirty());
        section.mark_clean();
        assert!(!section.is_dirty());
    }
}
