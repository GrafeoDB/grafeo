//! Folds the compacted base of a 0.5.x database file into its LPG store.
//!
//! The file of a database compacted with `compact()` (0.5.44 or older) holds
//! a columnar base (the `CompactStore` section), an overlay with the changes
//! made since (the `LpgStore` section) and the base nodes and edges deleted
//! since (the `OverlayDeletions` section). Opening such a file folds the base
//! into the store the overlay loaded into, so the database goes on with one
//! store, and its next checkpoint writes it as a plain one. Removed in 0.7.0
//! with the other 0.5.x readers.

use arcstr::ArcStr;
use grafeo_common::storage::SectionSource;
use grafeo_common::types::{EdgeId, NodeId};
use grafeo_common::utils::error::{Error, Result, StorageError};
use grafeo_common::utils::hash::FxHashSet;

use super::{CompactStore, labels_of_unescaped_key};
use crate::graph::lpg::LpgStore;

/// What [`fold_0_5_base`] created.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Folded {
    /// Base nodes created in the store.
    pub nodes: usize,
    /// Base edges created in the store.
    pub edges: usize,
    /// Base edges left out because an endpoint is gone: a node the deletion
    /// log lists while the edge is not listed. A delete of a node deleted its
    /// edges first, so a file holds none of them unless it is damaged.
    pub dangling_edges: usize,
    /// Overlay copies of base nodes given each of their labels in place of
    /// the one name joining them (see [`fold_0_5_base`]).
    pub relabeled: usize,
}

/// Folds the compacted base a 0.5.x database file holds into `store`, which
/// holds the file's LPG section (the overlay): reads the base from `base`
/// (the `CompactStore` section) and the deletes from `deletions` (the
/// `OverlayDeletions` section, if the file has one), then creates in `store`
/// every base node and edge the overlay neither copied nor deleted, under its
/// id, with its labels or its type and endpoints, and its properties.
///
/// The overlay holds the nodes and edges created after `compact()`, and a
/// whole copy of each base node and edge changed since, under the base's id,
/// which is the current version of it. Every other base node and edge is
/// current in the base, unless the deletion log lists it.
///
/// A copy of a base node with several labels can hold them as one label,
/// joined with `|`: the base served such a node so (#595). The copy gets
/// each of them in place of that one (see [`Folded::relabeled`]), and the
/// store no longer lists the joined name once no node has it.
///
/// No transaction may be open in `store`: the nodes and edges are created as
/// recovery creates them, committed at the store's current epoch. Build the
/// indexes of `store` after the fold, from all the data.
///
/// # Errors
///
/// Returns an error naming the section when a section does not decode or is
/// not the 0.5.x layout (see [`Error::Serialization`]), a corruption error
/// when a base without ids meets an overlay that holds data, or the error
/// of a record the store cannot allocate.
pub fn fold_0_5_base(
    base: &dyn SectionSource,
    deletions: Option<&dyn SectionSource>,
    store: &LpgStore,
) -> Result<Folded> {
    let base = super::section::read_base(base)?;
    let (deleted_nodes, deleted_edges) = match deletions {
        Some(source) => super::deletions_section::read_deletions(source)?,
        None => (Vec::new(), Vec::new()),
    };
    fold_into(&base, store, &deleted_nodes, &deleted_edges)
}

/// Creates in `store` every node and edge of `base` that `store` does not
/// hold and the deletion log (`deleted_nodes`, `deleted_edges`) does not
/// list, and splits the joined labels of the base nodes it holds (see
/// [`fold_0_5_base`]).
///
/// A base that does not keep the ids of its nodes and edges names them by
/// their table and row; it folds under those ids into an empty store only,
/// since an overlay's ids could name other entities.
fn fold_into(
    base: &CompactStore,
    store: &LpgStore,
    deleted_nodes: &[NodeId],
    deleted_edges: &[EdgeId],
) -> Result<Folded> {
    if !base.preserves_ids() && (store.node_count() > 0 || store.edge_count() > 0) {
        return Err(Error::Storage(StorageError::Corruption(
            "the compacted base does not keep the ids of its nodes and edges, and the data \
             written after it would mix with them"
                .to_string(),
        )));
    }
    let deleted_nodes: FxHashSet<NodeId> = deleted_nodes.iter().copied().collect();
    let deleted_edges: FxHashSet<EdgeId> = deleted_edges.iter().copied().collect();
    let mut folded = Folded::default();

    // The nodes the store holds: the overlay's, then the folded ones.
    let mut live: FxHashSet<NodeId> = store.node_ids().into_iter().collect();
    let base_nodes = base.node_ids();
    // The joined names the copies held, which the overlay's load added to
    // the store's labels.
    let mut joined_names: FxHashSet<ArcStr> = FxHashSet::default();
    for &id in &base_nodes {
        if live.contains(&id) {
            if split_joined_labels(base, store, id, &mut joined_names) {
                folded.relabeled += 1;
            }
            continue;
        }
        if deleted_nodes.contains(&id) {
            continue;
        }
        let Some(node) = base.get_node(id) else {
            continue;
        };
        let labels: Vec<&str> = node.labels.iter().map(|label| label.as_str()).collect();
        store.create_node_with_id(id, &labels)?;
        for (key, value) in node.properties {
            store.set_node_property(id, key.as_str(), value);
        }
        live.insert(id);
        folded.nodes += 1;
    }
    // Nobody wrote these names, so the store does not list them once no
    // node has them; one a node still has (a label written since) stays.
    for name in &joined_names {
        store.drop_unused_label(name);
    }

    // Every base edge once: from its source, as the outgoing edges of the
    // base's nodes (also of those the overlay copied or deleted).
    for &source in &base_nodes {
        for id in base.outgoing_edges(source) {
            if deleted_edges.contains(&id) || store.edge_type(id).is_some() {
                continue;
            }
            let Some(edge) = base.get_edge(id) else {
                continue;
            };
            if !live.contains(&edge.src) || !live.contains(&edge.dst) {
                folded.dangling_edges += 1;
                continue;
            }
            store.create_edge_with_id(id, edge.src, edge.dst, edge.edge_type.as_str())?;
            for (key, value) in edge.properties {
                store.set_edge_property(id, key.as_str(), value);
            }
            folded.edges += 1;
        }
    }
    Ok(folded)
}

/// Gives the overlay's copy of the base node `id` each label of the node's
/// base table in place of the one label that joins them, adds that label to
/// `joined_names`, and returns whether the copy held it.
///
/// Up to 0.5.44 the base served a node of several labels with one label,
/// the table's key: the labels joined with `|` (#595). A write after
/// `compact()` that found such a node without a label copied it to the
/// overlay with that label. The reader splits the key into the table's
/// labels, so a label of the copy that splits into exactly them is the key.
/// Every other label stays, one with a `|` written since included.
fn split_joined_labels(
    base: &CompactStore,
    store: &LpgStore,
    id: NodeId,
    joined_names: &mut FxHashSet<ArcStr>,
) -> bool {
    let Some((table_id, _)) = base.resolve_node(id) else {
        return false;
    };
    let labels = base.labels_of_table(table_id);
    if labels.len() < 2 {
        return false;
    }
    let Some(copy) = store.get_node(id) else {
        return false;
    };
    let mut joined = false;
    for label in &copy.labels {
        if labels_of_unescaped_key(label.as_str()) == labels && store.remove_label(id, label) {
            joined_names.insert(label.clone());
            joined = true;
        }
    }
    if joined {
        for label in labels {
            // False when the copy has the label already: a write since set it.
            store.add_label(id, label);
        }
    }
    joined
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use grafeo_common::storage::section::SectionType;
    use grafeo_common::storage::{ChunkMeta, ImageSource, MemoryImage, Section, SectionSink};
    use grafeo_common::types::{PropertyKey, Value};

    use super::*;

    const BASE: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.44/compact_store.bin"
    ));
    const DELETIONS: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.44/overlay_deletions.bin"
    ));
    const OVERLAY: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.44/lpg_store.bin"
    ));

    /// A 0.5.x image of the three sections, each one raw chunk; `None`
    /// leaves a section out.
    fn image(base: &[u8], deletions: Option<&[u8]>, overlay: Option<&[u8]>) -> MemoryImage {
        let mut image = MemoryImage::new();
        let mut section = |section_type, bytes: &[u8]| {
            image.begin_section(section_type, 1).unwrap();
            image.write_chunk(ChunkMeta::raw(), bytes).unwrap();
        };
        section(SectionType::CompactStore, base);
        if let Some(bytes) = deletions {
            section(SectionType::OverlayDeletions, bytes);
        }
        if let Some(bytes) = overlay {
            section(SectionType::LpgStore, bytes);
        }
        image
    }

    /// The store the overlay loads into, as the engine's load does, and the
    /// fold of the base into it.
    fn fold(image: &MemoryImage) -> (Arc<LpgStore>, Result<Folded>) {
        let store = Arc::new(LpgStore::new().unwrap());
        if let Some(source) = image.section_source(SectionType::LpgStore) {
            crate::graph::lpg::LpgStoreSection::new(Arc::clone(&store))
                .read_from(&*source)
                .unwrap();
        }
        let base = image.section_source(SectionType::CompactStore).unwrap();
        let deletions = image.section_source(SectionType::OverlayDeletions);
        let folded = fold_0_5_base(&*base, deletions.as_deref(), &store);
        (store, folded)
    }

    fn text(value: Option<Value>) -> Option<String> {
        value.and_then(|value| value.as_str().map(str::to_string))
    }

    /// The `name` of every node with `label`, sorted.
    fn names(store: &LpgStore, label: &str) -> Vec<String> {
        let key = PropertyKey::new("name");
        let mut names: Vec<String> = store
            .nodes_by_label(label)
            .into_iter()
            .filter_map(|id| text(store.get_node_property(id, &key)))
            .collect();
        names.sort();
        names
    }

    fn node_named(store: &LpgStore, label: &str, name: &str) -> NodeId {
        let key = PropertyKey::new("name");
        store
            .nodes_by_label(label)
            .into_iter()
            .find(|&id| text(store.get_node_property(id, &key)).as_deref() == Some(name))
            .unwrap_or_else(|| panic!("no {label} named {name}"))
    }

    /// The fixture: the first session of `scripts/released_fixtures.py`,
    /// `compact()`, then the second, with 0.5.44. The base holds the first
    /// session; the overlay a copy of Alix (her age set, her score removed),
    /// a copy of Amsterdam (the end of the museum's new edge) and what the
    /// second session created; the log Vincent.
    #[test]
    fn the_0_5_44_base_folds_into_its_overlay() {
        let (store, folded) = fold(&image(BASE, Some(DELETIONS), Some(OVERLAY)));
        let folded = folded.unwrap();
        assert_eq!(
            folded,
            Folded {
                nodes: 4,
                edges: 2,
                dangling_edges: 0,
                relabeled: 0
            },
            "Gus, Berlin and two documents; Alix knows Gus and lives in Amsterdam"
        );
        assert_eq!(
            names(&store, "Person"),
            ["Alix", "Gus", "Mia"],
            "Vincent stays deleted"
        );
        assert_eq!(
            names(&store, "City"),
            ["Amsterdam", "Berlin"],
            "the overlay's copy of Amsterdam, once"
        );

        let alix = node_named(&store, "Person", "Alix");
        assert_eq!(
            store.get_node_property(alix, &PropertyKey::new("age")),
            Some(Value::Int64(31)),
            "the overlay's copy of Alix wins over the base"
        );
        assert_eq!(
            store.get_node_property(alix, &PropertyKey::new("score")),
            None
        );

        let gus = node_named(&store, "Person", "Gus");
        let mut labels: Vec<String> = store
            .get_node(gus)
            .unwrap()
            .labels
            .iter()
            .map(ToString::to_string)
            .collect();
        labels.sort();
        assert_eq!(labels, ["Employee", "Person"], "0.5.44 never set Manager");

        let amsterdam = node_named(&store, "City", "Amsterdam");
        let mut alix_goes_to: Vec<NodeId> = store
            .edges_from(alix, crate::graph::Direction::Outgoing)
            .map(|(target, _)| target)
            .collect();
        alix_goes_to.sort_unstable();
        let mut expected = vec![gus, amsterdam];
        expected.sort_unstable();
        assert_eq!(alix_goes_to, expected, "the base's edges, folded");

        // A node created now gets an id past every id of the base.
        let base = super::super::section::read_base(
            &*image(BASE, None, None)
                .section_source(SectionType::CompactStore)
                .unwrap(),
        )
        .unwrap();
        let highest = base.node_ids().into_iter().max().unwrap();
        assert!(store.create_node(&["Person"]) > highest);
    }

    #[test]
    fn without_the_deletion_log_the_deleted_node_comes_back() {
        let (store, folded) = fold(&image(BASE, None, Some(OVERLAY)));
        assert_eq!(folded.unwrap().nodes, 5);
        assert_eq!(names(&store, "Person"), ["Alix", "Gus", "Mia", "Vincent"]);
    }

    /// The fixture's base, read alone.
    fn fixture_base() -> CompactStore {
        super::super::section::read_base(
            &*image(BASE, None, None)
                .section_source(SectionType::CompactStore)
                .unwrap(),
        )
        .unwrap()
    }

    /// The id the base gives the node named `name`.
    fn base_id(base: &CompactStore, name: &str) -> NodeId {
        let key = PropertyKey::new("name");
        base.node_ids()
            .into_iter()
            .find(|&id| {
                base.get_node(id)
                    .and_then(|node| text(node.properties.get(&key).cloned()))
                    .as_deref()
                    == Some(name)
            })
            .unwrap_or_else(|| panic!("the base holds no {name}"))
    }

    /// The labels of the node `id`, sorted.
    fn labels_of(store: &LpgStore, id: NodeId) -> Vec<String> {
        let mut labels: Vec<String> = store
            .get_node(id)
            .unwrap_or_else(|| panic!("no node {}", id.0))
            .labels
            .iter()
            .map(ToString::to_string)
            .collect();
        labels.sort();
        labels
    }

    /// Folds the fixture's base, without its deletion log, into `store`.
    fn fold_base_into(store: &LpgStore) -> Folded {
        let image = image(BASE, None, None);
        let base = image.section_source(SectionType::CompactStore).unwrap();
        fold_0_5_base(&*base, None, store).unwrap()
    }

    /// Up to 0.5.44 the base served a node of several labels with one label,
    /// its labels joined with `|`, and a write after `compact()` that found
    /// the node without a label copied it to the overlay with that label
    /// (#595). The fold gives the copy each label in its place: here Gus,
    /// a Person and an Employee in the base, whose copy got his age changed
    /// and the label Manager.
    #[test]
    fn an_overlay_copy_with_the_joined_labels_gets_each_label() {
        let base = fixture_base();
        let gus = base_id(&base, "Gus");
        let store = LpgStore::new().unwrap();
        store
            .create_node_with_id(gus, &["Employee|Person", "Manager"])
            .unwrap();
        store.set_node_property(gus, "name", Value::from("Gus"));
        store.set_node_property(gus, "age", Value::Int64(26));

        let folded = fold_base_into(&store);
        assert_eq!(
            labels_of(&store, gus),
            ["Employee", "Manager", "Person"],
            "each label of the base, and the overlay's own"
        );
        assert_eq!(folded.relabeled, 1, "the copy of Gus");
        assert_eq!(folded.nodes, 6, "every other base node");
        assert_eq!(
            store.get_node_property(gus, &PropertyKey::new("age")),
            Some(Value::Int64(26)),
            "the copy stays the current version"
        );
        assert_eq!(names(&store, "Employee"), ["Gus"]);
        assert_eq!(names(&store, "Manager"), ["Gus"]);
        assert_eq!(names(&store, "Person"), ["Alix", "Gus", "Vincent"]);
        assert_eq!(store.nodes_by_label_count("Person"), 3);
        assert!(
            store.nodes_by_label("Employee|Person").is_empty(),
            "no node keeps the joined name"
        );
        assert_eq!(
            sorted_labels(&store),
            ["City", "Document", "Employee", "Manager", "Person"],
            "the store no longer lists the joined name"
        );
    }

    /// The store's labels, sorted.
    fn sorted_labels(store: &LpgStore) -> Vec<String> {
        let mut labels = store.all_labels();
        labels.sort();
        labels
    }

    /// A joined name a node not in the base also has (a label written since
    /// that is the same text) is split on the copy and stays listed.
    #[test]
    fn a_joined_name_another_node_has_stays_listed() {
        let base = fixture_base();
        let gus = base_id(&base, "Gus");
        let jules = NodeId(base.node_ids().into_iter().max().unwrap().0 + 3);
        let store = LpgStore::new().unwrap();
        store
            .create_node_with_id(gus, &["Employee|Person"])
            .unwrap();
        store
            .create_node_with_id(jules, &["Employee|Person"])
            .unwrap();

        let folded = fold_base_into(&store);
        assert_eq!(labels_of(&store, gus), ["Employee", "Person"]);
        assert_eq!(labels_of(&store, jules), ["Employee|Person"]);
        assert_eq!(store.nodes_by_label("Employee|Person"), [jules]);
        assert!(
            sorted_labels(&store).contains(&"Employee|Person".to_string()),
            "Jules has it"
        );
        assert_eq!(folded.relabeled, 1);
    }

    /// Only the joined name of the base node's own labels is split: a label
    /// with a `|` on a node the base does not hold, or on a copy whose base
    /// labels it does not join, was written after `compact()` and stays.
    #[test]
    fn other_labels_with_a_bar_stay_as_they_are() {
        let base = fixture_base();
        let alix = base_id(&base, "Alix");
        let gus = base_id(&base, "Gus");
        let jules = NodeId(base.node_ids().into_iter().max().unwrap().0 + 88);
        let store = LpgStore::new().unwrap();
        store
            .create_node_with_id(alix, &["Person", "Employee|Person"])
            .unwrap();
        store.create_node_with_id(gus, &["In|Out"]).unwrap();
        store
            .create_node_with_id(jules, &["Employee|Person"])
            .unwrap();

        let folded = fold_base_into(&store);
        assert_eq!(
            labels_of(&store, alix),
            ["Employee|Person", "Person"],
            "Alix is a Person alone in the base"
        );
        assert_eq!(
            labels_of(&store, gus),
            ["In|Out"],
            "the copy of Gus no longer holds his joined labels"
        );
        assert_eq!(
            labels_of(&store, jules),
            ["Employee|Person"],
            "Jules is not in the base"
        );
        assert_eq!(folded.relabeled, 0);
    }

    const BASE_0_5_41: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.41/compact_store.bin"
    ));
    const OVERLAY_0_5_41: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/compact-0.5.41/lpg_store.bin"
    ));

    /// 0.5.40 and 0.5.41 wrote encoding 1 and no deletion log: the same
    /// sessions fold like the 0.5.44 fixture, but Vincent, whom the second
    /// session deleted from the base, comes back, as he did when 0.5.42 to
    /// 0.5.44 opened such a file.
    #[test]
    fn a_0_5_41_base_folds_and_its_unlogged_delete_comes_back() {
        let (store, folded) = fold(&image(BASE_0_5_41, None, Some(OVERLAY_0_5_41)));
        assert_eq!(
            folded.unwrap(),
            Folded {
                nodes: 5,
                edges: 2,
                dangling_edges: 0,
                relabeled: 0
            },
            "Gus, Vincent, Berlin and two documents; Alix knows Gus and lives in Amsterdam"
        );
        assert_eq!(names(&store, "Person"), ["Alix", "Gus", "Mia", "Vincent"]);
        assert_eq!(names(&store, "City"), ["Amsterdam", "Berlin"]);
        let alix = node_named(&store, "Person", "Alix");
        assert_eq!(
            store.get_node_property(alix, &PropertyKey::new("age")),
            Some(Value::Int64(31)),
            "the overlay's copy of Alix wins over the base"
        );
        let gus = node_named(&store, "Person", "Gus");
        let mut labels: Vec<String> = store
            .get_node(gus)
            .unwrap()
            .labels
            .iter()
            .map(ToString::to_string)
            .collect();
        labels.sort();
        assert_eq!(labels, ["Employee", "Person"]);
        assert_eq!(store.edge_count(), 3, "the base's two and the museum's");
    }

    /// The fixture's base with its flags byte cleared: it then names its
    /// nodes and edges by table and row (the id maps after it are left
    /// unread), as a base that does not keep ids does.
    fn base_without_ids() -> Vec<u8> {
        let mut bytes = BASE[..BASE.len() - 4].to_vec();
        bytes[5] = 0;
        let crc = crc32fast::hash(&bytes);
        bytes.extend_from_slice(&crc.to_le_bytes());
        bytes
    }

    #[test]
    fn a_base_without_ids_folds_into_an_empty_store_only() {
        let (store, folded) = fold(&image(&base_without_ids(), None, None));
        assert_eq!(
            folded.unwrap(),
            Folded {
                nodes: 7,
                edges: 2,
                dangling_edges: 0,
                relabeled: 0
            }
        );
        assert_eq!(names(&store, "Person"), ["Alix", "Gus", "Vincent"]);

        let (store, folded) = fold(&image(&base_without_ids(), None, Some(OVERLAY)));
        let error = folded.unwrap_err();
        assert!(
            matches!(error, Error::Storage(StorageError::Corruption(_))),
            "{error}"
        );
        assert_eq!(
            names(&store, "Person"),
            ["Alix", "Mia"],
            "nothing of the base is folded"
        );
    }

    #[test]
    fn a_damaged_section_fails_the_fold_and_names_it() {
        let mut damaged = DELETIONS.to_vec();
        damaged[10] ^= 0xFF;
        let (_, folded) = fold(&image(BASE, Some(&damaged), Some(OVERLAY)));
        let error = folded.unwrap_err().to_string();
        assert!(error.contains("OverlayDeletions"), "{error}");
    }
}
