//! Exact spill-aware vector read by `(label, property, NodeId)`.
//!
//! Proves `GrafeoDB::read_indexed_node_vector` against inline storage and the
//! ForceDisk checkpoint/reopen lifecycle where the inline property path is empty.
//!
//! ```bash
//! cargo test -p grafeo-engine --test exact_vector_read \
//!   --features "vector-index,mmap,grafeo-file"
//! ```

mod helpers;

use grafeo_common::memory::StorageTier;
use grafeo_common::storage::SectionType;
use grafeo_common::types::{NodeId, PropertyKey, Value};
use grafeo_engine::{GrafeoDB, IndexedVectorRead};
use helpers::{force_disk_config, inline_embedding, make_embedding, seed_indexed_node};

#[test]
fn inline_exact_read_returns_owned_floats() {
    let db = GrafeoDB::new_in_memory();
    let expected = vec![0.25, 0.5, 0.75, 1.0];
    let node = seed_indexed_node(&db, "Item", "embedding", &expected);

    let IndexedVectorRead::Found(got) = db
        .read_indexed_node_vector("Item", "embedding", node)
        .expect("exact read")
    else {
        panic!("expected Found");
    };
    assert_eq!(got, expected);
    // Owned copy: mutating the returned vec must not affect a second read.
    let mut owned = got;
    owned[0] = -1.0;
    let IndexedVectorRead::Found(again) = db
        .read_indexed_node_vector("Item", "embedding", node)
        .expect("second exact read")
    else {
        panic!("expected Found");
    };
    assert_eq!(again[0], 0.25);
}

#[test]
fn missing_node_and_missing_vector_return_absent() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("Item", "embedding", Some(4), None, None, None, None)
        .unwrap();

    let absent = NodeId::new(9_999);
    assert!(matches!(
        db.read_indexed_node_vector("Item", "embedding", absent)
            .unwrap(),
        IndexedVectorRead::Absent
    ));

    let node = db.create_node(&["Item"]).unwrap();
    db.set_node_property(node, "name", Value::from("no-vector"))
        .unwrap();
    assert!(matches!(
        db.read_indexed_node_vector("Item", "embedding", node)
            .unwrap(),
        IndexedVectorRead::Absent
    ));
}

#[test]
fn unregistered_or_wrong_label_property_is_typed() {
    let db = GrafeoDB::new_in_memory();
    let expected = make_embedding(3, 4);
    let node = seed_indexed_node(&db, "Item", "embedding", &expected);

    assert!(matches!(
        db.read_indexed_node_vector("Item", "missing_prop", node)
            .unwrap(),
        IndexedVectorRead::IndexNotRegistered
    ));

    assert!(matches!(
        db.read_indexed_node_vector("Other", "embedding", node)
            .unwrap(),
        IndexedVectorRead::IndexNotRegistered
    ));

    // Same property name on a different label: index exists for Item only, so
    // reading under a registered-but-mismatched label is already covered above.
    // A node lacking the indexed label returns Absent once an index exists for it.
    db.create_vector_index("Other", "embedding", Some(4), None, None, None, None)
        .unwrap();
    assert!(
        matches!(
            db.read_indexed_node_vector("Other", "embedding", node)
                .unwrap(),
            IndexedVectorRead::Absent
        ),
        "Item node must not satisfy Other-label index"
    );
}

#[test]
fn dimension_fidelity_matches_registered_width() {
    let db = GrafeoDB::new_in_memory();
    let dim = 16;
    let expected = make_embedding(11, dim);
    let node = seed_indexed_node(&db, "Doc", "emb", &expected);
    assert_eq!(db.vector_index_dimensions("Doc", "emb"), Some(dim));

    let IndexedVectorRead::Found(got) = db
        .read_indexed_node_vector("Doc", "emb", node)
        .expect("exact read")
    else {
        panic!("expected Found");
    };
    assert_eq!(got.len(), dim);
    assert_eq!(got, expected);
}

#[test]
fn uncommitted_session_vector_not_visible_on_database_api() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("Doc", "emb", Some(3), None, None, None, None)
        .unwrap();

    let mut session = db.session();
    session.begin_transaction().unwrap();
    let node = session
        .create_node_with_props(
            &["Doc"],
            [("emb", Value::Vector(vec![1.0, 0.0, 0.0].into()))],
        )
        .unwrap();

    assert!(
        matches!(
            db.read_indexed_node_vector("Doc", "emb", node).unwrap(),
            IndexedVectorRead::Absent
        ),
        "uncommitted vector must not be visible via GrafeoDB"
    );

    session.commit().unwrap();
    assert!(matches!(
        db.read_indexed_node_vector("Doc", "emb", node).unwrap(),
        IndexedVectorRead::Found(ref v) if v.as_slice() == [1.0, 0.0, 0.0]
    ));
}

#[test]
#[cfg(not(feature = "temporal"))]
fn force_disk_checkpoint_reopen_exact_read_without_inline_property() {
    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("exact_read.grafeo");
    let spill_path = dir.path().join("exact_read.spill");
    let expected = vec![0.125, 0.25, 0.375, 0.5];
    let node;

    {
        let db = GrafeoDB::with_config(force_disk_config(&db_path, &spill_path)).unwrap();
        node = seed_indexed_node(&db, "Item", "embedding", &expected);
        assert!(matches!(
            db.read_indexed_node_vector("Item", "embedding", node)
                .unwrap(),
            IndexedVectorRead::Found(ref v) if v.as_slice() == expected.as_slice()
        ));

        db.wal_checkpoint().unwrap();
        assert_eq!(
            db.storage_tiers().get(&SectionType::VectorStore),
            Some(&StorageTier::OnDisk)
        );

        // After ForceDisk spill, the inline property column must not supply
        // the vector — only the spill-aware exact API remains authoritative.
        assert!(
            inline_embedding(&db, node, "embedding").is_none(),
            "inline property path must be empty after ForceDisk checkpoint"
        );
        assert!(matches!(
            db.read_indexed_node_vector("Item", "embedding", node)
                .unwrap(),
            IndexedVectorRead::Found(ref v) if v.as_slice() == expected.as_slice()
        ));
        db.close().unwrap();
    }

    {
        let db = GrafeoDB::with_config(force_disk_config(&db_path, &spill_path)).unwrap();
        assert_eq!(
            db.storage_tiers().get(&SectionType::VectorStore),
            Some(&StorageTier::OnDisk)
        );
        assert!(
            inline_embedding(&db, node, "embedding").is_none(),
            "inline property path must stay empty after ForceDisk reopen"
        );
        let IndexedVectorRead::Found(got) = db
            .read_indexed_node_vector("Item", "embedding", node)
            .expect("exact read after reopen")
        else {
            panic!("spilled vector must be present");
        };
        assert_eq!(got, expected);
        // Property-key probe via the store must also miss (same as get_node).
        let prop_key = PropertyKey::new("embedding");
        assert!(db
            .get_node(node)
            .and_then(|n| n.properties.get(&prop_key).cloned())
            .is_none());
        db.close().unwrap();
    }
}
