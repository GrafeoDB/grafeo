//! Commit / visibility / preflight / WAL / vector coverage.

use super::common::person;
use grafeo_common::types::Value;
use grafeo_engine::session::{TransactionalEdgeCreate, TransactionalNodeCreate};
use grafeo_engine::{Config, GrafeoDB};

#[test]
fn session_transactional_batch_requires_explicit_transaction() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let err = session
        .create_nodes_with_props_transactional(&[person("A", 1)])
        .expect_err("batch without a transaction must fail");
    let msg = format!("{err}");
    assert!(
        msg.to_lowercase().contains("transaction"),
        "error should mention the missing transaction: {msg}"
    );

    let err = session
        .create_edges_with_props_transactional(&[])
        .expect_err("edge batch without a transaction must fail");
    assert!(format!("{err}").to_lowercase().contains("transaction"));
}

#[test]
fn session_transactional_batch_commit_makes_rows_visible() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    let ids = session
        .create_nodes_with_props_transactional(&[person("A", 1), person("B", 2), person("C", 3)])
        .expect("batch create");
    assert_eq!(ids.len(), 3);
    session.commit().expect("commit");

    for (id, (name, age)) in ids.iter().zip([("A", 1), ("B", 2), ("C", 3)]) {
        let node = db.get_node(*id).expect("committed node visible");
        assert!(node.has_label("Person"));
        assert_eq!(
            node.get_property("name").and_then(|v| v.as_str()),
            Some(name)
        );
        assert_eq!(
            node.get_property("age").and_then(|v| v.as_int64()),
            Some(age)
        );
    }
}

#[test]
fn session_transactional_batch_concurrent_visibility() {
    let db = GrafeoDB::new_in_memory();

    let mut session_a = db.session();
    session_a.begin_transaction().expect("begin A");
    let ids = session_a
        .create_nodes_with_props_transactional(&[person("A", 1), person("B", 2)])
        .expect("batch create");

    {
        let session_b = db.session();
        for id in &ids {
            assert!(
                session_b.get_node(*id).is_none(),
                "uncommitted batch row must be invisible to other sessions"
            );
        }
    }

    session_a.commit().expect("commit A");

    {
        let session_c = db.session();
        for id in &ids {
            assert!(
                session_c.get_node(*id).is_some(),
                "committed batch row must be visible after commit"
            );
        }
    }
}

#[test]
fn session_transactional_batch_preflight_is_atomic() {
    let config = Config::in_memory().with_max_property_size(8);
    let db = GrafeoDB::with_config(config).expect("open");
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    let oversized = TransactionalNodeCreate::new(["Person"]).with_property(
        "blob",
        Value::from("this value is far larger than eight bytes"),
    );
    let result = session.create_nodes_with_props_transactional(&[person("ok", 1), oversized]);

    assert!(result.is_err(), "oversized property must fail preflight");
    assert_eq!(db.node_count(), 0, "preflight failure must create nothing");
    session.rollback().ok();
}

#[test]
fn session_transactional_batch_edges_commit_and_order() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    let nodes = session
        .create_nodes_with_props_transactional(&[person("A", 1), person("B", 2), person("C", 3)])
        .expect("nodes");
    let edges = session
        .create_edges_with_props_transactional(&[
            TransactionalEdgeCreate::new(nodes[0], nodes[1], "KNOWS")
                .with_property("w", Value::from(1i64)),
            TransactionalEdgeCreate::new(nodes[1], nodes[2], "LIKES"),
        ])
        .expect("edges");
    session.commit().expect("commit");

    assert_eq!(edges.len(), 2);
    assert_eq!(edges[1].0, edges[0].0 + 1, "edge IDs preserve input order");
    let e0 = db.get_edge(edges[0]).expect("edge visible");
    assert_eq!(e0.src, nodes[0]);
    assert_eq!(e0.dst, nodes[1]);
    assert_eq!(e0.get_property("w").and_then(|v| v.as_int64()), Some(1));
}

#[cfg(feature = "wal")]
#[test]
fn session_transactional_batch_wal_durable_across_reopen() {
    use grafeo_engine::config::StorageFormat;

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("batchdb");

    let committed_ids = {
        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("open");
        let mut session = db.session();
        session.begin_transaction().expect("begin");
        let ids = session
            .create_nodes_with_props_transactional(&[person("A", 1), person("B", 2)])
            .expect("batch create");
        session.commit().expect("commit");
        drop(session);
        db.close().expect("close");
        ids
    };

    let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
    let db = GrafeoDB::with_config(config).expect("reopen");
    assert_eq!(
        db.node_count(),
        committed_ids.len(),
        "WAL replay restores batch"
    );
    for id in committed_ids {
        assert!(db.get_node(id).is_some(), "replayed batch row present");
    }
}

#[cfg(feature = "vector-index")]
#[test]
fn session_transactional_batch_vector_intents_deferred_to_commit() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");

    fn doc(x: f32) -> TransactionalNodeCreate {
        TransactionalNodeCreate::new(["Doc"])
            .with_property("emb", Value::Vector(vec![x, 0.0, 0.0].into()))
    }

    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let ids = session
        .create_nodes_with_props_transactional(&[doc(1.0), doc(0.0)])
        .expect("batch create with vectors");

    let pre = db.vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None);
    assert!(
        pre.as_ref().map_or(true, |r| r.is_empty()),
        "vector index must not see uncommitted batch rows"
    );

    session.commit().expect("commit");

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None)
        .expect("search after commit");
    assert_eq!(results.len(), 2, "both committed vectors become searchable");
    let found: Vec<u64> = results.iter().map(|(id, _)| id.as_u64()).collect();
    for id in &ids {
        assert!(found.contains(&id.as_u64()), "committed vector row indexed");
    }
}
