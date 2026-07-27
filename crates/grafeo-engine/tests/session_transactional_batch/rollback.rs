//! Full / savepoint / reopen rollback coverage for session batch APIs.

use super::common::person;
use grafeo_common::types::Value;
use grafeo_engine::session::TransactionalEdgeCreate;
use grafeo_engine::{Config, GrafeoDB};

#[test]
fn session_transactional_batch_rollback_discards_every_row() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    let ids = session
        .create_nodes_with_props_transactional(&[person("A", 1), person("B", 2)])
        .expect("batch create");
    session.rollback().expect("rollback");

    for id in ids {
        assert!(db.get_node(id).is_none(), "rolled-back row must be gone");
    }
    assert_eq!(db.node_count(), 0, "no nodes survive a full rollback");
}

#[test]
fn session_transactional_batch_rollback_cleans_secondaries_and_intents() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("name");

    {
        let mut s = db.session();
        s.begin_transaction().expect("begin baseline");
        s.create_nodes_with_props_transactional(&[person("keep", 9)])
            .expect("baseline");
        s.commit().expect("commit baseline");
    }

    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let nodes = session
        .create_nodes_with_props_transactional(&[person("gone", 1), person("gone2", 2)])
        .expect("nodes");
    let edges = session
        .create_edges_with_props_transactional(&[TransactionalEdgeCreate::new(
            nodes[0], nodes[1], "KNOWS",
        )
        .with_property("w", Value::from(1i64))])
        .expect("edges");
    session.rollback().expect("rollback");

    for id in &nodes {
        assert!(db.get_node(*id).is_none());
    }
    for id in &edges {
        assert!(db.get_edge(*id).is_none());
    }

    assert_eq!(
        db.find_nodes_by_property("name", &Value::from("keep"))
            .len(),
        1,
        "baseline property-index entry must survive"
    );
    assert!(
        db.find_nodes_by_property("name", &Value::from("gone"))
            .is_empty(),
        "property index must drop staged values"
    );
    assert!(
        db.find_nodes_by_property("name", &Value::from("gone2"))
            .is_empty()
    );
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.edge_count(), 0);

    session.begin_transaction().expect("begin again");
    let fresh = session
        .create_nodes_with_props_transactional(&[person("fresh", 3)])
        .expect("fresh");
    assert_ne!(fresh[0], nodes[0]);
    assert_ne!(fresh[0], nodes[1]);
    session.rollback().expect("rollback fresh");
}

#[cfg(feature = "vector-index")]
#[test]
fn session_transactional_batch_rollback_discards_vector_intents() {
    let db = GrafeoDB::new_in_memory();
    db.create_vector_index("Doc", "emb", Some(3), Some("cosine"), None, None, None)
        .expect("create vector index");

    let mut session = db.session();
    session.begin_transaction().expect("begin");
    let _ids = session
        .create_nodes_with_props_transactional(&[
            grafeo_engine::session::TransactionalNodeCreate::new(["Doc"])
                .with_property("emb", Value::Vector(vec![1.0, 0.0, 0.0].into())),
            grafeo_engine::session::TransactionalNodeCreate::new(["Doc"])
                .with_property("emb", Value::Vector(vec![0.0, 1.0, 0.0].into())),
        ])
        .expect("batch");
    session.rollback().expect("rollback");

    let results = db
        .vector_search("Doc", "emb", &[1.0, 0.0, 0.0], 5, None, None)
        .expect("search after rollback");
    assert!(
        results.is_empty(),
        "rolled-back vector intents must not be applied"
    );
    assert_eq!(db.node_count(), 0);
}

#[cfg(feature = "wal")]
#[test]
fn session_transactional_batch_rollback_absent_after_reopen() {
    use grafeo_engine::config::StorageFormat;

    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("batchdb_rollback");

    {
        let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
        let db = GrafeoDB::with_config(config).expect("open");
        let mut session = db.session();
        session.begin_transaction().expect("begin");
        session
            .create_nodes_with_props_transactional(&[person("A", 1), person("B", 2)])
            .expect("batch");
        session.rollback().expect("rollback");
        drop(session);
        db.close().expect("close");
    }

    let config = Config::persistent(&path).with_storage_format(StorageFormat::WalDirectory);
    let db = GrafeoDB::with_config(config).expect("reopen");
    assert_eq!(
        db.node_count(),
        0,
        "rolled-back batch must not resurrect after reopen"
    );
}

#[test]
fn session_transactional_batch_savepoint_cleans_secondaries() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("name");
    let mut session = db.session();

    session.begin_transaction().expect("begin");
    let kept = session
        .create_nodes_with_props_transactional(&[person("keep", 1)])
        .expect("pre-savepoint");
    session.savepoint("sp1").expect("savepoint");

    let staged = session
        .create_nodes_with_props_transactional(&[person("gone", 2), person("gone2", 3)])
        .expect("post-savepoint nodes");
    let edges = session
        .create_edges_with_props_transactional(&[TransactionalEdgeCreate::new(
            staged[0], staged[1], "TEMP",
        )])
        .expect("post-savepoint edges");

    session.rollback_to_savepoint("sp1").expect("rollback sp");

    assert!(db.get_node(kept[0]).is_none(), "still uncommitted");
    assert!(session.get_node(kept[0]).is_some(), "owner sees keep");
    for id in &staged {
        assert!(session.get_node(*id).is_none());
        assert!(db.get_node(*id).is_none());
    }
    for id in &edges {
        assert!(session.get_edge(*id).is_none());
        assert!(db.get_edge(*id).is_none());
    }
    assert!(
        db.find_nodes_by_property("name", &Value::from("gone"))
            .is_empty()
    );
    assert!(
        db.find_nodes_by_property("name", &Value::from("gone2"))
            .is_empty()
    );

    session.commit().expect("commit kept");
    assert_eq!(db.node_count(), 1);
    assert_eq!(db.edge_count(), 0);
    assert_eq!(
        db.find_nodes_by_property("name", &Value::from("keep")),
        vec![kept[0]]
    );
}
