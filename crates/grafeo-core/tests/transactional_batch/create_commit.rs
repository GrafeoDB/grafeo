//! Create / finalize / equivalence coverage for storage-level batch writes.

use super::common::{TX, node};
use grafeo_common::types::{PropertyKey, Value};
use grafeo_core::graph::lpg::{BatchEdgeCreate, BatchNodeCreate, LpgStore};

#[test]
fn transactional_batch_nodes_preserve_order_labels_and_props() {
    let store = LpgStore::new().unwrap();
    let epoch = store.new_epoch();

    let input = vec![
        node(
            &["Person"],
            vec![("name", Value::from("A")), ("age", Value::from(1i64))],
        ),
        node(&["Person", "Admin"], vec![("name", Value::from("B"))]),
        node(&["Robot"], vec![]),
    ];

    let ids = store.create_nodes_batch_versioned(&input, epoch, TX);
    assert_eq!(ids.len(), 3, "one id per input node");

    assert_eq!(ids[1].0, ids[0].0 + 1);
    assert_eq!(ids[2].0, ids[1].0 + 1);

    let commit = store.new_epoch();
    store.finalize_version_epochs(TX, commit);

    let n0 = store.get_node(ids[0]).unwrap();
    assert!(n0.has_label("Person"));
    assert_eq!(n0.get_property("name").and_then(|v| v.as_str()), Some("A"));
    assert_eq!(n0.get_property("age").and_then(|v| v.as_int64()), Some(1));

    let n1 = store.get_node(ids[1]).unwrap();
    assert!(n1.has_label("Person"));
    assert!(n1.has_label("Admin"));
    assert_eq!(n1.get_property("name").and_then(|v| v.as_str()), Some("B"));

    let n2 = store.get_node(ids[2]).unwrap();
    assert!(n2.has_label("Robot"));
    assert!(n2.properties.is_empty());

    assert_eq!(store.node_count(), 3);
}

#[test]
fn transactional_batch_edges_preserve_order_type_and_props() {
    let store = LpgStore::new().unwrap();
    let epoch = store.new_epoch();
    let nodes = store.create_nodes_batch_versioned(
        &[
            node(&["N"], vec![]),
            node(&["N"], vec![]),
            node(&["N"], vec![]),
        ],
        epoch,
        TX,
    );
    let commit = store.new_epoch();
    store.finalize_version_epochs(TX, commit);

    let epoch2 = store.new_epoch();
    let edges = vec![
        BatchEdgeCreate {
            source: nodes[0],
            target: nodes[1],
            edge_type: "KNOWS",
            properties: vec![(PropertyKey::new("since"), Value::from(2020i64))],
        },
        BatchEdgeCreate {
            source: nodes[1],
            target: nodes[2],
            edge_type: "LIKES",
            properties: vec![],
        },
    ];
    let eids = store.create_edges_batch_versioned(&edges, epoch2, TX);
    assert_eq!(eids.len(), 2);
    assert_eq!(eids[1].0, eids[0].0 + 1);

    let commit2 = store.new_epoch();
    store.finalize_version_epochs(TX, commit2);

    let e0 = store.get_edge(eids[0]).unwrap();
    assert_eq!(e0.src, nodes[0]);
    assert_eq!(e0.dst, nodes[1]);
    assert_eq!(e0.edge_type, "KNOWS");
    assert_eq!(
        e0.get_property("since").and_then(|v| v.as_int64()),
        Some(2020)
    );

    let e1 = store.get_edge(eids[1]).unwrap();
    assert_eq!(e1.edge_type, "LIKES");
    assert_eq!(store.edge_count(), 2);
}

#[test]
fn transactional_batch_pending_invisible_until_finalize() {
    let store = LpgStore::new().unwrap();
    let epoch = store.new_epoch();
    let ids = store.create_nodes_batch_versioned(
        &[node(&["P"], vec![("k", Value::from(1i64))])],
        epoch,
        TX,
    );

    // PENDING rows are invisible to other sessions (current-epoch read).
    assert!(store.get_node(ids[0]).is_none());
    assert!(store.get_node_versioned(ids[0], epoch, TX).is_some());

    let commit = store.new_epoch();
    store.finalize_version_epochs(TX, commit);
    assert!(store.get_node(ids[0]).is_some());
}

#[test]
fn transactional_batch_updates_property_index() {
    let store = LpgStore::new().unwrap();
    store.create_property_index("name");
    let epoch = store.new_epoch();

    let input = vec![
        node(&["P"], vec![("name", Value::from("match"))]),
        node(&["P"], vec![("name", Value::from("other"))]),
        node(&["P"], vec![("name", Value::from("match"))]),
    ];
    let ids = store.create_nodes_batch_versioned(&input, epoch, TX);
    let commit = store.new_epoch();
    store.finalize_version_epochs(TX, commit);

    let mut hits = store.find_nodes_by_property("name", &Value::from("match"));
    hits.sort_by_key(|n| n.0);
    let mut expected = vec![ids[0], ids[2]];
    expected.sort_by_key(|n| n.0);
    assert_eq!(
        hits, expected,
        "property index must reflect batch-created rows"
    );
}

#[test]
fn transactional_batch_equivalent_to_row_path() {
    let labels: &[&str] = &["Person"];
    let rows: Vec<(&str, i64)> = vec![("a", 1), ("b", 2), ("c", 3), ("d", 4)];

    let batched = LpgStore::new().unwrap();
    let epoch = batched.new_epoch();
    let input: Vec<BatchNodeCreate<'_>> = rows
        .iter()
        .map(|(name, age)| {
            node(
                labels,
                vec![("name", Value::from(*name)), ("age", Value::from(*age))],
            )
        })
        .collect();
    let b_ids = batched.create_nodes_batch_versioned(&input, epoch, TX);
    let bc = batched.new_epoch();
    batched.finalize_version_epochs(TX, bc);

    let rowed = LpgStore::new().unwrap();
    let repoch = rowed.new_epoch();
    let mut r_ids = Vec::new();
    for (name, age) in &rows {
        let id = rowed.create_node_with_props_versioned(
            labels,
            [("name", Value::from(*name)), ("age", Value::from(*age))],
            repoch,
            TX,
        );
        r_ids.push(id);
    }
    let rc = rowed.new_epoch();
    rowed.finalize_version_epochs(TX, rc);

    assert_eq!(batched.node_count(), rowed.node_count());
    assert_eq!(b_ids.len(), r_ids.len());
    for (b, r) in b_ids.iter().zip(r_ids.iter()) {
        let bn = batched.get_node(*b).unwrap();
        let rn = rowed.get_node(*r).unwrap();
        assert_eq!(bn.labels, rn.labels);
        assert_eq!(bn.properties.len(), rn.properties.len());
        for (key, value) in bn.properties.iter() {
            assert_eq!(rn.properties.get(key), Some(value), "mismatch on {key:?}");
        }
    }
}

#[test]
fn transactional_batch_empty_input_is_noop() {
    let store = LpgStore::new().unwrap();
    let epoch = store.new_epoch();
    let n = store.create_nodes_batch_versioned(&[], epoch, TX);
    let e = store.create_edges_batch_versioned(&[], epoch, TX);
    assert!(n.is_empty());
    assert!(e.is_empty());
    assert_eq!(store.node_count(), 0);
    assert_eq!(store.edge_count(), 0);
}
