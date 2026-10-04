//! Upserts create or update nodes and edges by a key property, in one
//! statement per call, and report what they did with every row.

#![cfg(all(feature = "lpg", feature = "gql"))]

use std::collections::HashMap;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::{EdgeUpsertOptions, UpsertSummary};

fn row(pairs: &[(&str, Value)]) -> HashMap<PropertyKey, Value> {
    pairs
        .iter()
        .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
        .collect()
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query).unwrap().rows().to_vec()
}

fn summary(created: usize, updated: usize, skipped_rows: &[usize]) -> UpsertSummary {
    UpsertSummary {
        created,
        updated,
        skipped: skipped_rows.len(),
        skipped_rows: skipped_rows.to_vec(),
    }
}

fn files(db: &GrafeoDB) {
    let files = [("f1", 1), ("f2", 2), ("f3", 3)]
        .iter()
        .map(|(id, size)| row(&[("id", Value::from(*id)), ("size", Value::Int64(*size))]))
        .collect();
    db.upsert_nodes(&["Graph", "File"], "id", files, false)
        .unwrap();
}

#[test]
fn nodes_are_created_then_updated_by_key() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .upsert_nodes(
            &["Graph", "File"],
            "id",
            vec![
                row(&[("id", Value::from("f1")), ("size", Value::Int64(3))]),
                row(&[("size", Value::Int64(4))]),
                row(&[("id", Value::from("f2")), ("size", Value::Int64(4))]),
                row(&[("id", Value::from("f1")), ("lang", Value::from("rs"))]),
            ],
            false,
        )
        .unwrap();
    assert_eq!(
        result,
        summary(2, 1, &[1]),
        "the row without the key is skipped"
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (n:Graph:File) RETURN n.id, n.size, n.lang ORDER BY n.id"
        ),
        [
            vec![Value::from("f1"), Value::Int64(3), Value::from("rs")],
            vec![Value::from("f2"), Value::Int64(4), Value::Null],
        ]
    );

    // Merge keeps the properties a row leaves out; replace removes them.
    let again = db
        .upsert_nodes(
            &["Graph", "File"],
            "id",
            vec![row(&[("id", Value::from("f1")), ("size", Value::Int64(5))])],
            false,
        )
        .unwrap();
    assert_eq!(again, summary(0, 1, &[]));
    assert_eq!(
        rows(&db, "MATCH (n {id: 'f1'}) RETURN n.size, n.lang"),
        [vec![Value::Int64(5), Value::from("rs")]]
    );
    db.upsert_nodes(
        &["Graph", "File"],
        "id",
        vec![row(&[("id", Value::from("f1")), ("size", Value::Int64(6))])],
        true,
    )
    .unwrap();
    assert_eq!(
        rows(&db, "MATCH (n {id: 'f1'}) RETURN n.size, n.lang, labels(n)"),
        [vec![
            Value::Int64(6),
            Value::Null,
            Value::List(vec![Value::from("File"), Value::from("Graph")].into()),
        ]]
    );
}

#[test]
fn a_node_matches_only_with_all_the_labels() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Graph {id: 'f1'})").unwrap();
    let result = db
        .upsert_nodes(
            &["Graph", "File"],
            "id",
            vec![row(&[("id", Value::from("f1"))])],
            false,
        )
        .unwrap();
    assert_eq!(result, summary(1, 0, &[]));
    assert_eq!(
        rows(&db, "MATCH (n {id: 'f1'}) RETURN count(n)"),
        [vec![Value::Int64(2)]]
    );
}

#[test]
fn edges_are_created_then_updated_between_existing_nodes() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id");
    files(&db);
    let edge = |src: &str, dst: &str, id: &str, w: i64| {
        row(&[
            ("src", Value::from(src)),
            ("dst", Value::from(dst)),
            ("id", Value::from(id)),
            ("w", Value::Int64(w)),
        ])
    };
    let result = db
        .upsert_edges(
            "Graph:USES",
            vec![
                edge("f1", "f2", "u1", 1),
                edge("f1", "missing", "u2", 1),
                edge("f1", "f2", "u1", 5),
                row(&[("src", Value::from("f2")), ("dst", Value::from("f3"))]),
                edge("f2", "f3", "u3", 2),
            ],
            &EdgeUpsertOptions::default(),
        )
        .unwrap();
    assert_eq!(
        result,
        summary(2, 1, &[1, 3]),
        "a missing endpoint or edge key skips the row"
    );
    assert_eq!(
        rows(
            &db,
            "MATCH (s)-[r]->(d) RETURN type(r), s.id, d.id, r.id, r.w ORDER BY r.id"
        ),
        [
            vec![
                Value::from("Graph:USES"),
                Value::from("f1"),
                Value::from("f2"),
                Value::from("u1"),
                Value::Int64(5),
            ],
            vec![
                Value::from("Graph:USES"),
                Value::from("f2"),
                Value::from("f3"),
                Value::from("u3"),
                Value::Int64(2),
            ],
        ]
    );

    // Replace mode: the edge's properties become the row's, endpoints aside.
    let replace = EdgeUpsertOptions {
        replace: true,
        ..EdgeUpsertOptions::default()
    };
    db.upsert_edges(
        "Graph:USES",
        vec![row(&[
            ("src", Value::from("f1")),
            ("dst", Value::from("f2")),
            ("id", Value::from("u1")),
        ])],
        &replace,
    )
    .unwrap();
    assert_eq!(
        rows(&db, "MATCH ()-[r {id: 'u1'}]->() RETURN properties(r)"),
        [vec![Value::Map(
            [(PropertyKey::new("id"), Value::from("u1"))]
                .into_iter()
                .collect::<std::collections::BTreeMap<_, _>>()
                .into()
        )]]
    );
}

/// A row that names two edges with the same key between the same endpoints
/// (written without an upsert) is written, not skipped: a row that comes back
/// once per edge of one pair of endpoints names no ambiguous endpoint.
#[test]
fn duplicate_keyed_edges_are_written_not_skipped() {
    let db = GrafeoDB::new_in_memory();
    files(&db);
    db.execute(
        "MATCH (a:File {id: 'f1'}), (b:File {id: 'f2'}) \
         INSERT (a)-[:USES {id: 'u1', w: 1}]->(b), (a)-[:USES {id: 'u1', w: 1}]->(b)",
    )
    .unwrap();
    let result = db
        .upsert_edges(
            "USES",
            vec![row(&[
                ("src", Value::from("f1")),
                ("dst", Value::from("f2")),
                ("id", Value::from("u1")),
                ("w", Value::Int64(2)),
            ])],
            &EdgeUpsertOptions::default(),
        )
        .unwrap();
    assert_eq!(result, summary(0, 1, &[]));
    assert!(
        rows(&db, "MATCH ()-[r:USES]->() RETURN r.w").contains(&vec![Value::Int64(2)]),
        "the row updates an edge"
    );
}

/// An endpoint key that more than one node has names no single endpoint:
/// the row is skipped and reported, and writes no edge at all.
#[test]
fn a_row_with_an_ambiguous_endpoint_is_skipped() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id");
    files(&db);
    db.execute("INSERT (:Other {id: 'f2'})").unwrap();
    let edge = |src: &str, dst: &str, id: &str| {
        row(&[
            ("src", Value::from(src)),
            ("dst", Value::from(dst)),
            ("id", Value::from(id)),
        ])
    };
    let result = db
        .upsert_edges(
            "USES",
            vec![
                edge("f1", "f2", "u1"),
                edge("f2", "f3", "u2"),
                edge("f1", "f3", "u3"),
            ],
            &EdgeUpsertOptions::default(),
        )
        .unwrap();
    assert_eq!(result, summary(1, 0, &[0, 1]));
    assert_eq!(
        rows(&db, "MATCH (s)-[r:USES]->(d) RETURN s.id, d.id, r.id"),
        [vec![
            Value::from("f1"),
            Value::from("f3"),
            Value::from("u3")
        ]]
    );

    // Inside a transaction the transaction's own writes stay.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Log {n: 1})").unwrap();
    let result = session
        .upsert_edges(
            "CALLS",
            vec![edge("f1", "f2", "c1"), edge("f3", "f1", "c2")],
            &EdgeUpsertOptions::default(),
        )
        .unwrap();
    assert_eq!(result, summary(1, 0, &[0]));
    session.commit().unwrap();
    assert_eq!(
        rows(&db, "MATCH (l:Log) RETURN l.n"),
        [vec![Value::Int64(1)]]
    );
    assert_eq!(
        rows(&db, "MATCH (s)-[r:CALLS]->(d) RETURN s.id, d.id"),
        [vec![Value::from("f3"), Value::from("f1")]]
    );

    // Without auto-commit and outside a transaction the call is still one
    // write: the undone attempt leaves nothing behind.
    let mut manual = db.session();
    manual.set_auto_commit(false);
    let result = manual
        .upsert_edges(
            "LINKS",
            vec![edge("f1", "f2", "l1"), edge("f3", "f1", "l2")],
            &EdgeUpsertOptions::default(),
        )
        .unwrap();
    assert_eq!(result, summary(1, 0, &[0]));
    assert_eq!(
        rows(&db, "MATCH (s)-[r:LINKS]->(d) RETURN s.id, d.id"),
        [vec![Value::from("f3"), Value::from("f1")]]
    );
}

/// The edge key and the two endpoint fields name three different fields of
/// a row; otherwise one would consume another and every row would be skipped.
#[test]
fn clashing_field_names_are_rejected() {
    let db = GrafeoDB::new_in_memory();
    files(&db);
    let rows_of = || {
        vec![row(&[
            ("src", Value::from("f1")),
            ("dst", Value::from("f2")),
            ("id", Value::from("u1")),
        ])]
    };
    for (key, src_field, dst_field) in [
        ("src", "src", "dst"),
        ("dst", "src", "dst"),
        ("id", "src", "src"),
    ] {
        let options = EdgeUpsertOptions {
            key: key.to_string(),
            src_field: src_field.to_string(),
            dst_field: dst_field.to_string(),
            ..EdgeUpsertOptions::default()
        };
        let error = db.upsert_edges("USES", rows_of(), &options).unwrap_err();
        assert!(
            error.to_string().contains("different fields"),
            "{key} {src_field} {dst_field}: {error}"
        );
    }
    assert_eq!(db.edge_count(), 0);
}

#[test]
fn endpoint_labels_and_field_names_are_configurable() {
    let db = GrafeoDB::new_in_memory();
    files(&db);
    db.execute("INSERT (:Other {id: 'f2'})").unwrap();
    let options = EdgeUpsertOptions {
        key: "rid".to_string(),
        endpoint_labels: vec!["File".to_string()],
        src_field: "from".to_string(),
        dst_field: "to".to_string(),
        ..EdgeUpsertOptions::default()
    };
    let result = db
        .upsert_edges(
            "CALLS",
            vec![row(&[
                ("from", Value::from("f1")),
                ("to", Value::from("f2")),
                ("rid", Value::from("c1")),
            ])],
            &options,
        )
        .unwrap();
    assert_eq!(result, summary(1, 0, &[]));
    assert_eq!(
        rows(&db, "MATCH (:File)-[r:CALLS]->(d) RETURN labels(d), r.rid"),
        [vec![
            Value::List(vec![Value::from("File"), Value::from("Graph")].into()),
            Value::from("c1"),
        ]],
        "the edge goes to the File, not the Other with the same id"
    );
}

#[test]
fn a_constraint_violation_writes_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT file_path FOR (n:File) ON (n.path) UNIQUE")
        .unwrap();
    let err = db
        .upsert_nodes(
            &["File"],
            "id",
            vec![
                row(&[("id", Value::from("f1")), ("path", Value::from("/a"))]),
                row(&[("id", Value::from("f2")), ("path", Value::from("/a"))]),
            ],
            false,
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    assert_eq!(
        rows(&db, "MATCH (n:File) RETURN count(n)"),
        [vec![Value::Int64(0)]]
    );
}

#[test]
fn upserts_follow_the_graph_and_the_transaction() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH model").unwrap();
    let model = db.graph("model").unwrap();
    let result = model
        .upsert_nodes(
            &["Component"],
            "id",
            vec![row(&[("id", Value::from("c1"))])],
            false,
        )
        .unwrap();
    assert_eq!(result, summary(1, 0, &[]));
    assert_eq!(
        model.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0],
        Value::Int64(1)
    );
    assert_eq!(
        rows(&db, "MATCH (n) RETURN count(n)"),
        [vec![Value::Int64(0)]]
    );

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .upsert_nodes(
            &["Doc"],
            "id",
            vec![row(&[("id", Value::from("d1"))])],
            false,
        )
        .unwrap();
    session.rollback().unwrap();
    assert_eq!(
        rows(&db, "MATCH (n:Doc) RETURN count(n)"),
        [vec![Value::Int64(0)]]
    );
}

#[test]
fn empty_names_are_rejected_and_no_rows_do_nothing() {
    let db = GrafeoDB::new_in_memory();
    assert!(db.upsert_nodes(&["File"], "", Vec::new(), false).is_err());
    assert!(
        db.upsert_edges("", Vec::new(), &EdgeUpsertOptions::default())
            .is_err()
    );
    assert_eq!(
        db.upsert_nodes(&["File"], "id", Vec::new(), false).unwrap(),
        UpsertSummary::default()
    );
}
