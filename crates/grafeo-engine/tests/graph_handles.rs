//! A graph handle works in one named graph without switching the database's
//! current graph: handles on different graphs can be used side by side and
//! from several threads, and a handle never falls back to another graph.

#![cfg(feature = "lpg")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Sorted `n.id` of every node the query target sees.
fn ids(result: grafeo_engine::database::QueryResult) -> Vec<String> {
    let mut ids: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(s) => s.to_string(),
            other => panic!("unexpected id {other:?}"),
        })
        .collect();
    ids.sort();
    ids
}

const ALL_IDS: &str = "MATCH (n) RETURN n.id";

fn db_with_graphs() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH extraction").unwrap();
    db.execute("CREATE GRAPH model").unwrap();
    db
}

#[test]
fn handles_interleave_without_switching() {
    let db = db_with_graphs();
    let extraction = db.graph("extraction").unwrap();
    let model = db.graph("model").unwrap();

    let file = extraction
        .session()
        .unwrap()
        .create_node_with_props(&["File"], [("id", Value::from("file::a"))])
        .unwrap();
    model
        .session()
        .unwrap()
        .create_node_with_props(&["Component"], [("id", Value::from("ac::a"))])
        .unwrap();
    extraction
        .session()
        .unwrap()
        .set_node_property(file, "size", Value::from(3_i64))
        .unwrap();
    model.execute("INSERT (:Component {id: 'ac::b'})").unwrap();

    assert_eq!(ids(extraction.execute(ALL_IDS).unwrap()), ["file::a"]);
    assert_eq!(ids(model.execute(ALL_IDS).unwrap()), ["ac::a", "ac::b"]);
    assert!(ids(db.execute(ALL_IDS).unwrap()).is_empty());
    assert_eq!(db.current_graph(), None);
    assert_eq!(
        extraction
            .session()
            .unwrap()
            .get_node(file)
            .unwrap()
            .properties
            .get(&"size".into()),
        Some(&Value::from(3_i64))
    );
    // The default graph is empty, so it has no node with that id.
    assert!(db.get_node(file).is_none());
}

#[test]
fn handles_are_safe_across_threads() {
    let db = db_with_graphs();
    let extraction = db.graph("extraction").unwrap();
    let model = db.graph("model").unwrap();

    std::thread::scope(|scope| {
        for (handle, prefix) in [
            (&extraction, "g"),
            (&model, "m"),
            (&extraction, "h"),
            (&model, "n"),
        ] {
            scope.spawn(move || {
                for i in 0..200 {
                    handle
                        .session()
                        .unwrap()
                        .create_node_with_props(
                            &["N"],
                            [("id", Value::from(format!("{prefix}{i}")))],
                        )
                        .unwrap();
                }
            });
        }
    });

    let first_letters = |ids: Vec<String>| {
        let mut letters: Vec<char> = ids.iter().filter_map(|id| id.chars().next()).collect();
        letters.dedup();
        (ids.len(), letters)
    };
    assert_eq!(
        first_letters(ids(extraction.execute(ALL_IDS).unwrap())),
        (400, vec!['g', 'h'])
    );
    assert_eq!(
        first_letters(ids(model.execute(ALL_IDS).unwrap())),
        (400, vec!['m', 'n'])
    );
}

#[test]
fn a_missing_graph_is_an_error_not_a_fallback() {
    let db = db_with_graphs();
    let err = db.graph("nowhere").unwrap_err();
    assert!(
        err.to_string().contains("'nowhere' does not exist"),
        "{err}"
    );

    let model = db.graph("model").unwrap();
    db.execute("DROP GRAPH model").unwrap();
    assert!(model.session().is_err());
    assert!(model.execute("INSERT (:Stray {id: 'x'})").is_err());
    assert!(
        ids(db.execute(ALL_IDS).unwrap()).is_empty(),
        "nothing reached the default graph"
    );
}

#[test]
fn use_graph_in_a_query_stays_in_that_query() {
    let db = db_with_graphs();
    let model = db.graph("model").unwrap();
    model.execute("USE GRAPH extraction").unwrap();
    model.execute("INSERT (:Component {id: 'ac::a'})").unwrap();

    assert_eq!(ids(model.execute(ALL_IDS).unwrap()), ["ac::a"]);
    assert!(ids(db.graph("extraction").unwrap().execute(ALL_IDS).unwrap()).is_empty());
    assert_eq!(db.current_graph(), None);
}

#[test]
fn a_handle_ignores_the_database_graph() {
    let db = db_with_graphs();
    db.set_current_graph(Some("extraction")).unwrap();
    let model = db.graph("model").unwrap();
    model.execute("INSERT (:Component {id: 'ac::a'})").unwrap();

    assert_eq!(ids(model.execute(ALL_IDS).unwrap()), ["ac::a"]);
    assert!(
        ids(db.execute(ALL_IDS).unwrap()).is_empty(),
        "extraction stays empty"
    );
    assert_eq!(db.current_graph().as_deref(), Some("extraction"));
}

#[test]
fn property_indexes_and_lookups_are_per_graph() {
    let db = db_with_graphs();
    let model = db.graph("model").unwrap();
    let extraction = db.graph("extraction").unwrap();
    let session = model.session().unwrap();
    session.create_property_index("id");
    let component = session
        .create_node_with_props(&["Component"], [("id", Value::from("x"))])
        .unwrap();
    extraction
        .session()
        .unwrap()
        .create_node_with_props(&["File"], [("id", Value::from("x"))])
        .unwrap();

    let model_session = model.session().unwrap();
    assert!(model_session.has_property_index("id"));
    assert!(!extraction.session().unwrap().has_property_index("id"));
    assert!(!db.has_property_index("id"));
    assert_eq!(
        model_session.find_nodes_by_property("id", &Value::from("x")),
        [component]
    );
}

#[test]
fn a_transaction_on_a_handle_session_groups_writes() {
    let db = db_with_graphs();
    let model = db.graph("model").unwrap();
    let mut session = model.session().unwrap();
    session.begin_transaction().unwrap();
    session
        .create_node_with_props(&["Component"], [("id", Value::from("ac::a"))])
        .unwrap();
    session
        .execute("INSERT (:Component {id: 'ac::b'})")
        .unwrap();
    assert_eq!(ids(session.execute(ALL_IDS).unwrap()), ["ac::a", "ac::b"]);
    session.rollback().unwrap();

    assert!(ids(model.execute(ALL_IDS).unwrap()).is_empty());

    let ids_created = model
        .session()
        .unwrap()
        .batch_create_nodes_with_props(
            "Component",
            vec![
                std::collections::HashMap::from([("id".into(), Value::from("ac::c"))]),
                std::collections::HashMap::from([("id".into(), Value::from("ac::d"))]),
            ],
        )
        .unwrap();
    assert_eq!(ids_created.len(), 2);
    assert_eq!(ids(model.execute(ALL_IDS).unwrap()), ["ac::c", "ac::d"]);
}

/// Node 0 of one graph and node 0 of another are different nodes: open
/// transactions writing them do not conflict.
#[test]
fn the_same_id_in_two_graphs_does_not_conflict() {
    let db = db_with_graphs();
    let mut in_model = db.graph("model").unwrap().session().unwrap();
    let mut in_default = db.session();
    in_model.begin_transaction().unwrap();
    in_default.begin_transaction().unwrap();

    let model_node = in_model.create_node(&["Component"]).unwrap();
    let default_node = in_default.create_node(&["Person"]).unwrap();
    assert_eq!(model_node, default_node, "both graphs start numbering at 0");
    in_model
        .set_node_property(model_node, "id", Value::from("ac::a"))
        .unwrap();
    in_default
        .set_node_property(default_node, "id", Value::from("alix"))
        .unwrap();

    in_model.commit().unwrap();
    in_default.commit().unwrap();
    assert_eq!(
        ids(db.graph("model").unwrap().execute(ALL_IDS).unwrap()),
        ["ac::a"]
    );
    assert_eq!(ids(db.execute(ALL_IDS).unwrap()), ["alix"]);
}

/// Within one graph, two open transactions writing the same node still
/// conflict.
#[test]
fn the_same_node_in_one_graph_still_conflicts() {
    let db = db_with_graphs();
    let model = db.graph("model").unwrap();
    let node = model
        .session()
        .unwrap()
        .create_node(&["Component"])
        .unwrap();

    let mut first = model.session().unwrap();
    let mut second = model.session().unwrap();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    first
        .set_node_property(node, "name", Value::from("Billing"))
        .unwrap();
    let err = second
        .set_node_property(node, "name", Value::from("Ledger"))
        .unwrap_err();
    assert!(err.to_string().contains("in graph 'model'"), "{err}");
}
