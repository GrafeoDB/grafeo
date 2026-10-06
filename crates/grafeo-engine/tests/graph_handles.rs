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
    assert!(
        ids(db.execute(ALL_IDS).unwrap()).is_empty(),
        "expected empty"
    );
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
    assert!(
        ids(db.graph("extraction").unwrap().execute(ALL_IDS).unwrap()).is_empty(),
        "expected empty"
    );
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
    session.create_property_index("id").unwrap();
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

    assert!(
        ids(model.execute(ALL_IDS).unwrap()).is_empty(),
        "expected empty"
    );

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

/// A handle's direct calls work in its graph, and once the graph is dropped
/// they fail without writing anywhere else.
#[test]
fn direct_calls_on_a_handle_stay_in_its_graph() {
    let db = db_with_graphs();
    let model = db.graph("model").unwrap();
    let billing = model
        .create_node_with_props(&["Component"], [("id", Value::from("ac::billing"))])
        .unwrap();
    let ledger = model.create_node(&["Component"]).unwrap();
    model
        .set_node_property(ledger, "id", Value::from("ac::ledger"))
        .unwrap();
    let uses = model.create_edge(billing, ledger, "USES").unwrap();
    model
        .set_edge_property(uses, "weight", Value::from(2_i64))
        .unwrap();
    assert!(model.add_node_label(ledger, "Store").unwrap());
    let ids_created = model
        .batch_create_nodes_with_props(
            "Component",
            vec![
                [("id".into(), Value::from("ac::audit"))]
                    .into_iter()
                    .collect(),
            ],
        )
        .unwrap();

    assert_eq!(
        ids(model.execute(ALL_IDS).unwrap()),
        ["ac::audit", "ac::billing", "ac::ledger"]
    );
    assert_eq!(
        model
            .execute("MATCH (:Component)-[r:USES]->(:Store) RETURN r.weight")
            .unwrap()
            .rows(),
        [vec![Value::from(2_i64)]]
    );
    assert_eq!(
        model
            .get_node(ids_created[0])
            .unwrap()
            .unwrap()
            .get_property("id"),
        Some(&Value::from("ac::audit"))
    );
    assert!(model.get_edge(uses).unwrap().is_some());
    assert!(
        ids(db.execute(ALL_IDS).unwrap()).is_empty(),
        "expected empty"
    );
    assert!(
        db.get_node(billing).is_none(),
        "the default graph got nothing"
    );

    db.execute("DROP GRAPH model").unwrap();
    let err = model.create_node(&["Component"]).unwrap_err();
    assert!(err.to_string().contains("does not exist"), "{err}");
    assert!(model.get_node(billing).is_err());
    assert!(
        ids(db.execute(ALL_IDS).unwrap()).is_empty(),
        "expected empty"
    );
}

#[test]
fn a_handle_batch_of_edges_stays_in_its_graph() {
    use grafeo_engine::database::BatchEdge;

    let db = db_with_graphs();
    let model = db.graph("model").unwrap();
    let ids = model
        .batch_create_nodes_with_labels(
            &["Graph", "Component"],
            vec![
                [("id".into(), Value::from("ac::a"))].into_iter().collect(),
                [("id".into(), Value::from("ac::b"))].into_iter().collect(),
            ],
        )
        .unwrap();
    model
        .batch_create_edges(vec![BatchEdge::new(ids[0], ids[1], "USES")])
        .unwrap();
    assert_eq!(
        model
            .execute("MATCH (:Graph:Component)-[r:USES]->() RETURN count(r)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(1)
    );
    assert_eq!(
        db.execute("MATCH ()-[r]->() RETURN count(r)")
            .unwrap()
            .rows()[0][0],
        Value::Int64(0)
    );
}

/// The number of nodes in the default graph.
fn default_graph_nodes(db: &GrafeoDB) -> Value {
    db.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0].clone()
}

fn assert_missing<T: std::fmt::Debug>(result: grafeo_common::utils::error::Result<T>) {
    let err = result.unwrap_err().to_string();
    assert!(err.contains("does not exist"), "{err}");
}

/// A schema's default graph exists only while the schema does: a handle on
/// it neither opens nor writes once the schema is gone.
#[test]
fn a_schema_default_graph_needs_its_schema() {
    let db = GrafeoDB::new_in_memory();
    assert_missing(db.graph_in(Some("missing"), "default"));

    db.execute("CREATE SCHEMA staging").unwrap();
    let staging = db.graph_in(Some("staging"), "default").unwrap();
    staging.execute("INSERT (:File {id: 'f0'})").unwrap();
    db.execute("DROP SCHEMA staging").unwrap();

    assert_missing(staging.execute("INSERT (:File {id: 'f1'})"));
    assert_missing(staging.create_node(&["File"]));
    assert_eq!(default_graph_nodes(&db), Value::Int64(0));
}

/// A session keeps its graph's name: after another session drops the graph,
/// its statements and direct calls fail instead of using the default graph.
#[test]
fn a_session_on_a_dropped_graph_writes_nowhere() {
    let db = db_with_graphs();
    let session = db.graph("model").unwrap().session().unwrap();
    session.execute("INSERT (:Component {id: 'c0'})").unwrap();
    db.execute("DROP GRAPH model").unwrap();

    assert_missing(session.execute("INSERT (:Component {id: 'c1'})"));
    assert_missing(session.execute("MATCH (n) RETURN n.id"));
    assert_missing(session.create_node(&["Component"]));
    assert_eq!(default_graph_nodes(&db), Value::Int64(0));

    // The session can still switch to a graph that exists.
    session.execute("USE GRAPH extraction").unwrap();
    session.execute("INSERT (:File {id: 'f0'})").unwrap();
}

/// The graph `set_current_graph` selects, dropped through another session:
/// the database's own calls fail instead of using the default graph.
#[test]
fn the_selected_graph_dropped_elsewhere() {
    let db = db_with_graphs();
    db.set_current_graph(Some("model")).unwrap();
    db.session().execute("DROP GRAPH model").unwrap();

    assert_missing(db.execute("INSERT (:Component {id: 'c0'})"));
    assert_missing(db.create_node(&["Component"]));
    db.set_current_graph(None).unwrap();
    assert_eq!(default_graph_nodes(&db), Value::Int64(0));
}
