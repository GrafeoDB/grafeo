//! Which graph an algorithm reads (#566): the selected graph, a named graph,
//! or a projection built over the graph selected when it was created.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_engine::{GrafeoDB, ProjectionSpec};

fn with_graph_g() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.execute("CREATE GRAPH g").unwrap();
    db.graph("g")
        .unwrap()
        .execute(
            "INSERT (:Graph {name: 'Gus'}), (:Graph {name: 'Vincent'}), (:Model {name: 'Jules'})",
        )
        .unwrap();
    db
}

#[test]
fn selected_graph_store_follows_set_current_graph() {
    let db = with_graph_g();
    assert_eq!(db.selected_graph_store().unwrap().node_count(), 1);
    db.set_current_graph(Some("g")).unwrap();
    assert_eq!(db.selected_graph_store().unwrap().node_count(), 3);
    db.set_current_graph(None).unwrap();
    assert_eq!(db.selected_graph_store().unwrap().node_count(), 1);
}

#[test]
fn a_graph_handle_reads_its_own_graph() {
    let db = with_graph_g();
    let store = db.graph("g").unwrap().graph_store().unwrap();
    assert_eq!(store.node_count(), 3);
}

#[test]
fn dropping_the_selected_graph_selects_the_default_graph() {
    let db = with_graph_g();
    db.set_current_graph(Some("g")).unwrap();
    db.execute("DROP GRAPH g").unwrap();
    assert_eq!(db.current_graph(), None);
    assert_eq!(db.selected_graph_store().unwrap().node_count(), 1);
}

#[test]
fn a_selected_graph_dropped_elsewhere_is_an_error_not_the_default_graph() {
    let db = with_graph_g();
    db.set_current_graph(Some("g")).unwrap();
    // Another session drops it: the database's selection still names g.
    db.session().execute("DROP GRAPH g").unwrap();
    assert_eq!(db.current_graph().as_deref(), Some("g"));
    let error = db.selected_graph_store().err().unwrap().to_string();
    assert!(error.contains("Graph 'g' does not exist"), "{error}");
    let error = db
        .create_projection("p", ProjectionSpec::new())
        .unwrap_err()
        .to_string();
    assert!(error.contains("Graph 'g' does not exist"), "{error}");
    assert!(db.projection("p").is_none());
}

#[test]
fn create_projection_follows_the_selected_graph_and_keeps_it() {
    let db = with_graph_g();
    db.set_current_graph(Some("g")).unwrap();
    let spec = ProjectionSpec::new().with_node_labels(["Graph"]);
    assert!(db.create_projection("extraction", spec.clone()).unwrap());
    assert!(!db.create_projection("extraction", spec).unwrap());
    db.set_current_graph(None).unwrap();
    // Built over g when it was created, whatever is selected now.
    assert_eq!(db.projection("extraction").unwrap().node_count(), 2);
}

#[test]
fn create_projection_without_a_selection_uses_the_default_graph() {
    let db = with_graph_g();
    assert!(
        db.create_projection("people", ProjectionSpec::new())
            .unwrap()
    );
    assert_eq!(db.projection("people").unwrap().node_count(), 1);
}

/// `CALL grafeo.<algorithm>({projection: ...})` runs on the projection (#566).
#[cfg(feature = "algos")]
#[test]
fn call_runs_an_algorithm_on_a_projection() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Graph {name: 'Alix'})-[:R]->(:Graph {name: 'Gus'}), (:Model {name: 'Mia'})",
    )
    .unwrap();
    db.create_projection(
        "extraction",
        ProjectionSpec::new().with_node_labels(["Graph"]),
    )
    .unwrap();

    let all = db.execute("CALL grafeo.pagerank() YIELD node_id").unwrap();
    assert_eq!(all.rows().len(), 3);
    let scoped = db
        .execute("CALL grafeo.pagerank({projection: 'extraction'}) YIELD node_id")
        .unwrap();
    assert_eq!(scoped.rows().len(), 2);
    let undirected = db
        .execute("CALL grafeo.pagerank({projection: 'extraction', directed: false}) YIELD score")
        .unwrap();
    for row in undirected.rows() {
        assert_eq!(row[0], grafeo_common::types::Value::Float64(0.5));
    }

    let error = db
        .execute("CALL grafeo.pagerank({projection: 'nope'})")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Projection 'nope' does not exist"),
        "{error}"
    );
}

/// A projection stays readable through `CALL` inside a named graph session:
/// projection names are database-wide.
#[cfg(feature = "algos")]
#[test]
fn call_on_a_projection_ignores_the_selected_graph() {
    let db = with_graph_g();
    db.create_projection("people", ProjectionSpec::new())
        .unwrap();
    db.set_current_graph(Some("g")).unwrap();
    let rows = db
        .execute("CALL grafeo.kcore({projection: 'people'}) YIELD node_id")
        .unwrap();
    assert_eq!(rows.rows().len(), 1);
}
