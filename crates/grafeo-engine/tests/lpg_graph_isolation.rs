//! Integration tests for LPG named graph data isolation (issue #133).
//!
//! Verifies that `USE GRAPH`, `SESSION SET SCHEMA`, and `SESSION SET GRAPH`
//! correctly route queries and mutations to the selected named graph,
//! not the default store.

use grafeo_engine::GrafeoDB;

fn db() -> GrafeoDB {
    GrafeoDB::new_in_memory()
}

// ── Basic data isolation ─────────────────────────────────────────

#[test]
fn use_graph_isolates_inserts() {
    let db = db();
    let session = db.session();

    // Insert into default graph
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Create and switch to named graph
    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();

    // Insert into named graph
    session.execute("INSERT (:Event {type: 'click'})").unwrap();

    // MATCH in named graph should only see Event, not Person
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Named graph should only have 1 node (Event)"
    );
}

#[test]
fn default_graph_unchanged_after_named_graph_insert() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("USE GRAPH analytics").unwrap();
    session.execute("INSERT (:Event {type: 'click'})").unwrap();

    // Switch back to default
    session.execute("USE GRAPH default").unwrap();

    // Default should have no data
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Default graph should be empty after inserting into named graph"
    );
}

#[test]
fn session_set_schema_isolates_data() {
    // ISO/IEC 39075 Section 7.1: SESSION SET SCHEMA sets schema independently.
    // Graphs created within a schema are isolated from the default schema.
    let db = db();
    let session = db.session();

    // Create a schema and a graph within it
    session.execute("CREATE SCHEMA reports").unwrap();
    session.execute("SESSION SET SCHEMA reports").unwrap();
    session.execute("CREATE GRAPH quarterly").unwrap();
    session.execute("SESSION SET GRAPH quarterly").unwrap();

    session.execute("INSERT (:Report {title: 'Q1'})").unwrap();

    let result = session.execute("MATCH (n:Report) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "Should see the Report node");

    // Reset session back to default (clears both schema and graph)
    session.execute("SESSION RESET").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Default graph should have no data after SESSION RESET"
    );
}

#[test]
fn session_set_graph_isolates_data() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH mydb").unwrap();
    session.execute("SESSION SET GRAPH mydb").unwrap();

    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1);

    // Switch back to default
    session.execute("SESSION SET GRAPH default").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 0, "Default graph should be empty");
}

// ── Cross-graph isolation ────────────────────────────────────────

#[test]
fn two_named_graphs_are_independent() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();
    session.execute("CREATE GRAPH beta").unwrap();

    // Insert into alpha
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Insert into beta
    session.execute("USE GRAPH beta").unwrap();
    session
        .execute("INSERT (:Animal {species: 'Cat'})")
        .unwrap();

    // beta should have 1 node
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "beta should have 1 node");

    // alpha should have 2 nodes
    session.execute("USE GRAPH alpha").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 2, "alpha should have 2 nodes");

    // default should have 0 nodes
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 0, "default should have 0 nodes");
}

// ── Cross-graph transactions ─────────────────────────────────────

#[test]
fn cross_graph_commit_persists_both_graphs() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();
    session.execute("CREATE GRAPH beta").unwrap();

    session.execute("START TRANSACTION").unwrap();

    // Insert into default graph
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Switch to alpha and insert
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Event {type: 'login'})").unwrap();

    // Switch to beta and insert
    session.execute("USE GRAPH beta").unwrap();
    session
        .execute("INSERT (:Animal {species: 'Cat'})")
        .unwrap();

    session.execute("COMMIT").unwrap();

    // Verify all three graphs have their data
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "default should have 1 node");

    session.execute("USE GRAPH alpha").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "alpha should have 1 node");

    session.execute("USE GRAPH beta").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "beta should have 1 node");
}

#[test]
fn cross_graph_rollback_discards_all_graphs() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();

    session.execute("START TRANSACTION").unwrap();

    // Insert into default
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Switch and insert into alpha
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Event {type: 'login'})").unwrap();

    session.execute("ROLLBACK").unwrap();

    // Both graphs should be empty
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "default should be empty after rollback"
    );

    session.execute("USE GRAPH alpha").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "alpha should be empty after rollback"
    );
}

#[test]
fn session_set_graph_works_in_transaction() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH analytics").unwrap();
    session.execute("START TRANSACTION").unwrap();

    // SESSION SET GRAPH should work within a transaction now
    session.execute("SESSION SET GRAPH analytics").unwrap();
    session.execute("INSERT (:Event {type: 'click'})").unwrap();
    session.execute("COMMIT").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "analytics should have 1 node");
}

#[test]
fn cross_graph_savepoint_rollback() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH alpha").unwrap();

    session.execute("START TRANSACTION").unwrap();

    // Insert into default
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Create savepoint while on default graph
    session.execute("SAVEPOINT sp1").unwrap();

    // Switch to alpha and insert
    session.execute("USE GRAPH alpha").unwrap();
    session.execute("INSERT (:Event {type: 'login'})").unwrap();

    // Also insert more into default
    session.execute("USE GRAPH default").unwrap();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    // Rollback to savepoint: should discard alpha's Event and default's Gus
    session.execute("ROLLBACK TO SAVEPOINT sp1").unwrap();

    session.execute("COMMIT").unwrap();

    // Default should only have Alix (from before savepoint)
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "default should have 1 node (Alix)");

    // Alpha should be empty (its insert was after savepoint)
    session.execute("USE GRAPH alpha").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "alpha should be empty after savepoint rollback"
    );
}

// ── Drop active graph ────────────────────────────────────────────

#[test]
fn drop_active_graph_resets_to_default() {
    let db = db();
    let session = db.session();

    // Create graph, switch to it, insert data
    session.execute("CREATE GRAPH temp").unwrap();
    session.execute("USE GRAPH temp").unwrap();
    session
        .execute("INSERT (:Temp {val: 'ephemeral'})")
        .unwrap();

    // Drop the graph we're currently on
    session.execute("DROP GRAPH temp").unwrap();

    // Should now be on default graph (which is empty)
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "After dropping active graph, should fall back to empty default"
    );
}

// ── Direct CRUD isolation ────────────────────────────────────────

#[test]
fn session_create_node_respects_active_graph() {
    let db = db();
    let session = db.session();

    session.execute("CREATE GRAPH mydb").unwrap();
    session.execute("USE GRAPH mydb").unwrap();

    // Direct CRUD via session
    session.create_node(&["Widget"]).unwrap();

    let result = session.execute("MATCH (n:Widget) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Direct create_node should write to active graph"
    );

    // Default should be empty
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Default graph should not have the Widget node"
    );
}

// ── Query cache isolation ────────────────────────────────────────

#[test]
fn same_query_different_graph_returns_correct_results() {
    let db = db();
    let session = db.session();

    // Insert data into default
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    // Execute query on default graph (warms cache)
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "Default graph has 1 node");

    // Create and switch to empty named graph
    session.execute("CREATE GRAPH empty").unwrap();
    session.execute("USE GRAPH empty").unwrap();

    // Same query text, different graph, should not return cached default result
    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        0,
        "Empty named graph should return 0 rows, not cached default result"
    );
}

// ── SESSION RESET restores default ───────────────────────────────

#[test]
fn session_reset_returns_to_default_graph() {
    let db = db();
    let session = db.session();

    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();

    session.execute("CREATE GRAPH other").unwrap();
    session.execute("USE GRAPH other").unwrap();

    // SESSION RESET should go back to default
    session.execute("SESSION RESET").unwrap();

    let result = session.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "After SESSION RESET, should see default graph data"
    );
}

// ── Edge isolation ───────────────────────────────────────────────

#[test]
fn edges_are_isolated_between_graphs() {
    let db = db();
    let session = db.session();

    // Create edge in default
    session
        .execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();

    session.execute("CREATE GRAPH social").unwrap();
    session.execute("USE GRAPH social").unwrap();

    // Insert different edge in named graph
    session
        .execute("INSERT (:Person {name: 'Vincent'})-[:WORKS_WITH]->(:Person {name: 'Jules'})")
        .unwrap();

    // Named graph should only see WORKS_WITH
    let result = session.execute("MATCH ()-[r]->() RETURN type(r)").unwrap();
    assert_eq!(result.row_count(), 1, "social graph should have 1 edge");

    // Default graph should only see KNOWS
    session.execute("USE GRAPH default").unwrap();
    let result = session.execute("MATCH ()-[r]->() RETURN type(r)").unwrap();
    assert_eq!(result.row_count(), 1, "default graph should have 1 edge");
}

// ── SHOW GRAPHS ─────────────────────────────────────────────────

#[test]
fn show_graphs_empty() {
    let db = db();
    let session = db.session();
    let result = session.execute("SHOW GRAPHS").unwrap();
    assert_eq!(result.columns, vec!["name"]);
    assert_eq!(result.row_count(), 0, "no named graphs initially");
}

#[test]
fn show_graphs_lists_created_graphs() {
    use grafeo_common::types::Value;

    let db = db();
    let session = db.session();
    session.execute("CREATE GRAPH beta").unwrap();
    session.execute("CREATE GRAPH alpha").unwrap();

    let result = session.execute("SHOW GRAPHS").unwrap();
    assert_eq!(result.columns, vec!["name"]);
    assert_eq!(result.row_count(), 2);
    // Results should be sorted alphabetically
    assert_eq!(result.rows()[0][0], Value::String("alpha".into()));
    assert_eq!(result.rows()[1][0], Value::String("beta".into()));
}

#[test]
fn show_graphs_reflects_drop() {
    let db = db();
    let session = db.session();
    session.execute("CREATE GRAPH temp").unwrap();
    assert_eq!(session.execute("SHOW GRAPHS").unwrap().row_count(), 1);

    session.execute("DROP GRAPH temp").unwrap();
    assert_eq!(session.execute("SHOW GRAPHS").unwrap().row_count(), 0);
}

// ── GrafeoDB-level graph context persistence ────────────────────

#[test]
fn db_execute_use_graph_persists_across_calls() {
    let db = db();
    db.execute("CREATE GRAPH analytics").unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap(); // default graph
    db.execute("USE GRAPH analytics").unwrap();
    db.execute("INSERT (:Event {type: 'click'})").unwrap(); // analytics graph

    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Named graph should have 1 node (Event)"
    );

    // Switch back to default
    db.execute("USE GRAPH default").unwrap();
    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Default graph should have 1 node (Person)"
    );
}

#[test]
fn db_execute_session_set_graph_persists() {
    let db = db();
    db.execute("CREATE GRAPH mydb").unwrap();
    db.execute("SESSION SET GRAPH mydb").unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1);

    db.execute("SESSION SET GRAPH default").unwrap();
    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 0, "Default graph should be empty");
}

#[test]
fn db_execute_session_reset_returns_to_default() {
    let db = db();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.execute("CREATE GRAPH other").unwrap();
    db.execute("USE GRAPH other").unwrap();
    db.execute("SESSION RESET").unwrap();

    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Should see default graph data after SESSION RESET"
    );
}

#[test]
fn db_current_graph_api() {
    let db = db();
    assert_eq!(db.current_graph(), None);

    db.execute("CREATE GRAPH mydb").unwrap();
    db.execute("USE GRAPH mydb").unwrap();
    assert_eq!(db.current_graph(), Some("mydb".to_string()));

    db.set_current_graph(None).unwrap();
    assert_eq!(db.current_graph(), None);

    db.set_current_graph(Some("mydb")).unwrap();
    assert_eq!(db.current_graph(), Some("mydb".to_string()));
}

#[test]
fn set_current_graph_rejects_nonexistent() {
    let db = db();
    let err = db.set_current_graph(Some("nope")).unwrap_err();
    assert!(
        err.to_string().contains("does not exist"),
        "expected 'does not exist' error, got: {err}"
    );
    // context should remain unchanged
    assert_eq!(db.current_graph(), None);
}

#[test]
fn set_current_graph_allows_default() {
    let db = db();
    // "default" is always valid even without creating it
    db.set_current_graph(Some("default")).unwrap();
    // setting None clears it
    db.set_current_graph(None).unwrap();
    assert_eq!(db.current_graph(), None);
}

#[test]
fn drop_graph_resets_active_context() {
    let db = db();
    db.execute("CREATE GRAPH ephemeral").unwrap();
    db.set_current_graph(Some("ephemeral")).unwrap();
    assert_eq!(db.current_graph(), Some("ephemeral".to_string()));

    db.drop_graph("ephemeral");
    assert_eq!(
        db.current_graph(),
        None,
        "current_graph should reset after dropping the active graph"
    );
}

#[test]
fn drop_graph_preserves_context_for_other_graph() {
    let db = db();
    db.execute("CREATE GRAPH keep").unwrap();
    db.execute("CREATE GRAPH other").unwrap();
    db.set_current_graph(Some("keep")).unwrap();

    db.drop_graph("other");
    assert_eq!(
        db.current_graph(),
        Some("keep".to_string()),
        "dropping a different graph should not touch current_graph"
    );
}

#[test]
fn set_current_schema_rejects_nonexistent() {
    let db = db();
    let err = db.set_current_schema(Some("fake")).unwrap_err();
    assert!(
        err.to_string().contains("does not exist"),
        "expected 'does not exist' error, got: {err}"
    );
    assert_eq!(db.current_schema(), None);
}

#[test]
fn set_current_schema_accepts_valid() {
    let db = db();
    db.execute("CREATE SCHEMA reporting").unwrap();
    db.set_current_schema(Some("reporting")).unwrap();
    assert_eq!(db.current_schema(), Some("reporting".to_string()));

    db.set_current_schema(None).unwrap();
    assert_eq!(db.current_schema(), None);
}

#[test]
fn db_execute_two_named_graphs_independent() {
    let db = db();
    db.execute("CREATE GRAPH alpha").unwrap();
    db.execute("CREATE GRAPH beta").unwrap();

    db.execute("USE GRAPH alpha").unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();

    db.execute("USE GRAPH beta").unwrap();
    db.execute("INSERT (:Animal {species: 'Cat'})").unwrap();

    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 1, "beta should have 1 node");

    db.execute("USE GRAPH alpha").unwrap();
    let result = db.execute("MATCH (n) RETURN n").unwrap();
    assert_eq!(result.row_count(), 2, "alpha should have 2 nodes");
}

#[test]
fn db_execute_language_respects_graph_context() {
    let db = db();
    db.execute("CREATE GRAPH mydb").unwrap();
    db.execute("USE GRAPH mydb").unwrap();
    db.execute("INSERT (:Widget {name: 'Gadget'})").unwrap();

    // execute_language should also see the graph context
    let result = db
        .execute_language("MATCH (n:Widget) RETURN n", "gql", None)
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "execute_language should respect graph context"
    );

    // Default should be empty
    db.execute("USE GRAPH default").unwrap();
    let result = db
        .execute_language("MATCH (n) RETURN n", "gql", None)
        .unwrap();
    assert_eq!(result.row_count(), 0);
}

// ── Concurrent named graph isolation (T3-19) ────────────────────

#[test]
fn concurrent_sessions_on_different_graphs() {
    let db = db();
    db.execute("CREATE GRAPH alpha").unwrap();
    db.execute("CREATE GRAPH beta").unwrap();

    let s1 = db.session();
    let s2 = db.session();

    s1.execute("USE GRAPH alpha").unwrap();
    s2.execute("USE GRAPH beta").unwrap();

    s1.execute("INSERT (:Item {name: 'widget'})").unwrap();
    s2.execute("INSERT (:Item {name: 'gadget'})").unwrap();

    let r1 = s1.execute("MATCH (i:Item) RETURN i.name").unwrap();
    let r2 = s2.execute("MATCH (i:Item) RETURN i.name").unwrap();

    assert_eq!(r1.rows().len(), 1, "alpha should have 1 item");
    assert_eq!(r2.rows().len(), 1, "beta should have 1 item");

    // Cross-check: each graph has its own data
    assert_eq!(
        r1.rows()[0][0],
        grafeo_common::types::Value::String("widget".into())
    );
    assert_eq!(
        r2.rows()[0][0],
        grafeo_common::types::Value::String("gadget".into())
    );
}
