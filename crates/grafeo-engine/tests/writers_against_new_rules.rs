//! A commit whose writes were checked before a rule changed is checked again against every kind of
//! rule, for every statement that tightens one; and a new rule on endpoints reads the edges there.
//!
//! Tests first: today the recheck reads only required properties and uniqueness, and only a new
//! type or constraint makes a commit recheck at all, so these commits go through and a committed
//! row breaks a committed rule.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main writers_against_new_rules::
//! ```

use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

fn count(session: &Session, query: &str) -> usize {
    session.execute(query).unwrap().rows().len()
}

/// Opens a writer on `graph`, writes `write`, runs `rule` from another session, and returns the
/// database with the writer's commit failing as a rule's violation, and its write gone.
fn the_commit_fails(setup: &[&str], graph: Option<&str>, write: &str, rule: &str, rows: &str) {
    let db = GrafeoDB::new_in_memory();
    run_all(&db.session(), setup);
    let mut writer = db.session();
    if let Some(graph) = graph {
        writer.execute(&format!("USE GRAPH {graph}")).unwrap();
    }
    writer.begin_transaction().unwrap();
    writer
        .execute(write)
        .unwrap_or_else(|error| panic!("{write}: {error}"));
    db.session()
        .execute(rule)
        .unwrap_or_else(|error| panic!("{rule}: {error}"));
    let error = writer
        .commit()
        .expect_err(&format!("{write}, then {rule}: the commit went through"));
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{rule}: {error}");
    assert_eq!(count(&writer, rows), 0, "{rule}: the write stays");
}

// ---------------------------------------------------------------------------
// What the recheck reads (catalog/mod.rs:2738)
// ---------------------------------------------------------------------------

#[test]
fn a_value_of_another_type_than_a_new_rule_fails_the_commit() {
    the_commit_fails(
        &[],
        None,
        "INSERT (:City {size: 'large'})",
        "CREATE NODE TYPE City (size INT64)",
        "MATCH (c:City) RETURN c.size",
    );
}

#[test]
fn an_edge_value_of_another_type_than_a_new_rule_fails_the_commit() {
    the_commit_fails(
        &[],
        None,
        "INSERT (:City)-[:ROAD {km: 'far'}]->(:City)",
        "CREATE EDGE TYPE ROAD (km INT64)",
        "MATCH ()-[r:ROAD]->() RETURN r.km",
    );
}

#[test]
fn an_edge_between_ends_a_new_rule_refuses_fails_the_commit() {
    the_commit_fails(
        &[],
        None,
        "INSERT (:Town)-[:ROAD]->(:Town)",
        "CREATE EDGE TYPE ROAD CONNECTING (City) TO (City)",
        "MATCH ()-[r:ROAD]->() RETURN id(r)",
    );
}

/// A closed type: the property is undeclared once the type takes it away.
#[test]
fn a_property_a_new_rule_no_longer_declares_fails_the_commit() {
    the_commit_fails(
        &[
            "CREATE NODE TYPE City (name STRING, size INT64)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City)",
            "CREATE GRAPH maps TYPED atlas",
        ],
        Some("maps"),
        "INSERT (:City {name: 'Paris', size: 3})",
        "ALTER NODE TYPE City DROP PROPERTY size",
        "MATCH (c:City) RETURN c.name",
    );
}

// ---------------------------------------------------------------------------
// What makes a commit recheck (catalog/mod.rs:45)
// ---------------------------------------------------------------------------

#[test]
fn a_new_required_property_of_a_node_type_fails_the_commit() {
    the_commit_fails(
        &["CREATE NODE TYPE City (name STRING)"],
        None,
        "INSERT (:City {name: 'Paris'})",
        "ALTER NODE TYPE City ADD PROPERTY size INTEGER NOT NULL",
        "MATCH (c:City) RETURN c.name",
    );
}

#[test]
fn a_new_required_property_of_an_edge_type_fails_the_commit() {
    the_commit_fails(
        &["CREATE EDGE TYPE ROAD (km INTEGER)"],
        None,
        "INSERT (:City)-[:ROAD {km: 3}]->(:City)",
        "ALTER EDGE TYPE ROAD ADD PROPERTY toll INTEGER NOT NULL",
        "MATCH ()-[r:ROAD]->() RETURN r.km",
    );
}

/// A label a closed graph type stops allowing: `ALTER GRAPH TYPE ... DROP NODE TYPE`.
#[test]
fn a_node_type_a_graph_type_stops_listing_fails_the_commit() {
    the_commit_fails(
        &[
            "CREATE NODE TYPE City (name STRING)",
            "CREATE NODE TYPE Town (name STRING)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, NODE TYPE Town)",
            "CREATE GRAPH maps TYPED atlas",
        ],
        Some("maps"),
        "INSERT (:City {name: 'Paris'})",
        "ALTER GRAPH TYPE atlas DROP NODE TYPE City",
        "MATCH (c:City) RETURN c.name",
    );
}

/// An edge type likewise.
#[test]
fn an_edge_type_a_graph_type_stops_listing_fails_the_commit() {
    the_commit_fails(
        &[
            "CREATE NODE TYPE City (name STRING)",
            "CREATE EDGE TYPE ROAD (km INT64)",
            "CREATE EDGE TYPE RAIL (km INT64)",
            "CREATE GRAPH TYPE atlas (NODE TYPE City, EDGE TYPE ROAD, EDGE TYPE RAIL)",
            "CREATE GRAPH maps TYPED atlas",
        ],
        Some("maps"),
        "INSERT (:City {name: 'Paris'})-[:ROAD {km: 3}]->(:City {name: 'Prague'})",
        "ALTER GRAPH TYPE atlas DROP EDGE TYPE ROAD",
        "MATCH ()-[r:ROAD]->() RETURN r.km",
    );
}

/// A type an open graph type starts listing: its rules now hold in the bound graph.
#[test]
fn a_type_an_open_graph_type_starts_listing_fails_the_commit() {
    the_commit_fails(
        &[
            "CREATE NODE TYPE Town (name STRING)",
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            // Another graph type lists it, so no rule by label holds in the open graph yet.
            "CREATE GRAPH TYPE elsewhere (NODE TYPE City)",
            "CREATE GRAPH TYPE o {node_types: [Town], edge_types: [], open: true}",
            "CREATE GRAPH free TYPED o",
        ],
        Some("free"),
        "INSERT (:City {size: 3})",
        "ALTER GRAPH TYPE o ADD NODE TYPE City",
        "MATCH (c:City) RETURN c.size",
    );
}

/// A writer that meets each new rule commits: the recheck takes nothing it should not.
#[test]
fn a_writer_that_meets_the_new_rule_commits() {
    let db = GrafeoDB::new_in_memory();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer
        .execute("INSERT (:City {name: 'Paris', size: 3})-[:ROAD {km: 3}]->(:City {name: 'Rome'})")
        .unwrap();
    let rules = db.session();
    run_all(
        &rules,
        &[
            "CREATE NODE TYPE City (name STRING, size INT64)",
            "CREATE EDGE TYPE ROAD CONNECTING (City) TO (City) (km INT64)",
        ],
    );
    writer.commit().unwrap();
    assert_eq!(count(&rules, "MATCH (:City)-[:ROAD]->(:City) RETURN 1"), 1);
}

// ---------------------------------------------------------------------------
// A rule on endpoints over the edges there (catalog/mod.rs:2890)
// ---------------------------------------------------------------------------

#[test]
fn endpoints_over_edges_that_break_them_are_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, &["INSERT (:City)-[:ROAD]->(:Town)"]);
    let error = session
        .execute("CREATE EDGE TYPE ROAD CONNECTING (Town) TO (Town)")
        .expect_err("an edge from a City is there");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
    assert_eq!(count(&session, "SHOW EDGE TYPES"), 0, "leaves nothing");
    session
        .execute("INSERT (:City)-[:ROAD]->(:Town)")
        .unwrap_or_else(|error| panic!("no rule was left: {error}"));
}

#[test]
fn endpoints_over_edges_that_meet_them_are_accepted() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "INSERT (:City)-[:ROAD]->(:Town)",
            "CREATE EDGE TYPE ROAD CONNECTING (City) TO (Town)",
        ],
    );
    assert!(session.execute("INSERT (:Town)-[:ROAD]->(:City)").is_err());
}

#[test]
fn a_copy_into_a_typed_graph_checks_the_endpoints() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(
        &session,
        &[
            "CREATE GRAPH TYPE strict { (:City {name STRING})-[:ROAD]->(:City), (:Town {name STRING}) }",
            "CREATE GRAPH loose",
            "USE GRAPH loose",
            "INSERT (:Town {name: 'Delft'})-[:ROAD]->(:City {name: 'Paris'})",
            "USE GRAPH default",
        ],
    );
    let error = session
        .execute("CREATE GRAPH checked TYPED strict AS COPY OF loose")
        .expect_err("the edge starts at a Town");
    assert_eq!(GqlStatus::from(&error).as_str(), "G2000", "{error}");
}
