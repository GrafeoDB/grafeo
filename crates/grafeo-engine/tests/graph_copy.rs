//! `CREATE GRAPH dst AS COPY OF src` copies the source graph (#423): its
//! nodes, edges, labels and properties as a reader at the statement's epoch
//! sees them (no open transaction's writes, current values), and its graph
//! type binding. The copy is independent of the source, and survives a
//! reopen: its WAL group holds the graph's creation and every copied entity.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test graph_copy
//! ```

#![cfg(feature = "gql")]
#![allow(
    missing_docs,
    reason = "a test crate has no public API: each test documents itself"
)]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// `(name, city)` of the people in `graph`, and `(from, to, since)` of their
/// KNOWS edges, sorted.
fn contents(db: &GrafeoDB, graph: &str) -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    let session = db.session();
    session.execute(&format!("USE GRAPH {graph}")).unwrap();
    let people = session
        .execute("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name")
        .unwrap()
        .rows()
        .to_vec();
    let knows = session
        .execute(
            "MATCH (a:Person)-[k:KNOWS]->(b:Person) RETURN a.name, b.name, k.since \
             ORDER BY a.name, b.name",
        )
        .unwrap()
        .rows()
        .to_vec();
    (people, knows)
}

fn row(values: &[Value]) -> Vec<Value> {
    values.to_vec()
}

/// Mia and Butch in Amsterdam and Berlin, Mia knows Butch since 2019.
fn trips(db: &GrafeoDB) {
    let session = db.session();
    session.execute("CREATE GRAPH trips").unwrap();
    session.execute("USE GRAPH trips").unwrap();
    session
        .execute(
            "INSERT (:Person {name: 'Mia', city: 'Amsterdam'})-[:KNOWS {since: 2019}]->\
             (:Person {name: 'Butch', city: 'Berlin'})",
        )
        .unwrap();
}

fn expected() -> (Vec<Vec<Value>>, Vec<Vec<Value>>) {
    (
        vec![
            row(&[Value::from("Butch"), Value::from("Berlin")]),
            row(&[Value::from("Mia"), Value::from("Amsterdam")]),
        ],
        vec![row(&[
            Value::from("Mia"),
            Value::from("Butch"),
            Value::Int64(2019),
        ])],
    )
}

/// The copy holds the source's nodes, edges, labels and properties.
#[test]
fn a_copy_holds_the_source_graph() {
    let db = GrafeoDB::new_in_memory();
    trips(&db);
    db.session()
        .execute("CREATE GRAPH copy AS COPY OF trips")
        .unwrap();
    assert_eq!(contents(&db, "copy"), expected());
    assert_eq!(
        contents(&db, "trips"),
        expected(),
        "the source is unchanged"
    );
}

/// The copy and the source change independently afterwards.
#[test]
fn a_copy_is_independent_of_its_source() {
    let db = GrafeoDB::new_in_memory();
    trips(&db);
    let session = db.session();
    session
        .execute("CREATE GRAPH copy AS COPY OF trips")
        .unwrap();
    session.execute("USE GRAPH copy").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Mia'}) SET p.city = 'Paris'")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent', city: 'Prague'})")
        .unwrap();
    session.execute("USE GRAPH trips").unwrap();
    session
        .execute("MATCH (p:Person {name: 'Butch'}) DETACH DELETE p")
        .unwrap();

    assert_eq!(
        contents(&db, "copy"),
        (
            vec![
                row(&[Value::from("Butch"), Value::from("Berlin")]),
                row(&[Value::from("Mia"), Value::from("Paris")]),
                row(&[Value::from("Vincent"), Value::from("Prague")]),
            ],
            vec![row(&[
                Value::from("Mia"),
                Value::from("Butch"),
                Value::Int64(2019),
            ])],
        )
    );
    assert_eq!(
        contents(&db, "trips"),
        (
            vec![row(&[Value::from("Mia"), Value::from("Amsterdam")])],
            vec![]
        )
    );
}

/// An open transaction's writes in the source are not in the copy: it copies
/// what a reader at the statement's epoch sees.
#[test]
fn a_copy_leaves_out_what_an_open_transaction_wrote() {
    let db = GrafeoDB::new_in_memory();
    trips(&db);
    let mut open = db.session();
    open.execute("USE GRAPH trips").unwrap();
    open.begin_transaction().unwrap();
    open.execute("INSERT (:Person {name: 'Jules', city: 'Barcelona'})")
        .unwrap();

    db.session()
        .execute("CREATE GRAPH copy AS COPY OF trips")
        .unwrap();
    open.commit().unwrap();

    assert_eq!(contents(&db, "copy"), expected(), "no Jules in the copy");
    assert_eq!(contents(&db, "trips").0.len(), 3, "Jules is in the source");
}

/// The copy takes the source's graph type binding.
#[test]
fn a_copy_takes_the_source_graph_type() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE NODE TYPE Person (name STRING, city STRING)",
        "CREATE GRAPH TYPE travel (NODE TYPE Person)",
        "CREATE GRAPH trips TYPED travel",
    ] {
        session.execute(statement).unwrap();
    }
    session.execute("USE GRAPH trips").unwrap();
    assert!(
        session.execute("INSERT (:City {name: 'Paris'})").is_err(),
        "the typed source refuses a label outside its type"
    );
    session.execute("USE GRAPH default").unwrap();
    session
        .execute("CREATE GRAPH copy AS COPY OF trips")
        .unwrap();
    session.execute("USE GRAPH copy").unwrap();
    assert!(
        session.execute("INSERT (:City {name: 'Paris'})").is_err(),
        "the copy is typed as the source"
    );
    session
        .execute("INSERT (:Person {name: 'Gus', city: 'Prague'})")
        .unwrap();
}

/// A copy of a graph with nothing in it is an empty graph.
#[test]
fn a_copy_of_an_empty_graph_is_empty() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE GRAPH empty").unwrap();
    session
        .execute("CREATE GRAPH copy AS COPY OF empty")
        .unwrap();
    assert_eq!(contents(&db, "copy"), (vec![], vec![]));
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[path = "common/replay.rs"]
mod replay;

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod persistence {
    use super::replay::reopened_after_crash;
    use super::*;

    /// The copy survives a reopen from the checkpoint `close()` writes.
    #[test]
    fn a_copy_survives_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("copies.grafeo");
        {
            let db = GrafeoDB::open(&path).unwrap();
            trips(&db);
            db.session()
                .execute("CREATE GRAPH copy AS COPY OF trips")
                .unwrap();
            db.close().unwrap();
        }
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(contents(&db, "copy"), expected());
        assert_eq!(contents(&db, "trips"), expected());
    }

    /// The copy survives a crash: its WAL group holds the graph and every
    /// copied entity, which the reopen replays, and a later write in the
    /// copy replays after it.
    #[test]
    fn a_copy_is_replayed_from_the_wal() {
        let (_dir, db) = reopened_after_crash(
            "persistence::a_copy_is_replayed_from_the_wal",
            |path| GrafeoDB::open(path).unwrap(),
            |db| {
                trips(db);
                let session = db.session();
                session
                    .execute("CREATE GRAPH copy AS COPY OF trips")
                    .unwrap();
                session.execute("USE GRAPH copy").unwrap();
                session
                    .execute("MATCH (p:Person {name: 'Mia'}) SET p.city = 'Paris'")
                    .unwrap();
            },
        );
        assert_eq!(
            contents(&db, "copy").0,
            vec![
                row(&[Value::from("Butch"), Value::from("Berlin")]),
                row(&[Value::from("Mia"), Value::from("Paris")]),
            ]
        );
        assert_eq!(contents(&db, "copy").1, expected().1);
        assert_eq!(contents(&db, "trips"), expected());
    }

    /// A copy whose WAL group cannot be written fails with `GRAFEO-T009`
    /// and leaves nothing: no graph, so the same statement runs again.
    #[cfg(feature = "testing-crash-injection")]
    #[test]
    fn a_copy_whose_group_is_not_written_leaves_nothing() {
        use grafeo_common::testing::crash::with_failure_at;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("failed.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        trips(&db);
        let session = db.session();
        let error = with_failure_at(1, || session.execute("CREATE GRAPH copy AS COPY OF trips"))
            .unwrap_err();
        assert!(error.to_string().contains("wal:write"), "{error}");
        assert_eq!(error.error_code().as_str(), "GRAFEO-T009", "{error}");
        assert!(
            !db.list_graphs().iter().any(|graph| graph == "copy"),
            "no graph is left: {:?}",
            db.list_graphs()
        );
        session
            .execute("CREATE GRAPH copy AS COPY OF trips")
            .unwrap();
        assert_eq!(contents(&db, "copy"), expected());
        db.close().unwrap();
        assert_eq!(
            contents(&GrafeoDB::open(&path).unwrap(), "copy"),
            expected()
        );
    }
}
