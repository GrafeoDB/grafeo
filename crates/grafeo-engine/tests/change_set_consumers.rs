//! The consumers of a transaction's change set: change data capture and the
//! WAL take every write from it, in every graph and through every write path,
//! also after `compact()` (#558).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test change_set_consumers
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Every write path in the named graph `trips` reports change events that
/// name `trips`, and none lands in the default graph, which holds nodes with
/// the same ids: statements on their own and in a transaction, the
/// session's direct API, and the graph handle's direct calls, single and
/// batched, also while a transaction is open.
#[cfg(feature = "cdc")]
#[test]
fn named_graph_writes_name_their_graph() {
    use grafeo_common::types::{EpochId, PropertyKey};
    use grafeo_engine::Config;

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute("CREATE GRAPH trips").unwrap();
    for name in [
        "Alix", "Gus", "Vincent", "Mia", "Jules", "Butch", "Django", "Hans",
    ] {
        db.create_node_with_props(&["Person"], [("name", Value::from(name))])
            .unwrap();
    }
    let start = db.current_epoch();

    let city = |via: &str| [("via", Value::from(via))];
    let mut session = db.session();
    session.use_graph("trips");
    session
        .execute("INSERT (:City {via: 'statement'})")
        .unwrap();
    session
        .execute("INSERT (:City {via: 'second statement'})")
        .unwrap();
    session
        .execute("MATCH (c:City {via: 'statement'}) SET c.visits = 3")
        .unwrap();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:City {via: 'transaction'})")
        .unwrap();
    session.commit().unwrap();
    session
        .create_node_with_props(&["City"], city("session direct"))
        .unwrap();
    let trips = db.graph("trips").unwrap();
    trips
        .create_node_with_props(&["City"], city("handle direct"))
        .unwrap();
    trips
        .batch_create_nodes_with_props(
            "City",
            vec![[(PropertyKey::new("via"), Value::from("handle batch"))].into()],
        )
        .unwrap();
    // While a transaction is open, a direct call is a private transaction.
    let mut open = db.session();
    open.begin_transaction().unwrap();
    open.execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Amsterdam'")
        .unwrap();
    trips
        .create_node_with_props(&["City"], city("handle direct beside a transaction"))
        .unwrap();
    open.commit().unwrap();

    let events: Vec<_> = db
        .changes_between(EpochId::new(start.as_u64() + 1), db.current_epoch())
        .unwrap();
    let in_default: Vec<_> = events
        .iter()
        .filter(|event| event.graph.is_none())
        .collect();
    assert_eq!(
        in_default.len(),
        1,
        "only the open transaction's update is in the default graph: {in_default:?}"
    );
    let in_trips = events
        .iter()
        .filter(|event| event.graph.as_deref() == Some("trips"))
        .count();
    assert_eq!(
        in_trips, 8,
        "seven creates and one update in trips, each naming the graph: {events:?}"
    );

    let people = db.execute("MATCH (n) RETURN count(n)").unwrap();
    assert_eq!(
        people.rows()[0][0],
        Value::Int64(8),
        "the default graph holds its people only"
    );
    let cities = trips.execute("MATCH (c:City) RETURN count(c)").unwrap();
    assert_eq!(cities.rows()[0][0], Value::Int64(7));
}

/// A call of a stored procedure whose body writes is a write statement,
/// although its plan shows no write: it runs in a transaction of its own,
/// whose commit reports the procedure's writes to change data capture, and
/// one that fails leaves nothing.
#[cfg(all(feature = "cdc", feature = "algos"))]
#[test]
fn a_call_of_a_procedure_that_writes_commits_as_a_transaction() {
    use grafeo_common::types::EpochId;
    use grafeo_engine::Config;
    use grafeo_engine::cdc::ChangeKind;

    let db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute(
        "CREATE PROCEDURE add_city(name STRING) RETURNS (n INTEGER) AS { \
         INSERT (c:City {name: $name}) RETURN 1 AS n }",
    )
    .unwrap();
    db.execute("CALL add_city('Paris') YIELD n RETURN n")
        .unwrap();
    let session = db.session();
    session
        .execute("CALL add_city('Prague') YIELD n RETURN n")
        .unwrap();
    assert!(!session.in_transaction());

    let creates = db
        .changes_between(EpochId::new(0), db.current_epoch())
        .unwrap()
        .into_iter()
        .filter(|event| event.kind == ChangeKind::Create)
        .count();
    assert_eq!(creates, 2, "each call's create is reported at its commit");

    // A procedure whose body fails after its insert (the delete of a node
    // with an edge) leaves nothing.
    db.execute(
        "CREATE PROCEDURE route_and_fail() RETURNS (n INTEGER) AS { \
         MATCH (p:City {name: 'Paris'}) INSERT (p)-[:ROUTE]->(:City {name: 'Berlin'}) \
         WITH p DELETE p RETURN 1 AS n }",
    )
    .unwrap();
    db.execute("CALL route_and_fail() YIELD n RETURN n")
        .unwrap_err();
    let cities = db.execute("MATCH (c:City) RETURN count(c)").unwrap();
    assert_eq!(
        cities.rows()[0][0],
        Value::Int64(2),
        "the failed call's insert is undone"
    );
}

/// Query writes and graph commands after `compact()` survive a crash, as
/// every write before it does (#558): each write path, in the default graph
/// and in a graph created after the compaction, replays from the WAL; a
/// rolled back transaction, a failed statement and a failed batch leave
/// nothing. The writes run in a child process that exits without `close()`.
#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::PathBuf;

    use grafeo_common::testing::child_process;
    use grafeo_common::types::{NodeId, Value};
    use grafeo_engine::GrafeoDB;
    use grafeo_engine::database::BatchEdge;

    /// Tells the child process where its database is.
    const PATH_VAR: &str = "GRAFEO_CHANGE_SET_CONSUMERS_CRASH_PATH";

    /// The writes: one before `compact()`, the rest after it.
    fn writes(db: &mut GrafeoDB) {
        db.execute("INSERT (:City {name: 'Amsterdam'})").unwrap();
        db.compact().unwrap();

        // A graph command, and statements in each mode.
        db.execute("CREATE GRAPH trips").unwrap();
        db.execute("INSERT (:City {name: 'Berlin'})").unwrap();
        let mut session = db.session();
        session
            .execute("MATCH (c:City {name: 'Amsterdam'}) SET c.visits = 3")
            .unwrap();
        session.use_graph("trips");
        session
            .execute(
                "INSERT (:City {name: 'Paris'})-[:ROUTE {hours: 19}]->(:City {name: 'Prague'})",
            )
            .unwrap();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:City {name: 'Barcelona'})")
            .unwrap();
        session
            .execute("MATCH (c:City {name: 'Paris'}) SET c:Capital REMOVE c.visits")
            .unwrap();
        session.commit().unwrap();
        session.begin_transaction().unwrap();
        session
            .execute("INSERT (:City {name: 'Rolled back'})")
            .unwrap();
        session.rollback().unwrap();
        // Fails on its delete of a node with an edge: nothing of it stays.
        session
            .execute(
                "MATCH (p:City {name: 'Prague'}) \
                 INSERT (p)-[:ROUTE]->(:City {name: 'Failed'}) WITH p DELETE p",
            )
            .unwrap_err();
        session
            .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
            .unwrap();
        session
            .execute("MATCH (c:City {name: 'Barcelona'}) DELETE c")
            .unwrap();

        // The direct API: a single call, a batch, a failed batch.
        let gus = db
            .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
            .unwrap();
        let trips = db.graph("trips").unwrap();
        trips
            .create_node_with_props(&["Person"], [("name", Value::from("Mia"))])
            .unwrap();
        db.batch_create_nodes_with_props(
            "Person",
            vec![[("name".into(), Value::from("Vincent"))].into()],
        )
        .unwrap();
        db.batch_create_edges(vec![
            BatchEdge::new(gus, gus, "KNOWS"),
            BatchEdge::new(gus, NodeId::new(88_000), "KNOWS"),
        ])
        .unwrap_err();
    }

    /// The rows of `query` on each graph, as sorted text.
    fn state(db: &GrafeoDB) -> Vec<String> {
        let query = concat!(
            "MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN ",
            "n.name + ' ' + coalesce(toString(n.visits), '-') + ' ' + ",
            "reduce(s = '', l IN labels(n) | s + ':' + l) + ' ' + ",
            "coalesce(type(r) + '-' + m.name + ' ' + coalesce(toString(r.hours), '-'), '-')"
        );
        let rows = |result: grafeo_engine::database::QueryResult, graph: &str| {
            result
                .rows()
                .iter()
                .map(|row| format!("{graph}: {}", row[0]))
                .collect::<Vec<_>>()
        };
        let mut all = rows(db.execute(query).unwrap(), "default");
        all.extend(rows(
            db.graph("trips").unwrap().execute(query).unwrap(),
            "trips",
        ));
        all.sort();
        all
    }

    /// Child-process entry: runs the writes on a new database and exits
    /// without `close()`; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let mut db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        writes(&mut db);
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }

    #[test]
    fn query_writes_and_graph_commands_after_compact_survive_a_crash() {
        let expected = {
            let mut db = GrafeoDB::new_in_memory();
            writes(&mut db);
            state(&db)
        };
        assert!(
            expected.iter().any(|row| row.contains("Paris")),
            "the expected state holds the writes after compact(): {expected:?}"
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("prague.grafeo");
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash::crash_child", "--nocapture"])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        assert!(
            PathBuf::from(sidecar).exists(),
            "the crash left the WAL for the reopen to replay"
        );

        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(state(&db), expected);
        db.close().unwrap();
    }
}
