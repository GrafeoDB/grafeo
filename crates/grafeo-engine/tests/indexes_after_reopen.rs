//! #638: an index made before a database file closes, or before its process
//! crashes, is there when the file opens again, read-write or read-only, and
//! an equality on its property looks the index up instead of scanning every
//! node. This holds for the default graph and named graphs, after a close
//! (the indexes come from the checkpoint), after a crash before any
//! checkpoint (the WAL replay makes them) and after a crash that follows a
//! checkpoint (the checkpoint has the index, the WAL has nodes written after
//! it, which the index must find). grafeo-server opens its `data.grafeo`
//! the same way, with `Sync` WAL durability: the tests run that open too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test indexes_after_reopen
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "grafeo-file"))]

#[cfg(feature = "wal")]
mod common;

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::{Config, DurabilityMode, GrafeoDB};

/// Opens the file at `path` read-write, creating it if it is missing.
fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// Opens the file at `path` as grafeo-server opens its database with
/// `wal_durability: sync`: read-write, its sidecar WAL synced at every
/// commit.
fn open_synced(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).with_wal_durability(DurabilityMode::Sync))
        .unwrap()
}

/// Opens the file at `path` read-write without a WAL: only a checkpoint
/// makes its writes durable.
fn open_without_wal(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path).without_wal()).unwrap()
}

/// Opens the file at `path` read-only.
fn open_read_only(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::read_only(path)).unwrap()
}

/// A property index by the API on `id` and one by DDL on `city` in the
/// default graph, a property index on `stop` in the named graph `trips`, and
/// the nodes they index, some written before an index and some after.
fn build(db: &GrafeoDB) {
    db.execute(
        "INSERT (:Person {id: 'p1', name: 'Alix', city: 'Amsterdam'}), \
         (:Person {id: 'p2', name: 'Gus', city: 'Berlin'})",
    )
    .unwrap();
    db.create_property_index("id").unwrap();
    db.execute("CREATE INDEX person_city FOR (p:Person) ON (p.city)")
        .unwrap();
    db.execute("INSERT (:Person {id: 'p3', name: 'Vincent', city: 'Paris'})")
        .unwrap();

    db.create_graph("trips").unwrap();
    let trips = db.graph("trips").unwrap();
    trips
        .execute("INSERT (:Stop {stop: 's1', city: 'Prague'})")
        .unwrap();
    trips
        .session()
        .unwrap()
        .create_property_index("stop")
        .unwrap();
    trips
        .execute("INSERT (:Stop {stop: 's2', city: 'Barcelona'})")
        .unwrap();
}

/// The text of the plan `{verb} {query}` gives in `run`, with `verb`
/// `EXPLAIN` or `PROFILE`.
fn plan(
    run: impl Fn(&str) -> grafeo_engine::database::QueryResult,
    verb: &str,
    query: &str,
) -> String {
    let result = run(&format!("{verb} {query}"));
    match &result.rows()[0][0] {
        Value::String(plan) => plan.to_string(),
        other => panic!("expected a plan, got {other:?}"),
    }
}

/// `query` plans a lookup of the index on `property` and returns `expected`.
fn assert_seeks(
    run: impl Fn(&str) -> grafeo_engine::database::QueryResult,
    query: &str,
    property: &str,
    expected: &str,
    stage: &str,
) {
    let plan = plan(&run, "EXPLAIN", query);
    assert!(
        plan.contains(&format!("[index: {property}]")),
        "{stage}: `{query}` looks up the index on {property}: {plan}"
    );
    assert_eq!(
        run(query).rows(),
        [[Value::from(expected)]],
        "{stage}: `{query}`"
    );
}

/// Everything [`build`] made is in `db`, and the lookups use it, also for
/// nodes written after the index.
fn assert_indexes(db: &GrafeoDB, stage: &str) {
    let run = |query: &str| db.execute(query).unwrap();
    assert!(db.has_property_index("id"), "{stage}: the API index");
    assert!(db.has_property_index("city"), "{stage}: the DDL index");
    for (id, name) in [("p1", "Alix"), ("p3", "Vincent")] {
        assert_seeks(
            run,
            &format!("MATCH (p:Person {{id: '{id}'}}) RETURN p.name"),
            "id",
            name,
            stage,
        );
    }
    assert_seeks(
        run,
        "MATCH (p:Person) WHERE p.city = 'Berlin' RETURN p.name",
        "city",
        "Gus",
        stage,
    );

    let trips = db.graph("trips").unwrap();
    let run = |query: &str| trips.execute(query).unwrap();
    assert!(
        trips.session().unwrap().has_property_index("stop"),
        "{stage}: the named graph's index"
    );
    assert!(
        !trips.session().unwrap().has_property_index("id"),
        "{stage}: indexes stay in their graph"
    );
    for (stop, city) in [("s1", "Prague"), ("s2", "Barcelona")] {
        assert_seeks(
            run,
            &format!("MATCH (s:Stop {{stop: '{stop}'}}) RETURN s.city"),
            "stop",
            city,
            stage,
        );
    }
}

/// The DDL index keeps its name, so `DROP INDEX` finds it.
fn assert_named(db: &GrafeoDB, stage: &str) {
    let names: Vec<Value> = db
        .execute("SHOW INDEXES")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect();
    assert_eq!(names, [Value::from("person_city")], "{stage}");
}

#[test]
fn indexes_survive_close_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = open(&path);
    build(&db);
    assert_indexes(&db, "before the close");
    db.close().unwrap();

    let db = open(&path);
    assert_indexes(&db, "read-write reopen");
    assert_named(&db, "read-write reopen");
    db.close().unwrap();

    let db = open_read_only(&path);
    assert_indexes(&db, "read-only reopen");
    db.close().unwrap();

    let db = GrafeoDB::open_read_only(&path).unwrap();
    assert_indexes(&db, "open_read_only");
    db.close().unwrap();
}

#[test]
fn indexes_survive_an_explicit_checkpoint_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = open(&path);
    build(&db);
    db.wal_checkpoint().unwrap();
    db.close().unwrap();

    let db = open(&path);
    assert_indexes(&db, "reopen after a checkpoint");
    // A second round: the reopened database writes the indexes it loaded.
    db.close().unwrap();
    let db = open(&path);
    assert_indexes(&db, "second reopen");
    db.close().unwrap();
}

/// Without a WAL the indexes come from the checkpoint of `close()` alone,
/// and they are there after a reopen without a WAL, twice, and read-only.
#[test]
fn indexes_survive_reopen_without_wal() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("db.grafeo");
    let db = open_without_wal(&path);
    build(&db);
    assert_indexes(&db, "before the close");
    db.close().unwrap();
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    assert!(
        !Path::new(&sidecar).exists(),
        "a database without a WAL has no sidecar WAL"
    );

    for stage in ["reopen without a WAL", "second reopen without a WAL"] {
        let db = open_without_wal(&path);
        assert_indexes(&db, stage);
        assert_named(&db, stage);
        db.close().unwrap();
    }

    let db = open_read_only(&path);
    assert_indexes(&db, "read-only reopen");
    db.close().unwrap();
}

/// grafeo-server's restart: a synced database closed and opened again, twice.
#[test]
fn indexes_survive_restarts_of_a_synced_database() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("data.grafeo");
    let db = open_synced(&path);
    build(&db);
    db.close().unwrap();

    for restart in ["first restart", "second restart"] {
        let db = open_synced(&path);
        assert_indexes(&db, restart);
        assert_named(&db, restart);
        db.close().unwrap();
    }
}

#[cfg(feature = "wal")]
mod after_a_crash {
    use super::*;
    use crate::common::replay::reopened_after_crash;

    /// Opens a new file read-write (in the child that crashes) and an
    /// existing one read-only (the reopen).
    fn open_new_or_read_only(path: &Path) -> GrafeoDB {
        if path.exists() {
            open_read_only(path)
        } else {
            open(path)
        }
    }

    /// No checkpoint before the crash: the WAL replay makes every index.
    #[test]
    fn indexes_survive_a_crash_before_any_checkpoint() {
        let (_dir, db) = reopened_after_crash(
            "after_a_crash::indexes_survive_a_crash_before_any_checkpoint",
            open,
            build,
        );
        assert_indexes(&db, "replayed");
        assert_named(&db, "replayed");
        db.close().unwrap();
    }

    #[test]
    fn indexes_survive_a_crash_before_any_checkpoint_read_only() {
        let (_dir, db) = reopened_after_crash(
            "after_a_crash::indexes_survive_a_crash_before_any_checkpoint_read_only",
            open_new_or_read_only,
            build,
        );
        assert!(db.is_read_only());
        assert_indexes(&db, "replayed read-only");
        db.close().unwrap();
    }

    /// The checkpoint has the indexes, the WAL the nodes written after it:
    /// the replay puts those nodes in the indexes the checkpoint restored.
    fn build_then_checkpoint_then_write(db: &GrafeoDB) {
        build(db);
        db.wal_checkpoint().unwrap();
        db.execute("INSERT (:Person {id: 'p4', name: 'Jules', city: 'Barcelona'})")
            .unwrap();
        db.graph("trips")
            .unwrap()
            .execute("INSERT (:Stop {stop: 's3', city: 'Berlin'})")
            .unwrap();
    }

    /// The lookups of [`assert_indexes`] for the nodes written after the
    /// checkpoint.
    fn assert_written_after_the_checkpoint(db: &GrafeoDB, stage: &str) {
        assert_indexes(db, stage);
        assert_seeks(
            |query: &str| db.execute(query).unwrap(),
            "MATCH (p:Person {id: 'p4'}) RETURN p.name",
            "id",
            "Jules",
            stage,
        );
        let trips = db.graph("trips").unwrap();
        assert_seeks(
            |query: &str| trips.execute(query).unwrap(),
            "MATCH (s:Stop {stop: 's3'}) RETURN s.city",
            "stop",
            "Berlin",
            stage,
        );
    }

    /// grafeo-server killed without a shutdown, before and after a
    /// checkpoint, then started twice: the first start replays the WAL, the
    /// second reads what the first wrote at its close.
    #[test]
    fn indexes_survive_a_crash_of_a_synced_database_before_any_checkpoint() {
        let (dir, db) = reopened_after_crash(
            "after_a_crash::indexes_survive_a_crash_of_a_synced_database_before_any_checkpoint",
            open_synced,
            build,
        );
        assert_indexes(&db, "synced, replayed");
        db.close().unwrap();
        let db = open_synced(&dir.path().join("db.grafeo"));
        assert_indexes(&db, "synced, started after the replay");
        db.close().unwrap();
    }

    #[test]
    fn indexes_survive_a_crash_of_a_synced_database_after_a_checkpoint() {
        let (dir, db) = reopened_after_crash(
            "after_a_crash::indexes_survive_a_crash_of_a_synced_database_after_a_checkpoint",
            open_synced,
            build_then_checkpoint_then_write,
        );
        assert_written_after_the_checkpoint(&db, "synced, replayed after a checkpoint");
        db.close().unwrap();
        let db = open_synced(&dir.path().join("db.grafeo"));
        assert_written_after_the_checkpoint(&db, "synced, started after the replay");
        db.close().unwrap();
    }

    #[test]
    fn indexes_take_the_nodes_replayed_after_a_checkpoint() {
        let (_dir, db) = reopened_after_crash(
            "after_a_crash::indexes_take_the_nodes_replayed_after_a_checkpoint",
            open,
            build_then_checkpoint_then_write,
        );
        assert_written_after_the_checkpoint(&db, "replayed after a checkpoint");
        db.close().unwrap();
    }

    #[test]
    fn indexes_take_the_nodes_replayed_after_a_checkpoint_read_only() {
        let (_dir, db) = reopened_after_crash(
            "after_a_crash::indexes_take_the_nodes_replayed_after_a_checkpoint_read_only",
            open_new_or_read_only,
            build_then_checkpoint_then_write,
        );
        assert!(db.is_read_only());
        assert_written_after_the_checkpoint(&db, "replayed read-only after a checkpoint");
        db.close().unwrap();
    }
}

/// The vector and text indexes of a named graph, which the sections leave
/// out (they hold the default graph's), are built from the data on reopen.
#[cfg(all(feature = "vector-index", feature = "text-index"))]
mod search_indexes {
    use super::*;

    const DOCS: &str = "INSERT (:Doc {id: 1, emb: vector([1.0, 0.0, 0.0]), body: 'canals of Amsterdam'}), \
                        (:Doc {id: 2, emb: vector([0.0, 1.0, 0.0]), body: 'bridges of Prague'})";
    const NEAREST: &str = "MATCH (d:Doc) \
                           WHERE cosine_similarity(d.emb, [1.0, 0.1, 0.0]) >= 0.9 RETURN d.id";
    const MATCHING: &str = "MATCH (d:Doc) WHERE text_score(d.body, 'canals') > 0.0 RETURN d.id";

    fn build(db: &GrafeoDB) {
        for graph in [None, Some("library")] {
            let session = db.session();
            if let Some(name) = graph {
                db.create_graph(name).unwrap();
                session.use_graph(name);
            }
            for statement in [
                DOCS,
                "CREATE VECTOR INDEX doc_emb ON :Doc(emb)",
                "CREATE INDEX doc_body FOR (d:Doc) ON (d.body) USING TEXT",
            ] {
                session.execute(statement).unwrap();
            }
        }
    }

    /// `query` in the graph `graph` runs `operator` and finds document 1.
    /// `PROFILE`, not `EXPLAIN`: `EXPLAIN` does not show the search-index
    /// rewrite.
    fn assert_scans(db: &GrafeoDB, graph: Option<&str>, query: &str, operator: &str, stage: &str) {
        let session = db.session();
        if let Some(name) = graph {
            session.use_graph(name);
        }
        let plan = plan(|query| session.execute(query).unwrap(), "PROFILE", query);
        assert!(plan.contains(operator), "{stage}, {graph:?}: {plan}");
        assert_eq!(
            session.execute(query).unwrap().rows(),
            [[Value::Int64(1)]],
            "{stage}, {graph:?}: `{query}`"
        );
    }

    fn assert_search_indexes(db: &GrafeoDB, stage: &str) {
        for graph in [None, Some("library")] {
            assert_scans(db, graph, NEAREST, "VectorScan", stage);
            assert_scans(db, graph, MATCHING, "TextScan", stage);
        }
    }

    #[test]
    fn vector_and_text_indexes_of_every_graph_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let db = open(&path);
        build(&db);
        assert_search_indexes(&db, "before the close");
        db.close().unwrap();

        let db = open(&path);
        assert_search_indexes(&db, "read-write reopen");
        db.close().unwrap();

        let db = open_read_only(&path);
        assert_search_indexes(&db, "read-only reopen");
        db.close().unwrap();
    }
}
