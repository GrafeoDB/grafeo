//! After `compact()`, sessions use the database's CDC, RDF store and
//! selected graph, and direct calls are written to the WAL.
//!
//! Queries after `compact()` are not written to the WAL yet: the session's
//! WAL records writes to a plain store, and a compacted database writes the
//! layered store (#448 takes the WAL from the transaction's change set).
//! After a crash, replay brings back what direct calls created after
//! `compact()`, but not their updates and deletes of data from before it
//! (#432 replaces the overlay).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test compact_sessions
//! ```

#![cfg(all(feature = "compact-store", feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// The `name`s of the `Person` nodes, sorted.
fn names(db: &GrafeoDB) -> Vec<Value> {
    let result = db
        .execute("MATCH (p:Person) RETURN p.name ORDER BY p.name")
        .unwrap();
    result.rows().iter().map(|row| row[0].clone()).collect()
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod crash {
    use std::path::Path;

    use grafeo_common::testing::child_process;
    use grafeo_engine::config::{DurabilityMode, StorageFormat};

    use super::*;

    const PATH_VAR: &str = "GRAFEO_COMPACT_SESSIONS_CRASH_PATH";
    const SCENARIO_VAR: &str = "GRAFEO_COMPACT_SESSIONS_CRASH_SCENARIO";

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(
            Config::persistent(path)
                .with_storage_format(StorageFormat::SingleFile)
                .with_wal_durability(DurabilityMode::Sync),
        )
        .unwrap()
    }

    /// Where the `concurrent_writes` scenario writes the state it left.
    fn expected_path(path: &Path) -> std::path::PathBuf {
        path.with_extension("expected")
    }

    /// Runs `scenario` on a new database at `path` in a child process that
    /// exits without `close()`, like a crash.
    fn crash_after(scenario: &str, path: &Path) {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "crash::crash_child", "--nocapture"])
                .env(PATH_VAR, path)
                .env(SCENARIO_VAR, scenario),
        )
        .unwrap();
        assert!(status.success(), "scenario {scenario} failed");
    }

    /// Child-process entry for [`crash_after`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let (Some(path), Ok(scenario)) = (std::env::var_os(PATH_VAR), std::env::var(SCENARIO_VAR))
        else {
            return;
        };
        let path = Path::new(&path);
        let mut db = open(path);
        match scenario.as_str() {
            "checkpointed_compact" => {
                db.create_graph("model").unwrap();
                db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
                db.compact().unwrap();
                // The file itself is compacted from here on: it reopens layered.
                db.wal_checkpoint().unwrap();
                let person = |name: &str| {
                    db.create_node_with_props(&["Person"], [("name", Value::from(name))])
                        .unwrap()
                };
                let gus = person("Gus");
                let jules = person("Jules");
                db.set_node_property(jules, "city", Value::from("Paris"))
                    .unwrap();
                db.create_edge(gus, jules, "KNOWS").unwrap();
                db.graph("model")
                    .unwrap()
                    .create_node_with_props(&["Component"], [("id", Value::from("c0"))])
                    .unwrap();
            }
            // No checkpoint: the file stays empty, the WAL holds everything.
            "compact_without_checkpoint" => {
                let alix = db
                    .create_node_with_props(&["Person"], [("name", Value::from("Alix"))])
                    .unwrap();
                db.compact().unwrap();
                db.create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
                    .unwrap();
                db.set_node_property(alix, "city", Value::from("Amsterdam"))
                    .unwrap();
            }
            "concurrent_writes" => {
                db.execute("INSERT (:Seed)").unwrap();
                db.compact().unwrap();
                let counters: Vec<_> = (0..4)
                    .map(|_| {
                        db.create_node_with_props(&["Counter"], [("v", Value::Int64(0))])
                            .unwrap()
                    })
                    .collect();
                std::thread::scope(|scope| {
                    for thread in 0..8_i64 {
                        let (db, counters) = (&db, &counters);
                        scope.spawn(move || {
                            for step in 0..200_i64 {
                                let counter = counters[usize::try_from(step + thread).unwrap() % 4];
                                let value = Value::Int64(thread * 1_000 + step);
                                // Writes to one node conflict (each call is a
                                // transaction here): retry, a bounded number of times.
                                let mut attempts = 0;
                                while let Err(error) =
                                    db.set_node_property(counter, "v", value.clone())
                                {
                                    attempts += 1;
                                    assert!(
                                        error.to_string().contains("conflict") && attempts < 10_000,
                                        "{error}"
                                    );
                                    std::thread::yield_now();
                                }
                            }
                        });
                    }
                });
                std::fs::write(expected_path(path), format!("{:?}", counter_values(&db))).unwrap();
            }
            other => panic!("unknown scenario {other}"),
        }
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }

    /// Direct calls after `compact()` are in the WAL, so a crash loses none
    /// of them, in the default graph and in a named graph.
    #[test]
    fn writes_after_compact_survive_a_crash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("checkpointed_compact", &path);

        let db = open(&path);
        assert_eq!(
            names(&db),
            [
                Value::from("Alix"),
                Value::from("Gus"),
                Value::from("Jules")
            ]
        );
        let city = db
            .execute("MATCH (p:Person {name: 'Jules'}) RETURN p.city")
            .unwrap();
        assert_eq!(city.rows(), [[Value::from("Paris")]]);
        let knows = db
            .execute("MATCH (:Person {name: 'Gus'})-[:KNOWS]->(p) RETURN p.name")
            .unwrap();
        assert_eq!(knows.rows(), [[Value::from("Jules")]]);
        let model = db
            .graph("model")
            .unwrap()
            .execute("MATCH (c:Component) RETURN c.id")
            .unwrap();
        assert_eq!(model.rows(), [[Value::from("c0")]]);
    }

    /// Before its first checkpoint a database has only its WAL, so direct
    /// calls after `compact()` must be in it to survive a crash; replay
    /// rebuilds a plain store, so updates of data from before `compact()`
    /// come back too.
    #[test]
    fn writes_after_compact_survive_a_crash_before_any_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("compact_without_checkpoint", &path);

        let db = open(&path);
        assert_eq!(names(&db), [Value::from("Alix"), Value::from("Gus")]);
        let city = db
            .execute("MATCH (p:Person {name: 'Alix'}) RETURN p.city")
            .unwrap();
        assert_eq!(city.rows(), [[Value::from("Amsterdam")]]);
        db.close().unwrap();
    }

    /// The `id` and `v` of every `Counter`, by id.
    fn counter_values(db: &GrafeoDB) -> Vec<Vec<Value>> {
        db.execute("MATCH (c:Counter) RETURN id(c), c.v ORDER BY id(c)")
            .unwrap()
            .rows()
            .to_vec()
    }

    /// Direct calls from several threads on a compacted database: the WAL ends
    /// with the state they left, so a reopen that replays it reads what memory
    /// held, never an older value that one call logged after another call's
    /// newer one.
    #[test]
    fn concurrent_writes_after_compact_replay_to_the_last_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        crash_after("concurrent_writes", &path);
        let expected = std::fs::read_to_string(expected_path(&path)).unwrap();

        let db = open(&path);
        assert_eq!(format!("{:?}", counter_values(&db)), expected);
        db.close().unwrap();
    }
}

/// Compacting again merges the overlay into a fresh base: inserts, updates
/// and deletes since the first `compact()` stay, and the database stays
/// writable. (The bindings have no `recompact()`; this is how they merge.)
#[test]
fn compacting_again_keeps_the_overlay_writes() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix', city: 'Paris'})-[:KNOWS]->(:Person {name: 'Gus', city: 'Berlin'})")
        .unwrap();
    db.compact().unwrap();
    db.execute("INSERT (:Person {name: 'Vincent', city: 'Prague'})")
        .unwrap();
    db.execute("MATCH (p:Person {name: 'Alix'}) SET p.city = 'Amsterdam'")
        .unwrap();
    db.execute("MATCH (p:Person {name: 'Gus'}) DETACH DELETE p")
        .unwrap();
    let cities = |db: &GrafeoDB| {
        db.execute("MATCH (p:Person) RETURN p.name, p.city ORDER BY p.name")
            .unwrap()
            .rows()
            .to_vec()
    };
    let before = cities(&db);
    assert_eq!(before.len(), 2);

    db.compact().unwrap();
    assert_eq!(cities(&db), before);
    db.execute("INSERT (:Person {name: 'Mia', city: 'Barcelona'})")
        .unwrap();
    assert_eq!(
        names(&db),
        [
            Value::from("Alix"),
            Value::from("Mia"),
            Value::from("Vincent")
        ]
    );
}

/// Direct calls and queries after `compact()` produce change events.
#[cfg(feature = "cdc")]
#[test]
fn writes_after_compact_reach_cdc() {
    let mut db = GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.compact().unwrap();

    let gus = db
        .create_node_with_props(&["Person"], [("name", Value::from("Gus"))])
        .unwrap();
    db.set_node_property(gus, "city", Value::from("Berlin"))
        .unwrap();
    assert_eq!(db.history(gus).unwrap().len(), 2, "create and update");

    let after_gus = grafeo_common::types::EpochId::new(db.current_epoch().as_u64() + 1);
    db.execute("INSERT (:Person {name: 'Jules'})").unwrap();
    let events = db.changes_between(after_gus, db.current_epoch()).unwrap();
    assert_eq!(events.len(), 1, "{events:?}");
}

/// SPARQL through a session reads and writes the database's RDF store after
/// `compact()`, not a store of its own.
#[cfg(feature = "sparql")]
#[test]
fn sparql_after_compact_uses_the_database_rdf_store() {
    let mut db = GrafeoDB::new_in_memory();
    db.execute_sparql("INSERT DATA { <http://ex/alix> <http://ex/knows> <http://ex/gus> }")
        .unwrap();
    db.compact().unwrap();
    db.session()
        .execute_sparql("INSERT DATA { <http://ex/gus> <http://ex/knows> <http://ex/vincent> }")
        .unwrap();
    let known = db
        .session()
        .execute_sparql("SELECT ?s WHERE { ?s <http://ex/knows> ?o }")
        .unwrap();
    assert_eq!(known.rows().len(), 2);
}

/// The graph `set_current_graph` selects holds after `compact()`, for
/// queries and direct calls.
#[test]
fn the_selected_graph_holds_after_compact() {
    let mut db = GrafeoDB::new_in_memory();
    db.create_graph("model").unwrap();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    db.compact().unwrap();

    db.set_current_graph(Some("model")).unwrap();
    db.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    db.create_node_with_props(&["Person"], [("name", Value::from("Jules"))])
        .unwrap();
    assert_eq!(db.current_graph().as_deref(), Some("model"));
    assert_eq!(names(&db), [Value::from("Gus"), Value::from("Jules")]);

    db.set_current_graph(None).unwrap();
    assert_eq!(names(&db), [Value::from("Alix")]);
}

/// `compact()` keeps a database opened read-only read-only: writes fail as
/// before, and `close()` has nothing to write back.
#[cfg(feature = "grafeo-file")]
#[test]
fn compact_keeps_a_read_only_database_read_only() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
        db.close().unwrap();
    }

    let mut db = GrafeoDB::open_read_only(&path).unwrap();
    db.compact().unwrap();
    assert!(db.is_read_only());
    assert!(db.execute("INSERT (:Person {name: 'Gus'})").is_err());
    assert!(db.create_node(&["Person"]).is_err());
    assert_eq!(names(&db), [Value::from("Alix")]);
    db.close().unwrap();
}
