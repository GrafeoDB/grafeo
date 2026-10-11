//! Graph commands, schema statements and the index API are standalone
//! changes: each statement or call is checked first, logged as a WAL group
//! of its own, then applied, and replay applies the same records. So
//! everything a schema statement or an index call made comes back after a
//! crash, with nothing left out (defaults, parent types, endpoints, every
//! vector index parameter), and a refused statement leaves nothing, in memory
//! or in the log.
//!
//! DDL inside a transaction takes effect at once and a rollback does not undo
//! it (the documented behavior of 0.6.0), and `DROP GRAPH` is refused while
//! an open transaction has changes in the graph, so no transaction commits
//! into a dropped graph and brings it back on replay.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test standalone_changes
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

/// The rows of `query`, which must succeed.
fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// Sorted `n.name` of every `:Person` in the session's current graph.
fn names_in(session: &Session) -> Vec<String> {
    session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => panic!("a name: {other:?}"),
        })
        .collect()
}

fn names_in_graph(db: &GrafeoDB, graph: &str) -> Vec<String> {
    let session = db.session();
    session.use_graph(graph);
    names_in(&session)
}

fn insert(session: &Session, name: &str) {
    session
        .execute(&format!("INSERT (:Person {{name: '{name}'}})"))
        .unwrap();
}

/// The error code of a refused call, which must fail.
fn refused<T: std::fmt::Debug>(outcome: grafeo_common::utils::error::Result<T>) -> &'static str {
    outcome.unwrap_err().error_code().as_str()
}

/// QG A: a graph another open transaction changed is not dropped, by a
/// statement of any session or by the API, nor by the transaction itself;
/// once that transaction commits or rolls back, the drop goes through. A
/// graph the open transaction did not change drops meanwhile.
#[test]
fn drop_graph_with_open_changes_is_refused() {
    let db = GrafeoDB::new_in_memory();
    for graph in ["trips", "visits", "routes"] {
        db.execute(&format!("CREATE GRAPH {graph}")).unwrap();
    }

    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.use_graph("trips");
    insert(&writer, "Alix");

    assert_eq!(
        refused(db.session().execute("DROP GRAPH trips")),
        "GRAFEO-T001",
        "another session's DROP GRAPH is a write conflict"
    );
    assert_eq!(
        refused(db.session().execute("DROP GRAPH IF EXISTS trips")),
        "GRAFEO-T001",
        "IF EXISTS spares only a graph that does not exist"
    );
    assert_eq!(refused(db.drop_graph("trips")), "GRAFEO-T001", "the API");
    assert_eq!(
        refused(writer.execute("DROP GRAPH trips")),
        "GRAFEO-T001",
        "the transaction that changed the graph"
    );
    assert!(
        db.drop_graph("visits").unwrap(),
        "a graph no open transaction changed drops"
    );

    writer.commit().unwrap();
    assert_eq!(
        names_in_graph(&db, "trips"),
        ["Alix"],
        "nothing was dropped: the transaction committed into the graph"
    );
    db.session().execute("DROP GRAPH trips").unwrap();

    // The same once the transaction rolls back.
    let mut undone = db.session();
    undone.begin_transaction().unwrap();
    undone.use_graph("routes");
    insert(&undone, "Gus");
    assert_eq!(refused(db.drop_graph("routes")), "GRAFEO-T001");
    undone.rollback().unwrap();
    assert!(db.drop_graph("routes").unwrap());
    assert!(db.list_graphs().is_empty(), "{:?}", db.list_graphs());
}

/// Q11 A, in memory: DDL and graph commands inside a transaction take effect
/// at once, and a rollback undoes the transaction's data but not them.
#[test]
fn ddl_in_a_rolled_back_transaction_takes_effect_and_stays() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for statement in ROLLED_BACK_DDL {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    insert(&session, "Vincent");
    session.rollback().unwrap();
    check_rolled_back_ddl(&db);
}

/// The DDL of the rolled-back transaction in
/// [`ddl_in_a_rolled_back_transaction_takes_effect_and_stays`].
const ROLLED_BACK_DDL: &[&str] = &[
    "CREATE GRAPH archive",
    "CREATE NODE TYPE Robot (name STRING NOT NULL, city STRING DEFAULT 'Berlin')",
    "CREATE INDEX robot_name FOR (r:Robot) ON (r.name)",
    "CREATE CONSTRAINT robot_serial FOR (r:Robot) ON (r.serial) UNIQUE",
];

/// What [`ROLLED_BACK_DDL`] made is there and works, and the rolled-back
/// data is not.
fn check_rolled_back_ddl(db: &GrafeoDB) {
    assert_eq!(db.list_graphs(), ["archive"]);
    assert!(names_in(&db.session()).is_empty(), "the data rolled back");
    assert!(db.has_property_index("name"), "the index");
    let names = |query: &str| -> Vec<Value> {
        rows(db, query)
            .into_iter()
            .map(|row| row[0].clone())
            .collect()
    };
    assert_eq!(names("SHOW INDEXES"), [Value::from("robot_name")]);
    assert_eq!(names("SHOW CONSTRAINTS"), [Value::from("robot_serial")]);
    db.execute("INSERT (:Robot {name: 'Gus', serial: 3})")
        .unwrap();
    assert_eq!(
        rows(db, "MATCH (r:Robot) RETURN r.city"),
        [[Value::from("Berlin")]],
        "the node type and its default"
    );
    let duplicate = db
        .execute("INSERT (:Robot {name: 'Mia', serial: 3})")
        .unwrap_err()
        .to_string();
    assert!(duplicate.contains("UNIQUE"), "the constraint: {duplicate}");
}

#[cfg(all(
    feature = "wal",
    feature = "grafeo-file",
    feature = "vector-index",
    feature = "text-index"
))]
mod replay {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_common::types::Value;
    use grafeo_core::index::vector::{DistanceMetric, QuantizationType};
    use grafeo_engine::{Config, GrafeoDB};

    use super::{ROLLED_BACK_DDL, check_rolled_back_ddl, insert, names_in, rows};

    const SCENARIO_VAR: &str = "GRAFEO_STANDALONE_SCENARIO";
    const PATH_VAR: &str = "GRAFEO_STANDALONE_PATH";

    fn db_path(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("standalone.grafeo")
    }

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path)).unwrap()
    }

    /// The sidecar WAL of the database file at `path`.
    fn wal_dir(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Runs `scenario` in a child process that exits without closing the
    /// database (a crash: no checkpoint, no destructors), then checks that
    /// it left its sidecar WAL, which the reopen replays.
    fn crash_after(scenario: &str, path: &Path) {
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "replay::crash_child", "--nocapture"])
                .env(SCENARIO_VAR, scenario)
                .env(PATH_VAR, path),
        )
        .unwrap();
        assert!(status.success(), "scenario {scenario} failed");
        assert!(
            wal_dir(path).exists(),
            "the crash left the sidecar WAL of {}",
            path.display()
        );
    }

    /// Child-process entry for [`crash_after`]; a no-op when run directly.
    #[test]
    fn crash_child() {
        let Ok(scenario) = std::env::var(SCENARIO_VAR) else {
            return;
        };
        let path = PathBuf::from(std::env::var_os(PATH_VAR).unwrap());
        // Built outside any crash point: nothing unwinds into its `Drop`.
        let db = open(&path);
        run_scenario(&scenario, &db);
        // Crash: no close(), no destructors.
        std::process::exit(0);
    }

    fn run(db: &GrafeoDB, statements: &[&str]) {
        for statement in statements {
            db.execute(statement)
                .unwrap_or_else(|error| panic!("{statement}: {error}"));
        }
    }

    fn run_scenario(scenario: &str, db: &GrafeoDB) {
        match scenario {
            "types" => run(db, TYPES),
            "indexes" => make_indexes(db),
            "dropped_graph" => {
                db.execute("CREATE GRAPH trips").unwrap();
                db.graph("trips")
                    .unwrap()
                    .execute("INSERT (:Person {name: 'Alix'})")
                    .unwrap();
                let mut writer = db.session();
                writer.begin_transaction().unwrap();
                writer.use_graph("trips");
                insert(&writer, "Gus");
                // Refused while the transaction has changes in the graph.
                // Before QG A it dropped the graph, the commit below wrote
                // its records into the dropped graph's store, and replay
                // created the graph again for them.
                let _ = db.session().execute("DROP GRAPH trips");
                writer.commit().unwrap();
                db.session().execute("DROP GRAPH IF EXISTS trips").unwrap();
                insert(&db.session(), "Vincent");
            }
            "rolled_back_ddl" => {
                let mut session = db.session();
                session.begin_transaction().unwrap();
                for statement in ROLLED_BACK_DDL {
                    session.execute(statement).unwrap();
                }
                insert(&session, "Vincent");
                session.rollback().unwrap();
            }
            "refused" => {
                // Each refused before anything changes.
                db.execute("CREATE GRAPH trips TYPED nowhere").unwrap_err();
                db.execute("CREATE NODE TYPE City (name STRING)").unwrap();
                db.execute("CREATE NODE TYPE City (name STRING, country STRING)")
                    .unwrap_err();
                let halfway = db
                    .execute(
                        "ALTER NODE TYPE City ADD PROPERTY zone STRING DEFAULT 'A' \
                         DROP PROPERTY nothing",
                    )
                    .unwrap_err()
                    .to_string();
                assert!(
                    halfway.contains("property nothing on City"),
                    "refused by its second alteration: {halfway}"
                );
            }
            other => panic!("unknown scenario {other}"),
        }
    }

    /// Every kind of catalog entry, with what the 0.5 WAL records left out:
    /// defaults, parent types, endpoints, `KEY` labels, a type replaced and
    /// types altered with defaults, a typed graph, then drops.
    const TYPES: &[&str] = &[
        "CREATE NODE TYPE City (name STRING NOT NULL, country STRING DEFAULT 'NL')",
        "CREATE NODE TYPE Capital EXTENDS City (since INT64)",
        "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64 DEFAULT 88)",
        "CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)",
        "CREATE CONSTRAINT person_email FOR (p:Person) ON (p.email) UNIQUE",
        "CREATE CONSTRAINT FOR (p:Person) ON (p.name) NOT NULL",
        "CREATE GRAPH trips TYPED travel",
        "CREATE SCHEMA archive",
        "CREATE PROCEDURE capitals() RETURNS (name STRING) AS { MATCH (c:Capital) RETURN c.name AS name }",
        "CREATE GRAPH TYPE itinerary (NODE TYPE Stop KEY (StopKey) (arrives ZONED DATETIME), \
         EDGE TYPE LEG KEY (LegKey) (departs LIST<LOCAL DATETIME>))",
        "CREATE NODE TYPE Event (begins ZONED DATETIME, tags LIST<STRING>)",
        "CREATE EDGE TYPE BOOKED CONNECTING (Event) TO (City) (stamps LIST<ZONED DATETIME>)",
        "CREATE NODE TYPE Museum (name STRING)",
        "CREATE OR REPLACE NODE TYPE Museum (name STRING, city STRING DEFAULT 'Paris')",
        "ALTER NODE TYPE Event ADD PROPERTY zone STRING DEFAULT 'CET'",
        "ALTER EDGE TYPE ROUTE ADD PROPERTY toll BOOL DEFAULT false",
        "ALTER GRAPH TYPE travel ADD NODE TYPE Capital",
        "CREATE NODE TYPE Doomed (name STRING)",
        "DROP NODE TYPE Doomed",
        "CREATE PROCEDURE doomed() RETURNS (name STRING) AS { MATCH (c:City) RETURN c.name AS name }",
        "DROP PROCEDURE doomed",
        "CREATE CONSTRAINT doomed FOR (c:City) ON (c.name) UNIQUE",
        "DROP CONSTRAINT doomed",
    ];

    /// The `SHOW` statements whose output a reopen must keep.
    const SHOWN: &[&str] = &[
        "SHOW NODE TYPES",
        "SHOW EDGE TYPES",
        "SHOW GRAPH TYPES",
        "SHOW CONSTRAINTS",
        "SHOW INDEXES",
        "SHOW GRAPHS",
        "SHOW SCHEMAS",
    ];

    /// Every row of every statement of [`SHOWN`], one line each, the rows of
    /// a statement sorted.
    fn show_everything(db: &GrafeoDB) -> Vec<String> {
        SHOWN
            .iter()
            .flat_map(|query| {
                let mut lines: Vec<String> = rows(db, query)
                    .iter()
                    .map(|row| format!("{query}: {row:?}"))
                    .collect();
                lines.sort();
                lines
            })
            .collect()
    }

    /// The row of `query` whose first column is `name`.
    fn row_of(db: &GrafeoDB, query: &str, name: &str) -> Vec<Value> {
        rows(db, query)
            .into_iter()
            .find(|row| row[0] == Value::from(name))
            .unwrap_or_else(|| panic!("{query} lists no {name}"))
    }

    fn single(db: &GrafeoDB, query: &str) -> Value {
        let rows = rows(db, query);
        assert_eq!(rows.len(), 1, "{query}: {rows:?}");
        rows[0][0].clone()
    }

    #[test]
    fn types_with_defaults_parents_endpoints_survive_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        crash_after("types", &path);

        let live = GrafeoDB::new_in_memory();
        run(&live, TYPES);
        let db = open(&path);
        assert_eq!(show_everything(&db), show_everything(&live));

        // The comparison covers what the 0.5 records left out.
        assert_eq!(
            row_of(&db, "SHOW NODE TYPES", "Capital")[3],
            Value::from("City"),
            "the parent type"
        );
        assert_eq!(
            row_of(&db, "SHOW EDGE TYPES", "ROUTE")[1..4],
            [
                Value::from("km INT64, toll BOOLEAN"),
                Value::from("City"),
                Value::from("City")
            ],
            "the endpoints and the property added"
        );
        assert_eq!(
            row_of(&db, "SHOW NODE TYPES", "Stop")[1..4],
            [
                Value::from("arrives ZONED DATETIME"),
                Value::from(""),
                Value::from("StopKey")
            ],
            "an inline element type, its key label also its parent"
        );
        assert_eq!(
            row_of(&db, "SHOW GRAPH TYPES", "travel")[2..4],
            [Value::from("City, Capital"), Value::from("ROUTE")],
            "the node type added"
        );

        // The defaults work.
        db.execute("INSERT (:City {name: 'Prague'})-[:ROUTE]->(:City {name: 'Berlin'})")
            .unwrap();
        assert_eq!(
            single(&db, "MATCH (c:City {name: 'Prague'}) RETURN c.country"),
            Value::from("NL")
        );
        assert_eq!(
            rows(&db, "MATCH ()-[r:ROUTE]->() RETURN r.km, r.toll"),
            [[Value::Int64(88), Value::Bool(false)]],
            "an edge type's defaults, also of a property added later"
        );
        db.execute("INSERT (:Museum {name: 'Louvre'}), (:Event {tags: ['jazz']})")
            .unwrap();
        assert_eq!(
            single(&db, "MATCH (m:Museum) RETURN m.city"),
            Value::from("Paris"),
            "the replacing type's default"
        );
        assert_eq!(
            single(&db, "MATCH (e:Event) RETURN e.zone"),
            Value::from("CET"),
            "the default of a property added"
        );
        // The constraints, the binding and the procedure.
        let duplicate = db
            .execute(
                "INSERT (:Person {name: 'Gus', email: 'gus@example.org'}), \
                 (:Person {name: 'Mia', email: 'gus@example.org'})",
            )
            .unwrap_err()
            .to_string();
        assert!(duplicate.contains("UNIQUE"), "{duplicate}");
        let outside = db
            .graph("trips")
            .unwrap()
            .execute("INSERT (:Person {name: 'Jules'})")
            .unwrap_err()
            .to_string();
        assert!(
            outside.contains("Person"),
            "trips keeps the type travel, which has no Person: {outside}"
        );
        db.execute("INSERT (:Capital {name: 'Amsterdam'})").unwrap();
        assert_eq!(
            single(&db, "CALL capitals() YIELD name RETURN name"),
            Value::from("Amsterdam")
        );
    }

    /// Indexes of every kind, by the API and by DDL, in the default graph
    /// and a named graph, some dropped again, with nodes written after an
    /// index was made.
    fn make_indexes(db: &GrafeoDB) {
        run(
            db,
            &[
                "INSERT (:Person {name: 'Alix', email: 'alix@example.org', city: 'Amsterdam', \
                 bio: 'canals and bridges', age: 19}), \
                 (:Person {name: 'Gus', email: 'gus@example.org', city: 'Berlin', bio: 'trains'})",
                "INSERT (:Doc {emb: vector([3.0, 19.0, 88.0]), body: 'boats on the canals'}), \
                 (:Doc {emb: vector([88.0, 19.0, 3.0]), body: 'bridges over the river'})",
                "INSERT (:Note {emb: vector([3.0, 3.0, 19.0])}), (:Tag {emb: vector([1.0, 0.0])})",
            ],
        );
        db.create_property_index("email").unwrap();
        db.create_vector_index(
            "Doc",
            "emb",
            Some(3),
            Some("euclidean"),
            Some(19),
            Some(88),
            Some("scalar"),
        )
        .unwrap();
        db.create_text_index("Doc", "body").unwrap();
        run(
            db,
            &[
                "CREATE INDEX person_name FOR (p:Person) ON (p.name)",
                "CREATE INDEX person_city FOR (p:Person) ON (p.city) USING BTREE",
                "CREATE INDEX person_bio FOR (p:Person) ON (p.bio) USING TEXT",
                "CREATE VECTOR INDEX note_emb ON :Note(emb) DIMENSION 3 METRIC 'manhattan'",
                "CREATE GRAPH model",
            ],
        );
        let model = db.session();
        model.use_graph("model");
        for statement in [
            "INSERT (:Part {serial: 'P3', emb: vector([1.0, 0.0])})",
            "CREATE INDEX part_serial FOR (p:Part) ON (p.serial)",
            "CREATE VECTOR INDEX part_emb ON :Part(emb)",
        ] {
            model.execute(statement).unwrap();
        }
        // Kept current while the WAL replays.
        db.execute("INSERT (:Doc {emb: vector([19.0, 3.0, 88.0]), body: 'trams in Prague'})")
            .unwrap();
        // Made and dropped again, each kind.
        db.create_property_index("bio").unwrap();
        assert!(db.drop_property_index("bio").unwrap());
        db.create_vector_index("Tag", "emb", None, None, None, None, None)
            .unwrap();
        assert!(db.drop_vector_index("Tag", "emb").unwrap());
        db.create_text_index("Person", "name").unwrap();
        assert!(db.drop_text_index("Person", "name").unwrap());
        run(
            db,
            &[
                "CREATE INDEX person_age FOR (p:Person) ON (p.age)",
                "DROP INDEX person_age",
            ],
        );
    }

    #[test]
    fn api_and_ddl_indexes_of_every_kind_survive_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        crash_after("indexes", &path);
        let db = open(&path);
        let store = db.store();

        for (property, indexed) in [
            ("email", true),
            ("name", true),
            ("city", true),
            ("bio", false),
            ("age", false),
        ] {
            assert_eq!(db.has_property_index(property), indexed, "{property}");
        }
        let mut names: Vec<String> = rows(&db, "SHOW INDEXES")
            .into_iter()
            .map(|row| match &row[0] {
                Value::String(name) => name.to_string(),
                other => panic!("an index name: {other:?}"),
            })
            .collect();
        names.sort();
        assert_eq!(
            names,
            ["part_serial", "person_bio", "person_city", "person_name"],
            "the names, without the one dropped"
        );

        let docs = store
            .get_vector_index("Doc", "emb")
            .expect("the API vector index");
        let config = docs.config();
        assert_eq!(
            (
                config.dimensions,
                config.metric,
                config.m,
                config.ef_construction
            ),
            (3, DistanceMetric::Euclidean, 19, 88),
            "every parameter of the API vector index"
        );
        assert_eq!(docs.quantization_type(), Some(QuantizationType::Scalar));
        assert_eq!(docs.len(), 3, "a node written after the index is in it");
        let notes = store
            .get_vector_index("Note", "emb")
            .expect("the DDL vector index");
        assert_eq!(
            (notes.config().dimensions, notes.config().metric),
            (3, DistanceMetric::Manhattan)
        );
        assert!(
            store.get_vector_index("Tag", "emb").is_none(),
            "a vector index dropped"
        );

        let found = db.text_search("Doc", "body", "Prague", 3, None).unwrap();
        assert_eq!(
            found.len(),
            1,
            "the API text index, kept current: {found:?}"
        );
        assert!(
            store.get_text_index("Person", "bio").is_some(),
            "the DDL text index"
        );
        assert!(
            store.get_text_index("Person", "name").is_none(),
            "a text index dropped"
        );

        let model = store.graph("model").expect("the named graph");
        assert!(model.has_property_index("serial"), "a named graph's index");
        assert_eq!(
            model
                .get_vector_index("Part", "emb")
                .map(|index| index.len()),
            Some(1),
            "a named graph's vector index"
        );
    }

    /// B4: a transaction that changed a graph never commits into it once it
    /// is dropped, so replay never creates the graph again for its records.
    #[test]
    fn a_transaction_never_resurrects_a_dropped_graph() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        crash_after("dropped_graph", &path);
        let db = open(&path);
        assert!(
            db.list_graphs().is_empty(),
            "the dropped graph came back: {:?}",
            db.list_graphs()
        );
        assert_eq!(names_in(&db.session()), ["Vincent"]);
    }

    /// Q11 A after a crash: the DDL of a rolled-back transaction was logged
    /// and comes back, and the transaction's data does not.
    #[test]
    fn ddl_in_a_rolled_back_transaction_survives_replay() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        crash_after("rolled_back_ddl", &path);
        check_rolled_back_ddl(&open(&path));
    }

    /// A statement refused by its checks changes nothing, in memory or in
    /// the log: a graph typed by a graph type that does not exist is not
    /// created, a node type that exists is not replaced, and an alteration
    /// that fails halfway adds no property.
    #[test]
    fn a_refused_statement_changes_nothing_in_memory_or_in_the_log() {
        let dir = tempfile::tempdir().unwrap();
        let path = db_path(&dir);
        crash_after("refused", &path);
        let db = open(&path);
        assert!(db.list_graphs().is_empty(), "{:?}", db.list_graphs());
        assert_eq!(
            rows(&db, "SHOW NODE TYPES"),
            [[
                Value::from("City"),
                Value::from("name STRING"),
                Value::from(""),
                Value::from("")
            ]]
        );
    }
}
