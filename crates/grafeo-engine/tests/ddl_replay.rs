//! Schema changes survive a WAL replay (#422).
//!
//! A single-file database replays its sidecar WAL when it reopens after a
//! crash, before its next checkpoint. The tests write in a child process that
//! exits without `close()`, so nothing is checkpointed. Replay has to give the
//! same catalog the statements built, fail on a record it does not understand,
//! and ignore records the checkpoint already contains.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test ddl_replay
//! ```

#![allow(
    missing_docs,
    reason = "a test crate; each test is documented by its name"
)]

#[cfg(all(feature = "wal", feature = "gql", feature = "grafeo-file"))]
mod common;

#[cfg(all(feature = "wal", feature = "gql", feature = "grafeo-file"))]
mod tests {
    use crate::common::replay::reopened_after_crash;
    use grafeo_common::types::{TransactionId, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::wal::{WalManager, WalRecord};
    use std::path::{Path, PathBuf};

    /// A single-file database.
    fn open_file(path: &Path) -> grafeo_common::utils::error::Result<GrafeoDB> {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(StorageFormat::Auto))
    }

    fn open(path: &Path) -> GrafeoDB {
        open_file(path).unwrap()
    }

    /// The sidecar WAL of the database file at `path`.
    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
        db.session().execute(query).unwrap().rows().to_vec()
    }

    /// Runs `statements` in one session of `db`.
    fn run(db: &GrafeoDB, statements: &[&str]) {
        let session = db.session();
        for statement in statements {
            session.execute(statement).unwrap();
        }
    }

    /// The schema as the SHOW statements report it.
    fn schema(db: &GrafeoDB) -> Vec<Vec<Vec<Value>>> {
        ["SHOW NODE TYPES", "SHOW EDGE TYPES", "SHOW GRAPH TYPES"]
            .iter()
            .map(|query| {
                let mut result = rows(db, query);
                result.sort_by_key(|row| format!("{row:?}"));
                result
            })
            .collect()
    }

    /// The schema `statements` build in memory.
    fn schema_of(statements: &[&str]) -> Vec<Vec<Vec<Value>>> {
        let reference = GrafeoDB::new_in_memory();
        run(&reference, statements);
        schema(&reference)
    }

    /// Runs `statements` on a new single-file database in a child process
    /// that exits without `close()`, then reopens it and checks that replay
    /// rebuilt the schema the statements build in memory, which the child
    /// also saw. `test` is the path of the calling test. Returns the
    /// directory and the reopened database: bind them as `(_dir, db)`, since
    /// bindings drop in reverse order and the database must close before the
    /// directory goes.
    fn replayed(test: &str, statements: &[&str]) -> (tempfile::TempDir, GrafeoDB) {
        let expected = schema_of(statements);
        let (dir, db) = reopened_after_crash(test, open, |db| {
            run(db, statements);
            // What the writer sees is what replay must rebuild.
            assert_eq!(schema(db), expected, "live schema");
        });
        assert_eq!(schema(&db), expected, "schema after replay");
        (dir, db)
    }

    fn graph_type(db: &GrafeoDB, name: &str) -> Vec<Value> {
        rows(db, &format!("SHOW GRAPH TYPE {name}")).remove(0)
    }

    #[test]
    fn alter_graph_type_survives_replay() {
        let (_dir, db) = replayed(
            "tests::alter_graph_type_survives_replay",
            &[
                "CREATE NODE TYPE Device (serial STRING)",
                "CREATE NODE TYPE Sensor (unit STRING)",
                "CREATE EDGE TYPE CONNECTS",
                "CREATE GRAPH TYPE iot (NODE TYPE Device)",
                "ALTER GRAPH TYPE iot ADD NODE TYPE Sensor",
                "ALTER GRAPH TYPE iot ADD EDGE TYPE CONNECTS",
                "ALTER GRAPH TYPE iot DROP NODE TYPE Device",
            ],
        );
        let iot = graph_type(&db, "iot");
        assert_eq!(iot[2], Value::from("Sensor"), "node types: {iot:?}");
        assert_eq!(iot[3], Value::from("CONNECTS"), "edge types: {iot:?}");
    }

    #[test]
    fn alter_node_and_edge_types_survive_replay() {
        let (_dir, db) = replayed(
            "tests::alter_node_and_edge_types_survive_replay",
            &[
                "CREATE NODE TYPE Sensor (unit STRING)",
                "ALTER NODE TYPE Sensor ADD PROPERTY location STRING",
                "ALTER NODE TYPE Sensor DROP PROPERTY unit",
                "CREATE EDGE TYPE READS (since INTEGER)",
                "ALTER EDGE TYPE READS ADD PROPERTY rate FLOAT",
            ],
        );
        let sensor = rows(&db, "SHOW NODE TYPES").remove(0);
        assert_eq!(sensor[0], Value::from("Sensor"));
        let properties = sensor[1].as_str().unwrap().to_string();
        assert!(properties.contains("location"), "{properties}");
        assert!(!properties.contains("unit"), "{properties}");
    }

    /// Named constraints and their drops survive a reopen: through replay of
    /// the sidecar WAL after a crash (#421), and through the catalog section
    /// of the file after `close()` (#420).
    #[test]
    fn constraints_survive_reopen() {
        const STATEMENTS: [&str; 3] = [
            "CREATE CONSTRAINT city_name FOR (c:City) ON (c.name) UNIQUE",
            "CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE",
            "DROP CONSTRAINT person_name",
        ];
        let (_replayed_dir, replayed) =
            reopened_after_crash("tests::constraints_survive_reopen", open, |db| {
                run(db, &STATEMENTS);
            });
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpointed.grafeo");
        {
            let db = open(&path);
            run(&db, &STATEMENTS);
            db.close().unwrap();
        }
        let checkpointed = open(&path);

        for (format, db) in [
            ("replayed from the sidecar WAL", replayed),
            ("read from the catalog section", checkpointed),
        ] {
            assert_eq!(
                rows(&db, "SHOW CONSTRAINTS"),
                vec![vec![
                    Value::from("city_name"),
                    Value::from("UNIQUE"),
                    Value::from("City"),
                    Value::from("name"),
                ]],
                "{format}"
            );
            let session = db.session();
            session.execute("INSERT (:City {name: 'Paris'})").unwrap();
            assert!(
                session.execute("INSERT (:City {name: 'Paris'})").is_err(),
                "{format}: the constraint is enforced after reopen"
            );
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
            session.execute("DROP CONSTRAINT city_name").unwrap();
            session.execute("INSERT (:City {name: 'Paris'})").unwrap();
            db.close().unwrap();
        }
    }

    /// An `ALTER` whose last alteration fails changes nothing. The ones before
    /// it used to stay applied without a WAL record, so the live schema and
    /// the one rebuilt from the WAL differed.
    #[test]
    fn failed_alter_changes_nothing() {
        let statements = [
            "CREATE NODE TYPE Sensor (unit STRING)",
            "CREATE EDGE TYPE READS (since INTEGER)",
        ];
        let expected = schema_of(&statements);
        let (_dir, db) = reopened_after_crash("tests::failed_alter_changes_nothing", open, |db| {
            run(db, &statements);
            let before = schema(db);
            assert_eq!(before, expected, "live schema");
            // (Graph type alterations cannot fail midway: adding or dropping a
            // member never fails.)
            let session = db.session();
            for failing in [
                "ALTER NODE TYPE Sensor ADD PROPERTY location STRING DROP PROPERTY missing",
                "ALTER EDGE TYPE READS DROP PROPERTY since ADD PROPERTY since STRING ADD PROPERTY since STRING",
            ] {
                assert!(session.execute(failing).is_err(), "{failing}");
                assert_eq!(schema(db), before, "live schema after {failing}");
            }
        });
        assert_eq!(schema(&db), expected, "schema after replay");
    }

    #[test]
    fn create_or_replace_types_survive_replay() {
        let (_dir, db) = replayed(
            "tests::create_or_replace_types_survive_replay",
            &[
                "CREATE NODE TYPE Widget (name STRING)",
                "CREATE OR REPLACE NODE TYPE Widget (name STRING, color STRING)",
                "CREATE EDGE TYPE HOLDS (since INTEGER)",
                "CREATE OR REPLACE EDGE TYPE HOLDS (until INTEGER)",
                "CREATE GRAPH TYPE shop (NODE TYPE Widget)",
                "CREATE OR REPLACE GRAPH TYPE shop (NODE TYPE Widget, EDGE TYPE HOLDS)",
            ],
        );
        let widget = rows(&db, "SHOW NODE TYPES").remove(0);
        assert!(
            widget[1].as_str().unwrap().contains("color"),
            "the replacement survives: {widget:?}"
        );
        let holds = rows(&db, "SHOW EDGE TYPES").remove(0);
        let holds_properties = holds[1].as_str().unwrap();
        assert!(
            holds_properties.contains("until") && !holds_properties.contains("since"),
            "{holds:?}"
        );
        assert_eq!(graph_type(&db, "shop")[3], Value::from("HOLDS"));
    }

    #[test]
    #[cfg(feature = "algos")]
    fn create_or_replace_procedure_survives_replay() {
        let (_dir, db) = replayed(
            "tests::create_or_replace_procedure_survives_replay",
            &[
                "INSERT (:Person {name: 'Alix'})",
                "CREATE PROCEDURE people() RETURNS (n INTEGER) AS { MATCH (p:Person) RETURN count(p) AS n }",
                "CREATE OR REPLACE PROCEDURE people() RETURNS (n INTEGER) AS { MATCH (p:Person) RETURN count(p) + 10 AS n }",
            ],
        );
        assert_eq!(rows(&db, "CALL people()"), vec![vec![Value::Int64(11)]]);
    }

    /// Writes `records` as one committed group to the WAL at `wal_dir`.
    fn append_wal(wal_dir: &Path, mut records: Vec<WalRecord>) {
        records.push(WalRecord::TransactionCommit {
            transaction_id: TransactionId::SYSTEM,
        });
        let wal = WalManager::open(wal_dir).unwrap();
        wal.log_batch(&records).unwrap();
        wal.sync().unwrap();
    }

    /// Opening a database whose WAL holds a schema record with a kind that
    /// replay does not know fails and names the record, instead of skipping
    /// it (graph type alterations) or guessing (constraints became UNIQUE).
    #[test]
    fn unknown_kinds_fail_the_open() {
        let cases = [
            WalRecord::AlterGraphType {
                name: "iot".to_string(),
                alterations: vec![("rename_node_type".to_string(), "Device".to_string())],
            },
            WalRecord::AlterNodeType {
                name: "Device".to_string(),
                alterations: vec![(
                    "rename".to_string(),
                    "serial".to_string(),
                    String::new(),
                    false,
                )],
            },
            WalRecord::CreateNodeType {
                name: "Gadget".to_string(),
                properties: Vec::new(),
                constraints: vec![("check".to_string(), vec!["serial".to_string()])],
            },
        ];
        for record in cases {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("db.grafeo");
            {
                let db = open(&path);
                db.session()
                    .execute("CREATE NODE TYPE Device (serial STRING)")
                    .unwrap();
                db.session()
                    .execute("CREATE GRAPH TYPE iot (NODE TYPE Device)")
                    .unwrap();
                db.close().unwrap();
            }
            let kind = match &record {
                WalRecord::AlterGraphType { alterations, .. } => alterations[0].0.clone(),
                WalRecord::AlterNodeType { alterations, .. } => alterations[0].0.clone(),
                WalRecord::CreateNodeType { constraints, .. } => constraints[0].0.clone(),
                _ => unreachable!(),
            };
            append_wal(&sidecar_wal(&path), vec![record]);

            let err = open_file(&path).err().expect("the open must fail");
            assert!(err.to_string().contains(&kind), "{kind}: {err}");
        }
    }

    /// A crash between writing a checkpoint and moving the WAL's checkpoint
    /// marker replays records the file already contains, and a WAL suffix
    /// can start in the middle of a type's history. Neither may fail the
    /// open or change the schema, and a record after them still applies.
    #[test]
    fn replaying_schema_records_already_in_the_container_is_a_no_op() {
        let statements = [
            "CREATE NODE TYPE Device (serial STRING)",
            "CREATE NODE TYPE Sensor (unit STRING)",
            "CREATE NODE TYPE Temp (v INTEGER)",
            "ALTER NODE TYPE Temp ADD PROPERTY w INTEGER",
            "DROP NODE TYPE Temp",
            "CREATE GRAPH TYPE iot (NODE TYPE Device)",
            "ALTER GRAPH TYPE iot ADD NODE TYPE Sensor",
            "ALTER GRAPH TYPE iot DROP NODE TYPE Device",
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        {
            let db = open(&path);
            run(&db, &statements);
            assert_eq!(schema(&db), schema_of(&statements), "live schema");
            db.close().unwrap();
        }

        let alter_graph = |action: &str, node_type: &str| WalRecord::AlterGraphType {
            name: "iot".to_string(),
            alterations: vec![(action.to_string(), node_type.to_string())],
        };
        append_wal(
            &sidecar_wal(&path),
            vec![
                // The suffix starts after `CREATE NODE TYPE Temp`.
                WalRecord::AlterNodeType {
                    name: "Temp".to_string(),
                    alterations: vec![(
                        "add".to_string(),
                        "w".to_string(),
                        "INTEGER".to_string(),
                        true,
                    )],
                },
                WalRecord::DropNodeType {
                    name: "Temp".to_string(),
                },
                WalRecord::CreateNodeType {
                    name: "Sensor".to_string(),
                    properties: vec![("unit".to_string(), "STRING".to_string(), true)],
                    constraints: Vec::new(),
                },
                alter_graph("add_node_type", "Sensor"),
                alter_graph("drop_node_type", "Device"),
                // A change the file does not hold yet, so the reopen must
                // replay the suffix to have it.
                WalRecord::CreateNodeType {
                    name: "Gadget".to_string(),
                    properties: vec![("serial".to_string(), "STRING".to_string(), true)],
                    constraints: Vec::new(),
                },
            ],
        );

        let db = open(&path);
        let mut with_gadget = statements.to_vec();
        with_gadget.push("CREATE NODE TYPE Gadget (serial STRING)");
        assert_eq!(schema(&db), schema_of(&with_gadget));
    }

    /// Statements the crash test runs before exiting without `close()`.
    const CRASH_STATEMENTS: [&str; 6] = [
        "CREATE NODE TYPE Device (serial STRING)",
        "CREATE NODE TYPE Sensor (unit STRING)",
        "CREATE GRAPH TYPE iot (NODE TYPE Device)",
        "ALTER GRAPH TYPE iot ADD NODE TYPE Sensor",
        "CREATE NODE TYPE Widget (name STRING)",
        "CREATE OR REPLACE NODE TYPE Widget (name STRING, color STRING)",
    ];

    /// A single-file database that crashes before its next checkpoint has
    /// its schema changes only in the sidecar WAL.
    #[test]
    fn schema_changes_survive_a_crash_before_the_checkpoint() {
        let (_dir, db) = replayed(
            "tests::schema_changes_survive_a_crash_before_the_checkpoint",
            &CRASH_STATEMENTS,
        );
        assert_eq!(graph_type(&db, "iot")[2], Value::from("Device, Sensor"));
    }
}
