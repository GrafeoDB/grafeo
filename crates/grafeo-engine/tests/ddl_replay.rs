//! Schema changes survive a WAL replay (#422).
//!
//! A WAL-directory database rebuilds its catalog from the WAL on every open,
//! and a single-file database replays the sidecar WAL after a crash. Replay
//! has to give the same catalog the statements built, fail on a record it
//! does not understand, and ignore records the checkpoint already contains.
//!
//! ```bash
//! cargo test -p grafeo-engine --features full --test ddl_replay
//! ```

#![allow(
    missing_docs,
    reason = "a test crate; each test is documented by its name"
)]

#[cfg(all(feature = "wal", feature = "gql"))]
mod tests {
    use grafeo_common::testing::child_process;
    use grafeo_common::types::{TransactionId, Value};
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};
    use grafeo_storage::wal::{WalManager, WalRecord};
    use std::path::Path;

    /// A WAL-directory database: its WAL is the only copy of the schema.
    fn open_wal_dir(path: &Path) -> grafeo_common::utils::error::Result<GrafeoDB> {
        GrafeoDB::with_config(
            Config::persistent(path).with_storage_format(StorageFormat::WalDirectory),
        )
    }

    fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
        db.session().execute(query).unwrap().rows().to_vec()
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

    /// Runs `statements` on a new WAL-directory database, then reopens it and
    /// checks that replay rebuilt the same schema. Returns the directory and
    /// the reopened database: bind them as `(_dir, db)`, since bindings drop
    /// in reverse order and the database must close before the directory goes.
    fn replayed(statements: &[&str]) -> (tempfile::TempDir, GrafeoDB) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let before = {
            let db = open_wal_dir(&path).unwrap();
            let session = db.session();
            for statement in statements {
                session.execute(statement).unwrap();
            }
            let before = schema(&db);
            db.close().unwrap();
            before
        };
        let db = open_wal_dir(&path).unwrap();
        assert_eq!(schema(&db), before, "schema after replay");
        (dir, db)
    }

    fn graph_type(db: &GrafeoDB, name: &str) -> Vec<Value> {
        rows(db, &format!("SHOW GRAPH TYPE {name}")).remove(0)
    }

    #[test]
    fn alter_graph_type_survives_replay() {
        let (_dir, db) = replayed(&[
            "CREATE NODE TYPE Device (serial STRING)",
            "CREATE NODE TYPE Sensor (unit STRING)",
            "CREATE EDGE TYPE CONNECTS",
            "CREATE GRAPH TYPE iot (NODE TYPE Device)",
            "ALTER GRAPH TYPE iot ADD NODE TYPE Sensor",
            "ALTER GRAPH TYPE iot ADD EDGE TYPE CONNECTS",
            "ALTER GRAPH TYPE iot DROP NODE TYPE Device",
        ]);
        let iot = graph_type(&db, "iot");
        assert_eq!(iot[2], Value::from("Sensor"), "node types: {iot:?}");
        assert_eq!(iot[3], Value::from("CONNECTS"), "edge types: {iot:?}");
    }

    #[test]
    fn alter_node_and_edge_types_survive_replay() {
        let (_dir, db) = replayed(&[
            "CREATE NODE TYPE Sensor (unit STRING)",
            "ALTER NODE TYPE Sensor ADD PROPERTY location STRING",
            "ALTER NODE TYPE Sensor DROP PROPERTY unit",
            "CREATE EDGE TYPE READS (since INTEGER)",
            "ALTER EDGE TYPE READS ADD PROPERTY rate FLOAT",
        ]);
        let sensor = rows(&db, "SHOW NODE TYPES").remove(0);
        assert_eq!(sensor[0], Value::from("Sensor"));
        let properties = sensor[1].as_str().unwrap().to_string();
        assert!(properties.contains("location"), "{properties}");
        assert!(!properties.contains("unit"), "{properties}");
    }

    /// Named constraints and their drops survive a reopen: through WAL replay
    /// in a WAL-directory database (#421), and through the catalog section
    /// of a `.grafeo` file (#420).
    #[test]
    fn constraints_survive_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let mut configs = vec![(
            "wal-directory",
            Config::persistent(dir.path().join("dir-db"))
                .with_storage_format(StorageFormat::WalDirectory),
        )];
        #[cfg(feature = "grafeo-file")]
        configs.push((
            "single-file",
            Config::persistent(dir.path().join("db.grafeo"))
                .with_storage_format(StorageFormat::SingleFile),
        ));

        for (format, config) in configs {
            {
                let db = GrafeoDB::with_config(config.clone()).unwrap();
                let session = db.session();
                for statement in [
                    "CREATE CONSTRAINT city_name FOR (c:City) ON (c.name) UNIQUE",
                    "CREATE CONSTRAINT person_name FOR (p:Person) ON (p.name) UNIQUE",
                    "DROP CONSTRAINT person_name",
                ] {
                    session.execute(statement).unwrap();
                }
                db.close().unwrap();
            }

            let db = GrafeoDB::with_config(config).unwrap();
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
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db");
        let before = {
            let db = open_wal_dir(&path).unwrap();
            let session = db.session();
            for statement in [
                "CREATE NODE TYPE Sensor (unit STRING)",
                "CREATE EDGE TYPE READS (since INTEGER)",
            ] {
                session.execute(statement).unwrap();
            }
            let before = schema(&db);
            // (Graph type alterations cannot fail midway: adding or dropping a
            // member never fails.)
            for failing in [
                "ALTER NODE TYPE Sensor ADD PROPERTY location STRING DROP PROPERTY missing",
                "ALTER EDGE TYPE READS DROP PROPERTY since ADD PROPERTY since STRING ADD PROPERTY since STRING",
            ] {
                assert!(session.execute(failing).is_err(), "{failing}");
                assert_eq!(schema(&db), before, "live schema after {failing}");
            }
            db.close().unwrap();
            before
        };
        let db = open_wal_dir(&path).unwrap();
        assert_eq!(schema(&db), before, "schema after replay");
    }

    #[test]
    fn create_or_replace_types_survive_replay() {
        let (_dir, db) = replayed(&[
            "CREATE NODE TYPE Widget (name STRING)",
            "CREATE OR REPLACE NODE TYPE Widget (name STRING, color STRING)",
            "CREATE EDGE TYPE HOLDS (since INTEGER)",
            "CREATE OR REPLACE EDGE TYPE HOLDS (until INTEGER)",
            "CREATE GRAPH TYPE shop (NODE TYPE Widget)",
            "CREATE OR REPLACE GRAPH TYPE shop (NODE TYPE Widget, EDGE TYPE HOLDS)",
        ]);
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
        let (_dir, db) = replayed(&[
            "INSERT (:Person {name: 'Alix'})",
            "CREATE PROCEDURE people() RETURNS (n INTEGER) AS { MATCH (p:Person) RETURN count(p) AS n }",
            "CREATE OR REPLACE PROCEDURE people() RETURNS (n INTEGER) AS { MATCH (p:Person) RETURN count(p) + 10 AS n }",
        ]);
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
            let path = dir.path().join("db");
            {
                let db = open_wal_dir(&path).unwrap();
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
            append_wal(&path.join("wal"), vec![record]);

            let err = open_wal_dir(&path).err().expect("the open must fail");
            assert!(err.to_string().contains(&kind), "{kind}: {err}");
        }
    }

    /// A crash between writing a checkpoint and moving the WAL's checkpoint
    /// marker replays records the file already contains, and a WAL suffix
    /// can start in the middle of a type's history. Neither may fail the
    /// open or change the schema.
    #[test]
    #[cfg(feature = "grafeo-file")]
    fn replaying_schema_records_already_in_the_container_is_a_no_op() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let open = || {
            GrafeoDB::with_config(
                Config::persistent(&path).with_storage_format(StorageFormat::SingleFile),
            )
            .unwrap()
        };

        let before = {
            let db = open();
            let session = db.session();
            for statement in [
                "CREATE NODE TYPE Device (serial STRING)",
                "CREATE NODE TYPE Sensor (unit STRING)",
                "CREATE NODE TYPE Temp (v INTEGER)",
                "ALTER NODE TYPE Temp ADD PROPERTY w INTEGER",
                "DROP NODE TYPE Temp",
                "CREATE GRAPH TYPE iot (NODE TYPE Device)",
                "ALTER GRAPH TYPE iot ADD NODE TYPE Sensor",
                "ALTER GRAPH TYPE iot DROP NODE TYPE Device",
            ] {
                session.execute(statement).unwrap();
            }
            let before = schema(&db);
            db.close().unwrap();
            before
        };

        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        let alter_graph = |action: &str, node_type: &str| WalRecord::AlterGraphType {
            name: "iot".to_string(),
            alterations: vec![(action.to_string(), node_type.to_string())],
        };
        append_wal(
            Path::new(&sidecar),
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
            ],
        );

        let db = open();
        assert_eq!(schema(&db), before);
    }

    const CRASH_PATH_VAR: &str = "GRAFEO_DDL_REPLAY_CRASH_PATH";

    /// Statements the crash test runs before exiting without `close()`.
    const CRASH_STATEMENTS: [&str; 6] = [
        "CREATE NODE TYPE Device (serial STRING)",
        "CREATE NODE TYPE Sensor (unit STRING)",
        "CREATE GRAPH TYPE iot (NODE TYPE Device)",
        "ALTER GRAPH TYPE iot ADD NODE TYPE Sensor",
        "CREATE NODE TYPE Widget (name STRING)",
        "CREATE OR REPLACE NODE TYPE Widget (name STRING, color STRING)",
    ];

    #[cfg(feature = "grafeo-file")]
    fn open_single_file(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(
            Config::persistent(path).with_storage_format(StorageFormat::SingleFile),
        )
        .unwrap()
    }

    /// Child-process entry for [`schema_changes_survive_a_crash_before_the_checkpoint`];
    /// a no-op when run directly.
    #[test]
    #[cfg(feature = "grafeo-file")]
    fn crash_child() {
        let Some(path) = std::env::var_os(CRASH_PATH_VAR) else {
            return;
        };
        let db = open_single_file(Path::new(&path));
        let session = db.session();
        for statement in CRASH_STATEMENTS {
            session.execute(statement).unwrap();
        }
        // Crash: no close(), no checkpoint, no destructors.
        std::process::exit(0);
    }

    /// A single-file database that crashes before its next checkpoint has
    /// its schema changes only in the sidecar WAL.
    #[test]
    #[cfg(feature = "grafeo-file")]
    fn schema_changes_survive_a_crash_before_the_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("db.grafeo");
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "tests::crash_child", "--nocapture"])
                .env(CRASH_PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success());

        let expected = {
            let reference = GrafeoDB::new_in_memory();
            let session = reference.session();
            for statement in CRASH_STATEMENTS {
                session.execute(statement).unwrap();
            }
            schema(&reference)
        };
        let db = open_single_file(&path);
        assert_eq!(schema(&db), expected);
        assert_eq!(graph_type(&db, "iot")[2], Value::from("Device, Sensor"));
    }
}
