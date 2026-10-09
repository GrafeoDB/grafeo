//! Type DDL refuses what it cannot keep, when the type is declared:
//!
//! - A property type is a type Grafeo supports (ISO/IEC 39075:2024 18.7
//!   `<property value type>`, Syntax Rule 4: "shall be a supported property
//!   value type"). Any other name, `INT32` or a typo, is refused instead of
//!   becoming `ANY`, which checked nothing. A name an older version wrote to
//!   its WAL still reads, an unknown one as `ANY`.
//! - A property name (18.6 `<property type>`) may be any name an INSERT
//!   takes as a property key, keywords such as `starts`, `ends` and
//!   `contains` included.
//! - A `DEFAULT` (a Grafeo extension of a property type) is a literal of the
//!   property's type: a signed number (21.2 `<signed numeric literal>`)
//!   included, and one of another type, or `NULL` for a `NOT NULL` property,
//!   refused by the DDL instead of by every insert that relies on it.
//! - `DROP GRAPH TYPE` takes the bindings of graphs to the type with it, so
//!   a graph type created later under the same name types no graph.
//!
//! Each holds after a reopen too: after a checkpoint and after a WAL replay.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test type_ddl_checks
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

mod common;

use std::path::{Path, PathBuf};

use grafeo_common::types::{TransactionId, Value};
use grafeo_engine::{Config, GrafeoDB};
use grafeo_storage::wal::{WalManager, WalRecord};

/// A new database at `path`, a single file with its sidecar WAL.
fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// The value of the only row and column of `query`.
fn single(db: &GrafeoDB, query: &str) -> Value {
    let rows = rows(db, query);
    assert_eq!(rows.len(), 1, "{query}: {rows:?}");
    rows[0][0].clone()
}

/// The error message of `query`, which must fail.
fn refusal(db: &GrafeoDB, query: &str) -> String {
    match db.execute(query) {
        Ok(result) => panic!("{query} was accepted: {:?}", result.rows()),
        Err(error) => error.to_string(),
    }
}

/// The schema as the SHOW statements report it, the rows of each sorted.
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

/// The `properties` column of the row of `type_name` in a SHOW statement.
fn property_list(db: &GrafeoDB, query: &str, type_name: &str) -> String {
    rows(db, query)
        .into_iter()
        .find(|row| row[0] == Value::from(type_name))
        .unwrap_or_else(|| panic!("{query} lists no {type_name}"))[1]
        .as_str()
        .unwrap()
        .to_string()
}

/// Runs `statements` on a new database file, closes it (a checkpoint) and
/// reopens it. Bind the result as `(_dir, db)`: the database closes before
/// its directory goes.
fn reopened_after_checkpoint(statements: &[&str]) -> (tempfile::TempDir, GrafeoDB) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("types.grafeo");
    let db = open(&path);
    for statement in statements {
        rows(&db, statement);
    }
    db.close().unwrap();
    drop(db);
    let db = open(&path);
    (dir, db)
}

/// The sidecar WAL of the database file at `path`.
fn sidecar_wal(path: &Path) -> PathBuf {
    let mut sidecar = path.as_os_str().to_owned();
    sidecar.push(".wal");
    PathBuf::from(sidecar)
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

// ── Property type names ─────────────────────────────────────────────

/// Statements that declare a property of an unknown type, each with that
/// type's name: every form of type DDL. `City` exists when they run.
const UNKNOWN_TYPES: &[(&str, &str)] = &[
    ("CREATE NODE TYPE Town (population INT32)", "INT32"),
    ("CREATE NODE TYPE Town (name STIRNG)", "STIRNG"),
    ("CREATE NODE TYPE Town (tags LIST<STIRNG>)", "STIRNG"),
    ("CREATE NODE TYPE Town (grid LIST<LIST<INT32>>)", "INT32"),
    ("CREATE EDGE TYPE ROAD (km INT32)", "INT32"),
    ("ALTER NODE TYPE City ADD population INT32", "INT32"),
    (
        "ALTER EDGE TYPE ROUTE ADD PROPERTY lanes SMALLINT",
        "SMALLINT",
    ),
    (
        "CREATE GRAPH TYPE atlas (NODE TYPE Town (population INT32))",
        "INT32",
    ),
    (
        "CREATE GRAPH TYPE atlas ((:Town {population INT32})-[:ROAD]->(:Town))",
        "INT32",
    ),
    (
        "CREATE GRAPH TYPE atlas { (:Town)-[:ROAD {km DECIMAL}]->(:Town) }",
        "DECIMAL",
    ),
];

/// A name that is not a property type Grafeo supports is refused, naming
/// it, and the statement declares nothing: it used to declare an `ANY`
/// property, which took every value.
#[test]
fn unknown_property_type_names_are_refused() {
    let db = GrafeoDB::new_in_memory();
    rows(&db, "CREATE NODE TYPE City (name STRING)");
    rows(&db, "CREATE EDGE TYPE ROUTE (km INT64)");
    let before = schema(&db);
    for (statement, name) in UNKNOWN_TYPES {
        let error = refusal(&db, statement);
        assert!(
            error.contains(name) && error.contains("not a property type"),
            "{statement}: {error}"
        );
        assert_eq!(schema(&db), before, "{statement} declared something");
    }
    // The town takes no population of any kind: no Town type exists.
    rows(&db, "INSERT (:Town {population: 'many'})");
}

/// Every type name `SHOW NODE TYPES` lists reads back in type DDL as the
/// same type, and so do the other spellings of each type and `ANY`.
#[test]
fn every_listed_type_name_reads_back() {
    let listed = "a STRING, b INT64, c FLOAT64, d BOOLEAN, e DATE, f TIME, g TIMESTAMP, \
                  h ZONED DATETIME, i LOCAL DATETIME, j DURATION, k LIST, l LIST<INT64>, \
                  m MAP, n BYTES, o NODE, p EDGE, q ANY";
    let db = GrafeoDB::new_in_memory();
    rows(&db, &format!("CREATE NODE TYPE Listed ({listed})"));
    assert_eq!(property_list(&db, "SHOW NODE TYPES", "Listed"), listed);

    rows(
        &db,
        "CREATE NODE TYPE Spelled (a varchar, b text, c int, d integer, e bigint, f float, \
         g double, h real, i bool, j datetime, k zoned_datetime, l localdatetime, \
         m interval, n array, o record, p binary, q blob, r relationship, s any)",
    );
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Spelled"),
        "a STRING, b STRING, c INT64, d INT64, e INT64, f FLOAT64, g FLOAT64, h FLOAT64, \
         i BOOLEAN, j TIMESTAMP, k ZONED DATETIME, l LOCAL DATETIME, m DURATION, n LIST, \
         o MAP, p BYTES, q BYTES, r EDGE, s ANY"
    );
}

/// Cypher's type DDL refuses an unknown property type as GQL's does.
#[cfg(feature = "cypher")]
#[test]
fn cypher_type_ddl_refuses_unknown_property_types() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Stop {zone :: INT32}) }")
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("INT32") && error.contains("not a property type"),
        "{error}"
    );
    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Stop {zone :: INTEGER}) }")
        .unwrap();
}

/// A WAL written by 0.5.x logged a property type as written, an unknown name
/// included. Replay still reads it, as `ANY` (what 0.5.x made of it), so
/// the database opens and the property keeps taking every value.
#[test]
fn an_unknown_type_name_in_an_older_wal_still_opens_as_any() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("older.grafeo");
    open(&path).close().unwrap();
    append_wal(
        &sidecar_wal(&path),
        vec![WalRecord::CreateNodeType {
            name: "Town".to_string(),
            properties: vec![
                ("population".to_string(), "INT32".to_string(), true),
                ("name".to_string(), "STRING".to_string(), true),
            ],
            constraints: Vec::new(),
        }],
    );

    let db = open(&path);
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Town"),
        "population ANY, name STRING"
    );
    rows(&db, "INSERT (:Town {population: 'many', name: 'Gus'})");
    assert!(refusal(&db, "INSERT (:Town {name: 3})").contains("name"));
}

// ── Keyword property names ──────────────────────────────────────────

/// The DDL that declares properties named like keywords, in every form.
const KEYWORD_NAMES: &[&str] = &[
    "CREATE NODE TYPE Trip (starts DATE, ends DATE, contains STRING, match INT64)",
    "CREATE EDGE TYPE LEG (starts INT64, order INT64 NOT NULL DEFAULT 3)",
    "ALTER NODE TYPE Trip ADD PROPERTY limit INT64",
    "ALTER NODE TYPE Trip ADD skip INT64",
    "ALTER NODE TYPE Trip ADD PROPERTY detach STRING",
    "ALTER NODE TYPE Trip DROP PROPERTY detach",
    "CREATE GRAPH TYPE tours ((:Tour {starts DATE, ends DATE})-[:STAGE {contains STRING}]->(:Tour))",
    "CREATE GRAPH TYPE hikes { (:Hike {starts DATE})-[:PART {ends INT64}]->(:Hike) }",
    "CREATE GRAPH TYPE rides (NODE TYPE Ride (starts DATE), EDGE TYPE HOP (ends INT64))",
];

/// The schema the statements of [`KEYWORD_NAMES`] declare, as SHOW lists it.
fn assert_keyword_schema(db: &GrafeoDB) {
    assert_eq!(
        property_list(db, "SHOW NODE TYPES", "Trip"),
        "starts DATE, ends DATE, contains STRING, match INT64, limit INT64, skip INT64"
    );
    assert_eq!(
        property_list(db, "SHOW EDGE TYPES", "LEG"),
        "starts INT64, order INT64 NOT NULL"
    );
    assert_eq!(
        property_list(db, "SHOW NODE TYPES", "Tour"),
        "starts DATE, ends DATE"
    );
    assert_eq!(property_list(db, "SHOW EDGE TYPES", "PART"), "ends INT64");
    assert_eq!(property_list(db, "SHOW NODE TYPES", "Ride"), "starts DATE");
}

/// Properties named like keywords (`starts`, `ends`, `contains`, `match`)
/// are declared in every form of type DDL, as INSERT takes them as property
/// keys, and writes are checked against them.
#[test]
fn keyword_property_names_are_declared_as_insert_takes_them() {
    let db = GrafeoDB::new_in_memory();
    for statement in KEYWORD_NAMES {
        rows(&db, statement);
    }
    assert_keyword_schema(&db);

    rows(
        &db,
        "INSERT (:Trip {starts: date('2024-03-19'), ends: date('2024-03-22'), \
         contains: 'Prague', match: 3})-[:LEG {starts: 19}]->(:Trip {match: 88})",
    );
    assert_eq!(
        single(&db, "MATCH ()-[l:LEG]->() RETURN l.order"),
        Value::Int64(3)
    );
    assert!(refusal(&db, "INSERT (:Trip {contains: 3})").contains("contains"));
    assert!(refusal(&db, "INSERT (:Trip {ends: 'later'})").contains("ends"));
    assert!(refusal(&db, "INSERT (:Hike {starts: 19})").contains("starts"));
}

#[test]
fn keyword_property_names_survive_a_checkpoint_and_a_reopen() {
    let (_dir, db) = reopened_after_checkpoint(KEYWORD_NAMES);
    assert_keyword_schema(&db);
}

#[test]
fn keyword_property_names_survive_a_wal_replay() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "keyword_property_names_survive_a_wal_replay",
        open,
        |db| {
            for statement in KEYWORD_NAMES {
                rows(db, statement);
            }
        },
    );
    // A v1 WAL record does not hold a default (see the `ddl-followups`
    // note), so LEG's `order` is compared by name and type only.
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Trip"),
        "starts DATE, ends DATE, contains STRING, match INT64, limit INT64, skip INT64"
    );
    assert!(property_list(&db, "SHOW EDGE TYPES", "LEG").starts_with("starts INT64, order INT64"));
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Tour"),
        "starts DATE, ends DATE"
    );
}

// ── Defaults ────────────────────────────────────────────────────────

/// Signed numbers as defaults (ISO's `<signed numeric literal>`), the
/// smallest INT64 included, and an integer default of a FLOAT64 property,
/// which is the float.
const SIGNED_DEFAULTS: &[&str] = &[
    "CREATE NODE TYPE Reading (celsius INT64 DEFAULT -3, trim FLOAT64 DEFAULT -0.5, \
     gain INT64 DEFAULT +19, floor INT64 DEFAULT -9223372036854775808, \
     scale FLOAT64 DEFAULT 88, mask INT64 DEFAULT -0x13)",
    "CREATE EDGE TYPE DRIFT (delta INT64 NOT NULL DEFAULT -88)",
    "ALTER NODE TYPE Reading ADD PROPERTY bias FLOAT64 DEFAULT -3",
];

fn assert_signed_defaults(db: &GrafeoDB) {
    rows(db, "INSERT (:Reading)-[:DRIFT]->(:Reading)");
    assert_eq!(
        rows(
            db,
            "MATCH (r:Reading)-[d:DRIFT]->() \
             RETURN r.celsius, r.trim, r.gain, r.floor, r.scale, r.mask, r.bias, d.delta"
        ),
        [[
            Value::Int64(-3),
            Value::Float64(-0.5),
            Value::Int64(19),
            Value::Int64(i64::MIN),
            Value::Float64(88.0),
            Value::Int64(-19),
            Value::Float64(-3.0),
            Value::Int64(-88),
        ]]
    );
}

/// A signed number is a default; `DEFAULT -3` was a syntax error. An
/// integer default of a FLOAT64 property is the float, so a node that relies
/// on it is written.
#[test]
fn signed_numbers_are_defaults() {
    let db = GrafeoDB::new_in_memory();
    for statement in SIGNED_DEFAULTS {
        rows(&db, statement);
    }
    assert_signed_defaults(&db);
}

#[test]
fn signed_defaults_survive_a_checkpoint_and_a_reopen() {
    let (_dir, db) = reopened_after_checkpoint(SIGNED_DEFAULTS);
    assert_signed_defaults(&db);
}

/// Replay of the DDL keeps the types (a v1 WAL record holds no default).
#[test]
fn signed_defaults_replay_their_types() {
    let (_dir, db) =
        common::replay::reopened_after_crash("signed_defaults_replay_their_types", open, |db| {
            for statement in SIGNED_DEFAULTS {
                rows(db, statement);
            }
        });
    assert_eq!(
        property_list(&db, "SHOW NODE TYPES", "Reading"),
        "celsius INT64, trim FLOAT64, gain INT64, floor INT64, scale FLOAT64, mask INT64, \
         bias FLOAT64"
    );
    assert_eq!(
        property_list(&db, "SHOW EDGE TYPES", "DRIFT"),
        "delta INT64 NOT NULL"
    );
}

/// Defaults a property cannot hold, each with a word of the refusal. `City`
/// and `ROUTE` exist when they run.
const WRONG_DEFAULTS: &[(&str, &str)] = &[
    ("CREATE NODE TYPE Town (x INT64 DEFAULT 'far')", "'far'"),
    ("CREATE NODE TYPE Town (x INT64 DEFAULT 2.5)", "2.5"),
    ("CREATE NODE TYPE Town (x INT64 DEFAULT TRUE)", "TRUE"),
    ("CREATE NODE TYPE Town (x STRING DEFAULT 3)", "3"),
    ("CREATE NODE TYPE Town (x STRING DEFAULT -3)", "-3"),
    ("CREATE NODE TYPE Town (x BOOLEAN DEFAULT 1)", "1"),
    ("CREATE NODE TYPE Town (x FLOAT64 DEFAULT 'far')", "'far'"),
    (
        "CREATE NODE TYPE Town (x DATE DEFAULT '2024-03-19')",
        "DATE",
    ),
    (
        "CREATE NODE TYPE Town (x LIST<STRING> DEFAULT 'Paris')",
        "LIST",
    ),
    ("CREATE NODE TYPE Town (x MAP DEFAULT 3)", "MAP"),
    (
        "CREATE NODE TYPE Town (x INT64 NOT NULL DEFAULT NULL)",
        "NOT NULL",
    ),
    (
        "CREATE NODE TYPE Town (x FLOAT64 DEFAULT 9007199254740993)",
        "9007199254740993",
    ),
    (
        "CREATE NODE TYPE Town (x INT64 DEFAULT 9223372036854775808)",
        "9223372036854775808",
    ),
    ("CREATE EDGE TYPE ROAD (km INT64 DEFAULT 'far')", "'far'"),
    (
        "ALTER NODE TYPE City ADD seats INT64 DEFAULT 'many'",
        "'many'",
    ),
    (
        "ALTER EDGE TYPE ROUTE ADD PROPERTY lanes INT64 DEFAULT 2.5",
        "2.5",
    ),
    (
        "CREATE GRAPH TYPE atlas (NODE TYPE Town (x INT64 DEFAULT 'far'))",
        "'far'",
    ),
    (
        "CREATE GRAPH TYPE atlas { (:Town {x INT64 DEFAULT 'far'})-[:ROAD]->(:Town) }",
        "'far'",
    ),
];

/// A default the property cannot hold is refused when the type is
/// declared, and the statement declares nothing: it used to be accepted,
/// and then every insert that relied on it failed.
#[test]
fn a_default_the_property_cannot_hold_is_refused() {
    let db = GrafeoDB::new_in_memory();
    rows(&db, "CREATE NODE TYPE City (name STRING)");
    rows(&db, "CREATE EDGE TYPE ROUTE (km INT64)");
    let before = schema(&db);
    for (statement, word) in WRONG_DEFAULTS {
        let error = refusal(&db, statement);
        assert!(
            error.contains("DEFAULT") && error.contains(word),
            "{statement}: {error}"
        );
        assert_eq!(schema(&db), before, "{statement} declared something");
    }
}

/// What every type takes as a default: a literal of its kind, `NULL` for a
/// property that may be null, anything for `ANY`, and a string with an
/// escaped quote as the string it spells (the backslash was kept).
#[test]
fn defaults_of_the_property_type_are_kept() {
    let db = GrafeoDB::new_in_memory();
    rows(
        &db,
        "CREATE NODE TYPE Stop (name STRING DEFAULT 'Gus\\'s stop', open BOOLEAN DEFAULT FALSE, \
         zone ANY DEFAULT 3, note ANY DEFAULT 'none', since DATE DEFAULT NULL, \
         ratio FLOAT64 DEFAULT 1.5e3)",
    );
    rows(&db, "INSERT (:Stop)");
    assert_eq!(
        rows(
            &db,
            "MATCH (s:Stop) RETURN s.name, s.open, s.zone, s.note, s.since, s.ratio"
        ),
        [[
            Value::from("Gus's stop"),
            Value::Bool(false),
            Value::Int64(3),
            Value::from("none"),
            Value::Null,
            Value::Float64(1500.0),
        ]]
    );
}

// ── DROP GRAPH TYPE ─────────────────────────────────────────────────

/// A closed graph type with one node type, and a graph typed by it.
#[cfg(feature = "cypher")]
const TYPED_GRAPH: &[&str] = &[
    "CREATE NODE TYPE City (name STRING)",
    "CREATE GRAPH TYPE atlas (NODE TYPE City)",
    "CREATE GRAPH europe TYPED atlas",
];

/// The graph type of the graph `europe`, as `SHOW CURRENT GRAPH TYPE` (in
/// Cypher) reports it: null for a graph without one.
#[cfg(feature = "cypher")]
fn type_of_europe(db: &GrafeoDB) -> Value {
    let session = db.session();
    session.execute("USE GRAPH europe").unwrap();
    let result = session.execute_cypher("SHOW CURRENT GRAPH TYPE").unwrap();
    assert_eq!(result.rows()[0][0], Value::from("europe"));
    result.rows()[0][1].clone()
}

/// The statements after which `europe` and `atlas` are new: the old graph
/// and graph type dropped, both created again, the graph without a type.
#[cfg(feature = "cypher")]
const RECREATED: &[&str] = &[
    "DROP GRAPH europe",
    "DROP GRAPH TYPE atlas",
    "CREATE GRAPH TYPE atlas (NODE TYPE City)",
    "CREATE GRAPH europe",
];

/// Dropping a graph type drops the bindings of graphs to it: a graph type
/// created later under the same name types no graph. The binding used to
/// stay and type the new graph `europe`, created without a type, by the new
/// graph type `atlas`.
#[cfg(feature = "cypher")]
#[test]
fn a_recreated_graph_type_types_no_graph() {
    let db = GrafeoDB::new_in_memory();
    for statement in TYPED_GRAPH {
        rows(&db, statement);
    }
    assert_eq!(type_of_europe(&db), Value::from("atlas"));
    for statement in RECREATED {
        rows(&db, statement);
    }
    assert_eq!(
        type_of_europe(&db),
        Value::Null,
        "the new europe has no type"
    );
}

/// The same after a WAL replay: the binding comes from the checkpoint, the
/// drops from the WAL.
#[cfg(feature = "cypher")]
#[test]
fn a_replayed_drop_of_a_graph_type_drops_the_bindings() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "a_replayed_drop_of_a_graph_type_drops_the_bindings",
        open,
        |db| {
            for statement in TYPED_GRAPH {
                rows(db, statement);
            }
            db.wal_checkpoint().unwrap();
            for statement in &RECREATED[..2] {
                rows(db, statement);
            }
        },
    );
    for statement in &RECREATED[2..] {
        rows(&db, statement);
    }
    assert_eq!(
        type_of_europe(&db),
        Value::Null,
        "the new europe has no type"
    );
}
