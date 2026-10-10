//! What type DDL declares is what the catalog keeps and what writes obey:
//!
//! - A graph type in ISO's brace form (ISO/IEC 39075:2024
//!   `<nested graph type specification>`, `CREATE GRAPH TYPE g { ... }`)
//!   declares the node and edge types its element patterns spell out, with
//!   their property types (`<property types specification>`), as the paren
//!   form does.
//! - A `MAP` property (Grafeo's map type) takes maps.
//! - An edge type's `DEFAULT` (a Grafeo extension of a property type) fills a
//!   property an insert leaves out, as a node type's does.
//!
//! Each holds in memory and after a reopen: after a checkpoint (catalog
//! section version 2) and after a WAL replay where the WAL's records carry
//! it.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test type_ddl_declarations
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file", feature = "wal"))]

mod common;

use std::path::Path;

use grafeo_common::types::{PropertyKey, Value};
use grafeo_engine::{Config, GrafeoDB};

/// A graph type in the brace form: a node type with a NOT NULL property, an
/// edge type with endpoints and a property, and a node type on its own.
const BRACE_FORM: &str = "CREATE GRAPH TYPE routes { \
     (:City {name STRING NOT NULL, population INT64})-[:ROUTE {km INT64}]->(:City), \
     (:Country {code STRING}) }";

/// [`BRACE_FORM`] in the paren form, which declared its types already.
const PAREN_FORM: &str = "CREATE GRAPH TYPE routes ( \
     (:City {name STRING NOT NULL, population INT64})-[:ROUTE {km INT64}]->(:City), \
     (:Country {code STRING}) )";

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

/// The row of `query` whose first column is `name`.
fn row_of(db: &GrafeoDB, query: &str, name: &str) -> Vec<Value> {
    rows(db, query)
        .into_iter()
        .find(|row| row[0] == Value::from(name))
        .unwrap_or_else(|| panic!("{query} lists no {name}"))
}

/// A new database at `path`, a single file with its sidecar WAL.
fn open(path: &Path) -> GrafeoDB {
    GrafeoDB::with_config(Config::persistent(path)).unwrap()
}

/// Runs `statements` on a new database file, closes it (a checkpoint) and
/// reopens it. Returns the directory and the reopened database: bind them
/// as `(_dir, db)`, so the database closes before its directory goes.
fn reopened_after_checkpoint(statements: &[&str]) -> (tempfile::TempDir, GrafeoDB) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("types.grafeo");
    let db = open(&path);
    for statement in statements {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db.close().unwrap();
    drop(db);
    let db = open(&path);
    (dir, db)
}

// ── Brace-form graph types ──────────────────────────────────────────

/// The brace form keeps the element types and their properties, which it
/// used to drop: it declared the names in the graph type only.
#[test]
fn a_brace_form_graph_type_declares_what_the_paren_form_declares() {
    let brace = GrafeoDB::new_in_memory();
    brace.execute(BRACE_FORM).unwrap();
    let paren = GrafeoDB::new_in_memory();
    paren.execute(PAREN_FORM).unwrap();

    assert_eq!(schema(&brace), schema(&paren));
    // What the comparison covers holds the declarations, so it cannot pass
    // on empty output.
    assert_eq!(
        row_of(&brace, "SHOW NODE TYPES", "City")[1],
        Value::from("name STRING NOT NULL, population INT64")
    );
    assert_eq!(
        row_of(&brace, "SHOW NODE TYPES", "Country")[1],
        Value::from("code STRING")
    );
    assert_eq!(
        row_of(&brace, "SHOW EDGE TYPES", "ROUTE"),
        [
            Value::from("ROUTE"),
            Value::from("km INT64"),
            Value::from("City"),
            Value::from("City")
        ]
    );
}

/// The writes obey the properties a brace-form graph type declares: their
/// types and NOT NULL.
fn assert_brace_form_checks_properties(db: &GrafeoDB) {
    let wrong_type = refusal(db, "INSERT (:City {name: 'Paris', population: 'many'})");
    assert!(wrong_type.contains("population"), "{wrong_type}");
    let missing = refusal(db, "INSERT (:City {population: 3})");
    assert!(
        missing.contains("missing required property 'name'"),
        "{missing}"
    );
    let wrong_edge = refusal(
        db,
        "INSERT (:City {name: 'Paris'})-[:ROUTE {km: 'far'}]->(:City {name: 'Prague'})",
    );
    assert!(wrong_edge.contains("km"), "{wrong_edge}");

    db.execute(
        "INSERT (:City {name: 'Amsterdam', population: 19})         -[:ROUTE {km: 88}]->(:City {name: 'Barcelona'})",
    )
    .unwrap();
    assert_eq!(
        single(db, "MATCH (:City)-[r:ROUTE]->(:City) RETURN r.km"),
        Value::Int64(88)
    );
}

/// The writes obey the endpoints of an edge type a brace-form graph type
/// declares.
fn assert_brace_form_checks_endpoints(db: &GrafeoDB) {
    let wrong_source = refusal(
        db,
        "INSERT (:Country {code: 'NL'})-[:ROUTE {km: 3}]->(:City {name: 'Berlin'})",
    );
    assert!(wrong_source.contains("ROUTE"), "{wrong_source}");
}

#[test]
fn a_brace_form_graph_type_checks_the_writes() {
    let db = GrafeoDB::new_in_memory();
    db.execute(BRACE_FORM).unwrap();
    assert_brace_form_checks_properties(&db);
    assert_brace_form_checks_endpoints(&db);
}

#[test]
fn a_brace_form_graph_type_survives_a_checkpoint_and_a_reopen() {
    let expected = {
        let reference = GrafeoDB::new_in_memory();
        reference.execute(BRACE_FORM).unwrap();
        schema(&reference)
    };
    let (_dir, db) = reopened_after_checkpoint(&[BRACE_FORM]);
    assert_eq!(schema(&db), expected);
    assert_brace_form_checks_properties(&db);
    assert_brace_form_checks_endpoints(&db);
}

/// Replay rebuilds the element types and their properties. It does not
/// rebuild an edge type's endpoints, whatever form declared them: the WAL's
/// `CreateEdgeType` record carries none (the records of WAL v2 will), so
/// the endpoint columns of `SHOW EDGE TYPES` are left out here.
#[test]
fn a_brace_form_graph_type_survives_a_wal_replay() {
    let (_dir, db) = common::replay::reopened_after_crash(
        "a_brace_form_graph_type_survives_a_wal_replay",
        open,
        |db| {
            db.execute(BRACE_FORM).unwrap();
        },
    );
    let reference = GrafeoDB::new_in_memory();
    reference.execute(BRACE_FORM).unwrap();
    let without_endpoints = |mut schema: Vec<Vec<Vec<Value>>>| {
        for edge_type in &mut schema[1] {
            edge_type.truncate(2);
        }
        schema
    };
    assert_eq!(
        without_endpoints(schema(&db)),
        without_endpoints(schema(&reference))
    );
    assert_eq!(
        row_of(&db, "SHOW EDGE TYPES", "ROUTE")[..2],
        [Value::from("ROUTE"), Value::from("km INT64")]
    );
    assert_brace_form_checks_properties(&db);
}

// ── MAP properties ──────────────────────────────────────────────────

/// A node type and an edge type with a MAP property each.
const MAP_TYPES: &[&str] = &[
    "CREATE NODE TYPE Config (settings MAP NOT NULL)",
    "CREATE EDGE TYPE TUNED (changes MAP)",
];

/// A MAP property takes maps, on insert, on SET and on an edge, and refuses
/// a string, a number and a list.
fn assert_map_properties_take_maps(db: &GrafeoDB) {
    let settings = single(
        db,
        "INSERT (c:Config {settings: {mode: 'fast', level: 3}}) RETURN c.settings",
    );
    let map = settings.as_map().unwrap_or_else(|| panic!("{settings:?}"));
    assert_eq!(
        map.get(&PropertyKey::from("mode")),
        Some(&Value::from("fast"))
    );
    assert_eq!(map.get(&PropertyKey::from("level")), Some(&Value::Int64(3)));

    for value in ["'fast'", "19", "[3, 19]"] {
        let refused = refusal(db, &format!("INSERT (:Config {{settings: {value}}})"));
        assert!(refused.contains("settings"), "{value}: {refused}");
    }
    assert_eq!(
        single(
            db,
            "MATCH (c:Config) SET c.settings = {mode: 'slow'} RETURN c.settings.mode"
        ),
        Value::from("slow")
    );
    let refused = refusal(db, "MATCH (c:Config) SET c.settings = 88");
    assert!(refused.contains("settings"), "{refused}");

    assert_eq!(
        single(
            db,
            "MATCH (c:Config) INSERT (c)-[t:TUNED {changes: {level: 19}}]->(c) \
             RETURN t.changes.level"
        ),
        Value::Int64(19)
    );
    let refused = refusal(
        db,
        "MATCH (c:Config) INSERT (c)-[:TUNED {changes: 'faster'}]->(c)",
    );
    assert!(refused.contains("changes"), "{refused}");
}

#[test]
fn a_map_property_takes_maps_and_refuses_other_values() {
    let db = GrafeoDB::new_in_memory();
    for statement in MAP_TYPES {
        db.execute(statement).unwrap();
    }
    assert_map_properties_take_maps(&db);
}

/// Cypher's writes check MAP properties as GQL's do, and Cypher's own type
/// DDL (`ALTER CURRENT GRAPH TYPE SET`) declares one.
#[cfg(feature = "cypher")]
#[test]
fn cypher_writes_obey_map_properties() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE Config (settings MAP)")
        .unwrap();
    db.execute_cypher("CREATE (:Config {settings: {mode: 'fast'}})")
        .unwrap();
    let refused = db
        .execute_cypher("CREATE (:Config {settings: 'fast'})")
        .unwrap_err()
        .to_string();
    assert!(refused.contains("settings"), "{refused}");

    db.execute_cypher("ALTER CURRENT GRAPH TYPE SET { (:Setting {values :: MAP}) }")
        .unwrap();
    db.execute_cypher("CREATE (:Setting {values: {level: 3}})")
        .unwrap();
    let refused = db
        .execute_cypher("CREATE (:Setting {values: 3})")
        .unwrap_err()
        .to_string();
    assert!(refused.contains("values"), "{refused}");
    assert_eq!(
        single(&db, "MATCH (s:Setting) RETURN s.values.level"),
        Value::Int64(3)
    );
}

#[test]
fn map_properties_survive_a_checkpoint_and_a_reopen() {
    let (_dir, db) = reopened_after_checkpoint(MAP_TYPES);
    assert_eq!(
        row_of(&db, "SHOW NODE TYPES", "Config")[1],
        Value::from("settings MAP NOT NULL")
    );
    assert_eq!(
        row_of(&db, "SHOW EDGE TYPES", "TUNED")[1],
        Value::from("changes MAP")
    );
    assert_map_properties_take_maps(&db);
}

#[test]
fn map_properties_survive_a_wal_replay() {
    let (_dir, db) =
        common::replay::reopened_after_crash("map_properties_survive_a_wal_replay", open, |db| {
            for statement in MAP_TYPES {
                db.execute(statement).unwrap();
            }
        });
    assert_eq!(
        row_of(&db, "SHOW NODE TYPES", "Config")[1],
        Value::from("settings MAP NOT NULL")
    );
    assert_map_properties_take_maps(&db);
}

// ── Edge type defaults ──────────────────────────────────────────────

/// City with a node type default, and ROUTE between cities with an edge type
/// default, as `scripts/released_fixtures.py` declares them.
const ROUTES: &[&str] = &[
    "CREATE NODE TYPE City (name STRING, country STRING DEFAULT 'NL')",
    "CREATE EDGE TYPE ROUTE CONNECTING (City) TO (City) (km INT64 DEFAULT 88, toll BOOL)",
];

fn routes() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in ROUTES {
        db.execute(statement).unwrap();
    }
    db
}

/// An insert that leaves the property out gets the default; one that gives
/// it a value keeps the value; a property without a default stays missing.
fn assert_route_defaults(db: &GrafeoDB) {
    db.execute("INSERT (:City {name: 'Paris'}), (:City {name: 'Prague'})")
        .unwrap();
    assert_eq!(
        single(
            db,
            "MATCH (p:City {name: 'Paris'}), (q:City {name: 'Prague'}) \
             INSERT (q)-[r:ROUTE]->(p) RETURN r.km"
        ),
        Value::Int64(88)
    );
    assert_eq!(
        single(
            db,
            "MATCH (p:City {name: 'Paris'}), (q:City {name: 'Prague'}) \
             INSERT (p)-[r:ROUTE {km: 3}]->(q) RETURN r.km"
        ),
        Value::Int64(3)
    );
    assert_eq!(
        rows(
            db,
            "INSERT (:City {name: 'Berlin'})-[r:ROUTE]->(c:City {name: 'Amsterdam'}) \
             RETURN r.km, r.toll, c.country"
        ),
        [[Value::Int64(88), Value::Null, Value::from("NL")]]
    );
}

#[test]
fn an_edge_type_default_fills_a_property_an_insert_leaves_out() {
    assert_route_defaults(&routes());
}

/// A NOT NULL property with a default is accepted without a value because
/// the default gives it one: the edge must have it.
#[test]
fn a_not_null_edge_property_with_a_default_is_never_missing() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE EDGE TYPE LEG (minutes INT64 NOT NULL DEFAULT 19)")
        .unwrap();
    assert_eq!(
        single(&db, "INSERT (:Stop)-[l:LEG]->(:Stop) RETURN l.minutes"),
        Value::Int64(19)
    );
    assert_eq!(
        single(
            &db,
            "MATCH ()-[l:LEG]->() WHERE l.minutes IS NULL RETURN count(l)"
        ),
        Value::Int64(0)
    );
}

/// Every way to create an edge applies the default: Cypher's CREATE and
/// MERGE (with and without ON CREATE expressions that read the new edge),
/// and the direct API.
#[cfg(feature = "cypher")]
#[test]
fn every_edge_write_applies_the_default() {
    let db = routes();
    db.execute("INSERT (:City {name: 'Paris'}), (:City {name: 'Prague'})")
        .unwrap();
    let cypher_km = |query: &str| {
        let rows = db.execute_cypher(query).unwrap().rows().to_vec();
        assert_eq!(rows.len(), 1, "{query}: {rows:?}");
        rows[0][0].clone()
    };
    assert_eq!(
        cypher_km(
            "MATCH (p:City {name: 'Paris'}), (q:City {name: 'Prague'}) \
             CREATE (p)-[r:ROUTE]->(q) RETURN r.km"
        ),
        Value::Int64(88),
        "Cypher CREATE"
    );
    assert_eq!(
        cypher_km(
            "MATCH (p:City {name: 'Paris'}), (q:City {name: 'Prague'}) \
             MERGE (q)-[r:ROUTE]->(p) RETURN r.km"
        ),
        Value::Int64(88),
        "Cypher MERGE"
    );
    assert_eq!(
        cypher_km(
            "MATCH (p:City {name: 'Paris'}), (q:City {name: 'Prague'}) \
             MERGE (q)-[r:ROUTE {toll: true}]->(p) \
             ON CREATE SET r.via = p.name + '-' + q.name RETURN r.km"
        ),
        Value::Int64(88),
        "Cypher MERGE with an ON CREATE expression"
    );

    let paris = db
        .create_node_with_props(&["City"], [("name", "Paris")])
        .unwrap();
    let berlin = db
        .create_node_with_props(&["City"], [("name", "Berlin")])
        .unwrap();
    let route = db.create_edge(paris, berlin, "ROUTE").unwrap();
    assert_eq!(
        db.get_edge(route).unwrap().get_property("km").cloned(),
        Some(Value::Int64(88)),
        "the direct API"
    );
}

/// The default applies in a named graph, and in a graph typed with a graph
/// type that lists the edge type.
#[test]
fn edge_type_defaults_apply_in_named_and_typed_graphs() {
    let db = routes();
    db.execute("CREATE GRAPH TYPE travel (NODE TYPE City, EDGE TYPE ROUTE)")
        .unwrap();
    db.execute("CREATE GRAPH atlas").unwrap();
    db.execute("CREATE GRAPH trips TYPED travel").unwrap();
    for graph in ["atlas", "trips"] {
        let session = db.session();
        session
            .execute(&format!("SESSION SET GRAPH {graph}"))
            .unwrap();
        let result = session
            .execute(
                "INSERT (:City {name: 'Paris'})-[r:ROUTE]->(:City {name: 'Prague'}) \
                 RETURN r.km",
            )
            .unwrap();
        assert_eq!(result.rows(), [[Value::Int64(88)]], "graph {graph}");
    }
}

#[test]
fn edge_type_defaults_survive_a_checkpoint_and_a_reopen() {
    let (_dir, db) = reopened_after_checkpoint(ROUTES);
    assert_route_defaults(&db);
}

// ── Inline element type defaults ────────────────────────────────────

/// The three forms of a graph type's inline element types, each with a node
/// type default (`zone`) and an edge type default (`minutes`).
const INLINE_DEFAULTS: &[(&str, &str)] = &[
    (
        "paren",
        "CREATE GRAPH TYPE trips ((:Stop {name STRING, zone STRING DEFAULT 'A'})\
         -[:LEG {minutes INT64 DEFAULT 19}]->(:Stop))",
    ),
    (
        "brace",
        "CREATE GRAPH TYPE trips { (:Stop {name STRING, zone STRING DEFAULT 'A'})\
         -[:LEG {minutes INT64 DEFAULT 19}]->(:Stop) }",
    ),
    (
        "verbose",
        "CREATE GRAPH TYPE trips (NODE TYPE Stop (name STRING, zone STRING DEFAULT 'A'), \
         EDGE TYPE LEG (minutes INT64 DEFAULT 19))",
    ),
];

fn assert_inline_defaults(form: &str, db: &GrafeoDB) {
    assert_eq!(
        rows(
            db,
            "INSERT (a:Stop {name: 'Berlin'})-[l:LEG]->(b:Stop {name: 'Prague', zone: 'B'}) \
             RETURN a.zone, l.minutes, b.zone"
        ),
        [[Value::from("A"), Value::Int64(19), Value::from("B")]],
        "{form} form"
    );
}

/// A default in an inline element type of a graph type fills a property an
/// insert leaves out, as one in CREATE NODE TYPE or CREATE EDGE TYPE does: in
/// the paren, brace and verbose forms, and after a checkpoint and a reopen.
#[test]
fn inline_element_type_defaults_fill_missing_properties() {
    for (form, ddl) in INLINE_DEFAULTS {
        let db = GrafeoDB::new_in_memory();
        db.execute(ddl).unwrap();
        assert_inline_defaults(form, &db);
        let (_dir, db) = reopened_after_checkpoint(&[ddl]);
        assert_inline_defaults(form, &db);
    }
}
