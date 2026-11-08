//! A schema's closed graph type lists its node and edge types however they came into it: by `CREATE
//! GRAPH TYPE`, by `ALTER GRAPH TYPE ... ADD`, or bare, as 0.6.1's `ALTER` stored an added one under a
//! schema. Each is held while a graph is bound to the type (`DROP NODE TYPE` is refused), `ALTER GRAPH
//! TYPE ... DROP` takes each out, and the refusal names a statement that runs as written.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main schema_graph_type_entries::
//! ```

use std::path::Path;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const TYPES: &[&str] = &[
    "CREATE SCHEMA s",
    "SESSION SET SCHEMA s",
    "CREATE NODE TYPE Held (name STRING)",
    "CREATE NODE TYPE Other (k INTEGER)",
    "CREATE NODE TYPE Extra (k INTEGER)",
    "CREATE EDGE TYPE Link (w INTEGER)",
    "CREATE EDGE TYPE Lone (w INTEGER)",
    "CREATE EDGE TYPE Added (w INTEGER)",
    "CREATE GRAPH TYPE held (NODE TYPE Held, NODE TYPE Other, EDGE TYPE Link, EDGE TYPE Lone)",
    "CREATE GRAPH g TYPED held",
];

fn run_all(session: &Session, statements: &[&str]) {
    for statement in statements {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
}

/// The schema `s`, its types, the closed graph type `held` and the empty graph `g` bound to it, with
/// `Extra` and `Added` taken in by `ALTER GRAPH TYPE ... ADD`.
fn in_a_schema() -> (GrafeoDB, Session) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    run_all(&session, TYPES);
    run_all(
        &session,
        &[
            "ALTER GRAPH TYPE held ADD NODE TYPE Extra",
            "ALTER GRAPH TYPE held ADD EDGE TYPE Added",
        ],
    );
    (db, session)
}

fn refused_as_listed(session: &Session, statement: &str) {
    let error = session.execute(statement).expect_err(statement).to_string();
    assert!(error.contains("closed graph type"), "{statement}: {error}");
}

/// Whether `SHOW {kind} TYPES` lists `name`, bare or under the schema.
fn lists(session: &Session, kind: &str, name: &str) -> bool {
    let shown = session
        .execute(&format!("SHOW {kind} TYPES"))
        .unwrap_or_else(|error| panic!("SHOW {kind} TYPES: {error}"));
    let qualified = format!("/{name}");
    shown.rows().iter().any(|row| match row.first() {
        Some(Value::String(text)) => text.as_str() == name || text.as_str().ends_with(&qualified),
        _ => false,
    })
}

/// `ALTER GRAPH TYPE held DROP {kind} TYPE {name}` takes the type out, and the type goes with it, as
/// no other graph type lists it (ISO/IEC 39075 12.7, as
/// `type_scope_across_graphs::an_altered_drop_takes_a_type_only_it_listed` pins).
fn alter_drop_takes(session: &Session, kind: &str, name: &str) {
    let statement = format!("ALTER GRAPH TYPE held DROP {kind} TYPE {name}");
    let status = session
        .execute(&statement)
        .unwrap_or_else(|error| panic!("{statement}: {error}"))
        .status_message
        .unwrap_or_default();
    assert!(
        status.contains(&format!("and its types {name}")),
        "{statement}: {status}"
    );
    assert!(
        !lists(session, kind, name),
        "{statement}: the type went with it"
    );
}

/// A type `ALTER GRAPH TYPE ... ADD` took in under a schema is held as one `CREATE GRAPH TYPE` listed:
/// its definition stays, and still applies in the graph.
#[test]
fn an_added_type_is_held_like_a_created_one() {
    for statement in [
        "DROP NODE TYPE Held",
        "DROP NODE TYPE Extra",
        "DROP EDGE TYPE Link",
        "DROP EDGE TYPE Added",
    ] {
        let (_db, session) = in_a_schema();
        refused_as_listed(&session, statement);
        run_all(&session, &["USE GRAPH g"]);
        // The definitions stayed: they still decide what a write may hold.
        run_all(
            &session,
            &["INSERT (:Extra {k: 1})-[:Added {w: 1}]->(:Held {name: 'Alix'})"],
        );
        assert!(
            session.execute("INSERT (:Extra {k: 'one'})").is_err(),
            "{statement}: Extra's k is still an integer"
        );
    }
}

/// `ALTER GRAPH TYPE ... DROP` takes a type out, whether `CREATE GRAPH TYPE` listed it or `ALTER ...
/// ADD` took it in: the type goes with it, and the graph refuses it from then on.
#[test]
fn alter_drop_takes_a_type_out_however_it_came_in() {
    for (kind, name, write) in [
        ("NODE", "Other", "INSERT (:Other {k: 1})"),
        ("NODE", "Extra", "INSERT (:Extra {k: 1})"),
        (
            "EDGE",
            "Lone",
            "INSERT (:Held {name: 'Gus'})-[:Lone {w: 1}]->(:Held {name: 'Mia'})",
        ),
        (
            "EDGE",
            "Added",
            "INSERT (:Held {name: 'Gus'})-[:Added {w: 1}]->(:Held {name: 'Mia'})",
        ),
    ] {
        let (_db, session) = in_a_schema();
        alter_drop_takes(&session, kind, name);
        run_all(&session, &["USE GRAPH g"]);
        assert!(
            session.execute(write).is_err(),
            "{name}: the closed type no longer lists it"
        );
    }
}

/// The refusal suggests the `ALTER GRAPH TYPE` statement that takes the type out, with the graph type
/// named as a statement writes it (`held`, not the stored `s/held`), and that statement runs and takes
/// the type with it.
#[test]
fn the_refusal_names_a_statement_that_runs() {
    for (kind, name) in [
        ("NODE", "Held"),
        ("NODE", "Extra"),
        ("EDGE", "Link"),
        ("EDGE", "Added"),
    ] {
        let (_db, session) = in_a_schema();
        let statement = format!("DROP {kind} TYPE {name}");
        let error = session
            .execute(&statement)
            .expect_err(&statement)
            .to_string();
        let suggested = format!("ALTER GRAPH TYPE held DROP {kind} TYPE {name}");
        assert!(error.contains(&suggested), "{statement}: {error}");
        assert!(!error.contains("s/held"), "{statement}: {error}");
        alter_drop_takes(&session, kind, name);
    }
}

/// The file 0.6.1 wrote (`fixtures/schema_alter_bare`, made by `scripts/schema_alter_fixture.py`): its
/// `Extra` and `Added` are listed bare. They are held as qualified entries are, `ALTER ... DROP` takes
/// them out, with their types, once the graph holds none, and so it stays after a reopen.
#[test]
fn a_bare_entry_from_0_6_1_is_held_and_taken_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema_alter_bare.grafeo");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/schema_alter_bare/0.6.1.grafeo"),
        &path,
    )
    .unwrap();
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        run_all(&session, &["SESSION SET SCHEMA s"]);
        for statement in [
            "DROP NODE TYPE Extra",
            "DROP EDGE TYPE Added",
            "DROP NODE TYPE Held",
        ] {
            refused_as_listed(&session, statement);
        }
        // The graph holds an Extra and an Added: the type cannot go while they are there.
        assert!(
            session
                .execute("ALTER GRAPH TYPE held DROP NODE TYPE Extra")
                .is_err(),
            "the graph holds an Extra"
        );
        run_all(
            &session,
            &["USE GRAPH g", "MATCH (n:Extra) DETACH DELETE n"],
        );
        alter_drop_takes(&session, "NODE", "Extra");
        alter_drop_takes(&session, "EDGE", "Added");
        assert!(session.execute("INSERT (:Extra {k: 2})").is_err());
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    run_all(&session, &["SESSION SET SCHEMA s", "USE GRAPH g"]);
    assert!(
        session.execute("INSERT (:Extra {k: 2})").is_err(),
        "taken out after a reopen"
    );
    assert!(!lists(&session, "NODE", "Extra"), "gone after a reopen");
    assert!(!lists(&session, "EDGE", "Added"), "gone after a reopen");
    refused_as_listed(&session, "DROP NODE TYPE Held");
    db.close().unwrap();
}

/// The same file, opened, checkpointed and reopened: the entries it listed bare stay held across the
/// checkpoint, and the graph still takes their writes and refuses an unlisted label.
#[test]
fn a_bare_entry_from_0_6_1_stays_held_after_a_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("schema_alter_bare.grafeo");
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/schema_alter_bare/0.6.1.grafeo"),
        &path,
    )
    .unwrap();
    {
        let db = GrafeoDB::open(&path).unwrap();
        db.wal_checkpoint().unwrap();
        db.close().unwrap();
    }
    for round in 0..2 {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        run_all(&session, &["SESSION SET SCHEMA s"]);
        for (kind, name) in [("NODE", "Extra"), ("EDGE", "Added"), ("NODE", "Held")] {
            let statement = format!("DROP {kind} TYPE {name}");
            let error = session
                .execute(&statement)
                .expect_err(&statement)
                .to_string();
            assert!(
                error.contains(&format!("ALTER GRAPH TYPE held DROP {kind} TYPE {name}")),
                "round {round}, {statement}: {error}"
            );
            assert!(lists(&session, kind, name), "round {round}: {name} stays");
        }
        run_all(
            &session,
            &[
                "USE GRAPH g",
                "INSERT (:Extra {k: 3})-[:Added {w: 3}]->(:Held {name: 'Mia'})",
            ],
        );
        assert!(
            session.execute("INSERT (:Stranger {k: 1})").is_err(),
            "round {round}: the closed type still refuses what it does not list"
        );
        db.wal_checkpoint().unwrap();
        db.close().unwrap();
    }
}
