//! A schema statement belongs to its transaction, as a data write does (#727).
//!
//! Tests first, from the issue: a rollback, or a rollback to a savepoint, undoes it with the data
//! writes. A commit makes it durable with those writes, as one group. The transaction's later
//! statements see its own schema changes, and other transactions do not until the commit. A crash
//! before the commit leaves nothing, and a crash after a rollback leaves nothing either. A
//! statement claims its name: another statement on that name fails at once with `GRAFEO-T001`.
//! The catalog is read with SHOW, or a CALL for procedures.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main schema_statements_in_a_transaction::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "wal"))]

use std::path::{Path, PathBuf};

use grafeo_common::testing::child_process;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const INDEXES: &str = "SHOW INDEXES";
const NODE_TYPES: &str = "SHOW NODE TYPES";
const EDGE_TYPES: &str = "SHOW EDGE TYPES";
const GRAPH_TYPES: &str = "SHOW GRAPH TYPES";
const CONSTRAINTS: &str = "SHOW CONSTRAINTS";
const GRAPHS: &str = "SHOW GRAPHS";
const SCHEMAS: &str = "SHOW SCHEMAS";
const PROCEDURE: &str = "CALL p1()";

const HASH_INDEX: &str = "CREATE INDEX i1 FOR (n:Person) ON (n.age)";
const BTREE_INDEX: &str = "CREATE INDEX i1 FOR (n:Person) ON (n.age) USING BTREE";
const TEXT_INDEX: &str = "CREATE INDEX i1 FOR (n:Person) ON (n.name) USING TEXT";
const VECTOR_INDEX: &str =
    "CREATE VECTOR INDEX i1 ON :Person(embedding) DIMENSION 2 METRIC 'cosine'";
const NODE_TYPE: &str = "CREATE NODE TYPE City (name STRING)";
const EDGE_TYPE: &str = "CREATE EDGE TYPE ROAD (km INTEGER)";
const GRAPH_TYPE: &str = "CREATE GRAPH TYPE atlas { (:Town {size INT64})-[:ROAD]->(:Town) }";
const UNIQUE: &str = "CREATE CONSTRAINT c1 FOR (n:Person) ON (n.email) UNIQUE";
const STORED: &str = "CREATE PROCEDURE p1() RETURNS (v INTEGER) AS { RETURN 1 AS v }";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Language {
    Gql,
    Cypher,
}
use Language::{Cypher, Gql};

/// One statement form: what runs before the transaction, what runs in it, what reads it.
struct Kind {
    name: &'static str,
    language: Language,
    before: Option<&'static str>,
    statement: &'static str,
    read: &'static str,
}

const fn kind(
    name: &'static str,
    language: Language,
    before: Option<&'static str>,
    statement: &'static str,
    read: &'static str,
) -> Kind {
    Kind {
        name,
        language,
        before,
        statement,
        read,
    }
}

/// Every schema statement form, in GQL and Cypher.
fn kinds() -> Vec<Kind> {
    let mut all = vec![
        kind("create graph", Gql, None, "CREATE GRAPH g2", GRAPHS),
        kind(
            "drop graph",
            Gql,
            Some("CREATE GRAPH g2"),
            "DROP GRAPH g2",
            GRAPHS,
        ),
        kind("create schema", Gql, None, "CREATE SCHEMA s2", SCHEMAS),
        kind(
            "drop schema",
            Gql,
            Some("CREATE SCHEMA s2"),
            "DROP SCHEMA s2",
            SCHEMAS,
        ),
        kind("create hash index", Gql, None, HASH_INDEX, INDEXES),
        kind("create btree index", Gql, None, BTREE_INDEX, INDEXES),
        kind(
            "drop hash index",
            Gql,
            Some(HASH_INDEX),
            "DROP INDEX i1",
            INDEXES,
        ),
        kind(
            "drop btree index",
            Gql,
            Some(BTREE_INDEX),
            "DROP INDEX i1",
            INDEXES,
        ),
        kind(
            "create node type",
            Gql,
            None,
            "CREATE NODE TYPE City (name STRING NOT NULL)",
            NODE_TYPES,
        ),
        kind(
            "alter node type",
            Gql,
            Some(NODE_TYPE),
            "ALTER NODE TYPE City ADD PROPERTY size INTEGER",
            NODE_TYPES,
        ),
        kind(
            "drop node type",
            Gql,
            Some(NODE_TYPE),
            "DROP NODE TYPE City",
            NODE_TYPES,
        ),
        kind("create edge type", Gql, None, EDGE_TYPE, EDGE_TYPES),
        kind(
            "alter edge type",
            Gql,
            Some(EDGE_TYPE),
            "ALTER EDGE TYPE ROAD ADD PROPERTY toll STRING",
            EDGE_TYPES,
        ),
        kind(
            "drop edge type",
            Gql,
            Some(EDGE_TYPE),
            "DROP EDGE TYPE ROAD",
            EDGE_TYPES,
        ),
        kind("create graph type", Gql, None, GRAPH_TYPE, GRAPH_TYPES),
        kind(
            "alter graph type",
            Gql,
            Some(GRAPH_TYPE),
            "ALTER GRAPH TYPE atlas ADD EDGE TYPE RAIL",
            GRAPH_TYPES,
        ),
        kind(
            "drop graph type",
            Gql,
            Some(GRAPH_TYPE),
            "DROP GRAPH TYPE atlas",
            GRAPH_TYPES,
        ),
        kind(
            "create not null constraint",
            Gql,
            None,
            "CREATE CONSTRAINT c1 FOR (n:Person) ON (n.email) NOT NULL",
            CONSTRAINTS,
        ),
        kind("create unique constraint", Gql, None, UNIQUE, CONSTRAINTS),
        kind(
            "drop constraint",
            Gql,
            Some(UNIQUE),
            "DROP CONSTRAINT c1",
            CONSTRAINTS,
        ),
        kind("create procedure", Gql, None, STORED, PROCEDURE),
        kind(
            "drop procedure",
            Gql,
            Some(STORED),
            "DROP PROCEDURE p1",
            PROCEDURE,
        ),
        kind("cypher create index", Cypher, None, HASH_INDEX, INDEXES),
        kind(
            "cypher drop index",
            Cypher,
            Some(HASH_INDEX),
            "DROP INDEX i1",
            INDEXES,
        ),
        kind(
            "cypher not null constraint",
            Cypher,
            None,
            "CREATE CONSTRAINT c1 FOR (n:Person) REQUIRE n.email IS NOT NULL",
            CONSTRAINTS,
        ),
        kind(
            "cypher unique constraint",
            Cypher,
            None,
            "CREATE CONSTRAINT c1 FOR (n:Person) REQUIRE n.email IS UNIQUE",
            CONSTRAINTS,
        ),
        kind(
            "cypher drop constraint",
            Cypher,
            Some(UNIQUE),
            "DROP CONSTRAINT c1",
            CONSTRAINTS,
        ),
    ];
    if cfg!(feature = "text-index") {
        all.push(kind("create text index", Gql, None, TEXT_INDEX, INDEXES));
        all.push(kind(
            "drop text index",
            Gql,
            Some(TEXT_INDEX),
            "DROP INDEX i1",
            INDEXES,
        ));
    }
    if cfg!(feature = "vector-index") {
        all.push(kind(
            "create vector index",
            Gql,
            None,
            VECTOR_INDEX,
            INDEXES,
        ));
        all.push(kind(
            "drop vector index",
            Gql,
            Some(VECTOR_INDEX),
            "DROP INDEX i1",
            INDEXES,
        ));
    }
    all
}

fn run(
    session: &Session,
    language: Language,
    statement: &str,
) -> grafeo_common::utils::error::Result<()> {
    match language {
        Gql => session.execute(statement).map(|_| ()),
        Cypher => session.execute_cypher(statement).map(|_| ()),
    }
}

/// The catalog as a read shows it, in an order that does not matter; a read that fails (a
/// procedure that is not there) shows as its error.
fn catalog(session: &Session, read: &str) -> Vec<String> {
    match session.execute(read) {
        Ok(result) => {
            let mut rows: Vec<String> =
                result.rows().iter().map(|row| format!("{row:?}")).collect();
            rows.sort();
            rows
        }
        Err(_) => vec!["no such thing".to_string()],
    }
}

/// What a database holds before each transaction: data the statements can act on.
fn seed(db: &GrafeoDB) {
    db.session()
        .execute(
            "INSERT (:Person {name: 'Alix', age: 30, email: 'alix@example.net', embedding: vector([1.0, 0.0])}),
                    (:Person {name: 'Gus', age: 40, email: 'gus@example.net', embedding: vector([0.0, 1.0])})",
        )
        .unwrap();
}

fn prepared(kind: &Kind) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    if let Some(before) = kind.before {
        db.session().execute(before).unwrap();
    }
    db
}

/// What the catalog must read after the statement: run plainly on a database of its own.
fn after_the_statement(kind: &Kind) -> Vec<String> {
    let db = prepared(kind);
    let session = db.session();
    run(&session, kind.language, kind.statement).unwrap();
    catalog(&session, kind.read)
}

fn prepared_on_disk(kind: &Kind, path: &Path) -> GrafeoDB {
    let db = GrafeoDB::open(path).unwrap();
    seed(&db);
    if let Some(before) = kind.before {
        db.session().execute(before).unwrap();
    }
    db
}

// ---------------------------------------------------------------------------
// Every form
// ---------------------------------------------------------------------------

#[test]
fn the_reads_tell_each_statement_apart() {
    // Without this, the tests below would pass on a read that shows nothing.
    for kind in kinds() {
        let db = prepared(&kind);
        let before = catalog(&db.session(), kind.read);
        assert_ne!(before, after_the_statement(&kind), "{}", kind.name);
    }
}

#[test]
fn a_rollback_undoes_it() {
    for kind in kinds() {
        let db = prepared(&kind);
        let before = catalog(&db.session(), kind.read);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        session.rollback().unwrap();
        assert_eq!(
            catalog(&session, kind.read),
            before,
            "{}: own session",
            kind.name
        );
        assert_eq!(
            catalog(&db.session(), kind.read),
            before,
            "{}: a new session",
            kind.name
        );
    }
}

#[test]
fn a_rollback_to_a_savepoint_undoes_it() {
    for kind in kinds() {
        let db = prepared(&kind);
        let before = catalog(&db.session(), kind.read);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.savepoint("s").unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        session.rollback_to_savepoint("s").unwrap();
        assert_eq!(
            catalog(&session, kind.read),
            before,
            "{}: in the transaction",
            kind.name
        );
        session.commit().unwrap();
        assert_eq!(
            catalog(&db.session(), kind.read),
            before,
            "{}: committed",
            kind.name
        );
    }
}

#[test]
fn the_transaction_sees_it_and_another_session_does_not_before_the_commit() {
    for kind in kinds() {
        let db = prepared(&kind);
        let before = catalog(&db.session(), kind.read);
        let after = after_the_statement(&kind);
        let other = db.session();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        assert_eq!(
            catalog(&session, kind.read),
            after,
            "{}: its own view",
            kind.name
        );
        assert_eq!(
            catalog(&other, kind.read),
            before,
            "{}: before the commit",
            kind.name
        );
        session.commit().unwrap();
        assert_eq!(
            catalog(&other, kind.read),
            after,
            "{}: after the commit",
            kind.name
        );
    }
}

#[test]
fn a_rollback_holds_after_a_reopen_and_a_commit_too() {
    let dir = tempfile::tempdir().unwrap();
    for (number, kind) in kinds().iter().enumerate() {
        let after = after_the_statement(kind);

        let path = dir.path().join(format!("rolled-{number}.grafeo"));
        let db = prepared_on_disk(kind, &path);
        let before = catalog(&db.session(), kind.read);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        session.rollback().unwrap();
        drop(session);
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            catalog(&db.session(), kind.read),
            before,
            "{}: rolled back",
            kind.name
        );
        db.close().unwrap();

        let path = dir.path().join(format!("committed-{number}.grafeo"));
        let db = prepared_on_disk(kind, &path);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        session.commit().unwrap();
        drop(session);
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(
            catalog(&db.session(), kind.read),
            after,
            "{}: committed",
            kind.name
        );
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// A crash
// ---------------------------------------------------------------------------

const PATH_VAR: &str = "GRAFEO_SCHEMA_TX_PATH";
const KIND_VAR: &str = "GRAFEO_SCHEMA_TX_KIND";
const END_VAR: &str = "GRAFEO_SCHEMA_TX_END";

#[derive(Clone, Copy, Debug)]
enum End {
    /// The process dies with the transaction open.
    Open,
    /// It dies right after the rollback.
    Rollback,
    /// It dies right after the commit.
    Commit,
}

impl End {
    fn name(self) -> &'static str {
        match self {
            End::Open => "open",
            End::Rollback => "rollback",
            End::Commit => "commit",
        }
    }
}

/// Child-process entry: a data write beside the statement, then the end, no close and no
/// destructors, so that the reopen replays what the WAL has.
#[test]
fn crash_child() {
    let Some(path) = std::env::var_os(PATH_VAR) else {
        return;
    };
    let number: usize = std::env::var(KIND_VAR).unwrap().parse().unwrap();
    let end = std::env::var(END_VAR).unwrap();
    let all = kinds();
    let kind = &all[number];
    let db = prepared_on_disk(kind, &PathBuf::from(path));
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run(&session, kind.language, kind.statement).unwrap();
    session
        .execute("INSERT (:Marker {name: 'Vincent'})")
        .unwrap();
    match end.as_str() {
        "rollback" => session.rollback().unwrap(),
        "commit" => session.commit().unwrap(),
        _ => {}
    }
    std::process::exit(0);
}

fn markers(session: &Session) -> usize {
    session
        .execute("MATCH (m:Marker) RETURN m.name")
        .unwrap()
        .row_count()
}

fn crash_and_reopen(dir: &Path, number: usize, end: End) -> GrafeoDB {
    let path = dir.join(format!("crash-{number}-{}.grafeo", end.name()));
    let status = child_process::run(
        grafeo_common::child_test!("crash_child")
            .env(PATH_VAR, &path)
            .env(KIND_VAR, number.to_string())
            .env(END_VAR, end.name()),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    GrafeoDB::open(&path).unwrap()
}

/// A crash after a rollback: the same schema as before the transaction, the data gone, and a
/// write that a leftover rule would refuse passes.
#[test]
fn a_crash_after_a_rollback_keeps_the_schema_as_it_was() {
    let dir = tempfile::tempdir().unwrap();
    for (number, kind) in kinds().iter().enumerate() {
        let before = catalog(&prepared(kind).session(), kind.read);
        let db = crash_and_reopen(dir.path(), number, End::Rollback);
        let session = db.session();
        assert_eq!(catalog(&session, kind.read), before, "{}", kind.name);
        assert_eq!(markers(&session), 0, "{}: the data write", kind.name);
        db.close().unwrap();
    }
}

/// A crash before the commit leaves nothing.
#[test]
fn a_crash_before_the_commit_leaves_nothing() {
    let dir = tempfile::tempdir().unwrap();
    for (number, kind) in kinds().iter().enumerate() {
        let before = catalog(&prepared(kind).session(), kind.read);
        let db = crash_and_reopen(dir.path(), number, End::Open);
        let session = db.session();
        assert_eq!(catalog(&session, kind.read), before, "{}", kind.name);
        assert_eq!(markers(&session), 0, "{}: the data write", kind.name);
        db.close().unwrap();
    }
}

/// A crash after the commit keeps the statement and the write, as one group.
#[test]
fn a_crash_after_the_commit_keeps_both() {
    let dir = tempfile::tempdir().unwrap();
    for (number, kind) in kinds().iter().enumerate() {
        let after = after_the_statement(kind);
        let db = crash_and_reopen(dir.path(), number, End::Commit);
        let session = db.session();
        assert_eq!(catalog(&session, kind.read), after, "{}", kind.name);
        assert_eq!(markers(&session), 1, "{}: the data write", kind.name);
        db.close().unwrap();
    }
}

// ---------------------------------------------------------------------------
// Rules: what later writes meet
// ---------------------------------------------------------------------------

/// `(name, language, the rule, a write it refuses)`.
const RULES: [(&str, Language, &str, &str); 5] = [
    (
        "not null node type",
        Gql,
        "CREATE NODE TYPE City (name STRING NOT NULL)",
        "INSERT (:City {size: 3})",
    ),
    (
        "not null constraint",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.must) NOT NULL",
        "INSERT (:File {id: 'b'})",
    ),
    (
        "unique constraint",
        Gql,
        "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
        "INSERT (:File {id: 'a', must: 2})",
    ),
    (
        "cypher not null constraint",
        Cypher,
        "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.must IS NOT NULL",
        "INSERT (:File {id: 'b'})",
    ),
    (
        "cypher unique constraint",
        Cypher,
        "CREATE CONSTRAINT c1 FOR (n:File) REQUIRE n.id IS UNIQUE",
        "INSERT (:File {id: 'a', must: 2})",
    ),
];

fn with_a_file(db: &GrafeoDB) {
    db.session()
        .execute("INSERT (:File {id: 'a', must: 1})")
        .unwrap();
}

/// The way Deriva hits it: a rule made in a transaction that rolls back still refused later
/// writes with `GRAFEO-V001`, also after a reopen.
#[test]
fn a_rolled_back_rule_refuses_no_later_write() {
    for (name, language, rule, write) in RULES {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, language, rule).unwrap();
        session.rollback().unwrap();
        session
            .execute(write)
            .unwrap_or_else(|error| panic!("{name}: refused after the rollback: {error}"));
    }
}

#[test]
fn a_rolled_back_rule_refuses_no_later_write_after_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    for (number, (name, language, rule, write)) in RULES.into_iter().enumerate() {
        let path = dir.path().join(format!("rule-{number}.grafeo"));
        let db = GrafeoDB::open(&path).unwrap();
        with_a_file(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, language, rule).unwrap();
        session.rollback().unwrap();
        drop(session);
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        db.session()
            .execute(write)
            .unwrap_or_else(|error| panic!("{name}: refused after a reopen: {error}"));
        db.close().unwrap();
    }
}

const RULE_PATH_VAR: &str = "GRAFEO_SCHEMA_TX_RULE_PATH";
const RULE_VAR: &str = "GRAFEO_SCHEMA_TX_RULE";

#[test]
fn rule_crash_child() {
    let Some(path) = std::env::var_os(RULE_PATH_VAR) else {
        return;
    };
    let number: usize = std::env::var(RULE_VAR).unwrap().parse().unwrap();
    let (_, language, rule, _) = RULES[number];
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    with_a_file(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    run(&session, language, rule).unwrap();
    session.rollback().unwrap();
    std::process::exit(0);
}

#[test]
fn a_rolled_back_rule_refuses_no_later_write_after_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    for (number, (name, _, _, write)) in RULES.into_iter().enumerate() {
        let path = dir.path().join(format!("rule-crash-{number}.grafeo"));
        let status = child_process::run(
            grafeo_common::child_test!("rule_crash_child")
                .env(RULE_PATH_VAR, &path)
                .env(RULE_VAR, number.to_string()),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        let db = GrafeoDB::open(&path).unwrap();
        db.session()
            .execute(write)
            .unwrap_or_else(|error| panic!("{name}: refused after a crash: {error}"));
        db.close().unwrap();
    }
}

/// A rule that commits does refuse, so the tests above do not pass on a rule that never works.
#[test]
fn a_committed_rule_refuses_a_later_write() {
    for (name, language, rule, write) in RULES {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, language, rule).unwrap();
        session.commit().unwrap();
        let error = session
            .execute(write)
            .err()
            .unwrap_or_else(|| panic!("{name}: a write the rule refuses went through"));
        assert!(error.to_string().contains("GRAFEO-V001"), "{name}: {error}");
    }
}

/// The rule applies to the transaction's own later writes, and to no other session's until the
/// commit.
#[test]
fn the_rule_applies_to_its_own_transaction_and_no_other() {
    for (name, language, rule, write) in RULES {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, language, rule).unwrap();
        assert!(session.execute(write).is_err(), "{name}: its own write");
        session.rollback().unwrap();

        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, language, rule).unwrap();
        db.session()
            .execute(write)
            .unwrap_or_else(|error| panic!("{name}: another session was refused: {error}"));
        session.rollback().unwrap();
    }
}

/// A rule that goes back up: a dropped rule that rolls back refuses again, and a dropped node
/// type's rule too.
#[test]
fn a_rolled_back_drop_brings_the_rule_back() {
    let db = GrafeoDB::new_in_memory();
    with_a_file(&db);
    db.session()
        .execute("CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("DROP CONSTRAINT c1").unwrap();
    session.rollback().unwrap();
    let error = session
        .execute("INSERT (:File {id: 'a'})")
        .expect_err("the rule is back");
    assert!(error.to_string().contains("GRAFEO-V001"), "{error}");

    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("DROP NODE TYPE City").unwrap();
    session.rollback().unwrap();
    assert!(
        session.execute("INSERT (:City {size: 3})").is_err(),
        "the type's rule is back"
    );
}

// ---------------------------------------------------------------------------
// Together with the data, and with what is in a graph
// ---------------------------------------------------------------------------

/// Catalog and data statements in one transaction: both stand after the commit, and a later
/// statement of it reads the new type.
// ISO: GP18
#[test]
fn a_commit_keeps_the_schema_and_the_data_together() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
        .unwrap();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    assert!(
        session.execute("INSERT (:City {size: 3})").is_err(),
        "a later statement reads the new type"
    );
    session.execute("INSERT (:City {name: 'Paris'})").unwrap();
    session.commit().unwrap();
    assert_eq!(catalog(&db.session(), NODE_TYPES).len(), 1);
    assert_eq!(
        db.session()
            .execute("MATCH (c:City) RETURN c.name")
            .unwrap()
            .row_count(),
        2
    );
}

// ISO: GP18
#[test]
fn a_rollback_undoes_the_schema_and_the_data_together() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE NODE TYPE City (name STRING)")
        .unwrap();
    session
        .execute("INSERT (:City {name: 'Amsterdam'})")
        .unwrap();
    session.rollback().unwrap();
    assert_eq!(
        catalog(&session, NODE_TYPES),
        catalog(&GrafeoDB::new_in_memory().session(), NODE_TYPES)
    );
    assert_eq!(
        session
            .execute("MATCH (c:City) RETURN c.name")
            .unwrap()
            .row_count(),
        0
    );
}

#[test]
fn a_rollback_to_a_savepoint_keeps_what_came_before_it() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE GRAPH kept").unwrap();
    session.savepoint("s").unwrap();
    session.execute("CREATE GRAPH gone").unwrap();
    session.rollback_to_savepoint("s").unwrap();
    session.commit().unwrap();
    let graphs = catalog(&db.session(), GRAPHS);
    assert!(graphs.iter().any(|row| row.contains("kept")), "{graphs:?}");
    assert!(!graphs.iter().any(|row| row.contains("gone")), "{graphs:?}");
}

/// A name is free again after a rollback, and the statement works a second time.
#[test]
fn a_name_is_free_again_after_a_rollback() {
    for kind in kinds() {
        let db = prepared(&kind);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        session.rollback().unwrap();
        run(&session, kind.language, kind.statement)
            .unwrap_or_else(|error| panic!("{}: the second time: {error}", kind.name));
        assert_eq!(
            catalog(&session, kind.read),
            after_the_statement(&kind),
            "{}",
            kind.name
        );
    }
}

/// `DROP GRAPH` that rolls back keeps the graph and its data, in memory and after a reopen and a
/// crash (the issue's data loss).
#[test]
fn a_rolled_back_drop_of_a_graph_keeps_its_data() {
    let vincent = |db: &GrafeoDB| {
        let session = db.session();
        session.execute("USE GRAPH g2").unwrap();
        session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count()
    };
    let fill = |db: &GrafeoDB| {
        db.session().execute("CREATE GRAPH g2").unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("USE GRAPH g2").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        session.commit().unwrap();
    };
    let drop_and_roll_back = |db: &GrafeoDB| {
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("DROP GRAPH g2").unwrap();
        session.rollback().unwrap();
    };
    let memory = GrafeoDB::new_in_memory();
    fill(&memory);
    drop_and_roll_back(&memory);
    assert_eq!(vincent(&memory), 1, "in memory");

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graphs.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    fill(&db);
    drop_and_roll_back(&db);
    assert_eq!(vincent(&db), 1, "on disk");
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    assert_eq!(vincent(&db), 1, "after a reopen");
    db.close().unwrap();
}

const GRAPH_PATH_VAR: &str = "GRAFEO_SCHEMA_TX_GRAPH_PATH";

#[test]
fn graph_crash_child() {
    let Some(path) = std::env::var_os(GRAPH_PATH_VAR) else {
        return;
    };
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    db.session().execute("CREATE GRAPH g2").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("USE GRAPH g2").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    session.commit().unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH g2").unwrap();
    session.rollback().unwrap();
    std::process::exit(0);
}

#[test]
fn a_rolled_back_drop_of_a_graph_survives_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("graphs.grafeo");
    let status = child_process::run(
        grafeo_common::child_test!("graph_crash_child").env(GRAPH_PATH_VAR, &path),
    )
    .unwrap();
    assert!(status.success(), "the child process failed");
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session.execute("USE GRAPH g2").unwrap();
    assert_eq!(
        session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count(),
        1
    );
    drop(session);
    db.close().unwrap();
}

/// A rolled-back drop of an index leaves it working: it still finds what was there and what is
/// written later. The rows are the same with an index that was never there.
#[test]
fn a_rolled_back_drop_of_an_index_leaves_it_working() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    db.session().execute(HASH_INDEX).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("DROP INDEX i1").unwrap();
    session
        .execute("INSERT (:Person {name: 'Jules', age: 30})")
        .unwrap();
    session.rollback().unwrap();
    session
        .execute("INSERT (:Person {name: 'Mia', age: 30})")
        .unwrap();
    let names = |session: &Session| {
        let mut names: Vec<String> = session
            .execute("MATCH (p:Person) WHERE p.age = 30 RETURN p.name")
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect();
        names.sort();
        names
    };
    let got = names(&session);
    assert_eq!(got.len(), 2, "{got:?}");
    assert!(got[0].contains("Alix") && got[1].contains("Mia"), "{got:?}");
    assert!(
        catalog(&session, INDEXES)
            .iter()
            .any(|row| row.contains("i1")),
        "the index is gone"
    );
}

/// An index made in the transaction serves its own writes, and rolled back it answers nothing
/// wrong: no plan keeps reading an index that is gone.
#[test]
fn an_index_made_in_the_transaction_serves_its_own_writes_only_until_a_rollback() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(HASH_INDEX).unwrap();
    session
        .execute("INSERT (:Person {name: 'Jules', age: 30})")
        .unwrap();
    let query = "MATCH (p:Person) WHERE p.age = 30 RETURN p.name";
    assert_eq!(session.execute(query).unwrap().row_count(), 2);
    session.rollback().unwrap();
    assert_eq!(
        session.execute(query).unwrap().row_count(),
        1,
        "the same session"
    );
    assert_eq!(
        db.session().execute(query).unwrap().row_count(),
        1,
        "a new session"
    );
    session
        .execute("INSERT (:Person {name: 'Mia', age: 30})")
        .unwrap();
    assert_eq!(session.execute(query).unwrap().row_count(), 2);
}

/// Created and dropped in one transaction, it leaves nothing, whichever way it ends; dropped and
/// created again, the new one stands after a commit and the old one after a rollback.
#[test]
fn several_statements_on_one_name_add_up() {
    for commit in [true, false] {
        let db = GrafeoDB::new_in_memory();
        let before = catalog(&db.session(), NODE_TYPES);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute(NODE_TYPE).unwrap();
        session.execute("DROP NODE TYPE City").unwrap();
        if commit {
            session.commit().unwrap();
        } else {
            session.rollback().unwrap();
        }
        assert_eq!(
            catalog(&db.session(), NODE_TYPES),
            before,
            "commit {commit}"
        );
    }
    for commit in [true, false] {
        let db = GrafeoDB::new_in_memory();
        db.session().execute(NODE_TYPE).unwrap();
        let old = catalog(&db.session(), NODE_TYPES);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("DROP NODE TYPE City").unwrap();
        session
            .execute("CREATE NODE TYPE City (name STRING, size INTEGER)")
            .unwrap();
        let new = catalog(&session, NODE_TYPES);
        assert_ne!(new, old);
        if commit {
            session.commit().unwrap();
            assert_eq!(catalog(&db.session(), NODE_TYPES), new);
        } else {
            session.rollback().unwrap();
            assert_eq!(catalog(&db.session(), NODE_TYPES), old);
        }
    }
}

/// A session dropped with a transaction open rolls back, as for data.
#[test]
fn a_session_dropped_with_an_open_transaction_undoes_it() {
    for kind in kinds() {
        let db = prepared(&kind);
        let before = catalog(&db.session(), kind.read);
        {
            let mut session = db.session();
            session.begin_transaction().unwrap();
            run(&session, kind.language, kind.statement).unwrap();
        }
        assert_eq!(catalog(&db.session(), kind.read), before, "{}", kind.name);
    }
}

/// A statement that fails inside a transaction leaves it usable and changes nothing itself.
#[test]
fn a_failing_schema_statement_leaves_the_transaction_as_it_was() {
    let db = GrafeoDB::new_in_memory();
    db.session().execute(NODE_TYPE).unwrap();
    let before = catalog(&db.session(), NODE_TYPES);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    assert!(session.execute(NODE_TYPE).is_err(), "the name is taken");
    assert_eq!(catalog(&session, NODE_TYPES), before);
    session
        .execute("CREATE EDGE TYPE ROAD (km INTEGER)")
        .unwrap();
    session.commit().unwrap();
    assert_eq!(catalog(&db.session(), NODE_TYPES), before);
    assert_eq!(catalog(&db.session(), EDGE_TYPES).len(), 1);
}

// ---------------------------------------------------------------------------
// Clashes at the commit, and other sessions
// ---------------------------------------------------------------------------

/// Statements that claim a name, and what shows it.
const NAMED: [(&str, &str, &str); 5] = [
    ("graph", "CREATE GRAPH g2", GRAPHS),
    ("schema", "CREATE SCHEMA s2", SCHEMAS),
    ("index", HASH_INDEX, INDEXES),
    ("node type", NODE_TYPE, NODE_TYPES),
    ("constraint", UNIQUE, CONSTRAINTS),
];

fn is_a_conflict(error: &grafeo_common::utils::error::Error) -> bool {
    error.to_string().contains("GRAFEO-T001")
}

/// A statement claims its name: a second transaction's statement on it fails at once with
/// `GRAFEO-T001`, the first commit stands, and the second's own rollback leaves nothing of it.
#[test]
fn a_name_is_claimed_by_the_first_statement() {
    for (name, statement, read) in NAMED {
        let db = GrafeoDB::new_in_memory();
        seed(&db);
        let before = catalog(&db.session(), read);
        let mut first = db.session();
        let mut second = db.session();
        first.begin_transaction().unwrap();
        second.begin_transaction().unwrap();
        second
            .execute("INSERT (:Marker {name: 'Vincent'})")
            .unwrap();
        first.execute(statement).unwrap();
        let error = second
            .execute(statement)
            .err()
            .unwrap_or_else(|| panic!("{name}: both claimed the name"));
        assert!(is_a_conflict(&error), "{name}: {error}");
        assert_eq!(
            catalog(&db.session(), read),
            before,
            "{name}: seen before the commit"
        );
        second.rollback().unwrap();
        first.commit().unwrap();
        assert_ne!(
            catalog(&db.session(), read),
            before,
            "{name}: the first commit"
        );
        assert_eq!(markers(&db.session()), 0, "{name}: the second's write");
    }
}

/// A statement outside a transaction meets the claim too.
#[test]
fn a_statement_outside_a_transaction_meets_the_claim() {
    for (name, statement, read) in NAMED {
        let db = GrafeoDB::new_in_memory();
        seed(&db);
        let before = catalog(&db.session(), read);
        let mut first = db.session();
        first.begin_transaction().unwrap();
        first.execute(statement).unwrap();
        let error = db
            .session()
            .execute(statement)
            .err()
            .unwrap_or_else(|| panic!("{name}: went past the claim"));
        assert!(is_a_conflict(&error), "{name}: {error}");
        first.rollback().unwrap();
        assert_eq!(catalog(&db.session(), read), before, "{name}");
        // Released: the statement now passes.
        db.session()
            .execute(statement)
            .unwrap_or_else(|error| panic!("{name}: after the rollback: {error}"));
    }
}

/// A drop claims the name as a create does.
#[test]
fn a_drop_claims_its_name() {
    for (name, setup, drop_it, recreate) in [
        (
            "graph",
            "CREATE GRAPH g2",
            "DROP GRAPH g2",
            "CREATE GRAPH g2",
        ),
        (
            "schema",
            "CREATE SCHEMA s2",
            "DROP SCHEMA s2",
            "CREATE SCHEMA s2",
        ),
        ("index", HASH_INDEX, "DROP INDEX i1", HASH_INDEX),
        ("node type", NODE_TYPE, "DROP NODE TYPE City", NODE_TYPE),
        ("constraint", UNIQUE, "DROP CONSTRAINT c1", UNIQUE),
    ] {
        let db = GrafeoDB::new_in_memory();
        seed(&db);
        db.session().execute(setup).unwrap();
        let mut first = db.session();
        first.begin_transaction().unwrap();
        first.execute(drop_it).unwrap();
        for other in [drop_it, recreate] {
            let error = db
                .session()
                .execute(other)
                .err()
                .unwrap_or_else(|| panic!("{name}: {other} went past the claim"));
            assert!(is_a_conflict(&error), "{name}: {other}: {error}");
        }
        first.rollback().unwrap();
    }
}

/// What releases a claim: a rollback, a savepoint rollback, a session dropped with its
/// transaction open. A commit turns it into the name's plain existence.
#[test]
fn a_claim_is_released_by_what_undoes_it() {
    for (name, statement, _) in NAMED {
        let db = GrafeoDB::new_in_memory();
        seed(&db);
        let mut first = db.session();
        first.begin_transaction().unwrap();
        first.savepoint("s").unwrap();
        first.execute(statement).unwrap();
        first.rollback_to_savepoint("s").unwrap();
        let mut second = db.session();
        second.begin_transaction().unwrap();
        second
            .execute(statement)
            .unwrap_or_else(|error| panic!("{name}: after the savepoint rollback: {error}"));
        second.rollback().unwrap();
        first.rollback().unwrap();

        let db = GrafeoDB::new_in_memory();
        seed(&db);
        {
            let mut gone = db.session();
            gone.begin_transaction().unwrap();
            gone.execute(statement).unwrap();
        }
        db.session()
            .execute(statement)
            .unwrap_or_else(|error| panic!("{name}: after the session went: {error}"));
    }
}

/// Names of different kinds do not clash, and neither do different names.
#[test]
fn a_claim_covers_its_own_name_only() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    let mut first = db.session();
    first.begin_transaction().unwrap();
    first.execute("CREATE GRAPH shared").unwrap();
    let mut second = db.session();
    second.begin_transaction().unwrap();
    second.execute("CREATE SCHEMA shared").unwrap();
    second.execute("CREATE GRAPH other").unwrap();
    db.session()
        .execute("CREATE NODE TYPE shared (a STRING)")
        .unwrap();
    first.commit().unwrap();
    second.commit().unwrap();
    let graphs = catalog(&db.session(), GRAPHS);
    assert!(
        graphs.iter().any(|row| row.contains("shared")),
        "{graphs:?}"
    );
    assert!(graphs.iter().any(|row| row.contains("other")), "{graphs:?}");
}

const CLASH_PATH_VAR: &str = "GRAFEO_SCHEMA_TX_CLASH_PATH";
const CLASH_VAR: &str = "GRAFEO_SCHEMA_TX_CLASH";

/// A clash, then the process dies with no close: the log holds nothing of the loser.
#[test]
fn clash_crash_child() {
    let Some(path) = std::env::var_os(CLASH_PATH_VAR) else {
        return;
    };
    let number: usize = std::env::var(CLASH_VAR).unwrap().parse().unwrap();
    let (_, statement, _) = NAMED[number];
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    seed(&db);
    let mut first = db.session();
    let mut second = db.session();
    first.begin_transaction().unwrap();
    second.begin_transaction().unwrap();
    second
        .execute("INSERT (:Marker {name: 'Vincent'})")
        .unwrap();
    first.execute(statement).unwrap();
    assert!(second.execute(statement).is_err());
    second.rollback().unwrap();
    first.commit().unwrap();
    std::process::exit(0);
}

#[test]
fn a_clash_leaves_nothing_in_the_log() {
    let dir = tempfile::tempdir().unwrap();
    for (number, (name, statement, read)) in NAMED.into_iter().enumerate() {
        // The catalog as one plain statement leaves it.
        let plain = GrafeoDB::new_in_memory();
        seed(&plain);
        plain.session().execute(statement).unwrap();
        let want = catalog(&plain.session(), read);

        let path = dir.path().join(format!("clash-crash-{number}.grafeo"));
        let status = child_process::run(
            grafeo_common::child_test!("clash_crash_child")
                .env(CLASH_PATH_VAR, &path)
                .env(CLASH_VAR, number.to_string()),
        )
        .unwrap();
        assert!(status.success(), "{name}: the child process failed");
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(catalog(&db.session(), read), want, "{name}: after a crash");
        assert_eq!(markers(&db.session()), 0, "{name}: the loser's data write");
        db.close().unwrap();
        // And once more, from the checkpoint the close wrote.
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(catalog(&db.session(), read), want, "{name}: after a reopen");
        assert_eq!(markers(&db.session()), 0, "{name}: the loser's data write");
        db.close().unwrap();
    }
}

/// A crash with the commit's mixed group written and not yet published: both parts or neither,
/// never the schema without the data or the other way round.
#[cfg(feature = "testing-statement-injection")]
mod between_log_and_publish {
    use super::*;
    use grafeo_common::testing::commit_hook::after_next_commit_written;

    const PATH_VAR: &str = "GRAFEO_SCHEMA_TX_PUBLISH_PATH";
    const KIND_VAR: &str = "GRAFEO_SCHEMA_TX_PUBLISH_KIND";

    #[test]
    fn publish_crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let number: usize = std::env::var(KIND_VAR).unwrap().parse().unwrap();
        let all = kinds();
        let kind = &all[number];
        let db = prepared_on_disk(kind, &PathBuf::from(path));
        let mut session = db.session();
        session.begin_transaction().unwrap();
        run(&session, kind.language, kind.statement).unwrap();
        session
            .execute("INSERT (:Marker {name: 'Vincent'})")
            .unwrap();
        after_next_commit_written(|| std::process::exit(0));
        session.commit().unwrap();
        panic!("the commit did not run the hook");
    }

    #[test]
    fn a_crash_between_log_and_publish_leaves_the_group_whole() {
        let dir = tempfile::tempdir().unwrap();
        for (number, kind) in kinds().iter().enumerate() {
            let before = catalog(&prepared(kind).session(), kind.read);
            let after = after_the_statement(kind);
            let path = dir.path().join(format!("publish-{number}.grafeo"));
            let status = child_process::run(
                grafeo_common::child_test!("publish_crash_child")
                    .env(PATH_VAR, &path)
                    .env(KIND_VAR, number.to_string()),
            )
            .unwrap();
            assert!(status.success(), "{}: the child process failed", kind.name);
            let db = GrafeoDB::open(&path).unwrap();
            let session = db.session();
            let shown = catalog(&session, kind.read);
            match markers(&session) {
                0 => assert_eq!(shown, before, "{}: schema without its data", kind.name),
                1 => assert_eq!(shown, after, "{}: data without its schema", kind.name),
                other => panic!("{}: {other} markers", kind.name),
            }
            drop(session);
            db.close().unwrap();
        }
    }
}

/// The new graph and the writes into it, with the group written and not published.
#[cfg(feature = "testing-statement-injection")]
mod new_graph_between_log_and_publish {
    use super::*;
    use grafeo_common::testing::commit_hook::after_next_commit_written;

    const PATH_VAR: &str = "GRAFEO_SCHEMA_TX_PUBLISH_GRAPH_PATH";

    #[test]
    fn publish_graph_crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute("CREATE GRAPH g2").unwrap();
        session.execute("USE GRAPH g2").unwrap();
        session
            .execute("INSERT (:Person {name: 'Vincent'})")
            .unwrap();
        after_next_commit_written(|| std::process::exit(0));
        session.commit().unwrap();
        panic!("the commit did not run the hook");
    }

    #[test]
    fn a_new_graph_and_its_writes_stay_one_group() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("publish-graph.grafeo");
        let status = child_process::run(
            grafeo_common::child_test!("publish_graph_crash_child").env(PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        if catalog(&session, GRAPHS)
            .iter()
            .any(|row| row.contains("g2"))
        {
            session.execute("USE GRAPH g2").unwrap();
            assert_eq!(
                session
                    .execute("MATCH (p:Person) RETURN p.name")
                    .unwrap()
                    .row_count(),
                1,
                "the graph came back without its write"
            );
        }
        drop(session);
        db.close().unwrap();
    }
}

/// A transaction's own catalog does not move when another session commits: not before a savepoint
/// rollback, which rebuilds it, and not after.
#[test]
fn a_savepoint_rollback_does_not_let_another_commit_into_its_view() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE NODE TYPE Kept (a STRING)").unwrap();
    let at_the_savepoint = catalog(&session, NODE_TYPES);
    session.savepoint("s").unwrap();
    session.execute("CREATE NODE TYPE Gone (a STRING)").unwrap();
    db.session()
        .execute("CREATE NODE TYPE Theirs (a STRING)")
        .unwrap();
    assert!(
        !catalog(&session, NODE_TYPES)
            .iter()
            .any(|row| row.contains("Theirs")),
        "before the savepoint rollback"
    );
    session.rollback_to_savepoint("s").unwrap();
    assert_eq!(
        catalog(&session, NODE_TYPES),
        at_the_savepoint,
        "the view moved at the savepoint rollback"
    );
    session.commit().unwrap();
    let last = catalog(&db.session(), NODE_TYPES);
    for (name, present) in [("Kept", true), ("Theirs", true), ("Gone", false)] {
        assert_eq!(
            last.iter().any(|row| row.contains(name)),
            present,
            "{name}: {last:?}"
        );
    }
}

/// A rule another session commits while a transaction with its own catalog view is open: the
/// transaction's write that breaks it is refused at the statement or at the commit, and no
/// row that breaks the rule stands.
#[test]
fn a_concurrent_new_rule_holds_against_a_transaction_with_its_own_view() {
    for (name, rule, write, broken) in [
        (
            "unique",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE",
            "INSERT (:File {id: 'a', must: 2})",
            "MATCH (f:File) WHERE f.id = 'a' AND f.must = 2 RETURN f.id",
        ),
        (
            "not null",
            "CREATE CONSTRAINT c1 FOR (n:File) ON (n.must) NOT NULL",
            "INSERT (:File {id: 'b'})",
            "MATCH (f:File) WHERE f.id = 'b' RETURN f.id",
        ),
    ] {
        let db = GrafeoDB::new_in_memory();
        with_a_file(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("CREATE EDGE TYPE ROAD (km INTEGER)")
            .unwrap();
        db.session().execute(rule).unwrap();
        let written = session.execute(write).is_ok();
        let committed = written && session.commit().is_ok();
        if !committed {
            let _ = session.rollback();
        }
        assert_eq!(
            db.session().execute(broken).unwrap().row_count(),
            0,
            "{name}: a row that breaks the committed rule stands"
        );
    }
}

/// A transaction's own rule against what others committed since its statement: the commit checks
/// again, fails, and leaves nothing.
#[test]
fn a_unique_rule_checks_what_others_committed_meanwhile() {
    let db = GrafeoDB::new_in_memory();
    with_a_file(&db);
    let before = catalog(&db.session(), CONSTRAINTS);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE CONSTRAINT c1 FOR (n:File) ON (n.id) UNIQUE")
        .unwrap();
    db.session()
        .execute("INSERT (:File {id: 'a', must: 2})")
        .unwrap();
    assert!(
        session.commit().is_err(),
        "a rule the data breaks was committed"
    );
    assert_eq!(catalog(&db.session(), CONSTRAINTS), before);
}

/// SPARQL's graph operations queue like the others: a rolled-back `DROP GRAPH` keeps the graph
/// and its triples, and `CREATE GRAPH` leaves nothing.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod sparql_graphs {
    use super::*;

    const PARIS: &str = "http://example.org/paris";

    fn triples(session: &Session) -> usize {
        session
            .execute_sparql(&format!(
                "SELECT ?s WHERE {{ GRAPH <{PARIS}> {{ ?s ?p ?o }} }}"
            ))
            .unwrap()
            .row_count()
    }

    fn with_a_graph(db: &GrafeoDB) {
        db.session()
            .execute_sparql(&format!(
                "INSERT DATA {{ GRAPH <{PARIS}> {{ <http://example.org/gus> <http://example.org/name> 'Gus' }} }}"
            ))
            .unwrap();
    }

    #[test]
    fn a_rolled_back_drop_keeps_the_graph() {
        let db = GrafeoDB::new_in_memory();
        with_a_graph(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!("DROP GRAPH <{PARIS}>"))
            .unwrap();
        assert_eq!(
            triples(&db.session()),
            1,
            "another session before the commit"
        );
        session.rollback().unwrap();
        assert_eq!(triples(&session), 1);
    }

    #[test]
    fn a_rolled_back_drop_keeps_the_graph_after_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rdf.grafeo");
        let db = GrafeoDB::open(&path).unwrap();
        with_a_graph(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!("DROP GRAPH <{PARIS}>"))
            .unwrap();
        session.rollback().unwrap();
        drop(session);
        db.close().unwrap();
        let db = GrafeoDB::open(&path).unwrap();
        assert_eq!(triples(&db.session()), 1);
        db.close().unwrap();
    }

    #[test]
    fn a_committed_drop_removes_it() {
        let db = GrafeoDB::new_in_memory();
        with_a_graph(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!("DROP GRAPH <{PARIS}>"))
            .unwrap();
        session.commit().unwrap();
        assert_eq!(triples(&db.session()), 0);
    }

    #[test]
    fn a_rolled_back_create_leaves_nothing_and_the_name_free() {
        let db = GrafeoDB::new_in_memory();
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!("CREATE GRAPH <{PARIS}>"))
            .unwrap();
        session.rollback().unwrap();
        db.session()
            .execute_sparql(&format!("CREATE GRAPH <{PARIS}>"))
            .unwrap();
    }

    #[test]
    fn a_savepoint_rollback_undoes_the_drop() {
        let db = GrafeoDB::new_in_memory();
        with_a_graph(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.savepoint("s").unwrap();
        session
            .execute_sparql(&format!("DROP GRAPH <{PARIS}>"))
            .unwrap();
        session.rollback_to_savepoint("s").unwrap();
        session.commit().unwrap();
        assert_eq!(triples(&db.session()), 1);
    }
}

/// `DROP GRAPH` with another transaction writing in that graph: the drop is refused at its commit
/// and the writer's commit keeps its data. With the writer gone the drop passes.
#[test]
fn drop_graph_with_another_open_writer_is_refused_at_the_commit() {
    let db = GrafeoDB::new_in_memory();
    db.session().execute("CREATE GRAPH g2").unwrap();
    let mut writer = db.session();
    writer.begin_transaction().unwrap();
    writer.execute("USE GRAPH g2").unwrap();
    writer
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();

    let mut dropper = db.session();
    dropper.begin_transaction().unwrap();
    dropper.execute("DROP GRAPH g2").unwrap();
    assert!(
        dropper.commit().is_err(),
        "the graph was dropped under its writer"
    );
    assert!(
        catalog(&db.session(), GRAPHS)
            .iter()
            .any(|row| row.contains("g2"))
    );

    writer.commit().unwrap();
    let reader = db.session();
    reader.execute("USE GRAPH g2").unwrap();
    assert_eq!(
        reader
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count(),
        1
    );
    let mut dropper = db.session();
    dropper.begin_transaction().unwrap();
    dropper.execute("DROP GRAPH g2").unwrap();
    dropper.commit().unwrap();
    assert!(
        !catalog(&db.session(), GRAPHS)
            .iter()
            .any(|row| row.contains("g2"))
    );
}

/// In a transaction a dropped graph cannot be resolved, and the others still can.
#[test]
fn a_dropped_graph_is_gone_for_its_transaction_only() {
    let db = GrafeoDB::new_in_memory();
    db.session().execute("CREATE GRAPH g2").unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("DROP GRAPH g2").unwrap();
    assert!(session.execute("USE GRAPH g2").is_err(), "its own view");
    let other = db.session();
    other.execute("USE GRAPH g2").unwrap();
    other.execute("INSERT (:Person {name: 'Mia'})").unwrap();
    session.rollback().unwrap();
    assert_eq!(
        other
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count(),
        1
    );
}

/// A graph made in the transaction takes its writes, and a savepoint rollback across it drops the
/// graph and them: a graph of that name later is new and empty. Writes before the savepoint stay.
#[test]
fn a_savepoint_rollback_across_create_graph_drops_its_writes() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Marker {name: 'Jules'})").unwrap();
    session.savepoint("s").unwrap();
    session.execute("CREATE GRAPH g2").unwrap();
    session.execute("USE GRAPH g2").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    assert_eq!(
        session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count(),
        1,
        "its own write in its own graph"
    );
    session.rollback_to_savepoint("s").unwrap();
    assert!(
        session.execute("USE GRAPH g2").is_err(),
        "the graph is gone for the transaction"
    );
    session.commit().unwrap();
    assert!(
        !catalog(&db.session(), GRAPHS)
            .iter()
            .any(|row| row.contains("g2"))
    );
    assert_eq!(markers(&db.session()), 1, "the write before the savepoint");
    db.session().execute("CREATE GRAPH g2").unwrap();
    let reader = db.session();
    reader.execute("USE GRAPH g2").unwrap();
    assert_eq!(
        reader
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count(),
        0,
        "a new g2 is empty"
    );
}

/// A graph made and filled in one committed transaction holds its data, also after a reopen.
#[test]
fn a_committed_create_graph_keeps_what_was_written_into_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("filled.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE GRAPH g2").unwrap();
    session.execute("USE GRAPH g2").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    assert!(
        !catalog(&db.session(), GRAPHS)
            .iter()
            .any(|row| row.contains("g2")),
        "seen before the commit"
    );
    session.commit().unwrap();
    drop(session);
    db.close().unwrap();
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session.execute("USE GRAPH g2").unwrap();
    assert_eq!(
        session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .row_count(),
        1
    );
    drop(session);
    db.close().unwrap();
}

const FILLED_PATH_VAR: &str = "GRAFEO_SCHEMA_TX_FILLED_PATH";
const FILLED_END_VAR: &str = "GRAFEO_SCHEMA_TX_FILLED_END";

/// Child-process entry: a new graph and writes into it, then the process dies.
#[test]
fn filled_crash_child() {
    let Some(path) = std::env::var_os(FILLED_PATH_VAR) else {
        return;
    };
    let commit = std::env::var(FILLED_END_VAR).unwrap() == "commit";
    let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE GRAPH g2").unwrap();
    session.execute("USE GRAPH g2").unwrap();
    session
        .execute("INSERT (:Person {name: 'Vincent'})")
        .unwrap();
    if commit {
        session.commit().unwrap();
    }
    std::process::exit(0);
}

/// The graph and the writes into it are one group: a crash leaves both or neither.
#[test]
fn a_new_graph_and_its_writes_survive_a_crash_as_one_group() {
    let dir = tempfile::tempdir().unwrap();
    for (end, rows) in [("commit", Some(1)), ("open", None)] {
        let path = dir.path().join(format!("filled-{end}.grafeo"));
        let status = child_process::run(
            grafeo_common::child_test!("filled_crash_child")
                .env(FILLED_PATH_VAR, &path)
                .env(FILLED_END_VAR, end),
        )
        .unwrap();
        assert!(status.success(), "the child process failed");
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        match rows {
            Some(rows) => {
                session.execute("USE GRAPH g2").unwrap();
                assert_eq!(
                    session
                        .execute("MATCH (p:Person) RETURN p.name")
                        .unwrap()
                        .row_count(),
                    rows
                );
            }
            None => assert!(
                !catalog(&session, GRAPHS)
                    .iter()
                    .any(|row| row.contains("g2")),
                "the graph stayed without its commit"
            ),
        }
        drop(session);
        db.close().unwrap();
    }
}

/// Rollback puts the session back in the graph it had: a `USE GRAPH` of a graph the transaction
/// made leaves no session pointing at nothing.
#[test]
fn a_rollback_leaves_the_session_in_a_graph_that_exists() {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute("INSERT (:Marker {name: 'Butch'})")
        .unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE GRAPH g2").unwrap();
    session.execute("USE GRAPH g2").unwrap();
    session.rollback().unwrap();
    assert_eq!(markers(&session), 1, "back in the default graph");
}

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

fn explain(session: &Session, query: &str) -> String {
    let result = session.execute(&format!("EXPLAIN {query}")).unwrap();
    format!("{:?}", result.rows())
}

/// In a transaction a new index shows and no plan uses it, with the same rows; at the commit it
/// holds the transaction's own writes and what others committed since.
#[test]
fn a_new_index_shows_but_plans_only_after_the_commit() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    let query = "MATCH (p:Person) WHERE p.age = 30 RETURN p.name";
    let plain = explain(&db.session(), query);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute(HASH_INDEX).unwrap();
    session
        .execute("INSERT (:Person {name: 'Jules', age: 30})")
        .unwrap();
    assert!(
        catalog(&session, INDEXES)
            .iter()
            .any(|row| row.contains("i1")),
        "SHOW INDEXES"
    );
    assert_eq!(
        explain(&session, query),
        plain,
        "a plan used the unbuilt index"
    );
    assert_eq!(
        session.execute(query).unwrap().row_count(),
        2,
        "the same rows"
    );
    db.session()
        .execute("INSERT (:Person {name: 'Mia', age: 30})")
        .unwrap();
    session.commit().unwrap();
    let reader = db.session();
    assert_ne!(
        explain(&reader, query),
        plain,
        "the index is not in use after the commit"
    );
    assert_eq!(
        reader.execute(query).unwrap().row_count(),
        3,
        "the index holds all three"
    );
}

/// A plan made in a transaction with schema statements does not reach another session through the
/// plan cache, and the other way round. The index stays in use for others during a drop.
#[test]
fn plans_of_a_transaction_do_not_leak_through_the_cache() {
    let db = GrafeoDB::new_in_memory();
    seed(&db);
    db.session().execute(HASH_INDEX).unwrap();
    let query = "MATCH (p:Person) WHERE p.age = 30 RETURN p.name";
    let reader = db.session();
    let with_index = explain(&reader, query);
    assert_eq!(reader.execute(query).unwrap().row_count(), 1);

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("DROP INDEX i1").unwrap();
    assert_eq!(session.execute(query).unwrap().row_count(), 1);
    assert_eq!(
        explain(&reader, query),
        with_index,
        "the drop reached another session's plan"
    );
    assert_eq!(
        explain(&db.session(), query),
        with_index,
        "and a new session's"
    );
    session.rollback().unwrap();
    assert_eq!(
        explain(&session, query),
        with_index,
        "its own plan, after the rollback"
    );

    // A graph of the transaction's own: the same text reads another graph elsewhere.
    let marker_query = "MATCH (m:Marker) RETURN m.name";
    db.session()
        .execute("INSERT (:Marker {name: 'Django'})")
        .unwrap();
    assert_eq!(reader.execute(marker_query).unwrap().row_count(), 1);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("CREATE GRAPH g2").unwrap();
    session.execute("USE GRAPH g2").unwrap();
    assert_eq!(
        session.execute(marker_query).unwrap().row_count(),
        0,
        "the new graph"
    );
    assert_eq!(
        reader.execute(marker_query).unwrap().row_count(),
        1,
        "the old graph"
    );
    session.rollback().unwrap();
    assert_eq!(
        session.execute(marker_query).unwrap().row_count(),
        1,
        "back in the old graph"
    );
}

/// A type the transaction made checks the transaction's writes through the plan it makes, and
/// another session writes by the old catalog with the same text. The text is planned once first,
/// and rolled back, since a rule is refused over nodes that break it.
#[test]
fn a_cached_write_plan_does_not_skip_the_new_rule() {
    let db = GrafeoDB::new_in_memory();
    let write = "INSERT (:City {size: 3})";
    let mut planned = db.session();
    planned.begin_transaction().unwrap();
    planned.execute(write).unwrap();
    planned.rollback().unwrap();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("CREATE NODE TYPE City (name STRING NOT NULL)")
        .unwrap();
    assert!(
        session.execute(write).is_err(),
        "the transaction's own rule"
    );
    db.session()
        .execute(write)
        .unwrap_or_else(|error| panic!("another session: {error}"));
    session.rollback().unwrap();
    session.execute(write).unwrap();
}

// ---------------------------------------------------------------------------
// What a transaction refuses
// ---------------------------------------------------------------------------

fn status(error: &grafeo_common::utils::error::Error) -> String {
    grafeo_common::utils::gqlstatus::GqlStatus::from(error)
        .as_str()
        .to_string()
}

/// A graph copy reads the source as it stands, which the transaction's queue cannot give, so it
/// is refused in a transaction with 25G01 and `GRAFEO-T004`, and the transaction goes on.
#[test]
fn a_graph_copy_in_a_transaction_is_refused_and_the_transaction_goes_on() {
    let db = GrafeoDB::new_in_memory();
    db.session().execute("CREATE GRAPH source").unwrap();
    let before = catalog(&db.session(), GRAPHS);
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Marker {name: 'Jules'})").unwrap();
    let error = session
        .execute("CREATE GRAPH clone AS COPY OF source")
        .expect_err("a copy in a transaction");
    assert_eq!(status(&error), "25G01", "{error}");
    assert!(error.to_string().contains("GRAFEO-T004"), "{error}");
    assert_eq!(catalog(&session, GRAPHS), before, "its own view");
    session
        .execute("INSERT (:Marker {name: 'Mia'})")
        .unwrap_or_else(|error| panic!("the transaction stopped: {error}"));
    session.commit().unwrap();
    assert_eq!(catalog(&db.session(), GRAPHS), before);
    assert_eq!(markers(&db.session()), 2, "both writes");
    // Outside a transaction the same statement passes.
    db.session()
        .execute("CREATE GRAPH clone AS COPY OF source")
        .unwrap();
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
mod sparql_reads_of_queued_graphs {
    use super::*;

    const PARIS: &str = "http://example.org/paris";
    const BERLIN: &str = "http://example.org/berlin";
    const ROME: &str = "http://example.org/rome";

    fn read(graph: &str) -> String {
        format!("SELECT ?s ?o WHERE {{ GRAPH <{graph}> {{ ?s <http://example.org/name> ?o }} }}")
    }

    fn answer(session: &Session, graph: &str) -> usize {
        session.execute_sparql(&read(graph)).unwrap().row_count()
    }

    fn with_graphs(db: &GrafeoDB) {
        for (graph, name) in [(PARIS, "Gus"), (BERLIN, "Hans"), (ROME, "Mia")] {
            db.session()
                .execute_sparql(&format!(
                    "INSERT DATA {{ GRAPH <{graph}> {{ <http://example.org/x> <http://example.org/name> '{name}' }} }}"
                ))
                .unwrap();
        }
    }

    /// `(the operation, the graphs it affects)`.
    fn operations() -> [(String, Vec<&'static str>); 5] {
        [
            (format!("DROP GRAPH <{PARIS}>"), vec![PARIS]),
            (format!("CLEAR GRAPH <{PARIS}>"), vec![PARIS]),
            (format!("COPY <{PARIS}> TO <{BERLIN}>"), vec![BERLIN]),
            (format!("MOVE <{PARIS}> TO <{BERLIN}>"), vec![PARIS, BERLIN]),
            (format!("ADD <{PARIS}> TO <{BERLIN}>"), vec![BERLIN]),
        ]
    }

    /// Triples apply ahead of a queued graph operation, so in that transaction every later SPARQL
    /// statement fails with `GRAFEO-T004`, reads and updates alike, whichever graph they name.
    /// Its GQL work, its commit and other sessions are untouched.
    #[test]
    fn sparql_after_a_queued_graph_operation_fails_until_the_commit() {
        for (operation, affected) in operations() {
            let db = GrafeoDB::new_in_memory();
            with_graphs(&db);
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute_sparql(&operation).unwrap();
            let update = format!(
                "INSERT DATA {{ GRAPH <{ROME}> {{ <http://example.org/y> <http://example.org/name> 'Lucas' }} }}"
            );
            for statement in [read(PARIS), read(BERLIN), read(ROME), update] {
                let error = session
                    .execute_sparql(&statement)
                    .expect_err(&format!("{operation}: {statement}"));
                assert!(
                    error.to_string().contains("GRAFEO-T004"),
                    "{operation}: {statement}: {error}"
                );
            }
            session
                .execute("INSERT (:Marker {name: 'Jules'})")
                .unwrap_or_else(|error| panic!("{operation}: GQL work: {error}"));
            // Another session reads what is committed.
            for graph in [PARIS, BERLIN, ROME] {
                assert_eq!(
                    answer(&db.session(), graph),
                    1,
                    "{operation}: {graph} elsewhere"
                );
            }
            session.commit().unwrap();
            for graph in &affected {
                session
                    .execute_sparql(&read(graph))
                    .unwrap_or_else(|error| panic!("{operation}: {graph} after: {error}"));
            }
            assert_eq!(answer(&session, ROME), 1, "{operation}: after the commit");
            assert_eq!(markers(&db.session()), 1, "{operation}: the data write");
        }
    }

    /// The refusal holds only inside a transaction: outside one the operation applies at once, and
    /// the same session goes on reading and updating. A transaction that holds no queued operation
    /// runs SPARQL as before.
    #[test]
    fn outside_a_transaction_sparql_after_a_graph_operation_runs() {
        for (operation, _) in operations() {
            let db = GrafeoDB::new_in_memory();
            with_graphs(&db);
            let session = db.session();
            session.execute_sparql(&operation).unwrap();
            let want = after_commit(&operation);
            assert_eq!(
                (answer(&session, PARIS), answer(&session, BERLIN)),
                want,
                "{operation}"
            );
            session
                .execute_sparql(&format!(
                    "INSERT DATA {{ GRAPH <{ROME}> {{ <http://example.org/y> <http://example.org/name> 'Lucas' }} }}"
                ))
                .unwrap_or_else(|error| panic!("{operation}: an update after it: {error}"));
            assert_eq!(answer(&session, ROME), 2, "{operation}");
        }

        let db = GrafeoDB::new_in_memory();
        with_graphs(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        assert_eq!(answer(&session, ROME), 1, "a read with nothing queued");
        session
            .execute_sparql(&format!(
                "INSERT DATA {{ GRAPH <{ROME}> {{ <http://example.org/y> <http://example.org/name> 'Lucas' }} }}"
            ))
            .unwrap_or_else(|error| panic!("an update with nothing queued: {error}"));
        assert_eq!(answer(&session, ROME), 2, "its own write");
        session.commit().unwrap();
        assert_eq!(answer(&db.session(), ROME), 2);
    }

    /// A savepoint rollback past the operation lets SPARQL run again, and the operation is gone.
    #[test]
    fn a_savepoint_rollback_past_the_operation_lets_sparql_run() {
        for (operation, _) in operations() {
            let db = GrafeoDB::new_in_memory();
            with_graphs(&db);
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.savepoint("s").unwrap();
            session.execute_sparql(&operation).unwrap();
            assert!(session.execute_sparql(&read(ROME)).is_err(), "{operation}");
            session.rollback_to_savepoint("s").unwrap();
            assert_eq!(answer(&session, PARIS), 1, "{operation}: its own read");
            session.commit().unwrap();
            assert_eq!(answer(&session, PARIS), 1, "{operation}: committed");
            assert_eq!(answer(&session, BERLIN), 1, "{operation}: committed");
        }
    }

    /// What the operations leave: `(paris, berlin)` rows after a commit.
    fn after_commit(operation: &str) -> (usize, usize) {
        if operation.starts_with("DROP") || operation.starts_with("CLEAR") {
            (0, 1)
        } else if operation.starts_with("COPY") {
            (1, 1)
        } else if operation.starts_with("MOVE") {
            (0, 1)
        } else {
            (1, 2)
        }
    }

    #[test]
    fn rollback_and_commit_leave_what_they_should() {
        for (operation, _) in operations() {
            let db = GrafeoDB::new_in_memory();
            with_graphs(&db);
            let mut session = db.session();
            session.begin_transaction().unwrap();
            session.execute_sparql(&operation).unwrap();
            session.rollback().unwrap();
            assert_eq!(answer(&session, PARIS), 1, "{operation}: rolled back");
            assert_eq!(answer(&session, BERLIN), 1, "{operation}: rolled back");

            session.begin_transaction().unwrap();
            session.execute_sparql(&operation).unwrap();
            session.commit().unwrap();
            let want = after_commit(&operation);
            assert_eq!(
                (answer(&session, PARIS), answer(&session, BERLIN)),
                want,
                "{operation}: committed"
            );
        }
    }

    const PATH_VAR: &str = "GRAFEO_SCHEMA_TX_SPARQL_PATH";
    const OPERATION_VAR: &str = "GRAFEO_SCHEMA_TX_SPARQL_OPERATION";
    const END_VAR: &str = "GRAFEO_SCHEMA_TX_SPARQL_END";

    #[test]
    fn sparql_crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let number: usize = std::env::var(OPERATION_VAR).unwrap().parse().unwrap();
        let end = std::env::var(END_VAR).unwrap();
        let (operation, _) = &operations()[number];
        let db = GrafeoDB::open(PathBuf::from(path)).unwrap();
        with_graphs(&db);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session.execute_sparql(operation).unwrap();
        session
            .execute("INSERT (:Marker {name: 'Vincent'})")
            .unwrap();
        match end.as_str() {
            "commit" => session.commit().unwrap(),
            "rollback" => session.rollback().unwrap(),
            _ => {}
        }
        std::process::exit(0);
    }

    #[test]
    fn a_crash_keeps_the_operation_only_after_its_commit() {
        let dir = tempfile::tempdir().unwrap();
        for (number, (operation, _)) in operations().iter().enumerate() {
            for (end, committed) in [("open", false), ("rollback", false), ("commit", true)] {
                let path = dir.path().join(format!("sparql-{number}-{end}.grafeo"));
                let status = child_process::run(
                    grafeo_common::child_test!("sparql_crash_child")
                        .env(PATH_VAR, &path)
                        .env(OPERATION_VAR, number.to_string())
                        .env(END_VAR, end),
                )
                .unwrap();
                assert!(
                    status.success(),
                    "{operation}, {end}: the child process failed"
                );
                let db = GrafeoDB::open(&path).unwrap();
                let session = db.session();
                let want = if committed {
                    after_commit(operation)
                } else {
                    (1, 1)
                };
                assert_eq!(
                    (answer(&session, PARIS), answer(&session, BERLIN)),
                    want,
                    "{operation}, {end}"
                );
                assert_eq!(
                    markers(&session),
                    usize::from(committed),
                    "{operation}, {end}"
                );
                drop(session);
                db.close().unwrap();
            }
        }
    }
}
