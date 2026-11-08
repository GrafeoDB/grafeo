//! A session without external access refuses, at planning and before any file is opened, every
//! statement that reaches outside the database: LOAD DATA in every form (Cypher `LOAD CSV`, GQL
//! `LOAD DATA`, CSV, JSONL, Parquet), the load procedures, SPARQL `LOAD`, and every procedure the
//! registry marks as reading or writing outside. The refusal is the permission-denied error of a
//! role denial, saying what reached outside. External access belongs to the identity: Admin (and
//! `db.session()`) has it, ReadOnly and ReadWrite do not unless granted with
//! `Identity::with_external_access(true)`. It is independent of `START TRANSACTION READ ONLY`, and
//! it applies to query text only: the caller's own API calls (`load_nodes_csv`) are unaffected.
//!
//! The oracle: the same statement on a session with external access answers as today, and a
//! refused one changes nothing and opens no file. A file that does not exist tells: a session that
//! opened it would say it is not found, a refused one says it has no external access.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main external_access::
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::path::{Path, PathBuf};

use grafeo_common::utils::error::Error;
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::auth::{Identity, Role};
use grafeo_engine::procedures::{BuiltinProcedures, ExternalAccess};
use grafeo_engine::session::Session;

/// The language a statement is written in.
#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
    #[cfg(feature = "sparql")]
    Sparql,
    #[cfg(feature = "sql-pgq")]
    Sql,
}

fn run(session: &Session, language: Language, query: &str) -> Result<usize, Error> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
        #[cfg(feature = "sparql")]
        Language::Sparql => session.execute_sparql(query),
        #[cfg(feature = "sql-pgq")]
        Language::Sql => session.execute_sql(query),
    };
    result.map(|result| result.rows().len())
}

/// The files the statements read: a CSV with headers, one without, and a JSON Lines file, of
/// two people; a Turtle file of one triple. Paths use forward slashes, as a query writes them.
struct Files {
    _dir: tempfile::TempDir,
    csv: String,
    bare_csv: String,
    jsonl: String,
    turtle: String,
    missing: String,
}

fn files() -> Files {
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str| -> PathBuf { dir.path().join(name) };
    let text = |path: &Path| path.to_string_lossy().replace('\\', "/");
    std::fs::write(path("people.csv"), "id,name\n1,Alix\n2,Gus\n").unwrap();
    std::fs::write(path("bare.csv"), "1,Alix\n2,Gus\n").unwrap();
    std::fs::write(
        path("people.jsonl"),
        "{\"id\": 1, \"name\": \"Alix\"}\n{\"id\": 2, \"name\": \"Gus\"}\n",
    )
    .unwrap();
    std::fs::write(path("people.ttl"), "<urn:alix> <urn:knows> <urn:gus> .\n").unwrap();
    Files {
        csv: text(&path("people.csv")),
        bare_csv: text(&path("bare.csv")),
        jsonl: text(&path("people.jsonl")),
        turtle: text(&path("people.ttl")),
        missing: text(&path("nobody-wrote-this.csv")),
        _dir: dir,
    }
}

/// One statement that reaches outside: its name, language, text, and whether a session with
/// access answers it (false for a URL the test does not fetch, a missing file, and SPARQL
/// `LOAD`, which a session with access refuses as not supported yet).
struct Reach {
    name: &'static str,
    language: Language,
    query: String,
    runs: bool,
}

fn reach(name: &'static str, language: Language, query: String, runs: bool) -> Reach {
    Reach {
        name,
        language,
        query,
        runs,
    }
}

/// Every form the matrix names, each a statement the planner refuses on a session without
/// external access.
fn reaching(files: &Files) -> Vec<Reach> {
    let Files {
        csv,
        bare_csv,
        jsonl,
        turtle,
        missing,
        ..
    } = files;
    let load = |path: &str| -> String {
        format!("CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Person'], key: 'id'}})")
    };
    let mut all = vec![
        reach(
            "GQL LOAD DATA CSV with headers",
            Language::Gql,
            format!(
                "LOAD DATA FROM '{csv}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name"
            ),
            true,
        ),
        reach(
            "GQL LOAD DATA CSV without headers",
            Language::Gql,
            format!("LOAD DATA FROM '{bare_csv}' FORMAT CSV AS row RETURN row[1] AS name"),
            true,
        ),
        reach(
            "GQL LOAD DATA from a file:// URL",
            Language::Gql,
            format!(
                "LOAD DATA FROM 'file:///{}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name",
                csv.trim_start_matches('/')
            ),
            true,
        ),
        reach(
            "GQL LOAD DATA from a relative path",
            Language::Gql,
            "LOAD DATA FROM 'people.csv' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name"
                .to_string(),
            false,
        ),
        reach(
            "GQL LOAD DATA from an https URL",
            Language::Gql,
            "LOAD DATA FROM 'https://example.invalid/people.csv' FORMAT CSV WITH HEADERS AS row \
             RETURN row.name AS name"
                .to_string(),
            false,
        ),
        reach(
            "GQL LOAD DATA from a file that does not exist",
            Language::Gql,
            format!(
                "LOAD DATA FROM '{missing}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name"
            ),
            false,
        ),
        reach("GQL load_nodes_csv", Language::Gql, load(csv), true),
        // GQL takes no MATCH straight after a leading CALL ... YIELD yet, though ISO allows
        // it (14.3): the NEXT form here, the direct form in Cypher below.
        reach(
            "GQL load_nodes_csv with YIELD and a following MATCH",
            Language::Gql,
            format!(
                "{} YIELD rows RETURN rows NEXT MATCH (p:Person) RETURN count(p) AS people",
                load(csv)
            ),
            true,
        ),
        reach(
            "GQL load_nodes_csv of a file that does not exist",
            Language::Gql,
            load(missing),
            false,
        ),
        reach(
            "GQL load_edges_csv",
            Language::Gql,
            format!(
                "CALL grafeo.load_edges_csv({{path: '{csv}', type: 'KNOWS', \
                 source: {{label: 'Person', property: 'id', column: 'id'}}, \
                 target: {{label: 'Person', property: 'id', column: 'id'}}}})"
            ),
            true,
        ),
        reach(
            "Cypher LOAD CSV with headers",
            Language::Cypher,
            format!(
                "LOAD CSV WITH HEADERS FROM 'file:///{}' AS row RETURN row.name AS name",
                csv.trim_start_matches('/')
            ),
            true,
        ),
        reach(
            "Cypher LOAD CSV without headers",
            Language::Cypher,
            format!("LOAD CSV FROM '{bare_csv}' AS row RETURN row[1] AS name"),
            true,
        ),
    ];
    all.push(reach(
        "Cypher load_nodes_csv",
        Language::Cypher,
        format!("{} YIELD rows RETURN rows", load(csv)),
        true,
    ));
    all.push(reach(
        "Cypher load_nodes_csv with YIELD and a following MATCH",
        Language::Cypher,
        format!(
            "{} YIELD rows MATCH (p:Person) RETURN p.name AS name",
            load(csv)
        ),
        true,
    ));
    #[cfg(feature = "jsonl-import")]
    all.push(reach(
        "GQL LOAD DATA JSONL",
        Language::Gql,
        format!("LOAD DATA FROM '{jsonl}' FORMAT JSONL AS row RETURN row.name AS name"),
        true,
    ));
    #[cfg(feature = "parquet-import")]
    all.push(reach(
        "GQL LOAD DATA Parquet (a file that does not exist)",
        Language::Gql,
        format!("LOAD DATA FROM '{missing}.parquet' FORMAT PARQUET AS row RETURN row"),
        false,
    ));
    #[cfg(feature = "sparql")]
    {
        let turtle = turtle.trim_start_matches('/');
        all.push(reach(
            "SPARQL LOAD",
            Language::Sparql,
            format!("LOAD <file:///{turtle}>"),
            false,
        ));
        all.push(reach(
            "SPARQL LOAD INTO GRAPH",
            Language::Sparql,
            format!("LOAD <file:///{turtle}> INTO GRAPH <urn:people>"),
            false,
        ));
    }
    #[cfg(feature = "sql-pgq")]
    all.push(reach(
        "SQL/PGQ CALL load_nodes_csv",
        Language::Sql,
        load(csv),
        true,
    ));
    let _ = (jsonl, turtle);
    all
}

/// A ReadWrite session as B1 rules it: no external access unless granted.
fn without_access(db: &GrafeoDB) -> Session {
    db.session_with_identity(Identity::new("deriva", [Role::ReadWrite]))
}

fn with_access(db: &GrafeoDB) -> Session {
    db.session_with_identity(Identity::new("deriva", [Role::ReadWrite]).with_external_access(true))
}

/// The status a role denial has today: the refusal is in the same class.
fn role_denial(db: &GrafeoDB) -> String {
    let error = db
        .session_with_role(Role::ReadOnly)
        .execute("INSERT (:Denied)")
        .expect_err("a ReadOnly session may not write");
    GqlStatus::from(&error).as_str().to_string()
}

/// `result` is the external-access refusal: a role denial's status, saying the session has no
/// external access.
fn refused(result: Result<usize, Error>, denial: &str, what: &str) -> Option<String> {
    match result {
        Ok(rows) => Some(format!(
            "{what}: ran ({rows} rows), want the external-access refusal"
        )),
        Err(error) => {
            let status = GqlStatus::from(&error).as_str().to_string();
            let text = error.to_string();
            (status != denial || !text.contains("external access"))
                .then(|| format!("{what}: {status} {text}, want {denial} naming external access"))
        }
    }
}

fn people(db: &GrafeoDB) -> usize {
    db.execute("MATCH (p:Person) RETURN p")
        .unwrap()
        .rows()
        .len()
}

// ---------------------------------------------------------------------------
// The forms
// ---------------------------------------------------------------------------

/// Every form is refused on a session without external access, a file that does not exist
/// included (the refusal comes before the open), and nothing changes.
#[test]
fn every_statement_that_reaches_outside_is_refused_without_access() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let denial = role_denial(&db);
    let session = without_access(&db);
    let wrong: Vec<String> = reaching(&files)
        .iter()
        .filter_map(|r| refused(run(&session, r.language, &r.query), &denial, r.name))
        .collect();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(people(&db), 0, "a refused load changed the graph");
}

/// The same statements answer on a session with access, as they do today: by the grant, on an
/// Admin identity, and on `db.session()`.
#[test]
fn every_such_statement_runs_with_access() {
    let files = files();
    let sessions: [(&str, fn(&GrafeoDB) -> Session); 3] = [
        ("granted", with_access),
        ("Admin", |db| {
            db.session_with_identity(Identity::new("ops", [Role::Admin]))
        }),
        ("db.session()", GrafeoDB::session),
    ];
    for (who, session_of) in sessions {
        for r in reaching(&files).iter().filter(|r| r.runs) {
            // A database per statement: one load does not meet another's nodes. The edges find
            // their ends among people loaded through the API first.
            let db = GrafeoDB::new_in_memory();
            if r.name.contains("edges") {
                db.load_nodes_csv(
                    Path::new(&files.csv),
                    &grafeo_engine::loader::NodeLoad {
                        labels: vec!["Person".to_string()],
                        key: Some("id".to_string()),
                        ..Default::default()
                    },
                )
                .unwrap();
            }
            if let Err(error) = run(&session_of(&db), r.language, &r.query) {
                panic!("{who}, {}: {error}", r.name);
            }
        }
    }
}

/// A file that does not exist, on a session with access, fails as a file it cannot open: so the
/// refusal on a session without access is not that error, and comes before the open.
#[test]
fn a_missing_file_fails_differently_with_access() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let denial = role_denial(&db);
    let session = with_access(&db);
    for r in reaching(&files)
        .iter()
        .filter(|r| r.name.contains("does not exist"))
    {
        let error = run(&session, r.language, &r.query).expect_err(r.name);
        assert!(
            !error.to_string().contains("external access")
                && GqlStatus::from(&error).as_str() != denial,
            "{}: {error}",
            r.name
        );
    }
}

// ---------------------------------------------------------------------------
// The identities
// ---------------------------------------------------------------------------

/// External access follows the role (B1): ReadOnly and ReadWrite, from `session_with_role` or
/// `Identity::new`, have none; the grant gives it to either; Admin has it.
#[test]
fn external_access_follows_the_role_unless_granted() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let denial = role_denial(&db);
    let query = format!(
        "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name",
        files.csv
    );
    let cases = [
        (
            "session_with_role(ReadOnly)",
            db.session_with_role(Role::ReadOnly),
            false,
        ),
        (
            "session_with_role(ReadWrite)",
            db.session_with_role(Role::ReadWrite),
            false,
        ),
        (
            "Identity::new(ReadOnly)",
            db.session_with_identity(Identity::new("a", [Role::ReadOnly])),
            false,
        ),
        (
            "Identity::new(ReadWrite)",
            db.session_with_identity(Identity::new("b", [Role::ReadWrite])),
            false,
        ),
        (
            "ReadOnly granted",
            db.session_with_identity(
                Identity::new("c", [Role::ReadOnly]).with_external_access(true),
            ),
            true,
        ),
        ("ReadWrite granted", with_access(&db), true),
        (
            "Admin",
            db.session_with_identity(Identity::new("d", [Role::Admin])),
            true,
        ),
        ("db.session()", db.session(), true),
    ];
    let mut wrong = Vec::new();
    for (who, session, allowed) in cases {
        let result = run(&session, Language::Gql, &query);
        if allowed {
            if let Err(error) = result {
                wrong.push(format!("{who}: {error}"));
            }
        } else if let Some(problem) = refused(result, &denial, who) {
            wrong.push(problem);
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert!(Identity::new("e", [Role::Admin]).can_access_external());
    assert!(!Identity::new("f", [Role::ReadWrite]).can_access_external());
    assert!(
        Identity::new("g", [Role::ReadWrite])
            .with_external_access(true)
            .can_access_external()
    );
    assert!(
        !Identity::new("h", [Role::Admin])
            .with_external_access(false)
            .can_access_external(),
        "the explicit setting decides, also against the role"
    );
}

/// External access applies to query text only: the caller's own API calls load on a session
/// without it, and on the database.
#[test]
fn api_calls_are_not_bound_by_it() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let session = without_access(&db);
    let report = session.load_nodes_csv(
        Path::new(&files.csv),
        &grafeo_engine::loader::NodeLoad {
            labels: vec!["Person".to_string()],
            key: Some("id".to_string()),
            ..Default::default()
        },
    );
    assert!(report.is_ok(), "{report:?}");
    assert_eq!(people(&db), 2);
}

// ---------------------------------------------------------------------------
// The views
// ---------------------------------------------------------------------------

/// The refusal holds wherever the statement runs: auto-commit, under EXPLAIN and PROFILE, in a
/// read-write and a read-only transaction, after a rollback, after `USE GRAPH`, and for a plan an
/// unrestricted session already cached. A statement composed after one that writes refuses the
/// whole, so nothing runs.
#[test]
fn the_refusal_holds_in_every_view() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH elsewhere").unwrap();
    let files = files();
    let denial = role_denial(&db);
    let query = format!(
        "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name",
        files.csv
    );
    let mut wrong = Vec::new();
    let mut check = |what: &str, result: Result<usize, Error>| {
        if let Some(problem) = refused(result, &denial, what) {
            wrong.push(problem);
        }
    };

    let session = without_access(&db);
    check("auto-commit", run(&session, Language::Gql, &query));
    check(
        "EXPLAIN",
        run(&session, Language::Gql, &format!("EXPLAIN {query}")),
    );
    check(
        "PROFILE",
        run(&session, Language::Gql, &format!("PROFILE {query}")),
    );

    let mut session = without_access(&db);
    session.begin_transaction().unwrap();
    check(
        "in a read-write transaction",
        run(&session, Language::Gql, &query),
    );
    session.rollback().unwrap();
    check("after its rollback", run(&session, Language::Gql, &query));

    let session = without_access(&db);
    session.execute("START TRANSACTION READ ONLY").unwrap();
    check(
        "in a read-only transaction",
        run(&session, Language::Gql, &query),
    );
    session.execute("ROLLBACK").unwrap();

    // No statement gives the session access: a reset of the session leaves it without.
    let session = without_access(&db);
    session.execute("SESSION RESET").unwrap();
    check("after SESSION RESET", run(&session, Language::Gql, &query));

    let session = without_access(&db);
    session.execute("USE GRAPH elsewhere").unwrap();
    check("after USE GRAPH", run(&session, Language::Gql, &query));

    // A plan an unrestricted session cached is checked again for this one.
    run(&with_access(&db), Language::Gql, &query).unwrap();
    check(
        "a plan cached by another session",
        run(&without_access(&db), Language::Gql, &query),
    );

    let composed = format!("INSERT (:Marker) RETURN 1 AS one NEXT {query}");
    check(
        "composed after a write",
        run(&without_access(&db), Language::Gql, &composed),
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(
        db.execute("MATCH (m:Marker) RETURN m")
            .unwrap()
            .rows()
            .len(),
        0,
        "the write before the refused statement ran"
    );
}

/// READ ONLY and external access are independent: a session with access runs `LOAD DATA ...
/// RETURN` inside a read-only transaction.
#[test]
fn a_read_only_transaction_keeps_external_access() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let session = with_access(&db);
    session.execute("START TRANSACTION READ ONLY").unwrap();
    let rows = run(
        &session,
        Language::Gql,
        &format!(
            "LOAD DATA FROM '{}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name",
            files.csv
        ),
    )
    .unwrap();
    assert_eq!(rows, 2);
    session.execute("COMMIT").unwrap();
}

// ---------------------------------------------------------------------------
// The class: the procedure registry
// ---------------------------------------------------------------------------

/// Every registered procedure, the load procedures planned by name included, states whether it
/// reaches outside, and a call without access is refused before its arguments are read exactly
/// when it does (`EXPLAIN CALL name()`, so nothing runs and no argument is needed). A procedure
/// added later is covered here with no change.
#[test]
fn a_procedure_is_refused_without_access_exactly_when_it_reaches_outside() {
    let db = GrafeoDB::new_in_memory();
    let denial = role_denial(&db);
    let session = without_access(&db);
    let procedures = BuiltinProcedures::new().list();
    for name in ["grafeo.load_nodes_csv", "grafeo.load_edges_csv"] {
        let load = procedures
            .iter()
            .find(|procedure| procedure.name == name)
            .unwrap_or_else(|| panic!("{name} is in the registry CALL resolution reads"));
        assert_ne!(
            load.external_access,
            ExternalAccess::None,
            "{name} reads a file"
        );
    }
    let mut wrong = Vec::new();
    let mut reaching = 0;
    for procedure in &procedures {
        let result = run(
            &session,
            Language::Gql,
            &format!("EXPLAIN CALL {}()", procedure.name),
        );
        let reaches = procedure.external_access != ExternalAccess::None;
        if reaches {
            reaching += 1;
            if let Some(problem) = refused(result, &denial, &procedure.name) {
                wrong.push(problem);
            }
        } else if let Err(error) = result
            && error.to_string().contains("external access")
        {
            wrong.push(format!(
                "{}: refused, but it states no external access",
                procedure.name
            ));
        }
    }
    assert!(reaching >= 2, "{reaching} procedures reach outside");
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The external-access check reads the logical plan and needs no values: EXPLAIN of a load
/// procedure whose path is a parameter, without its value, is refused on a session without
/// access and gives the plan on one with it. (LOAD DATA and LOAD CSV take only a literal path.)
#[test]
fn explain_of_a_load_from_a_parameter_is_judged_without_values() {
    let db = GrafeoDB::new_in_memory();
    let denial = role_denial(&db);
    for (language, query) in [
        (
            Language::Gql,
            "EXPLAIN CALL grafeo.load_nodes_csv({path: $path, labels: ['Person']})",
        ),
        (
            Language::Cypher,
            "EXPLAIN CALL grafeo.load_nodes_csv({path: $path, labels: ['Person']}) \
             YIELD rows RETURN rows",
        ),
    ] {
        if let Some(problem) = refused(run(&without_access(&db), language, query), &denial, query) {
            panic!("{problem}");
        }
        run(&with_access(&db), language, query)
            .unwrap_or_else(|error| panic!("{query} with access: {error}"));
    }
}

/// A session without external access runs everything that stays inside the database: reads,
/// writes (its role allows them), procedures that state no external access, and EXPLAIN.
#[test]
fn a_session_without_access_runs_everything_else() {
    let db = GrafeoDB::new_in_memory();
    let session = without_access(&db);
    for (language, query, rows) in [
        (Language::Gql, "INSERT (:Person {name: 'Alix'})", 0),
        (Language::Cypher, "CREATE (:Person {name: 'Gus'})", 0),
        (Language::Gql, "MATCH (n) RETURN count(n) AS n", 1),
        (
            Language::Cypher,
            "MATCH (p:Person) RETURN p.name AS name",
            2,
        ),
        (
            Language::Gql,
            "CALL grafeo.procedures() YIELD name RETURN count(name) AS n",
            1,
        ),
        (Language::Gql, "EXPLAIN MATCH (p:Person) RETURN p.name", 1),
    ] {
        match run(&session, language, query) {
            Ok(found) => assert_eq!(found, rows, "{query}"),
            Err(error) => panic!("{query}: {error}"),
        }
    }
    assert_eq!(people(&db), 2);
}

/// Deriva's case: an untrusted session (ReadOnly, so without external access by B1) on a
/// database a pipeline shares. A secret file on disk, both of Deriva's probe queries inside a
/// transaction it rolls back: each is refused, nothing is loaded, and the session still counts
/// the graph. On 0.6.1 both queries returned the file's rows. The load procedures also write,
/// and the external-access refusal comes before the role's write denial, so all three name it.
#[test]
fn an_untrusted_session_reads_no_file() {
    let db = GrafeoDB::new_in_memory();
    let dir = tempfile::tempdir().unwrap();
    let secret = dir.path().join("secret.csv");
    std::fs::write(&secret, "name\nHans\nShosanna\n").unwrap();
    let secret = secret.to_string_lossy().replace('\\', "/");
    let denial = role_denial(&db);

    let mut session = db.session_with_identity(Identity::new("studio", [Role::ReadOnly]));
    session.begin_transaction().unwrap();
    let mut wrong = Vec::new();
    for (language, query) in [
        (
            Language::Cypher,
            format!(
                "CALL grafeo.load_nodes_csv({{path: '{secret}', labels: ['Leak']}}) \
                 YIELD rows MATCH (n:Leak) RETURN n.name"
            ),
        ),
        (
            Language::Gql,
            format!("CALL grafeo.load_nodes_csv({{path: '{secret}', labels: ['Leak']}})"),
        ),
        (
            Language::Cypher,
            format!(
                "LOAD CSV WITH HEADERS FROM 'file:///{}' AS row RETURN row.name",
                secret.trim_start_matches('/')
            ),
        ),
    ] {
        if let Some(problem) = refused(run(&session, language, &query), &denial, &query) {
            wrong.push(problem);
        }
    }
    session.rollback().unwrap();
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(
        run(&session, Language::Gql, "MATCH (n) RETURN count(n) AS n").unwrap(),
        1
    );
    assert_eq!(
        db.execute("MATCH (n:Leak) RETURN n").unwrap().rows().len(),
        0,
        "a refused load left nodes"
    );
}

/// The order of refusals for one statement: external access, then the role's write denial,
/// then the READ ONLY transaction (25G03), then argument validation (a value of the wrong type,
/// 22G03, or a name that is no parameter, 42002, as for every procedure). A load procedure both
/// reaches outside and writes, so each limit lifted in turn shows the next refusal.
#[test]
fn the_refusals_come_in_their_order() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let denial = role_denial(&db);
    let load = format!(
        "CALL grafeo.load_nodes_csv({{path: '{}', labels: ['Person']}})",
        files.csv
    );
    let bad_argument = "CALL grafeo.load_nodes_csv({path: 5, labels: ['Person']})".to_string();
    let unknown_name = format!(
        "CALL grafeo.load_nodes_csv({{path: '{}', labels: ['Person'], nonsense: 1}})",
        files.csv
    );
    let read_only = || Identity::new("studio", [Role::ReadOnly]);
    let read_write = || Identity::new("pipeline", [Role::ReadWrite]);
    let admin = || Identity::new("ops", [Role::Admin]);
    // (who, identity, inside START TRANSACTION READ ONLY, query, the refusal expected)
    let cases = [
        ("ReadOnly", read_only(), false, &load, "access"),
        (
            "ReadOnly, a bad argument",
            read_only(),
            false,
            &bad_argument,
            "access",
        ),
        (
            "ReadOnly granted",
            read_only().with_external_access(true),
            false,
            &load,
            "role",
        ),
        (
            "ReadOnly granted, a bad argument",
            read_only().with_external_access(true),
            false,
            &bad_argument,
            "role",
        ),
        (
            "ReadWrite in READ ONLY",
            read_write(),
            true,
            &load,
            "access",
        ),
        (
            "ReadWrite granted in READ ONLY",
            read_write().with_external_access(true),
            true,
            &load,
            "read-only transaction",
        ),
        (
            "ReadWrite granted in READ ONLY, a bad argument",
            read_write().with_external_access(true),
            true,
            &bad_argument,
            "read-only transaction",
        ),
        (
            "Admin in READ ONLY",
            admin(),
            true,
            &load,
            "read-only transaction",
        ),
        (
            "Admin without access in READ ONLY",
            admin().with_external_access(false),
            true,
            &load,
            "access",
        ),
        (
            "ReadWrite granted, a bad argument",
            read_write().with_external_access(true),
            false,
            &bad_argument,
            "argument",
        ),
        (
            "ReadOnly, an unknown name",
            read_only(),
            false,
            &unknown_name,
            "access",
        ),
        (
            "ReadOnly granted, an unknown name",
            read_only().with_external_access(true),
            false,
            &unknown_name,
            "role",
        ),
        (
            "ReadWrite granted in READ ONLY, an unknown name",
            read_write().with_external_access(true),
            true,
            &unknown_name,
            "read-only transaction",
        ),
        (
            "ReadWrite granted, an unknown name",
            read_write().with_external_access(true),
            false,
            &unknown_name,
            "unknown name",
        ),
    ];
    let mut wrong = Vec::new();
    for (who, identity, in_read_only, query, expected) in cases {
        let session = db.session_with_identity(identity);
        if in_read_only {
            session.execute("START TRANSACTION READ ONLY").unwrap();
        }
        let got = match run(&session, Language::Gql, query) {
            Ok(rows) => format!("ran ({rows} rows)"),
            Err(error) => {
                let status = GqlStatus::from(&error).as_str().to_string();
                let text = error.to_string();
                if status == denial && text.contains("external access") {
                    "access".to_string()
                } else if status == denial && text.contains("permission denied") {
                    "role".to_string()
                } else if status == "25G03" {
                    "read-only transaction".to_string()
                } else if status == "22G03" {
                    "argument".to_string()
                } else if status == "42002" && text.contains("nonsense") {
                    "unknown name".to_string()
                } else {
                    format!("{status} {text}")
                }
            }
        };
        if got != expected {
            wrong.push(format!("{who}: {got}, want {expected}"));
        }
        if in_read_only {
            session.execute("ROLLBACK").unwrap();
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(people(&db), 0, "a refused load changed the graph");
}

// ---------------------------------------------------------------------------
// The load procedures in the registry
// ---------------------------------------------------------------------------

/// The two load procedures are in the one table CALL resolution reads, in every build (the
/// registry is not behind `algos`): `BuiltinProcedures::list()` and `grafeo.procedures()` name
/// them.
#[test]
fn the_load_procedures_are_listed_in_every_build() {
    let listed: Vec<String> = BuiltinProcedures::new()
        .list()
        .into_iter()
        .map(|procedure| procedure.name.clone())
        .collect();
    let db = GrafeoDB::new_in_memory();
    let called: Vec<String> = db
        .execute("CALL grafeo.procedures() YIELD name RETURN name")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].to_string())
        .collect();
    for name in ["grafeo.load_nodes_csv", "grafeo.load_edges_csv"] {
        assert!(listed.iter().any(|n| n == name), "{name} not in list()");
        assert!(
            called.iter().any(|n| n.contains(name)),
            "{name} not in grafeo.procedures(): {called:?}"
        );
    }
}

/// A bad or missing load argument is an argument error, 22023 or 22G03, as for every procedure,
/// in the map and the positional form: `path` is required for both, `labels` for nodes, the
/// type and both ends for edges. A name that is no parameter is 42002, as for every procedure.
/// On a session without external access the access refusal comes first for each of them.
#[test]
fn a_bad_load_argument_is_an_argument_error() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let denial = role_denial(&db);
    let missing = &files.missing;
    let bad = [
        "CALL grafeo.load_nodes_csv()".to_string(),
        "CALL grafeo.load_nodes_csv({labels: ['Person']})".to_string(),
        format!("CALL grafeo.load_nodes_csv({{path: '{missing}'}})"),
        format!("CALL grafeo.load_nodes_csv({{path: '{missing}', labels: 5}})"),
        "CALL grafeo.load_nodes_csv(5, ['Person'])".to_string(),
        format!("CALL grafeo.load_nodes_csv('{missing}')"),
        "CALL grafeo.load_nodes_csv(null, ['Person'])".to_string(),
        "CALL grafeo.load_edges_csv()".to_string(),
        format!("CALL grafeo.load_edges_csv({{path: '{missing}'}})"),
        format!("CALL grafeo.load_edges_csv({{path: '{missing}', type: 'KNOWS'}})"),
    ];
    let mut wrong = Vec::new();
    for query in &bad {
        match run(&with_access(&db), Language::Gql, query) {
            Ok(rows) => wrong.push(format!("{query}: ran ({rows} rows)")),
            Err(error) => {
                let status = GqlStatus::from(&error).as_str().to_string();
                if status != "22023" && status != "22G03" {
                    wrong.push(format!("{query}: {status} {error}, want 22023 or 22G03"));
                }
            }
        }
        if let Some(problem) = refused(
            run(&without_access(&db), Language::Gql, query),
            &denial,
            query,
        ) {
            wrong.push(format!("without access, {problem}"));
        }
    }
    for procedure in ["grafeo.load_nodes_csv", "grafeo.load_edges_csv"] {
        let query = format!("CALL {procedure}({{path: '{missing}', nonsense: 1}})");
        match run(&with_access(&db), Language::Gql, &query) {
            Ok(rows) => wrong.push(format!("{query}: ran ({rows} rows)")),
            Err(error) => {
                let status = GqlStatus::from(&error).as_str().to_string();
                if status != "42002" || !error.to_string().contains("nonsense") {
                    wrong.push(format!(
                        "{query}: {status} {error}, want 42002 naming nonsense"
                    ));
                }
            }
        }
        if let Some(problem) = refused(
            run(&without_access(&db), Language::Gql, &query),
            &denial,
            &query,
        ) {
            wrong.push(format!("without access, {problem}"));
        }
    }
    // The positional form with good arguments is read as the map form: the missing file is
    // then what fails, as an open, not as an argument.
    let positional = format!("CALL grafeo.load_nodes_csv('{missing}', ['Person'])");
    match run(&with_access(&db), Language::Gql, &positional) {
        Ok(rows) => wrong.push(format!("{positional}: ran ({rows} rows)")),
        Err(error) => {
            let status = GqlStatus::from(&error).as_str().to_string();
            if status.starts_with("42") || status == "22023" || status == "22G03" {
                wrong.push(format!(
                    "{positional}: {status} {error}, want the open's error"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(people(&db), 0);
}

/// `labels` takes one label as a string or a list of them, and both load the same nodes. The
/// widening is declared on `labels` only: a lone string for another procedure's List parameter
/// is still a type error.
#[test]
fn labels_take_a_string_or_a_list() {
    let files = files();
    for labels in ["'Person'", "['Person']"] {
        let db = GrafeoDB::new_in_memory();
        let query = format!(
            "CALL grafeo.load_nodes_csv({{path: '{}', labels: {labels}}})",
            files.csv
        );
        if let Err(error) = run(&with_access(&db), Language::Gql, &query) {
            panic!("labels: {labels}: {error}");
        }
        assert_eq!(people(&db), 2, "labels: {labels}");
    }
    let db = GrafeoDB::new_in_memory();
    let error = run(
        &with_access(&db),
        Language::Gql,
        "CALL grafeo.search.vector('Doc', 'emb', 'abc', 3)",
    )
    .expect_err("a string is not a list");
    assert_eq!(GqlStatus::from(&error).as_str(), "22G03", "{error}");
}

/// A load option with a value outside its domain is 22023, naming the procedure, the argument
/// and the key; a non-map value where a map is declared is 22G03. On a session without external
/// access the access refusal comes first.
#[test]
fn a_load_option_outside_its_domain_is_refused() {
    let db = GrafeoDB::new_in_memory();
    let files = files();
    let denial = role_denial(&db);
    let csv = &files.csv;
    let nodes = |extra: &str| {
        format!("CALL grafeo.load_nodes_csv({{path: '{csv}', labels: ['Person']{extra}}})")
    };
    let end = "{label: 'Person', property: 'id', column: 'id'}";
    let edges = |source: &str, target: &str, extra: &str| {
        format!(
            "CALL grafeo.load_edges_csv({{path: '{csv}', type: 'KNOWS', source: {source}, \
             target: {target}{extra}}})"
        )
    };
    // (query, status, the words the message names: procedure, argument, key)
    let cases: Vec<(String, &str, Vec<&str>)> = vec![
        (
            nodes(", delimiter: ''"),
            "22023",
            vec!["grafeo.load_nodes_csv", "delimiter"],
        ),
        (
            nodes(", delimiter: ';;'"),
            "22023",
            vec!["grafeo.load_nodes_csv", "delimiter"],
        ),
        (
            nodes(", types: {id: 'nonsense'}"),
            "22023",
            vec!["grafeo.load_nodes_csv", "types", "id"],
        ),
        (
            nodes(", types: 5"),
            "22G03",
            vec!["grafeo.load_nodes_csv", "types"],
        ),
        (
            edges(end, end, ", missing: 'maybe'"),
            "22023",
            vec!["grafeo.load_edges_csv", "missing"],
        ),
        (
            edges("{label: 'Person', property: 'id'}", end, ""),
            "22023",
            vec!["grafeo.load_edges_csv", "source", "column"],
        ),
        (
            edges(
                "{label: 'Person', property: 'id', column: 'id', bogus: 1}",
                end,
                "",
            ),
            "22023",
            vec!["grafeo.load_edges_csv", "source", "bogus"],
        ),
        (
            edges(end, "{property: 'id', column: 'id'}", ""),
            "22023",
            vec!["grafeo.load_edges_csv", "target", "label"],
        ),
        (
            edges("5", end, ""),
            "22G03",
            vec!["grafeo.load_edges_csv", "source"],
        ),
        (
            edges(end, "'Person'", ""),
            "22G03",
            vec!["grafeo.load_edges_csv", "target"],
        ),
    ];
    let mut wrong = Vec::new();
    for (query, status, words) in &cases {
        match run(&with_access(&db), Language::Gql, query) {
            Ok(rows) => wrong.push(format!("{query}: ran ({rows} rows)")),
            Err(error) => {
                let got = GqlStatus::from(&error).as_str().to_string();
                let text = error.to_string();
                let unnamed: Vec<&&str> = words.iter().filter(|w| !text.contains(**w)).collect();
                if got != *status || !unnamed.is_empty() {
                    wrong.push(format!(
                        "{query}: {got} {text}, want {status} naming {words:?}"
                    ));
                }
            }
        }
        if let Some(problem) = refused(
            run(&without_access(&db), Language::Gql, query),
            &denial,
            query,
        ) {
            wrong.push(format!("without access, {problem}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    assert_eq!(people(&db), 0);
}

/// `grafeo.procedures()` shows the loads' declared arguments: their types, that `labels` takes
/// a string or a list, and the domain of `missing`.
#[test]
fn the_procedure_list_shows_the_load_arguments() {
    let db = GrafeoDB::new_in_memory();
    let result = db
        .execute(
            "CALL grafeo.procedures() YIELD name, parameters \
             RETURN name, parameters",
        )
        .unwrap();
    let parameters = |procedure: &str| -> String {
        result
            .rows()
            .iter()
            .find(|row| row[0].to_string().contains(procedure))
            .map_or_else(
                || panic!("{procedure} is not listed"),
                |row| row[1].to_string(),
            )
    };
    let nodes = parameters("grafeo.load_nodes_csv");
    for word in [
        "path",
        "String",
        "labels",
        "List",
        "delimiter",
        "types",
        "key",
    ] {
        assert!(
            nodes.contains(word),
            "load_nodes_csv lists no {word}: {nodes}"
        );
    }
    let edges = parameters("grafeo.load_edges_csv");
    for word in [
        "path", "type", "source", "target", "missing", "error", "skip",
    ] {
        assert!(
            edges.contains(word),
            "load_edges_csv lists no {word}: {edges}"
        );
    }
}
