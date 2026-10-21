//! `Session::statement_writes` tells, without running anything, whether a
//! statement writes: whether a read-only database refuses it. The command
//! line tool opens a database read-only for a statement that does not, so
//! the answer must match what the read-only database then does.

#![cfg(feature = "gql")]

use grafeo_engine::GrafeoDB;

fn writes(db: &GrafeoDB, query: &str, language: &str) -> bool {
    db.session()
        .statement_writes(query, language)
        .unwrap_or_else(|error| panic!("{language} {query}: {error}"))
}

#[test]
fn gql_queries_read_and_changes_write() {
    let db = GrafeoDB::new_in_memory();
    for query in [
        "MATCH (n) RETURN n",
        "MATCH (n:Person) WHERE n.age > 3 RETURN count(n) AS c",
        "RETURN 1 AS one",
        "OPTIONAL MATCH (n) RETURN n",
    ] {
        assert!(!writes(&db, query, "gql"), "{query}");
    }
    for query in [
        "INSERT (:Person {name: 'Alix'})",
        "MATCH (n) SET n.seen = true",
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) REMOVE n:Person",
        "MERGE (:City {name: 'Amsterdam'})",
        "MATCH (n) INSERT (n)-[:KNOWS]->(:Person {name: 'Gus'}) RETURN n",
    ] {
        assert!(writes(&db, query, "gql"), "{query}");
    }
}

#[test]
fn explain_runs_nothing_and_profile_runs_its_statement() {
    let db = GrafeoDB::new_in_memory();
    assert!(!writes(
        &db,
        "EXPLAIN INSERT (:Person {name: 'Alix'})",
        "gql"
    ));
    assert!(!writes(&db, "PROFILE MATCH (n) RETURN n", "gql"));
    assert!(writes(
        &db,
        "PROFILE INSERT (:Person {name: 'Alix'})",
        "gql"
    ));
}

#[test]
fn schema_ddl_and_graph_commands_write_and_session_commands_do_not() {
    let db = GrafeoDB::new_in_memory();
    for query in [
        "CREATE NODE TYPE Person (name STRING)",
        "DROP NODE TYPE Person",
        "CREATE GRAPH g",
        "DROP GRAPH g",
    ] {
        assert!(writes(&db, query, "gql"), "{query}");
    }
    for query in [
        "SESSION SET TIME ZONE '+01:00'",
        "SESSION RESET",
        "START TRANSACTION",
        "COMMIT",
    ] {
        assert!(!writes(&db, query, "gql"), "{query}");
    }
}

/// A call of a stored procedure writes when its body does, which the plan of
/// the call does not show.
#[cfg(feature = "algos")]
#[test]
fn a_call_writes_when_the_procedure_body_does() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE PROCEDURE add_person(name STRING) RETURNS (n INTEGER) AS { INSERT (:Person {name: $name}) RETURN 1 AS n }")
        .unwrap();
    session
        .execute("CREATE PROCEDURE answer() RETURNS (v INTEGER) AS { RETURN 42 AS v }")
        .unwrap();
    assert!(writes(
        &db,
        "CALL add_person('Alix') YIELD n RETURN n",
        "gql"
    ));
    assert!(!writes(&db, "CALL answer() YIELD v RETURN v", "gql"));
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_statements() {
    let db = GrafeoDB::new_in_memory();
    assert!(!writes(&db, "MATCH (n) RETURN n", "cypher"));
    assert!(!writes(&db, "SHOW INDEXES", "cypher"));
    assert!(writes(&db, "CREATE (:Person {name: 'Alix'})", "cypher"));
    assert!(writes(&db, "MATCH (n) DETACH DELETE n", "cypher"));
    assert!(writes(
        &db,
        "CREATE INDEX person_name FOR (n:Person) ON (n.name)",
        "cypher"
    ));
}

#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_statements() {
    let db = GrafeoDB::new_in_memory();
    assert!(!writes(
        &db,
        "SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))",
        "sql"
    ));
    assert!(writes(
        &db,
        "CREATE PROPERTY GRAPH g NODE TABLES (Person (id BIGINT PRIMARY KEY, name VARCHAR))",
        "sql"
    ));
}

#[cfg(feature = "gremlin")]
#[test]
fn gremlin_statements() {
    let db = GrafeoDB::new_in_memory();
    assert!(!writes(&db, "g.V().count()", "gremlin"));
    assert!(writes(
        &db,
        "g.addV('Person').property('name', 'Alix')",
        "gremlin"
    ));
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_statements() {
    let db = GrafeoDB::new_in_memory();
    assert!(!writes(&db, "SELECT ?s WHERE { ?s ?p ?o }", "sparql"));
    assert!(writes(
        &db,
        "INSERT DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> }",
        "sparql"
    ));
}

#[test]
fn a_statement_that_does_not_parse_fails_as_running_it_would() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .session()
        .statement_writes("MATCH (n RETURN n", "gql")
        .unwrap_err();
    let run = db.execute("MATCH (n RETURN n").unwrap_err();
    assert_eq!(error.to_string(), run.to_string());
}

#[test]
fn an_unknown_language_is_an_error() {
    let db = GrafeoDB::new_in_memory();
    let error = db
        .session()
        .statement_writes("MATCH (n) RETURN n", "klingon")
        .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("Unknown query language: 'klingon'"),
        "{error}"
    );
}

/// The guarantee: a read-only database runs every statement that does not
/// write, and refuses, before it changes anything, every one that does.
#[test]
fn a_read_only_database_refuses_exactly_the_writing_statements() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("people.grafeo");
    let db = GrafeoDB::open(&path).unwrap();
    db.execute("INSERT (:Person {name: 'Alix', age: 30})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    db.close().unwrap();
    drop(db);

    let db = GrafeoDB::open_read_only(&path).unwrap();
    let mut cases = vec![
        ("gql", "MATCH (n:Person) RETURN n.name AS name"),
        ("gql", "EXPLAIN INSERT (:Person {name: 'Mia'})"),
        ("gql", "PROFILE MATCH (n) RETURN count(n) AS c"),
        ("gql", "SESSION SET TIME ZONE '+01:00'"),
        ("gql", "INSERT (:Person {name: 'Mia'})"),
        ("gql", "MATCH (n:Person {name: 'Alix'}) SET n.age = 31"),
        ("gql", "MATCH (n) DETACH DELETE n"),
        ("gql", "PROFILE INSERT (:Person {name: 'Mia'})"),
        ("gql", "CREATE NODE TYPE City (name STRING)"),
        ("gql", "CREATE GRAPH other"),
    ];
    #[cfg(feature = "cypher")]
    cases.extend([
        ("cypher", "MATCH (n:Person) RETURN n.name AS name"),
        ("cypher", "EXPLAIN CREATE (:Person {name: 'Mia'})"),
        ("cypher", "CREATE (:Person {name: 'Mia'})"),
        (
            "cypher",
            "CREATE INDEX person_name FOR (n:Person) ON (n.name)",
        ),
    ]);
    for (language, query) in cases {
        let writes = writes(&db, query, language);
        let result = db.session().execute_language(query, language, None);
        if writes {
            assert!(
                result.is_err(),
                "{language} {query} wrote on a read-only database"
            );
        } else if let Err(error) = result {
            panic!(
                "{language} {query} does not write, but a read-only database refused it: {error}"
            );
        }
    }
    assert_eq!(db.node_count(), 2);
}

/// A read-only transaction refuses the same statements: it runs `EXPLAIN` of
/// a write, which used to fail in GQL, and refuses the write itself.
#[test]
fn a_read_only_transaction_explains_a_write_and_refuses_it() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION READ ONLY").unwrap();
    let explained = session.execute("EXPLAIN INSERT (:Person {name: 'Alix'})");
    assert!(explained.is_ok(), "{:?}", explained.err());
    assert!(session.execute("INSERT (:Person {name: 'Alix'})").is_err());
    session.execute("ROLLBACK").unwrap();
    assert_eq!(db.node_count(), 0);
}
