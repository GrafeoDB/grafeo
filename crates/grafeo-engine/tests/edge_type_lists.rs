//! An edge pattern names one edge type, or alternatives joined by `|`; a
//! second `:` (`-[:Graph:CONTAINS]->`) is a syntax error that names both ways
//! to write what was meant, in every pattern of every statement.
//!
//! Reported downstream (Deriva, 2026-10-09): its edge types carry a namespace
//! (`Graph:CONTAINS`), and `MATCH (r)-[:Graph:CONTAINS*]->(f)` returned no
//! rows where `` -[:`Graph:CONTAINS`*]-> `` returned the files: the parsers
//! read `:Graph:CONTAINS` as the alternatives `Graph` or `CONTAINS`, and
//! CREATE, MERGE and INSERT created an edge of type `Graph`.
//!
//! openCypher 9 writes a relationship's types as `:A`, or `:A|B` and `:A|:B`
//! for alternatives (`RelationshipTypes` in its grammar). In GQL the types of
//! an edge are a label expression (ISO/IEC 39075:2024, 16.8): `|` joins
//! alternatives and `&` labels the edge must all carry; an edge has exactly
//! one type here, so `:A&B` would match no edge and is an error too. An edge
//! created by CREATE, MERGE or INSERT gets one type, never alternatives.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test edge_type_lists
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_common::utils::error::Result;
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// A repository that contains a directory that contains a file, over edges
/// of type `Graph:CONTAINS`, and a plain `CONTAINS` edge from the directory
/// to a second file and a `Graph` edge from the repository to a third.
fn code_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (r:Repository {repoName: 'deriva'})-[:`Graph:CONTAINS`]->(d:Directory {name: 'Amsterdam'})\
               -[:`Graph:CONTAINS`]->(:File {filePath: 'Amsterdam/Alix.py'}), \
               (d)-[:CONTAINS]->(:File {filePath: 'Amsterdam/Gus.py'}), \
               (r)-[:Graph]->(:File {filePath: 'Berlin.py'})",
    )
    .unwrap();
    db
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> Result<QueryResult> {
    match language {
        "cypher" => db.execute_cypher(query),
        _ => db.execute(query),
    }
}

/// The values of the one column `query` returns, sorted.
fn column(db: &GrafeoDB, language: &str, query: &str) -> Vec<String> {
    let result = run(db, language, query).unwrap_or_else(|e| panic!("{language}: {query}: {e}"));
    let mut values: Vec<String> = result
        .rows()
        .iter()
        .map(|row| match &row[0] {
            Value::String(text) => text.to_string(),
            value => format!("{value}"),
        })
        .collect();
    values.sort();
    values
}

/// Asserts that `query` is a syntax error whose message contains every one
/// of `names`, and that it wrote nothing.
fn assert_refused(db: &GrafeoDB, language: &str, query: &str, names: &[&str]) {
    let edges_before = column(db, "gql", "MATCH ()-[e]->() RETURN count(e)");
    let error = match run(db, language, query) {
        Ok(result) => panic!(
            "{language}: {query}: expected a syntax error, got {:?}",
            result.rows()
        ),
        Err(error) => error.to_string(),
    };
    for name in names {
        assert!(
            error.contains(name),
            "{language}: {query}: the error names `{name}`: {error}"
        );
    }
    assert_eq!(
        column(db, "gql", "MATCH ()-[e]->() RETURN count(e)"),
        edges_before,
        "{language}: {query} wrote an edge"
    );
}

#[test]
fn a_second_colon_in_a_relationship_type_is_a_syntax_error_in_cypher() {
    let db = code_graph();
    let advice = ["`Graph:CONTAINS`", "Graph|CONTAINS"];
    for query in [
        // One relationship, a variable-length one, from the start and in a
        // later relationship of the path
        "MATCH (r:Repository)-[:Graph:CONTAINS]->(d) RETURN d.name",
        "MATCH (r:Repository)-[:Graph:CONTAINS*]->(f:File) RETURN f.filePath",
        "MATCH (r:Repository)-[e:Graph:CONTAINS*1..3]->(f) RETURN f.filePath",
        "MATCH (r:Repository)-->(d)<-[:Graph:CONTAINS]-(r) RETURN d.name",
        "OPTIONAL MATCH (r:Repository)-[:Graph:CONTAINS]->(d) RETURN d.name",
        // Subqueries and predicates
        "MATCH (d:Directory) WHERE EXISTS { (d)-[:Graph:CONTAINS*]->(f:File) } RETURN d.name",
        "MATCH (d:Directory) WHERE (d)-[:Graph:CONTAINS]->(:File) RETURN d.name",
        "MATCH (d:Directory) RETURN [(d)-[:Graph:CONTAINS]->(f) | f.filePath] AS files",
        "MATCH (d:Directory) RETURN COUNT { (d)-[:Graph:CONTAINS]->() } AS files",
        "MATCH p = shortestPath((r:Repository)-[:Graph:CONTAINS*]->(f:File)) RETURN length(p)",
        // Writes
        "MATCH (d:Directory) CREATE (d)-[:Graph:CONTAINS]->(:File {filePath: 'Paris.py'})",
        "MATCH (d:Directory) MERGE (d)-[:Graph:CONTAINS]->(:File {filePath: 'Prague.py'})",
    ] {
        assert_refused(&db, "cypher", query, &advice);
    }
}

#[test]
fn a_second_colon_or_an_ampersand_in_an_edge_type_is_a_syntax_error_in_gql() {
    let db = code_graph();
    let advice = ["`Graph:CONTAINS`", "Graph|CONTAINS"];
    for query in [
        "MATCH (r:Repository)-[:Graph:CONTAINS]->(d) RETURN d.name",
        "MATCH (r:Repository)-[:Graph:CONTAINS*]->(f:File) RETURN f.filePath",
        "MATCH (r:Repository)-[:Graph:CONTAINS]->{1,3}(f:File) RETURN f.filePath",
        "MATCH (d:Directory)<-[:Graph:CONTAINS]-(r) RETURN r.repoName",
        "MATCH (d:Directory)~[:Graph:CONTAINS]~(r) RETURN r.repoName",
        "MATCH (d:Directory) WHERE EXISTS { MATCH (d)-[:Graph:CONTAINS]->(:File) } RETURN d.name",
        "MATCH (d:Directory) INSERT (d)-[:Graph:CONTAINS]->(:File {filePath: 'Paris.py'})",
        "MATCH (d:Directory) MERGE (d)-[:Graph:CONTAINS]->(f:File {filePath: 'Prague.py'})",
    ] {
        assert_refused(&db, "gql", query, &advice);
    }
    // An edge has one type: one that is both Graph and CONTAINS is none
    let advice = ["`Graph&CONTAINS`", "Graph|CONTAINS"];
    for query in [
        "MATCH (r:Repository)-[:Graph&CONTAINS]->(d) RETURN d.name",
        "MATCH (r:Repository)-[:Graph&CONTAINS*]->(f:File) RETURN f.filePath",
        "MATCH (d:Directory) INSERT (d)-[:Graph&CONTAINS]->(:File {filePath: 'Paris.py'})",
    ] {
        assert_refused(&db, "gql", query, &advice);
    }
}

#[test]
fn a_created_edge_gets_one_type_not_alternatives() {
    let db = code_graph();
    for (language, query) in [
        (
            "cypher",
            "MATCH (d:Directory) CREATE (d)-[:CONTAINS|LINKS]->(:File {filePath: 'Paris.py'})",
        ),
        (
            "cypher",
            "MATCH (d:Directory) MERGE (d)-[:CONTAINS|LINKS]->(:File {filePath: 'Prague.py'})",
        ),
        (
            "gql",
            "MATCH (d:Directory) INSERT (d)-[:CONTAINS|LINKS]->(:File {filePath: 'Paris.py'})",
        ),
        (
            "gql",
            "MATCH (d:Directory) MERGE (d)-[:CONTAINS|LINKS]->(f:File {filePath: 'Prague.py'})",
        ),
    ] {
        assert_refused(&db, language, query, &["one type", "CONTAINS|LINKS"]);
    }
}

#[test]
fn quoted_types_and_alternatives_match_what_they_name() {
    let db = code_graph();
    let both = vec!["Amsterdam/Alix.py".to_string()];
    for language in ["gql", "cypher"] {
        // The quoted name is one type, also in a variable-length pattern
        let query = "MATCH (r:Repository)-[:`Graph:CONTAINS`*]->(f:File) RETURN f.filePath";
        assert_eq!(column(&db, language, query), both, "{language}: {query}");
        // Alternatives match either type
        let query = "MATCH (r:Repository)-[:`Graph:CONTAINS`*]->(d)-[:`Graph:CONTAINS`|CONTAINS]->(f:File) \
             RETURN f.filePath";
        assert_eq!(
            column(&db, language, query),
            vec![
                "Amsterdam/Alix.py".to_string(),
                "Amsterdam/Gus.py".to_string()
            ],
            "{language}: {query}"
        );
    }
    // openCypher 9 also writes the alternatives `:A|:B`
    let query = "MATCH (r:Repository)-[:Graph|:`Graph:CONTAINS`]->(n) RETURN labels(n)[0]";
    assert_eq!(
        column(&db, "cypher", query),
        vec!["Directory".to_string(), "File".to_string()],
        "{query}"
    );
}
