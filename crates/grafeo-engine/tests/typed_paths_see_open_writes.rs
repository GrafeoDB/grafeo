//! A typed variable-length pattern, a chain of typed expands and a typed
//! shortest path see the edges of their own transaction that it has not
//! committed yet: those an earlier statement of an open transaction created,
//! and those an earlier clause of the same statement created, as a single
//! typed expand and an untyped pattern do. They read the type of an edge as
//! their transaction sees it (the committed type misses the edges it
//! created), and a shortest path sees only the edges its transaction may see:
//! not those another transaction created and has not committed, nor those its
//! own transaction deleted. In GQL and Cypher, with and without factorized
//! execution. These queries write, so they are no cases of the differential
//! test corpus (which only reads).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test typed_paths_see_open_writes
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB, Session};

#[derive(Debug, Clone, Copy)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?} `{query}` failed: {error}"))
        .rows()
        .to_vec()
}

fn int(value: i64) -> Value {
    Value::Int64(value)
}

/// Writes `(h:Hub)-[:R]->(:Q {i: 3})-[:S]->(t:T {i: 19})` and a shortcut
/// `(h)-[:X]->(t)` of another type, binding `h` and `t`.
fn write(language: Language) -> &'static str {
    match language {
        Language::Gql => "INSERT (h:Hub)-[:R]->(:Q {i: 3})-[:S]->(t:T {i: 19}), (h)-[:X]->(t)",
        Language::Cypher => "CREATE (h:Hub)-[:R]->(:Q {i: 3})-[:S]->(t:T {i: 19}), (h)-[:X]->(t)",
    }
}

/// A pattern from `h` (and to `t`, for the shortest paths) of the written
/// graph, with the rows it returns: the typed patterns leave the shortcut out.
struct Case {
    gql: &'static str,
    cypher: &'static str,
    expected: Vec<Vec<Value>>,
}

impl Case {
    fn pattern(&self, language: Language) -> &'static str {
        match language {
            Language::Gql => self.gql,
            Language::Cypher => self.cypher,
        }
    }
}

fn cases() -> Vec<Case> {
    vec![
        // A variable-length expand of one type: only `q`.
        Case {
            gql: "MATCH (h)-[:R]->{1,2}(n) RETURN count(n) AS found",
            cypher: "MATCH (h)-[:R*1..2]->(n) RETURN count(n) AS found",
            expected: vec![vec![int(1)]],
        },
        // Of two types: `q` and `t`, not `t` again through the shortcut.
        Case {
            gql: "MATCH (h)-[:R|S]->{1,2}(n) RETURN count(n) AS found",
            cypher: "MATCH (h)-[:R|S*1..2]->(n) RETURN count(n) AS found",
            expected: vec![vec![int(2)]],
        },
        // A named path of a variable-length expand.
        Case {
            gql: "MATCH p = (h)-[:R|S]->{2}(n) RETURN length(p) AS hops, n.i AS i",
            cypher: "MATCH p = (h)-[:R|S*2]->(n) RETURN length(p) AS hops, n.i AS i",
            expected: vec![vec![int(2), int(19)]],
        },
        // A chain of typed expands (a factorized chain when that is on).
        Case {
            gql: "MATCH (h)-[:R]->(q)-[:S]->(n) RETURN q.i AS q, n.i AS n",
            cypher: "MATCH (h)-[:R]->(q)-[:S]->(n) RETURN q.i AS q, n.i AS n",
            expected: vec![vec![int(3), int(19)]],
        },
        // The same chain, counted (a factorized aggregate when that is on).
        Case {
            gql: "MATCH (h)-[:R]->(q)-[:S]->(n) RETURN count(*) AS found",
            cypher: "MATCH (h)-[:R]->(q)-[:S]->(n) RETURN count(*) AS found",
            expected: vec![vec![int(1)]],
        },
        // A typed shortest path takes two edges, not the shortcut.
        Case {
            gql: "MATCH p = ANY SHORTEST (h)-[:R|S]->+(t) RETURN length(p) AS hops",
            cypher: "MATCH p = shortestPath((h)-[:R|S*]->(t)) RETURN length(p) AS hops",
            expected: vec![vec![int(2)]],
        },
        Case {
            gql: "MATCH p = ALL SHORTEST (h)-[:R|S]->+(t) RETURN length(p) AS hops",
            cypher: "MATCH p = allShortestPaths((h)-[:R|S*]->(t)) RETURN length(p) AS hops",
            expected: vec![vec![int(2)]],
        },
        // An untyped one takes the shortcut.
        Case {
            gql: "MATCH p = ANY SHORTEST (h)-[]->+(t) RETURN length(p) AS hops",
            cypher: "MATCH p = shortestPath((h)-[*]->(t)) RETURN length(p) AS hops",
            expected: vec![vec![int(1)]],
        },
    ]
}

/// A new database, with or without factorized execution.
fn database(factorized: bool) -> GrafeoDB {
    let config = if factorized {
        Config::in_memory()
    } else {
        Config::in_memory().without_factorized_execution()
    };
    GrafeoDB::with_config(config).unwrap()
}

/// In an open transaction, each pattern sees the graph an earlier statement
/// of it wrote, and still does after the commit.
#[test]
fn typed_patterns_see_the_edges_of_their_open_transaction() {
    for factorized in [true, false] {
        for language in LANGUAGES {
            let db = database(factorized);
            let mut session = db.session();
            session.begin_transaction().unwrap();
            run(&session, language, write(language));
            let read = |session: &Session, case: &Case| {
                let query = format!("MATCH (h:Hub), (t:T) {}", case.pattern(language));
                (run(session, language, &query), query)
            };
            for case in cases() {
                let (rows, query) = read(&session, &case);
                assert_eq!(
                    rows, case.expected,
                    "{language:?} `{query}` in the open transaction (factorized: {factorized})"
                );
            }
            session.commit().unwrap();
            for case in cases() {
                let (rows, query) = read(&session, &case);
                assert_eq!(
                    rows, case.expected,
                    "{language:?} `{query}` after the commit (factorized: {factorized})"
                );
            }
        }
    }
}

/// Within one statement, each pattern after the write sees what it wrote.
#[test]
fn typed_patterns_see_the_edges_their_statement_wrote() {
    for factorized in [true, false] {
        for language in LANGUAGES {
            for case in cases() {
                let db = database(factorized);
                let session = db.session();
                let query = format!("{} WITH h, t {}", write(language), case.pattern(language));
                assert_eq!(
                    run(&session, language, &query),
                    case.expected,
                    "{language:?} `{query}` (factorized: {factorized})"
                );
            }
        }
    }
}

/// The edge `(t)-[:R]->(h)` closes a cycle, so the shortest path from `t` to
/// `h` takes one edge while it is there. Another transaction's edge is not
/// there before that transaction commits, and an edge its own transaction
/// deleted is not there for it.
#[test]
fn a_shortest_path_sees_only_the_edges_its_transaction_may_see() {
    for language in LANGUAGES {
        let (any, all) = match language {
            Language::Gql => (
                "MATCH (t:T), (h:Hub) MATCH p = ANY SHORTEST (t)-[]->+(h) RETURN length(p) AS hops",
                "MATCH (t:T), (h:Hub) MATCH p = ALL SHORTEST (t)-[:R]->+(h) RETURN length(p) AS hops",
            ),
            Language::Cypher => (
                "MATCH (t:T), (h:Hub) MATCH p = shortestPath((t)-[*]->(h)) RETURN length(p) AS hops",
                "MATCH (t:T), (h:Hub) MATCH p = allShortestPaths((t)-[:R*]->(h)) RETURN length(p) AS hops",
            ),
        };
        let db = GrafeoDB::new_in_memory();
        let reader = db.session();
        run(&reader, language, write(language));

        let mut writer = db.session();
        writer.begin_transaction().unwrap();
        let insert = match language {
            Language::Gql => "INSERT",
            Language::Cypher => "CREATE",
        };
        run(
            &writer,
            language,
            &format!("MATCH (t:T), (h:Hub) {insert} (t)-[:R]->(h)"),
        );
        for query in [any, all] {
            assert_eq!(
                run(&writer, language, query),
                [[int(1)]],
                "{language:?} `{query}` in the transaction that created the edge"
            );
            assert!(
                run(&reader, language, query).is_empty(),
                "{language:?} `{query}` before the other transaction commits its edge"
            );
        }
        writer.commit().unwrap();
        for query in [any, all] {
            assert_eq!(
                run(&reader, language, query),
                [[int(1)]],
                "{language:?} `{query}` after the commit"
            );
        }

        let mut deleter = db.session();
        deleter.begin_transaction().unwrap();
        run(&deleter, language, "MATCH (:T)-[e:R]->(:Hub) DELETE e");
        for query in [any, all] {
            assert!(
                run(&deleter, language, query).is_empty(),
                "{language:?} `{query}` after its transaction deleted the edge"
            );
        }
        deleter.rollback().unwrap();
    }
}
