//! MERGE of a pattern with relationships, in GQL and Cypher: it matches the
//! whole pattern, or when nothing matches creates all of it, as openCypher
//! defines MERGE. The variables bound before are reused, never created, and
//! every other node and relationship is matched or created with the rest.
//!
//! Before, a pattern of more than one relationship was refused, an end node
//! with a label or a property was merged on its own first (so a node the
//! pattern does not connect was reused), and an end node with neither was
//! refused.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test merge_paths
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Alix, Gus and Vincent, and Alix KNOWS Gus.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix'}), (gus:Person {name: 'Gus'}), \
         (:Person {name: 'Vincent'}), (alix)-[:KNOWS]->(gus)",
    )
    .unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|e| panic!("{language} `{query}`: {e}"))
        .rows()
        .to_vec()
}

/// The one value a GQL `query` returns.
fn one(db: &GrafeoDB, query: &str) -> Value {
    let rows = db.execute(query).unwrap().rows().to_vec();
    assert_eq!(rows.len(), 1, "one row from {query}: {rows:?}");
    rows[0][0].clone()
}

/// The number of nodes and relationships.
fn size(db: &GrafeoDB) -> (Value, Value) {
    (
        one(db, "MATCH (n) RETURN count(n)"),
        one(db, "MATCH ()-[r]->() RETURN count(r)"),
    )
}

fn counts(nodes: i64, relationships: i64) -> (Value, Value) {
    (Value::Int64(nodes), Value::Int64(relationships))
}

/// `MERGE (a)-[:S]->(b)-[:T]->(:N {id: 3})` merged its S relationship and
/// dropped the rest, then was refused. It is created whole, then found whole.
#[test]
fn a_longer_pattern_is_created_then_matched() {
    for language in LANGUAGES {
        let db = people();
        let merge = "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) \
                     MERGE (a)-[:S]->(b)-[:T]->(n:N {id: 3}) RETURN n.id AS id";
        assert_eq!(
            rows(&db, language, merge),
            [[Value::Int64(3)]],
            "{language}"
        );
        assert_eq!(size(&db), counts(4, 3), "{language}: N, S and T created");
        assert_eq!(
            rows(&db, language, merge),
            [[Value::Int64(3)]],
            "{language}"
        );
        assert_eq!(size(&db), counts(4, 3), "{language}: the second finds them");
        assert_eq!(
            one(
                &db,
                "MATCH (:Person {name: 'Alix'})-[:S]->(:Person {name: 'Gus'})-[:T]->(n:N) \
                 RETURN n.id"
            ),
            Value::Int64(3),
            "{language}"
        );
    }
}

/// The pattern is matched as a whole: a Gus that Vincent does not know does
/// not match `(v)-[:KNOWS]->(:Person {name: 'Gus'})`, so the MERGE creates the
/// relationship with a new Gus (openCypher), and finds both after that.
#[test]
fn an_unconnected_end_node_is_not_reused() {
    for language in LANGUAGES {
        let db = people();
        let merge =
            "MATCH (v:Person {name: 'Vincent'}) MERGE (v)-[:KNOWS]->(:Person {name: 'Gus'})";
        rows(&db, language, merge);
        rows(&db, language, merge);
        assert_eq!(
            one(&db, "MATCH (p:Person {name: 'Gus'}) RETURN count(p)"),
            Value::Int64(2),
            "{language}: the first MERGE created a Gus with the relationship"
        );
        assert_eq!(
            one(
                &db,
                "MATCH (:Person {name: 'Vincent'})-[r:KNOWS]->(:Person {name: 'Gus'}) \
                 RETURN count(r)"
            ),
            Value::Int64(1),
            "{language}: the second found what the first created"
        );
    }
}

/// The start node may be anonymous as well; the node the MERGE creates takes
/// the relationship.
#[test]
fn an_anonymous_start_node_is_created_with_the_relationship() {
    for language in LANGUAGES {
        let db = people();
        let merge = "MATCH (g:Person {name: 'Gus'}) MERGE (:City {name: 'Prague'})-[:HOME_OF]->(g)";
        rows(&db, language, merge);
        rows(&db, language, merge);
        assert_eq!(
            rows(
                &db,
                "gql",
                "MATCH (c:City)-[:HOME_OF]->(p:Person) RETURN c.name, p.name"
            ),
            [[Value::from("Prague"), Value::from("Gus")]],
            "{language}"
        );
    }
}

/// An end node with neither a label nor a property is any node: the MERGE
/// matches a relationship to one, or creates an empty node with it.
#[test]
fn an_anonymous_end_node_matches_any_node() {
    for language in LANGUAGES {
        let db = people();
        let merge = "MATCH (a:Person {name: 'Alix'}) MERGE (a)-[:KNOWS]->(x) RETURN x.name AS name";
        assert_eq!(
            rows(&db, language, merge),
            [[Value::from("Gus")]],
            "{language}: Alix knows Gus"
        );
        let merge = "MATCH (v:Person {name: 'Vincent'}) MERGE (v)-[:KNOWS]->(x) \
                     RETURN labels(x) AS l";
        assert_eq!(
            rows(&db, language, merge),
            [[Value::List(Vec::new().into())]],
            "{language}"
        );
        assert_eq!(size(&db), counts(4, 2), "{language}: an empty node created");
    }
}

/// Every match is a row, as for MATCH.
#[test]
fn every_match_is_a_row() {
    for language in LANGUAGES {
        let db = people();
        db.execute(
            "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(g)",
        )
        .unwrap();
        assert_eq!(
            rows(
                &db,
                language,
                "MATCH (a:Person {name: 'Alix'}), (g:Person {name: 'Gus'}) \
                 MERGE (a)-[r:KNOWS]->(g) RETURN count(r) AS c"
            ),
            [[Value::Int64(2)]],
            "{language}"
        );
        assert_eq!(size(&db), counts(3, 2), "{language}: nothing created");
    }
}

/// `ON CREATE SET` runs on what the MERGE created and `ON MATCH SET` on what
/// it found, and either may set any element of the pattern.
#[test]
fn on_create_and_on_match_set_any_element() {
    for language in LANGUAGES {
        let db = people();
        let merge = "MATCH (a:Person {name: 'Alix'}) \
                     MERGE (a)-[r:LIVES_IN]->(c:City {name: 'Prague'}) \
                     ON CREATE SET r.since = 2020, c.created = true \
                     ON MATCH SET r.seen = true, a.visits = 1";
        let read = "MATCH (a)-[r:LIVES_IN]->(c:City) RETURN r.since, r.seen, c.created, a.visits";
        rows(&db, language, merge);
        assert_eq!(
            rows(&db, "gql", read),
            [[
                Value::Int64(2020),
                Value::Null,
                Value::Bool(true),
                Value::Null
            ]],
            "{language}: created"
        );
        rows(&db, language, merge);
        assert_eq!(
            rows(&db, "gql", read),
            [[
                Value::Int64(2020),
                Value::Bool(true),
                Value::Bool(true),
                Value::Int64(1)
            ]],
            "{language}: matched"
        );
    }
}

/// A row sees what the MERGE created for an earlier row of the same
/// statement: two rows with `x` 1 make one pattern.
#[test]
fn a_row_sees_what_earlier_rows_created() {
    for language in LANGUAGES {
        let db = GrafeoDB::new_in_memory();
        rows(
            &db,
            language,
            "UNWIND [1, 1, 2] AS x MERGE (:A {x: x})-[:R]->(:B {x: x})",
        );
        assert_eq!(size(&db), counts(4, 2), "{language}");
    }
}

/// An undirected relationship matches either way and is created from left to
/// right.
#[test]
fn an_undirected_relationship_matches_either_way() {
    for language in LANGUAGES {
        let db = people();
        rows(
            &db,
            language,
            "MATCH (g:Person {name: 'Gus'}), (a:Person {name: 'Alix'}) MERGE (g)-[:KNOWS]-(a)",
        );
        assert_eq!(size(&db), counts(3, 1), "{language}: Alix knows Gus");
        rows(
            &db,
            language,
            "MATCH (g:Person {name: 'Gus'}), (v:Person {name: 'Vincent'}) MERGE (g)-[:KNOWS]-(v)",
        );
        assert_eq!(
            one(
                &db,
                "MATCH (:Person {name: 'Gus'})-[r:KNOWS]->(:Person {name: 'Vincent'}) \
                 RETURN count(r)"
            ),
            Value::Int64(1),
            "{language}: created from Gus to Vincent"
        );
    }
}

/// A Cypher path variable is bound whether the path was matched or created.
#[test]
fn a_path_variable_is_bound_either_way() {
    let db = GrafeoDB::new_in_memory();
    let merge = "MERGE p = (:P {k: 1})-[:R]->(:Q {k: 2}) RETURN length(p) AS l";
    assert_eq!(rows(&db, "cypher", merge), [[Value::Int64(1)]], "created");
    assert_eq!(rows(&db, "cypher", merge), [[Value::Int64(1)]], "matched");
    assert_eq!(size(&db), counts(2, 1));
}

/// What MERGE cannot do is an error, before it writes anything.
#[test]
fn what_a_merge_cannot_do_is_an_error() {
    for (query, message) in [
        (
            "MATCH (a:Person {name: 'Alix'}) MERGE (a)-[]->(:City {name: 'Prague'})",
            "type",
        ),
        (
            "MATCH (a:Person {name: 'Alix'}) MERGE (a)-[:A|B]->(:City {name: 'Prague'})",
            "one type",
        ),
        (
            "MATCH (a:Person {name: 'Alix'}) MERGE (a)-[:R*1..2]->(:City {name: 'Prague'})",
            "variable-length",
        ),
    ] {
        for language in LANGUAGES {
            let db = people();
            let before = size(&db);
            let error = db
                .execute_language(query, language, None)
                .expect_err(query)
                .to_string();
            assert!(error.contains(message), "{language} `{query}`: {error}");
            assert_eq!(size(&db), before, "{language} `{query}` wrote nothing");
        }
    }
}

/// A Cypher node MERGE whose `SET n = {...}` replaces the node's properties
/// does, on create and on match (it merged the map into them), and one may
/// set a variable bound before.
#[test]
fn a_node_merge_sets_as_a_set_clause_does() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:X {id: 1, old: 1}), (:Y {name: 'Gus'})")
        .unwrap();
    assert_eq!(
        rows(
            &db,
            "cypher",
            "MERGE (n:X {id: 1}) ON MATCH SET n = {id: 1, fresh: 2} RETURN n.old, n.fresh"
        ),
        [[Value::Null, Value::Int64(2)]],
        "matched: replaced"
    );
    assert_eq!(
        rows(
            &db,
            "cypher",
            "MERGE (n:Z {id: 3}) ON CREATE SET n = {fresh: 4} RETURN n.id, n.fresh"
        ),
        [[Value::Null, Value::Int64(4)]],
        "created: replaced, the merged key too"
    );
    assert_eq!(
        rows(
            &db,
            "cypher",
            "MATCH (y:Y) MERGE (n:X {id: 1}) ON MATCH SET y.seen = true RETURN y.seen"
        ),
        [[Value::Bool(true)]],
        "another variable"
    );
}

/// A MERGE from or to a node an OPTIONAL MATCH did not find is an error that
/// says so, and writes nothing.
#[test]
fn a_merge_from_a_null_node_is_an_error() {
    for language in LANGUAGES {
        for query in [
            "OPTIONAL MATCH (n:Missing) MERGE (n)-[:R]->(m:Target {name: 'Alix'})",
            "MATCH (a:Person {name: 'Alix'}) OPTIONAL MATCH (b:Missing) MERGE (a)-[:R]->(b)",
        ] {
            let db = people();
            let before = size(&db);
            let error = db
                .execute_language(query, language, None)
                .expect_err(query)
                .to_string();
            assert!(
                error.contains("null") && error.contains("created relationship"),
                "{language} `{query}`: {error}"
            );
            assert_eq!(size(&db), before, "{language} `{query}` wrote nothing");
        }
    }
}

/// A pattern from a node an OPTIONAL MATCH did not find matches nothing (it
/// was an internal error).
#[test]
fn a_pattern_from_a_null_node_matches_nothing() {
    for language in LANGUAGES {
        let db = people();
        assert_eq!(
            rows(
                &db,
                language,
                "OPTIONAL MATCH (n:Missing) MATCH (n)-[:KNOWS]->(m) RETURN count(m) AS c"
            ),
            [[Value::Int64(0)]],
            "{language}"
        );
    }
}
