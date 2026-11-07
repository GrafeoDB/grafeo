//! A null inside a list or a record (ISO/IEC 39075 19.3 General Rule 2 b ii and iii): `=` is True when
//! every pair is True, False when the lengths differ or some pair is False, and Unknown otherwise, so
//! `[null] = [null]` and `[1] = [null]` are Unknown and `[1, 2] = [null, 3]` is False, at any depth and in
//! GQL and Cypher. A join, `IN` and an index lookup take a value with a null at any depth as meeting
//! nothing. DISTINCT, GROUP BY, set operations and UNIQUE keep treating nulls as not distinct.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main null_inside_values::
//! ```

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::PlannerRules;
use grafeo_engine::session::Session;

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

fn run(session: &Session, language: Language, query: &str) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute(query),
        Language::Cypher => session.execute_cypher(query),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?}: {query}: {error}"))
        .rows()
        .to_vec()
}

fn one(session: &Session, language: Language, query: &str) -> Value {
    run(session, language, query)[0][0].clone()
}

const NULL: Value = Value::Null;

/// Left, right, and what `=` gives; `<>` gives its negation, Unknown staying Unknown.
fn pairs() -> Vec<(&'static str, &'static str, Option<bool>)> {
    vec![
        ("[null]", "[null]", None),
        ("[1]", "[null]", None),
        ("[null]", "[1]", None),
        ("[1, null]", "[1, null]", None),
        ("[1, null]", "[1, 2]", None),
        ("[1, 2]", "[null, 3]", Some(false)),
        ("[null, 2]", "[1, 3]", Some(false)),
        ("[1]", "[1, null]", Some(false)),
        ("[]", "[]", Some(true)),
        ("[1, 2]", "[1, 2]", Some(true)),
        ("{a: 1}", "{a: null}", None),
        ("{a: null}", "{a: null}", None),
        ("{a: 1, b: null}", "{a: 2, b: null}", Some(false)),
        ("{a: 1, b: 2}", "{a: 1, b: 2}", Some(true)),
        // At any depth.
        ("[[null]]", "[[null]]", None),
        ("[[1, null]]", "[[1, 2]]", None),
        ("[[1, null]]", "[[2, null]]", Some(false)),
        ("{a: [null]}", "{a: [null]}", None),
        ("[{a: null}]", "[{a: 1}]", None),
        ("[{a: 1}]", "[{a: 2}]", Some(false)),
    ]
}

fn truth(value: Option<bool>) -> Value {
    value.map_or(NULL, Value::Bool)
}

#[test]
fn equality_of_lists_and_records_with_a_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for (left, right, equal) in pairs() {
            for (op, want) in [("=", truth(equal)), ("<>", truth(equal.map(|b| !b)))] {
                let query = format!("RETURN {left} {op} {right} AS r");
                let found = one(&session, language, &query);
                if found != want {
                    wrong.push(format!(
                        "{language:?}: {left} {op} {right}: {found:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Unknown filters out the row, and so does its negation: `NOT ([1] = [null])` keeps nothing.
#[test]
fn unknown_and_its_negation_both_drop_the_row() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:L {k: [1], name: 'Alix'}), (:L {k: [2], name: 'Gus'})")
        .unwrap();
    for language in [Language::Gql, Language::Cypher] {
        for (query, rows) in [
            ("MATCH (n:L) WHERE n.k = [null] RETURN n.name AS n", 0),
            ("MATCH (n:L) WHERE NOT (n.k = [null]) RETURN n.name AS n", 0),
            ("MATCH (n:L) WHERE n.k <> [null] RETURN n.name AS n", 0),
            ("MATCH (n:L) WHERE NOT ([1] = [null]) RETURN n.name AS n", 0),
            (
                "MATCH (n:L) WHERE ([1] = [null]) IS NULL RETURN n.name AS n",
                2,
            ),
            ("MATCH (n:L) WHERE n.k <> [1, null] RETURN n.name AS n", 2),
        ] {
            assert_eq!(
                run(&session, language, query).len(),
                rows,
                "{language:?}: {query}"
            );
        }
    }
}

/// A key with a null at any depth meets nothing in `IN`, a join, a lookup with and without an index.
#[test]
fn a_null_inside_a_key_meets_nothing() {
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session
            .execute(
                "INSERT (:A {k: [null], name: 'Alix'}), (:B {k: [null], name: 'Gus'}), \
                 (:A {k: [1, null], name: 'Mia'}), (:B {k: [1, null], name: 'Vincent'}), \
                 (:A {k: [1], name: 'Jules'}), (:B {k: [1], name: 'Hans'})",
            )
            .unwrap();
        if indexed {
            session
                .execute("CREATE INDEX on_a FOR (n:A) ON (n.k)")
                .unwrap();
        }
        let mut plain = db.session();
        plain.set_planner_rules(PlannerRules::plain());
        for language in [Language::Gql, Language::Cypher] {
            for (query, want) in [
                ("RETURN [null] IN [[null]] AS r", vec![vec![NULL]]),
                ("RETURN [null] IN [[null], [2]] AS r", vec![vec![NULL]]),
                (
                    "RETURN [1] IN [[null], [1]] AS r",
                    vec![vec![Value::Bool(true)]],
                ),
                ("RETURN [[null]] IN [[[null]]] AS r", vec![vec![NULL]]),
            ] {
                assert_eq!(
                    run(&session, language, query),
                    want,
                    "index {indexed}, {language:?}: {query}"
                );
            }
            for (query, names) in [
                ("MATCH (n:A {k: [null]}) RETURN n.name AS n", &[][..]),
                ("MATCH (n:A {k: [1, null]}) RETURN n.name AS n", &[]),
                ("MATCH (n:A) WHERE n.k = [null] RETURN n.name AS n", &[]),
                (
                    "MATCH (n:A) WHERE n.k IN [[null], [1]] RETURN n.name AS n",
                    &["Jules"],
                ),
                (
                    "MATCH (a:A), (b:B) WHERE a.k = b.k RETURN a.name AS n ORDER BY n",
                    &["Jules"],
                ),
                (
                    "MATCH (a:A) MATCH (b:B {k: a.k}) RETURN a.name AS n ORDER BY n",
                    &["Jules"],
                ),
                (
                    "MATCH (a:A) OPTIONAL MATCH (b:B) WHERE b.k = a.k RETURN a.name AS a, b.name AS b ORDER BY a",
                    &["Alix", "Jules", "Mia"],
                ),
            ] {
                for (route, reader) in [("usual", &session), ("plain", &plain)] {
                    let found: Vec<String> = run(reader, language, query)
                        .iter()
                        .map(|row| row[0].as_str().unwrap_or("null").to_string())
                        .collect();
                    assert_eq!(
                        found, names,
                        "index {indexed}, {route}, {language:?}: {query}"
                    );
                }
            }
            let optional = run(
                &session,
                language,
                "MATCH (a:A) OPTIONAL MATCH (b:B) WHERE b.k = a.k RETURN a.name AS a, b.name AS b ORDER BY a",
            );
            let joined: Vec<bool> = optional.iter().map(|row| row[1] != NULL).collect();
            assert_eq!(
                joined,
                [false, true, false],
                "index {indexed}, {language:?}: only Jules meets Hans"
            );
        }
    }
}

/// DISTINCT, GROUP BY and set operations treat two nulls as not distinct, inside a list too; so does a
/// UNIQUE rule.
#[test]
fn grouping_keeps_nulls_together() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for language in [Language::Gql, Language::Cypher] {
        for (query, rows) in [
            ("UNWIND [[null], [null]] AS x RETURN DISTINCT x", 1),
            (
                "UNWIND [[null], [null], [1, null], [1, null]] AS x RETURN DISTINCT x",
                2,
            ),
            ("UNWIND [[null], [null]] AS x RETURN x, count(*) AS c", 1),
            ("UNWIND [{a: null}, {a: null}] AS x RETURN DISTINCT x", 1),
            ("UNWIND [[[null]], [[null]]] AS x RETURN DISTINCT x", 1),
            ("RETURN [null] AS x UNION RETURN [null] AS x", 1),
            (
                "UNWIND [[null], [null]] AS x RETURN count(DISTINCT x) AS c",
                1,
            ),
        ] {
            assert_eq!(
                run(&session, language, query).len(),
                rows,
                "{language:?}: {query}"
            );
        }
        let counted = one(
            &session,
            language,
            "UNWIND [[null], [null]] AS x RETURN count(DISTINCT x) AS c",
        );
        assert_eq!(counted, Value::Int64(1), "{language:?}");
    }
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE CONSTRAINT one_k FOR (n:U) ON (n.k) UNIQUE")
        .unwrap();
    session.execute("INSERT (:U {k: [1, null]})").unwrap();
    assert!(
        session.execute("INSERT (:U {k: [1, null]})").is_err(),
        "a second [1, null] is a duplicate"
    );
    session.execute("INSERT (:U {k: [2, null]})").unwrap();
}
