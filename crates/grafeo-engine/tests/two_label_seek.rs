//! A match on several labels with a property map (`(n:Graph:Method {id: $id})`)
//! finds the nodes with every label: the rows are those of checking every
//! label of every node with the id, which each case compares against (the
//! nodes with the id and their `labels(n)`):
//!
//! - either label order, a label no node has, three labels with one twice;
//! - a WHERE label check over a property map, and over a range filter;
//! - the lookup per row of an UNWIND, and inside EXISTS and COUNT
//!   subqueries; a two-hop EXISTS on a node of two labels;
//! - a label added or removed in the transaction, and the rollback.
//!
//! With and without the index, in GQL and Cypher.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test two_label_seek
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

const LANGUAGES: [&str; 2] = ["gql", "cypher"];

/// Graph:Method `m1` to `m40` and Graph:File `f1` to `f40` (id 1 to 40); a
/// Method without Graph (`loose`, id 3) and a Graph:Method:Test (`t3`, id
/// 3). CALLS: m1 to m2 to m3, m5 to m6 to f6. Ref nodes with `ref` 3, 7 and
/// 50 (no node has id 50).
fn database(index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    if index {
        db.create_property_index("id").unwrap();
    }
    db.execute("UNWIND range(1, 40) AS i INSERT (:Graph:Method {id: i, name: 'm' + toString(i)})")
        .unwrap();
    db.execute("UNWIND range(1, 40) AS i INSERT (:Graph:File {id: i, name: 'f' + toString(i)})")
        .unwrap();
    db.execute(
        "INSERT (:Method {id: 3, name: 'loose'}), (:Graph:Method:Test {id: 3, name: 't3'}), \
         (:Ref {ref: 3}), (:Ref {ref: 7}), (:Ref {ref: 50})",
    )
    .unwrap();
    for (from, to) in [("m1", "m2"), ("m2", "m3"), ("m5", "m6"), ("m6", "f6")] {
        db.execute(&format!(
            "MATCH (a {{name: '{from}'}}), (b {{name: '{to}'}}) INSERT (a)-[:CALLS]->(b)"
        ))
        .unwrap();
    }
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<Vec<Value>> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .to_vec()
}

/// The first column of each row as text, sorted.
fn names(db: &GrafeoDB, language: &str, query: &str) -> Vec<String> {
    let mut names: Vec<String> = rows(db, language, query)
        .into_iter()
        .map(|row| match &row[0] {
            Value::String(name) => name.to_string(),
            other => format!("{other:?}"),
        })
        .collect();
    names.sort();
    names
}

/// The names of the nodes with `id` that have every one of `labels`, found
/// by reading each such node's labels.
fn having(db: &GrafeoDB, id: i64, labels: &[&str]) -> Vec<String> {
    let mut found: Vec<String> = rows(
        db,
        "gql",
        &format!("MATCH (n) WHERE n.id = {id} RETURN n.name, labels(n)"),
    )
    .into_iter()
    .filter(|row| {
        let Value::List(has) = &row[1] else {
            panic!("{:?}", row[1]);
        };
        labels
            .iter()
            .all(|label| has.iter().any(|l| *l == Value::from(*label)))
    })
    .map(|row| match &row[0] {
        Value::String(name) => name.to_string(),
        other => panic!("{other:?}"),
    })
    .collect();
    found.sort();
    found
}

#[test]
fn either_label_order_finds_the_nodes_with_both() {
    for index in [false, true] {
        let db = database(index);
        for language in LANGUAGES {
            for id in [1, 3, 40, 41] {
                let want = having(&db, id, &["Graph", "Method"]);
                for pattern in ["n:Graph:Method", "n:Method:Graph"] {
                    assert_eq!(
                        names(
                            &db,
                            language,
                            &format!("MATCH ({pattern} {{id: {id}}}) RETURN n.name")
                        ),
                        want,
                        "{language} {pattern} id {id}, index {index}"
                    );
                }
            }
            assert_eq!(
                having(&db, 3, &["Graph", "Method"]),
                ["m3", "t3"],
                "the oracle itself"
            );
        }
    }
}

#[test]
fn a_label_no_node_has_and_a_label_twice() {
    for index in [false, true] {
        let db = database(index);
        for language in LANGUAGES {
            for pattern in ["n:Graph:Nope", "n:Nope:Graph", "n:Method:File"] {
                assert_eq!(
                    names(
                        &db,
                        language,
                        &format!("MATCH ({pattern} {{id: 3}}) RETURN n.name")
                    ),
                    Vec::<String>::new(),
                    "{language} {pattern}, index {index}"
                );
            }
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Graph:Method:Graph {id: 3}) RETURN n.name"
                ),
                ["m3", "t3"],
                "{language}, index {index}"
            );
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Test:Graph:Method {id: 3}) RETURN n.name"
                ),
                ["t3"],
                "{language}, index {index}"
            );
        }
    }
}

#[test]
fn a_where_label_check_over_a_property_map_or_a_range() {
    for index in [false, true] {
        let db = database(index);
        for language in LANGUAGES {
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Graph {id: 3}) WHERE n:Method RETURN n.name"
                ),
                having(&db, 3, &["Graph", "Method"]),
                "{language}, index {index}"
            );
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Graph {id: 3}) WHERE n:Method AND n:Test RETURN n.name"
                ),
                ["t3"],
                "{language}, index {index}"
            );
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Graph:Method) WHERE n.id > 37 RETURN n.name"
                ),
                ["m38", "m39", "m40"],
                "{language}, index {index}"
            );
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Graph) WHERE n.id > 37 AND n:Method RETURN n.name"
                ),
                ["m38", "m39", "m40"],
                "{language}, index {index}"
            );
        }
    }
}

/// The lookup per row of an UNWIND: a value no node has gives no row.
#[test]
fn the_lookup_per_row_of_an_unwind() {
    for index in [false, true] {
        let db = database(index);
        for language in LANGUAGES {
            let got = rows(
                &db,
                language,
                "UNWIND [40, 3, 99, 1] AS x MATCH (n:Graph:Method {id: x}) \
                 RETURN x, n.name ORDER BY x, n.name",
            );
            let want: Vec<Vec<Value>> = [(1, "m1"), (3, "m3"), (3, "t3"), (40, "m40")]
                .into_iter()
                .map(|(x, name)| vec![Value::Int64(x), Value::from(name)])
                .collect();
            assert_eq!(got, want, "{language}, index {index}");
        }
    }
}

/// The lookup inside EXISTS and COUNT, correlated by the outer row's
/// `ref`; and a two-hop EXISTS from a node of two labels, against COUNT.
#[test]
fn inside_subqueries() {
    for index in [false, true] {
        let db = database(index);
        for language in LANGUAGES {
            assert_eq!(
                rows(
                    &db,
                    language,
                    "MATCH (r:Ref) WHERE EXISTS { MATCH (n:Graph:Method {id: r.ref}) } \
                     RETURN r.ref ORDER BY r.ref"
                ),
                [[Value::Int64(3)], [Value::Int64(7)]],
                "{language}, index {index}"
            );
            assert_eq!(
                rows(
                    &db,
                    language,
                    "MATCH (r:Ref) RETURN r.ref, COUNT { MATCH (n:Graph:Method {id: r.ref}) } \
                     AS c ORDER BY r.ref"
                ),
                [
                    [Value::Int64(3), Value::Int64(2)],
                    [Value::Int64(7), Value::Int64(1)],
                    [Value::Int64(50), Value::Int64(0)]
                ],
                "{language}, index {index}"
            );
            let hops = "MATCH (n)-[:CALLS]->(:Method)-[:CALLS]->()";
            let exists = names(
                &db,
                language,
                &format!("MATCH (n:Graph:Method) WHERE EXISTS {{ {hops} }} RETURN n.name"),
            );
            let counted = names(
                &db,
                language,
                &format!("MATCH (n:Graph:Method) WHERE COUNT {{ {hops} }} > 0 RETURN n.name"),
            );
            assert_eq!(exists, ["m1", "m5"], "{language}, index {index}");
            assert_eq!(exists, counted, "{language}, index {index}");
        }
    }
}

/// Inside a transaction: f3 gains Method, m3 loses Graph; the lookup sees
/// both, and after the rollback neither.
#[test]
fn a_label_added_or_removed_in_the_transaction() {
    for index in [false, true] {
        let db = database(index);
        let mut session = db.session();
        session.begin_transaction().unwrap();
        session
            .execute("MATCH (n {name: 'f3'}) SET n:Method")
            .unwrap();
        session
            .execute("MATCH (n {name: 'm3'}) REMOVE n:Graph")
            .unwrap();
        for language in LANGUAGES {
            let mut inside: Vec<String> = session
                .execute_language(
                    "MATCH (n:Graph:Method {id: 3}) RETURN n.name",
                    language,
                    None,
                )
                .unwrap()
                .rows()
                .iter()
                .map(|row| match &row[0] {
                    Value::String(name) => name.to_string(),
                    other => panic!("{other:?}"),
                })
                .collect();
            inside.sort();
            assert_eq!(inside, ["f3", "t3"], "{language}, index {index}");
        }
        session.rollback().unwrap();
        for language in LANGUAGES {
            assert_eq!(
                names(
                    &db,
                    language,
                    "MATCH (n:Graph:Method {id: 3}) RETURN n.name"
                ),
                ["m3", "t3"],
                "{language}, index {index}"
            );
        }
    }
}
