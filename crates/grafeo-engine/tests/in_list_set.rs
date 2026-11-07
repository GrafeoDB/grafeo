//! `x IN <a long constant list>` answers as the walk over the list does (key-lookup design, G).
//!
//! The walk is what a short list does, below any threshold, so the reference for each answer is the
//! same query with the list cut down to the item in question. Fillers are integers no needle equals.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main in_list_set::
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
use grafeo_engine::session::Session;

/// One literal of each kind and each spelling `=` treats specially, as in `equality_with_index`, and
/// the temporal kinds, which keep the walk as items and may meet a long list of scalars as needles.
const LITERALS: &[&str] = &[
    "42",
    "-42",
    "42.0",
    "42.5",
    "'42'",
    "'-42'",
    "'042'",
    "'+42'",
    "'42.0'",
    "'4.2e1'",
    "'abc'",
    "0.3",
    "'0.3'",
    "0",
    "0.0",
    "-0.0",
    "0.0e0",
    "-0.0e0",
    "'-0'",
    "1",
    "1.0",
    "1.0e0",
    "5",
    "'5'",
    "'05'",
    "NaN",
    "Infinity",
    "-Infinity",
    "'1e3'",
    "1000",
    "true",
    "false",
    "'true'",
    "9007199254740993",
    "9007199254740992",
    "9007199254740992.0",
    "9007199254740992.0e0",
    "'9007199254740993'",
    "'NaN'",
    "'inf'",
    "''",
];

/// Items that are not scalars, and a NULL: each keeps the walk or changes what a miss gives.
const OTHER_ITEMS: &[&str] = &[
    "null",
    "[42]",
    "[]",
    "{a: 42}",
    "date('2026-11-04')",
    "time('12:00:00')",
    "datetime('2026-11-04T12:00:00')",
    "duration('P1D')",
    "'2026-11-04'",
];

/// Needles beside the literals.
const OTHER_NEEDLES: &[&str] = &[
    "null",
    "[42]",
    "{a: 42}",
    "date('2026-11-04')",
    "datetime('2026-11-04T12:00:00')",
    "'2026-11-04'",
];

/// Around the guess of 16, and well over it: the constant may be measured again, and these
/// hold whatever it becomes up to the largest.
const LENGTHS: [usize; 9] = [2, 8, 15, 16, 17, 32, 64, 257, 1_100];

/// The list of `length` items with `item` at `position`, the others fillers no needle equals.
fn padded(item: &str, length: usize, position: usize) -> String {
    let items: Vec<String> = (0..length)
        .map(|at| {
            if at == position {
                item.to_string()
            } else {
                (1_000_000 + at).to_string()
            }
        })
        .collect();
    format!("[{}]", items.join(", "))
}

/// The answer of an expression, or that it raises, as the text the comparison reads.
fn answer(session: &Session, query: &str) -> String {
    match session.execute(query) {
        Ok(result) => format!("{:?}", result.rows()),
        Err(error) => format!(
            "raises {}",
            error.to_string().split(':').next().unwrap_or("")
        ),
    }
}

/// NaN as a value rather than a spelling the parser may not take, and the rest as written.
fn literal(text: &str) -> String {
    match text {
        "NaN" => "(0.0e0 / 0.0e0)".to_string(),
        "Infinity" => "(1.0e0 / 0.0e0)".to_string(),
        "-Infinity" => "(-1.0e0 / 0.0e0)".to_string(),
        other => other.to_string(),
    }
}

/// `x IN [item, fillers...]`, at each length and with the item first and last, answers as
/// `x IN [item]`, and as the list's other forms below.
fn agrees(session: &Session, needle: &str, item: &str, lengths: &[usize]) {
    let (needle, item) = (literal(needle), literal(item));
    let walk = answer(session, &format!("RETURN {needle} IN [{item}] AS found"));
    for &length in lengths {
        for position in [0, length / 2, length - 1] {
            let list = padded(&item, length, position);
            let query = format!("RETURN {needle} IN {list} AS found");
            let got = answer(session, &query);
            assert_eq!(
                got, walk,
                "{needle} IN [{item}], {length} items, at {position}"
            );
        }
    }
}

/// Each pair, as needle and as item.
#[test]
fn every_pair_of_scalars_answers_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for needle in LITERALS {
        for item in LITERALS {
            agrees(&session, needle, item, &[16, 40]);
        }
    }
}

#[test]
fn lengths_around_the_threshold_answer_alike() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (needle, item) in [
        ("1", "1.0"),
        ("1.0", "1"),
        ("5", "'05'"),
        ("'05'", "5"),
        ("9007199254740993", "9007199254740992"),
        ("9007199254740993", "9007199254740992.0"),
        ("9007199254740992", "9007199254740992.0e0"),
        ("'abc'", "'abc'"),
        ("0.0e0", "-0.0e0"),
        ("NaN", "NaN"),
        ("true", "'true'"),
        ("42", "'4.2e1'"),
    ] {
        agrees(&session, needle, item, &LENGTHS);
    }
}

/// The exact answers the set must keep, not only that it agrees with today.
#[test]
fn the_answers_are_those_of_equals() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let with_fillers = |needle: &str, item: &str| -> Value {
        let list = padded(&literal(item), 64, 30);
        let query = format!("RETURN {} IN {list} AS found", literal(needle));
        session.execute(&query).unwrap().rows()[0][0].clone()
    };
    for (needle, item, wanted) in [
        ("1", "1.0", true),
        // A string equals no number, whatever it spells.
        ("5", "'05'", false),
        ("'05'", "5", false),
        ("5", "'5'", false),
        ("'5'", "5", false),
        ("42", "'4.2e1'", false),
        ("true", "'true'", false),
        ("'05'", "'05'", true),
        ("0.0e0", "-0.0e0", true),
        ("NaN", "NaN", false),
        ("9007199254740993", "9007199254740992", false),
        ("9007199254740993", "9007199254740992.0", false),
        ("'abc'", "'abd'", false),
    ] {
        assert_eq!(
            with_fillers(needle, item),
            Value::Bool(wanted),
            "{needle} IN [..{item}..]"
        );
    }
}

/// A non-scalar or NULL item keeps the walk, whichever needle meets it.
#[test]
fn items_that_are_not_scalars_answer_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for needle in LITERALS.iter().chain(OTHER_NEEDLES) {
        for item in OTHER_ITEMS {
            agrees(&session, needle, item, &[40]);
        }
    }
}

/// A needle of another kind against scalars: a temporal, a list, a map.
#[test]
fn needles_that_are_not_scalars_answer_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for needle in OTHER_NEEDLES {
        for item in LITERALS {
            agrees(&session, needle, item, &[40]);
        }
    }
}

// ---------------------------------------------------------------------------
// NULL
// ---------------------------------------------------------------------------

/// A miss on a list holding NULL gives null, not false, and a NULL needle null: `NOT` and `IS NULL`
/// show it in `WHERE`, where false and null differ.
#[test]
fn a_list_holding_null_gives_null_on_a_miss() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(1, 5) AS i INSERT (:N {x: i})")
        .unwrap();
    session.execute("INSERT (:N {name: 'no x'})").unwrap();
    let names = |query: &str| -> usize { session.execute(query).unwrap().row_count() };
    for length in LENGTHS {
        // 1 and 2 are in the list, with a NULL at the end.
        let items: Vec<String> = (1..=length.max(2))
            .map(|at| {
                if at <= 2 {
                    at.to_string()
                } else {
                    (1_000_000 + at).to_string()
                }
            })
            .chain(["null".to_string()])
            .collect();
        let with_null = format!("[{}]", items.join(", "));
        assert_eq!(
            names(&format!("MATCH (n:N) WHERE n.x IN {with_null} RETURN n")),
            2,
            "{length}"
        );
        // 3, 4 and 5 miss: their IN is null, so `NOT` keeps none of them.
        assert_eq!(
            names(&format!(
                "MATCH (n:N) WHERE NOT n.x IN {with_null} RETURN n"
            )),
            0,
            "{length}"
        );
        // Three misses and the node with no x are null; the hits are true.
        assert_eq!(
            names(&format!(
                "MATCH (n:N) WHERE (n.x IN {with_null}) IS NULL RETURN n"
            )),
            4,
            "{length}"
        );
        let without: Vec<String> = items[..items.len() - 1].to_vec();
        let without = format!("[{}]", without.join(", "));
        assert_eq!(
            names(&format!("MATCH (n:N) WHERE NOT n.x IN {without} RETURN n")),
            3,
            "{length}"
        );
        assert_eq!(
            names(&format!(
                "MATCH (n:N) WHERE (n.x IN {without}) IS NULL RETURN n"
            )),
            1,
            "{length}"
        );
    }
}

#[test]
fn a_null_needle_gives_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for length in LENGTHS {
        for list in [padded("1", length, 0), padded("null", length, 0)] {
            assert_eq!(
                session
                    .execute(&format!("RETURN null IN {list} AS found"))
                    .unwrap()
                    .rows()[0][0],
                Value::Null,
                "{length}"
            );
        }
    }
}

#[test]
fn duplicates_in_the_list_count_once() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(1, 5) AS i INSERT (:N {x: i})")
        .unwrap();
    let items = vec!["3"; 40].join(", ");
    let rows = session
        .execute(&format!("MATCH (n:N) WHERE n.x IN [{items}] RETURN n.x"))
        .unwrap();
    assert_eq!(rows.rows(), vec![vec![Value::Int64(3)]]);
}

/// An empty list is false for a value and for NULL's sake unaffected.
#[test]
fn an_empty_list_finds_nothing() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:N {x: 1})").unwrap();
    let found = session
        .execute("MATCH (n:N) WHERE n.x IN [] RETURN n")
        .unwrap();
    assert_eq!(found.row_count(), 0);
    let kept = session
        .execute("MATCH (n:N) WHERE NOT n.x IN [] RETURN n")
        .unwrap();
    assert_eq!(kept.row_count(), 1);
}

// ---------------------------------------------------------------------------
// Nodes and edges as needles
// ---------------------------------------------------------------------------

/// A node or edge needle keeps today's behaviour against a long list, error or result, in both
/// languages: they differ (GQL's temporal rule raises where openCypher gives false), which is why the
/// short list is the reference and not a fixed answer.
#[test]
fn a_node_or_edge_needle_answers_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (a:N {x: 1})-[:R]->(b:N {x: 2})")
        .unwrap();
    let run = |cypher: bool, query: &str| -> String {
        let result = if cypher {
            session.execute_cypher(query)
        } else {
            session.execute(query)
        };
        match result {
            Ok(result) => format!("{:?}", result.rows()),
            Err(error) => format!(
                "raises {}",
                error.to_string().split(':').next().unwrap_or("")
            ),
        }
    };
    for cypher in [false, true] {
        for item in ["1", "0", "'n'", "1.0", "true", "null"] {
            for (needle, pattern) in [("n", "MATCH (n:N)"), ("r", "MATCH ()-[r:R]->()")] {
                // Two items, below any threshold: a filler can raise for a node needle, so the walk
                // it is compared with must meet a filler too.
                let list = padded(item, 2, 1);
                let short = format!("{pattern} RETURN {needle} IN {list} AS found");
                let walk = run(cypher, &short);
                for length in [16, 40, 257] {
                    let list = padded(item, length, length / 2);
                    let long = format!("{pattern} RETURN {needle} IN {list} AS found");
                    assert_eq!(
                        run(cypher, &long),
                        walk,
                        "cypher {cypher}: {short}, {length} items"
                    );
                }
            }
        }
    }
}

/// An entity evaluates to its id, so the needle of `n IN [...]` reads as an `Int64`, and a set keyed
/// by that integer would find the node's own id in the list. The walk's kind check decides, so a list
/// holding the node's and the edge's ids answers as the short list, in `WHERE` and in `RETURN`, in
/// GQL and in Cypher.
#[test]
fn a_list_holding_the_ids_of_the_needle_answers_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (a:N {x: 1})-[:R]->(b:N {x: 2})")
        .unwrap();
    let id_of = |query: &str| {
        session.execute(query).unwrap().rows()[0][0]
            .as_int64()
            .unwrap()
    };
    let node = id_of("MATCH (n:N {x: 1}) RETURN id(n)");
    let edge = id_of("MATCH ()-[r:R]->() RETURN id(r)");
    for cypher in [false, true] {
        let run = |query: &str| -> String {
            let result = if cypher {
                session.execute_cypher(query)
            } else {
                session.execute(query)
            };
            match result {
                Ok(result) => rows(result.rows()).join(";"),
                Err(error) => format!(
                    "raises {}",
                    error.to_string().split(':').next().unwrap_or("")
                ),
            }
        };
        for (needle, pattern, id) in [
            ("n", "MATCH (n:N {x: 1})", node),
            ("r", "MATCH ()-[r:R]->()", edge),
        ] {
            for form in [
                format!("{pattern} RETURN {needle} IN LIST AS found"),
                format!("{pattern} WHERE {needle} IN LIST RETURN 1 AS found"),
                format!("{pattern} WHERE NOT {needle} IN LIST RETURN 1 AS found"),
            ] {
                let walk = run(&form.replace("LIST", &format!("[{id}]")));
                for length in [15, 16, 17, 64, 257] {
                    let list = padded(&id.to_string(), length, length / 2);
                    assert_eq!(
                        run(&form.replace("LIST", &list)),
                        walk,
                        "cypher {cypher}: {form}, {length} items"
                    );
                }
            }
            // From a parameter, too.
            let list: Vec<Value> = std::iter::once(Value::Int64(id))
                .chain((0..40).map(|at| Value::Int64(1_000_000 + at)))
                .collect();
            let params =
                |list: Vec<Value>| HashMap::from([("list".to_string(), Value::List(list.into()))]);
            let query = format!("{pattern} RETURN {needle} IN $list AS found");
            let by_parameter = |list: Vec<Value>| {
                let result = if cypher {
                    session.execute_cypher_with_params(&query, params(list))
                } else {
                    session.execute_with_params(&query, params(list))
                };
                match result {
                    Ok(result) => rows(result.rows()).join(";"),
                    Err(error) => format!(
                        "raises {}",
                        error.to_string().split(':').next().unwrap_or("")
                    ),
                }
            };
            assert_eq!(
                by_parameter(list),
                by_parameter(vec![Value::Int64(id)]),
                "cypher {cypher}: {query}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Where the list comes from, and how often it is read
// ---------------------------------------------------------------------------

/// A literal and a parameter answer alike, and with a parameter that has NULL or another kind.
#[test]
fn a_literal_and_a_parameter_answer_alike() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(0, 99) AS i INSERT (:N {x: i, text: toString(i)})")
        .unwrap();
    for (key, build) in [
        ("x", (|i: i64| Value::Int64(i)) as fn(i64) -> Value),
        ("x", |i: i64| Value::Float64(i as f64)),
        ("x", |i: i64| Value::from(i.to_string().as_str())),
        ("text", |i: i64| Value::from(i.to_string().as_str())),
        ("text", |i: i64| Value::Int64(i)),
    ] {
        for length in [15_i64, 16, 17, 64, 200] {
            let list: Vec<Value> = (0..length).map(|i| build(i * 3)).collect();
            let query = format!("MATCH (n:N) WHERE n.{key} IN $list RETURN n.x");
            let params = HashMap::from([("list".to_string(), Value::List(list.clone().into()))]);
            let by_parameter = rows(session.execute_with_params(&query, params).unwrap().rows());
            // The walk, one value at a time.
            let mut by_walk: Vec<String> = Vec::new();
            for value in &list {
                let params = HashMap::from([("one".to_string(), value.clone())]);
                let one = format!("MATCH (n:N) WHERE n.{key} = $one RETURN n.x");
                by_walk.extend(rows(
                    session.execute_with_params(&one, params).unwrap().rows(),
                ));
            }
            by_walk.sort();
            assert_eq!(by_parameter, by_walk, "{key}, {length}");
        }
    }
    let literal_list: Vec<String> = (0..40).map(|i| (i * 3).to_string()).collect();
    let literal_query = format!(
        "MATCH (n:N) WHERE n.x IN [{}] RETURN n.x",
        literal_list.join(", ")
    );
    let list: Vec<Value> = (0..40).map(|i| Value::Int64(i * 3)).collect();
    let params = HashMap::from([("list".to_string(), Value::List(list.into()))]);
    assert_eq!(
        rows(session.execute(&literal_query).unwrap().rows()),
        rows(
            session
                .execute_with_params("MATCH (n:N) WHERE n.x IN $list RETURN n.x", params)
                .unwrap()
                .rows()
        )
    );
}

fn rows(rows: &[Vec<Value>]) -> Vec<String> {
    let mut rows: Vec<String> = rows.iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

/// A cached plan runs with another list, so nothing the first list built may serve the second.
#[test]
fn a_cached_plan_reads_each_runs_list() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(0, 999) AS i INSERT (:N {x: i})")
        .unwrap();
    let query = "MATCH (n:N) WHERE n.x IN $list RETURN count(n) AS found";
    for (first, count) in [(0_i64, 40_i64), (500, 40), (5_000, 0), (990, 10)] {
        let list: Vec<Value> = (first..first + 40).map(Value::Int64).collect();
        let params = HashMap::from([("list".to_string(), Value::List(list.into()))]);
        let result = session.execute_with_params(query, params).unwrap();
        assert_eq!(result.rows()[0][0].as_int64(), Some(count), "from {first}");
    }
    // A short list after a long one, and one that holds NULL.
    let params = HashMap::from([(
        "list".to_string(),
        Value::List(vec![Value::Int64(1)].into()),
    )]);
    let result = session.execute_with_params(query, params).unwrap();
    assert_eq!(result.rows()[0][0].as_int64(), Some(1));
    let mut with_null: Vec<Value> = (0..40).map(|i| Value::Int64(i + 5_000)).collect();
    with_null.push(Value::Null);
    let params = HashMap::from([("list".to_string(), Value::List(with_null.into()))]);
    let not_in = "MATCH (n:N) WHERE NOT n.x IN $list RETURN count(n) AS found";
    let result = session.execute_with_params(not_in, params).unwrap();
    assert_eq!(result.rows()[0][0].as_int64(), Some(0));
}

/// A list from the row, a property or an earlier variable, is read per row.
#[test]
fn a_list_from_the_row_keeps_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let long: Vec<String> = (0..40).map(|i| i.to_string()).collect();
    session
        .execute(&format!(
            "INSERT (:N {{x: 3, list: [{}]}})",
            long.join(", ")
        ))
        .unwrap();
    session.execute("INSERT (:N {x: 3, list: [1, 2]})").unwrap();
    session
        .execute("INSERT (:N {x: 50, list: [50, null]})")
        .unwrap();
    session
        .execute("INSERT (:N {x: 60, list: [50, null]})")
        .unwrap();
    let found = session
        .execute("MATCH (n:N) WHERE n.x IN n.list RETURN n.x")
        .unwrap();
    assert_eq!(
        rows(found.rows()),
        rows(&[vec![Value::Int64(3)], vec![Value::Int64(50)]])
    );
    let empty = session
        .execute("MATCH (n:N) WHERE NOT n.x IN n.list RETURN n.x")
        .unwrap();
    assert_eq!(rows(empty.rows()), rows(&[vec![Value::Int64(3)]]));
    let bound = session
        .execute_cypher(&format!(
            "WITH [{}] AS list MATCH (n:N) WHERE n.x IN list RETURN count(n) AS found",
            long.join(", ")
        ))
        .unwrap();
    assert_eq!(bound.rows()[0][0].as_int64(), Some(2));
}

/// A computed list that holds only constants is a list from the row as far as the answer goes.
#[test]
fn a_list_with_an_expression_from_the_row() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(1, 50) AS i INSERT (:N {x: i, y: i * 2})")
        .unwrap();
    let items: Vec<String> = (0..30).map(|i| (1_000_000 + i).to_string()).collect();
    let query = format!(
        "MATCH (n:N) WHERE n.y IN [n.x * 2, {}] RETURN count(n) AS found",
        items.join(", ")
    );
    assert_eq!(
        session.execute(&query).unwrap().rows()[0][0].as_int64(),
        Some(50)
    );
}

/// More rows than a chunk holds, and more than a morsel: every chunk reads the one set.
#[test]
fn several_chunks_answer_alike() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(0, 9999) AS i INSERT (:N {x: i})")
        .unwrap();
    let every_third: Vec<String> = (0..3334).map(|i| (i * 3).to_string()).collect();
    let query = format!(
        "MATCH (n:N) WHERE n.x IN [{}] RETURN n.x",
        every_third.join(", ")
    );
    let found = session.execute(&query).unwrap();
    assert_eq!(found.row_count(), 3334);
    assert!(
        found
            .rows()
            .iter()
            .all(|row| row[0].as_int64().unwrap() % 3 == 0)
    );
}

/// A subquery runs its filter once per outer row, and each run answers alike.
#[test]
fn a_subquery_reads_the_set_per_outer_row() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(0, 49) AS i INSERT (:Outer {x: i}), (:Inner {x: i * 2})")
        .unwrap();
    let list: Vec<String> = (0..40).map(|i| (i * 2).to_string()).collect();
    let query = format!(
        "MATCH (o:Outer) WHERE EXISTS {{ MATCH (i:Inner) WHERE i.x = o.x AND i.x IN [{}] }} RETURN o.x",
        list.join(", ")
    );
    let found = session.execute(&query).unwrap();
    assert_eq!(found.row_count(), 25, "{:?}", found.rows());
    let cypher = session.execute_cypher(&query).unwrap();
    assert_eq!(rows(found.rows()), rows(cypher.rows()));
}

// ---------------------------------------------------------------------------
// That it is a set
// ---------------------------------------------------------------------------

/// A long list of negative numbers answers as the walk, written and as a parameter, also next to
/// positive ones, floats and an expression that folds.
#[test]
fn negative_numbers_answer_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("UNWIND range(-50, 50) AS i INSERT (:N {x: i, f: i * 1.0e0})")
        .unwrap();
    for length in [15_i64, 16, 17, 40, 257] {
        let negatives: Vec<i64> = (1..=length).map(|i| -i).collect();
        let wanted = usize::try_from(length.min(50)).unwrap();
        let written: Vec<String> = negatives.iter().map(ToString::to_string).collect();
        let query = format!(
            "MATCH (n:N) WHERE n.x IN [{}] RETURN n.x",
            written.join(", ")
        );
        assert_eq!(
            session.execute(&query).unwrap().row_count(),
            wanted,
            "{length}"
        );
        let query = format!(
            "MATCH (n:N) WHERE n.f IN [{}] RETURN n.x",
            written.join(", ")
        );
        assert_eq!(
            session.execute(&query).unwrap().row_count(),
            wanted,
            "floats, {length}"
        );
        let list: Vec<Value> = negatives.iter().map(|n| Value::Int64(*n)).collect();
        let params = HashMap::from([("list".to_string(), Value::List(list.into()))]);
        let found = session
            .execute_with_params("MATCH (n:N) WHERE n.x IN $list RETURN n.x", params)
            .unwrap();
        assert_eq!(found.row_count(), wanted, "parameter, {length}");
        // Written as a float, a minus over a sum, and next to positive numbers.
        let mixed: Vec<String> = (1..=length)
            .map(|i| match i % 4 {
                0 => format!("-{i}.0e0"),
                1 => format!("-({i})"),
                2 => format!("(0 - {i})"),
                _ => format!("-{i}"),
            })
            .collect();
        let query = format!("MATCH (n:N) WHERE n.x IN [{}] RETURN n.x", mixed.join(", "));
        assert_eq!(
            session.execute(&query).unwrap().row_count(),
            wanted,
            "mixed, {length}"
        );
        let query = format!(
            "MATCH (n:N) WHERE NOT n.x IN [{}] RETURN n.x",
            written.join(", ")
        );
        assert_eq!(
            session.execute(&query).unwrap().row_count(),
            101 - wanted,
            "NOT, {length}"
        );
    }
    agrees(&session, "-5", "-5.0", &[16, 40]);
    agrees(&session, "-5", "'-5'", &[16, 40]);
    agrees(&session, "-0", "0", &[16, 40]);
    agrees(&session, "-0.0e0", "0", &[16, 40]);
}

/// The time per row does not grow with the list. A walk of misses reads each item per row, so 2,000
/// items over 20,000 rows take about 125 times the 16-item query; the set takes about the same.
///
/// A ratio against the same run, with the best of three, so a slow machine does not fail it and the
/// margin stays wide of both. A literal list and a parameter each.
#[test]
fn a_long_list_does_not_cost_per_item() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::KeyColumnLookup, false));
    let session = session;
    session
        .execute("UNWIND range(0, 19999) AS i INSERT (:N {x: i})")
        .unwrap();
    let best_of = |run: &dyn Fn() -> usize| -> Duration {
        (0..3)
            .map(|_| {
                let begun = Instant::now();
                assert_eq!(run(), 0);
                begun.elapsed()
            })
            .min()
            .unwrap()
    };
    let literal = |length: i64| {
        let items: Vec<String> = (0..length).map(|i| (1_000_000 + i).to_string()).collect();
        let query = format!("MATCH (n:N) WHERE n.x IN [{}] RETURN n.x", items.join(", "));
        best_of(&|| session.execute(&query).unwrap().row_count())
    };
    let parameter = |length: i64| {
        let items: Vec<Value> = (0..length).map(|i| Value::Int64(1_000_000 + i)).collect();
        let list = Value::List(items.into());
        best_of(&|| {
            let params = HashMap::from([("list".to_string(), list.clone())]);
            session
                .execute_with_params("MATCH (n:N) WHERE n.x IN $list RETURN n.x", params)
                .unwrap()
                .row_count()
        })
    };
    // Negative numbers are written as a minus over a literal: the list is constant all the same.
    let negative = |length: i64| {
        let items: Vec<String> = (0..length).map(|i| (-1_000_000 - i).to_string()).collect();
        let query = format!("MATCH (n:N) WHERE n.x IN [{}] RETURN n.x", items.join(", "));
        best_of(&|| session.execute(&query).unwrap().row_count())
    };
    for (form, short, long) in [
        ("literal", literal(16), literal(2_000)),
        ("parameter", parameter(16), parameter(2_000)),
        ("negative literal", negative(16), negative(2_000)),
    ] {
        assert!(
            long < short * 20,
            "{form}: 2,000 items took {long:?} and 16 items {short:?}: the list is walked per row"
        );
    }
}
