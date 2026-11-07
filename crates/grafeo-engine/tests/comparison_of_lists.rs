//! `<` on two lists, element by element, in GQL and Cypher (GA04 step b, the ruling of 2026-11-06):
//! at the first position where the elements are not equal, that pair's comparison is the answer,
//! unknown when either is null, and for two kinds GQL's order of kinds in GQL and null in Cypher. A
//! list that is a prefix of the other is the lesser; two equal lists are not less. ISO/IEC 39075 19.3
//! (General Rule 2, b, ii, 5, B) leaves the order of lists to the implementation, and a path compares
//! as its list of elements (b, iv); the openCypher TCK pins five answers (`Comparison2.feature`,
//! scenario 4). 0.6.1 answered null for every `<` on two lists, `[1] < [1]` included.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main comparison_of_lists::
//! ```

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Language {
    Gql,
    Cypher,
}

const LANGUAGES: [Language; 2] = [Language::Gql, Language::Cypher];

fn run(
    session: &Session,
    language: Language,
    query: &str,
    params: HashMap<String, Value>,
) -> Vec<Vec<Value>> {
    let result = match language {
        Language::Gql => session.execute_with_params(query, params),
        Language::Cypher => session.execute_cypher_with_params(query, params),
    };
    result
        .unwrap_or_else(|error| panic!("{language:?}: {query}: {error}"))
        .rows()
        .to_vec()
}

/// A three-valued answer as the tables write it.
fn answer(value: &Value) -> &'static str {
    match value {
        Value::Bool(true) => "true",
        Value::Bool(false) => "false",
        Value::Null => "null",
        other => panic!("a truth value, not {other:?}"),
    }
}

fn single(session: &Session, language: Language, expression: &str) -> &'static str {
    let rows = run(
        session,
        language,
        &format!("RETURN {expression} AS v"),
        HashMap::new(),
    );
    answer(&rows[0][0])
}

/// Expression, GQL's answer, Cypher's answer.
const CASES: &[(&str, &str, &str)] = &[
    // openCypher TCK, Comparison2.feature, scenario 4 "Comparing lists".
    ("[1, 0] >= [1]", "true", "true"),
    ("[1, null] >= [1]", "true", "true"),
    ("[1, 2] >= [1, null]", "null", "null"),
    ("[1, 'a'] >= [1, null]", "null", "null"),
    ("[1, 2] >= [3, null]", "false", "false"),
    // Equal lists: not less, and `<=` holds (0.6.1 gave null).
    ("[1] < [1]", "false", "false"),
    ("[1] <= [1]", "true", "true"),
    ("[1] > [1]", "false", "false"),
    ("[1] >= [1]", "true", "true"),
    ("[] < []", "false", "false"),
    ("[] <= []", "true", "true"),
    ("['a', 'b'] >= ['a', 'b']", "true", "true"),
    // Equal numbers of two numeric kinds are equal elements.
    ("[1] < [1.0]", "false", "false"),
    ("[1] <= [1.0]", "true", "true"),
    ("[1.0, 2] < [1, 3]", "true", "true"),
    // A prefix is the lesser.
    ("[] < [1]", "true", "true"),
    ("[1] < [1, 2]", "true", "true"),
    ("[1, 2] > [1]", "true", "true"),
    ("[] < [null]", "true", "true"),
    ("[1, null] < [1]", "false", "false"),
    ("[1, null] > [1]", "true", "true"),
    // The first pair that is not equal decides, whatever follows it.
    ("[1, 2] < [1, 3]", "true", "true"),
    ("[2] > [1, 5]", "true", "true"),
    ("[1, 9] < [2, 0]", "true", "true"),
    ("['a'] < ['b']", "true", "true"),
    ("[1, 'a'] < [2]", "true", "true"),
    ("[1, 'a'] < [1, 'b']", "true", "true"),
    // A null in the deciding pair: unknown.
    ("[1, null] < [1, 2]", "null", "null"),
    ("[null, 1] < [2, 0]", "null", "null"),
    ("[1] < [null]", "null", "null"),
    ("[null] < [null]", "null", "null"),
    ("[null] <= [null]", "null", "null"),
    // Nested lists, by the same walk.
    ("[[1]] < [[2]]", "true", "true"),
    ("[[1], 2] < [[1], 3]", "true", "true"),
    ("[[1, 2]] > [[1]]", "true", "true"),
    ("[[1, null]] < [[1, 2]]", "null", "null"),
    // Two kinds in the deciding pair: GQL's order of kinds (a string, then a boolean, then a
    // number; a list before all three), Cypher's null.
    ("[1] < ['1']", "false", "null"),
    ("['1'] < [1]", "true", "null"),
    ("[true] < [1]", "true", "null"),
    ("[1, 'a'] < [1, 2]", "true", "null"),
    ("[[1]] < [1]", "true", "null"),
    // A pair of two kinds after the deciding pair plays no part.
    ("[1, 'a'] < [2, 3]", "true", "true"),
];

#[test]
fn lists_compare_element_by_element() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut wrong = Vec::new();
    for (expression, gql, cypher) in CASES {
        for (language, want) in [(Language::Gql, *gql), (Language::Cypher, *cypher)] {
            let found = single(&session, language, expression);
            if found != want {
                wrong.push(format!(
                    "{language:?}: {expression} gave {found}, want {want}"
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "{} of {} wrong:\n{}",
        wrong.len(),
        CASES.len() * 2,
        wrong.join("\n")
    );
}

/// A stored list compared in a filter: every plan gives the rows the expression gives, GQL keeping
/// the rows its order of kinds puts below, Cypher dropping the pairs of two kinds.
#[test]
fn a_stored_list_compares_alike_in_every_plan() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute(
            "INSERT (:L {r: 0, k: [1]}), (:L {r: 1, k: [1, 2]}), (:L {r: 2, k: [2]}), \
             (:L {r: 3, k: []}), (:L {r: 4, k: [1, null]}), (:L {r: 5, k: ['a']}), \
             (:L {r: 6, k: 5}), (:L {r: 7}), (:A {k: 1})",
        )
        .unwrap();
    // r4's deciding pair holds a null; r5's is a string against a number; r6 is a number, which
    // GQL orders after every list; r7 has no k.
    for (condition, gql, cypher) in [
        ("n.k < [1, 2]", vec![0, 3, 5], vec![0, 3]),
        ("[1, 2] > n.k", vec![0, 3, 5], vec![0, 3]),
        ("n.k <= [1, 2]", vec![0, 1, 3, 5], vec![0, 1, 3]),
        ("n.k > [1]", vec![1, 2, 4, 6], vec![1, 2, 4]),
        ("n.k >= [1]", vec![0, 1, 2, 4, 6], vec![0, 1, 2, 4]),
        ("n.k < []", vec![], vec![]),
        ("n.k >= [1] AND n.k < [2]", vec![0, 1, 4], vec![0, 1, 4]),
    ] {
        for (language, want) in [(Language::Gql, &gql), (Language::Cypher, &cypher)] {
            for query in [
                format!("MATCH (n:L) WHERE {condition} RETURN n.r AS r"),
                format!("MATCH (a:A), (n:L) WHERE {condition} RETURN n.r AS r"),
                format!("MATCH (n:L), (a:A) WHERE {condition} RETURN n.r AS r"),
                format!("UNWIND [1] AS x MATCH (n:L) WHERE {condition} RETURN n.r AS r"),
            ] {
                let mut found: Vec<i64> = run(&session, language, &query, HashMap::new())
                    .iter()
                    .map(|row| row[0].as_int64().unwrap())
                    .collect();
                found.sort_unstable();
                assert_eq!(&found, want, "{language:?}: {query}");
            }
        }
    }
}

/// A path compares as its list of elements: equal paths are not less, and a path that is a prefix
/// of a longer one is the lesser.
#[test]
fn a_path_compares_as_its_elements() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:P {name: 'Alix'})-[:K]->(:P {name: 'Gus'})-[:K]->(:P {name: 'Vincent'})")
        .unwrap();
    for language in LANGUAGES {
        let rows = run(
            &session,
            language,
            "MATCH p = (a:P {name: 'Alix'})-[:K]->(b) MATCH q = (a)-[:K]->(b)-[:K]->(c) \
             RETURN p < p AS a, p <= p AS b, p < q AS c, q > p AS d, q < p AS e",
            HashMap::new(),
        );
        let found: Vec<&str> = rows[0].iter().map(answer).collect();
        assert_eq!(
            found,
            ["false", "true", "true", "true", "false"],
            "{language:?}"
        );
    }
}

/// The walk the ruling describes, built from the language's own `=` and ordering of two elements:
/// at the first position whose elements are not equal (their `=` is false or unknown), that pair's
/// answer; with no such position, the lengths decide.
fn reference(
    elements: &HashMap<(usize, usize, &'static str), &'static str>,
    left: &[usize],
    right: &[usize],
    op: &'static str,
) -> &'static str {
    for (a, b) in left.iter().zip(right) {
        if elements[&(*a, *b, "=")] != "true" {
            return elements[&(*a, *b, op)];
        }
    }
    let (l, r) = (left.len(), right.len());
    let holds = match op {
        "<" => l < r,
        "<=" => l <= r,
        ">" => l > r,
        ">=" => l >= r,
        _ => unreachable!("{op}"),
    };
    if holds { "true" } else { "false" }
}

/// Every pair of lists of up to two elements drawn from numbers of two kinds, NaN, a string, a
/// boolean, null and a nested list: each ordering operator answers as the walk over the language's
/// own element comparison. A later change to how two elements compare cannot leave lists behind.
#[test]
fn every_pair_of_short_lists_answers_as_the_walk() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let pool = [
        Value::Int64(0),
        Value::Int64(1),
        Value::Float64(1.0),
        Value::Float64(f64::NAN),
        Value::from("a"),
        Value::Bool(true),
        Value::Null,
        Value::List(vec![Value::Int64(0)].into()),
    ];
    let mut lists: Vec<Vec<usize>> = vec![vec![]];
    for a in 0..pool.len() {
        lists.push(vec![a]);
        for b in 0..pool.len() {
            lists.push(vec![a, b]);
        }
    }
    let as_value = |list: &[usize]| Value::List(list.iter().map(|i| pool[*i].clone()).collect());
    let ops = ["<", "<=", ">", ">="];
    let mut wrong = Vec::new();
    let mut checked = 0;
    for language in LANGUAGES {
        let mut elements = HashMap::new();
        for a in 0..pool.len() {
            for b in 0..pool.len() {
                let params = HashMap::from([
                    ("a".to_string(), pool[a].clone()),
                    ("b".to_string(), pool[b].clone()),
                ]);
                let row = run(
                    &session,
                    language,
                    "RETURN $a = $b AS eq, $a < $b AS lt, $a <= $b AS le, $a > $b AS gt, $a >= $b AS ge",
                    params,
                )
                .remove(0);
                for (op, value) in ["=", "<", "<=", ">", ">="].into_iter().zip(&row) {
                    elements.insert((a, b, op), answer(value));
                }
            }
        }
        for left in &lists {
            for right in &lists {
                let params = HashMap::from([
                    ("a".to_string(), as_value(left)),
                    ("b".to_string(), as_value(right)),
                ]);
                let row = run(
                    &session,
                    language,
                    "RETURN $a < $b AS lt, $a <= $b AS le, $a > $b AS gt, $a >= $b AS ge",
                    params,
                )
                .remove(0);
                for (op, value) in ops.into_iter().zip(&row) {
                    checked += 1;
                    let want = reference(&elements, left, right, op);
                    let found = answer(value);
                    if found != want {
                        wrong.push(format!(
                            "{language:?}: {:?} {op} {:?} gave {found}, want {want}",
                            as_value(left),
                            as_value(right)
                        ));
                    }
                }
            }
        }
    }
    assert!(checked > 20_000, "{checked} checked");
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong, the first 40:\n{}",
        wrong.len(),
        wrong
            .iter()
            .take(40)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}

/// Lists of one kind are totally ordered: for any two exactly one of `<`, `=` and `>` holds, and
/// ORDER BY sorts them as `<` does.
#[test]
fn lists_of_one_kind_sort_as_they_compare() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let pool = "[[], [0], [1], [1, 0], [1, 2], [2], [0, 5], [1, 2, 3], [1.5], [2, 0]]";
    for language in LANGUAGES {
        let rows = run(
            &session,
            language,
            &format!(
                "UNWIND {pool} AS a UNWIND {pool} AS b \
                 RETURN a, b, a < b AS lt, a = b AS eq, a > b AS gt"
            ),
            HashMap::new(),
        );
        for row in &rows {
            let truths: Vec<&str> = row[2..].iter().map(answer).collect();
            assert_eq!(
                truths.iter().filter(|t| **t == "true").count(),
                1,
                "{language:?}: {:?} against {:?}: {truths:?}",
                row[0],
                row[1]
            );
        }
        let sorted = run(
            &session,
            language,
            &format!("UNWIND {pool} AS a RETURN a ORDER BY a"),
            HashMap::new(),
        );
        for pair in sorted.windows(2) {
            let params = HashMap::from([
                ("a".to_string(), pair[0][0].clone()),
                ("b".to_string(), pair[1][0].clone()),
            ]);
            let row = run(&session, language, "RETURN $a <= $b AS le", params);
            assert_eq!(
                answer(&row[0][0]),
                "true",
                "{language:?}: ORDER BY put {:?} before {:?}",
                pair[0][0],
                pair[1][0]
            );
        }
    }
}
