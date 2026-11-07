//! GA04, comparison across kinds: `=` is false and `<>` true for two values of different kinds in
//! GQL and Cypher; `<`, `<=`, `>` and `>=` order them by kind in GQL, the order of `ORDER BY`, and are
//! null in Cypher. Numbers are one kind, so `2.0 = 2`. Two values of one kind compare as before, pinned
//! from the evaluator before the change. Every other form that compares (a stored value with and
//! without an index, `IN`, `CASE`, `DISTINCT`, grouping, a join, `min` and `max`) answers as `=` and `<`
//! do. `comparison_across_kinds.gtest` holds the literal pairs.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main comparison_across_kinds::
//! ```

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use grafeo_common::types::{
    Date, Decimal, Duration, PropertyKey, Time, Timestamp, Value, ZonedDatetime,
};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn map(pairs: &[(&str, Value)]) -> Value {
    Value::Map(Arc::new(
        pairs
            .iter()
            .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    ))
}

/// Name, family and rank of each value: two of one family compare as before, two of different
/// families by the rule.
fn values() -> Vec<(&'static str, &'static str, u8, Value)> {
    vec![
        ("int0", "number", 15, Value::Int64(0)),
        ("int1", "number", 15, Value::Int64(1)),
        ("int2", "number", 15, Value::Int64(-1)),
        ("int3", "number", 15, Value::Int64(40)),
        ("int4", "number", 15, Value::Int64(50)),
        ("int5", "number", 15, Value::Int64(5)),
        ("int6", "number", 15, Value::Int64(1000)),
        ("int7", "number", 15, Value::Int64(9_007_199_254_740_993)),
        ("float8", "number", 15, Value::Float64(0.0)),
        ("float9", "number", 15, Value::Float64(-0.0)),
        ("float10", "number", 15, Value::Float64(1.5)),
        ("float11", "number", 15, Value::Float64(50.0)),
        (
            "float12",
            "number",
            15,
            Value::Float64(9_007_199_254_740_992.0),
        ),
        ("float13", "number", 15, Value::Float64(f64::NAN)),
        ("float14", "number", 15, Value::Float64(f64::INFINITY)),
        (
            "decimal15",
            "number",
            15,
            Value::Decimal(Decimal::parse("1.5").unwrap()),
        ),
        (
            "decimal16",
            "number",
            15,
            Value::Decimal(Decimal::parse("50.00").unwrap()),
        ),
        ("string17", "string", 11, Value::from("")),
        ("string18", "string", 11, Value::from("0")),
        ("string19", "string", 11, Value::from("1.5")),
        ("string20", "string", 11, Value::from("40")),
        ("string21", "string", 11, Value::from("50")),
        ("string22", "string", 11, Value::from("042")),
        ("string23", "string", 11, Value::from("4.2e1")),
        ("string24", "string", 11, Value::from("NaN")),
        ("string25", "string", 11, Value::from("true")),
        ("string26", "string", 11, Value::from("+5")),
        ("string27", "string", 11, Value::from("5.0")),
        ("string28", "string", 11, Value::from("1e3")),
        ("string29", "string", 11, Value::from("Alix")),
        ("string30", "string", 11, Value::from("2024-01-01")),
        ("bool31", "bool", 12, Value::Bool(false)),
        ("bool32", "bool", 12, Value::Bool(true)),
        (
            "date33",
            "date",
            6,
            Value::Date(Date::parse("2024-01-01").unwrap()),
        ),
        (
            "date34",
            "date",
            6,
            Value::Date(Date::parse("2024-06-15").unwrap()),
        ),
        (
            "datetime35",
            "datetime",
            5,
            Value::Timestamp(Timestamp::from_micros(1_704_103_200_000_000)),
        ),
        (
            "zoned36",
            "datetime",
            4,
            Value::ZonedDatetime(ZonedDatetime::parse("2024-01-01T10:00:00+01:00").unwrap()),
        ),
        (
            "zoned37",
            "datetime",
            4,
            Value::ZonedDatetime(ZonedDatetime::parse("2024-01-01T09:00:00Z").unwrap()),
        ),
        (
            "time38",
            "time",
            8,
            Value::Time(Time::parse("10:00:00").unwrap()),
        ),
        (
            "zoned_time39",
            "time",
            7,
            Value::Time(Time::parse("10:00:00+01:00").unwrap()),
        ),
        (
            "duration40",
            "duration",
            9,
            Value::Duration(Duration::parse("P1D").unwrap()),
        ),
        (
            "duration41",
            "duration",
            9,
            Value::Duration(Duration::parse("PT1H").unwrap()),
        ),
        (
            "list42",
            "list",
            1,
            Value::List(vec![Value::Int64(1)].into()),
        ),
        (
            "list43",
            "list",
            1,
            Value::List(vec![Value::Float64(1.0)].into()),
        ),
        (
            "list44",
            "list",
            1,
            Value::List(vec![Value::from("1")].into()),
        ),
        ("list45", "list", 1, Value::List(vec![].into())),
        ("map46", "map", 0, map(&[("a", Value::Int64(1))])),
        ("map47", "map", 0, map(&[("a", Value::from("1"))])),
        ("bytes48", "bytes", 10, Value::Bytes(vec![1_u8, 2].into())),
        (
            "vector49",
            "vector",
            3,
            Value::Vector(vec![1.0_f32, 2.0].into()),
        ),
        ("null50", "null", 16, Value::Null),
    ]
}

#[derive(Clone, Copy, Debug)]
enum Language {
    Gql,
    Cypher,
}

fn run(
    session: &Session,
    language: Language,
    query: &str,
    params: HashMap<String, Value>,
) -> Result<Vec<Vec<Value>>, String> {
    let result = match language {
        Language::Gql => session.execute_with_params(query, params),
        Language::Cypher => session.execute_cypher_with_params(query, params),
    };
    result
        .map(|result| result.rows().to_vec())
        .map_err(|error| error.to_string())
}

fn pair(a: &Value, b: &Value) -> HashMap<String, Value> {
    HashMap::from([("a".to_string(), a.clone()), ("b".to_string(), b.clone())])
}

/// What one comparison gave: `true`, `false`, `null`, or the error.
fn outcome(session: &Session, language: Language, op: &str, a: &Value, b: &Value) -> String {
    match run(
        session,
        language,
        &format!("RETURN $a {op} $b AS r"),
        pair(a, b),
    ) {
        Ok(rows) => match &rows[0][0] {
            Value::Bool(flag) => flag.to_string(),
            Value::Null => "null".to_string(),
            other => format!("{other:?}"),
        },
        Err(error) => format!("error {}", error.split(':').next().unwrap_or("")),
    }
}

const OPS: [&str; 6] = ["=", "<>", "<", "<=", ">", ">="];

/// The design's rule for two values of different families.
fn across(op: &str, a: (&str, u8), b: (&str, u8), language: Language) -> &'static str {
    if a.0 == "null" || b.0 == "null" {
        return "null";
    }
    match (op, language) {
        ("=", _) => "false",
        ("<>", _) => "true",
        (_, Language::Cypher) => "null",
        _ => {
            let order = a.1.cmp(&b.1);
            let held = match op {
                "<" => order.is_lt(),
                "<=" => order.is_le(),
                ">" => order.is_gt(),
                _ => order.is_ge(),
            };
            if held { "true" } else { "false" }
        }
    }
}

/// Every pair of values of different families, each operator, both languages: the rule.
#[test]
fn two_kinds_compare_by_the_rule() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let values = values();
    let mut wrong = Vec::new();
    let mut checked = 0;
    for (a_name, a_family, a_rank, a) in &values {
        for (b_name, b_family, b_rank, b) in &values {
            if a_family == b_family {
                continue;
            }
            for language in [Language::Gql, Language::Cypher] {
                for op in OPS {
                    let want = across(op, (a_family, *a_rank), (b_family, *b_rank), language);
                    let found = outcome(&session, language, op, a, b);
                    checked += 1;
                    if found != want {
                        wrong.push(format!(
                            "{language:?}: {a_name} {op} {b_name}: {found}, want {want}"
                        ));
                    }
                }
            }
        }
    }
    let shown: Vec<&str> = wrong.iter().take(80).map(String::as_str).collect();
    assert!(
        wrong.is_empty(),
        "{} of {checked} wrong, the first of them:\n{}",
        wrong.len(),
        shown.join("\n")
    );
}

/// Two values of one family, as the evaluator answered them before the change: `name op name answer`,
/// one per line, both languages. Two lists order element by element since 2026-11-06 (0.6.1 gave
/// null for every ordering of two lists, `[1] < [1]` included): their rows hold that answer.
const SAME_FAMILY: &str = include_str!("comparison_across_kinds.golden");

/// Every pair of one family compares as it did.
#[test]
fn one_kind_compares_as_before() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let values = values();
    let found = same_family_answers(&session, &values);
    if std::env::var_os("GRAFEO_WRITE_GOLDEN").is_some() {
        std::fs::write(
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/tests/comparison_across_kinds.golden"
            ),
            found.join("\n") + "\n",
        )
        .unwrap();
        return;
    }
    let want: Vec<&str> = SAME_FAMILY.lines().collect();
    let wrong: Vec<String> = found
        .iter()
        .zip(&want)
        .filter(|(found, want)| found != *want)
        .map(|(found, want)| format!("{found}, was {want}"))
        .collect();
    assert_eq!(
        found.len(),
        want.len(),
        "the table and the values agree in length"
    );
    assert!(
        wrong.is_empty(),
        "{} changed:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// Two lists or two maps of one family whose elements differ in kind (`[1]` and `['1']`): the rule
/// reaches them through their elements, so `two_kinds_inside_compare_by_the_rule` holds them, not
/// the table of answers from before the change.
const KINDS_INSIDE_DIFFER: [(&str, &str); 3] = [
    ("list42", "list44"),
    ("list43", "list44"),
    ("map46", "map47"),
];

fn kinds_inside_differ(a: &str, b: &str) -> bool {
    KINDS_INSIDE_DIFFER
        .iter()
        .any(|&(x, y)| (a, b) == (x, y) || (a, b) == (y, x))
}

fn same_family_answers(session: &Session, values: &[(&str, &str, u8, Value)]) -> Vec<String> {
    let mut lines = Vec::new();
    for (a_name, a_family, _, a) in values {
        for (b_name, b_family, _, b) in values {
            if a_family != b_family || kinds_inside_differ(a_name, b_name) {
                continue;
            }
            for language in [Language::Gql, Language::Cypher] {
                for op in OPS {
                    lines.push(format!(
                        "{language:?} {a_name} {op} {b_name} {}",
                        outcome(session, language, op, a, b)
                    ));
                }
            }
        }
    }
    lines
}

/// A list or a map whose elements differ in kind from the other's is not equal to it, in GQL and
/// Cypher (`[1] = ['1']` is false, as `1 = '1'` is): the step a rule, reached through the elements.
#[test]
fn two_kinds_inside_compare_by_the_rule() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let values = values();
    let named = |name: &str| values.iter().find(|v| v.0 == name).unwrap().3.clone();
    let mut wrong = Vec::new();
    for (a_name, b_name) in KINDS_INSIDE_DIFFER {
        let (a, b) = (named(a_name), named(b_name));
        for (x_name, x, y_name, y) in [(a_name, &a, b_name, &b), (b_name, &b, a_name, &a)] {
            for language in [Language::Gql, Language::Cypher] {
                for (op, want) in [("=", "false"), ("<>", "true")] {
                    let found = outcome(&session, language, op, x, y);
                    if found != want {
                        wrong.push(format!(
                            "{language:?}: {x_name} {op} {y_name}: {found}, want {want}"
                        ));
                    }
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// Step b: two lists order by their first elements that differ, so a list whose first element is a
/// number comes after one whose first element is a string in GQL (`[1] > ['1']`, the rank of
/// numbers above strings) and the order is null in Cypher, as `1 < '1'` is. Maps whose values differ
/// in kind are not ordered here: no ruling yet.
#[test]
fn two_kinds_inside_a_list_order_by_the_rule() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let values = values();
    let named = |name: &str| values.iter().find(|v| v.0 == name).unwrap().3.clone();
    let mut wrong = Vec::new();
    for number_list in ["list42", "list43"] {
        let (numbers, strings) = (named(number_list), named("list44"));
        for (op, gql_want) in [
            ("<", "false"),
            ("<=", "false"),
            (">", "true"),
            (">=", "true"),
        ] {
            for language in [Language::Gql, Language::Cypher] {
                let want = match language {
                    Language::Gql => gql_want,
                    Language::Cypher => "null",
                };
                let found = outcome(&session, language, op, &numbers, &strings);
                if found != want {
                    wrong.push(format!(
                        "{language:?}: {number_list} {op} list44: {found}, want {want}"
                    ));
                }
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// One value of each family, and the pairs the other forms read.
fn sample() -> Vec<(&'static str, Value)> {
    let mut seen = Vec::new();
    let mut sample = Vec::new();
    for (name, family, _, value) in values() {
        // Two of each family, so a family meets itself too.
        let count = seen.iter().filter(|known| **known == family).count();
        if count < 2 {
            seen.push(family);
            sample.push((name, value));
        }
    }
    sample
}

fn truth(session: &Session, language: Language, op: &str, a: &Value, b: &Value) -> Option<bool> {
    match outcome(session, language, op, a, b).as_str() {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

/// A stored value read by a key with and without an index, by `IN`, by `CASE` and in a `WHERE`, against
/// `=` and `<` of the two values: the rows are those for which the comparison holds.
#[test]
fn every_form_answers_as_the_comparison() {
    let sample = sample();
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for indexed in [false, true] {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            for (row, (_, value)) in sample.iter().enumerate() {
                let props = HashMap::from([
                    ("row".to_string(), Value::Int64(i64::try_from(row).unwrap())),
                    ("v".to_string(), value.clone()),
                ]);
                run(
                    &session,
                    Language::Gql,
                    "INSERT (:L {row: $row, k: $v})",
                    props,
                )
                .unwrap();
            }
            if indexed {
                run(
                    &session,
                    Language::Gql,
                    "CREATE INDEX by_k FOR (n:L) ON (n.k)",
                    HashMap::new(),
                )
                .unwrap();
            }
            for (b_name, b) in &sample {
                let params = HashMap::from([("b".to_string(), b.clone())]);
                for (op, form) in [
                    (
                        "=",
                        "MATCH (n:L) WHERE n.k = $b RETURN n.row ORDER BY n.row",
                    ),
                    ("=", "MATCH (n:L {k: $b}) RETURN n.row ORDER BY n.row"),
                    (
                        "=",
                        "MATCH (n:L) WHERE n.k IN [$b] RETURN n.row ORDER BY n.row",
                    ),
                    (
                        "=",
                        "MATCH (n:L) WHERE n.k IN [$b, 'nothing at all'] RETURN n.row ORDER BY n.row",
                    ),
                    (
                        "=",
                        "MATCH (n:L) WHERE CASE n.k WHEN $b THEN true ELSE false END RETURN n.row ORDER BY n.row",
                    ),
                    (
                        "<>",
                        "MATCH (n:L) WHERE n.k <> $b RETURN n.row ORDER BY n.row",
                    ),
                    (
                        "<",
                        "MATCH (n:L) WHERE n.k < $b RETURN n.row ORDER BY n.row",
                    ),
                    (
                        "<=",
                        "MATCH (n:L) WHERE n.k <= $b RETURN n.row ORDER BY n.row",
                    ),
                    (
                        ">",
                        "MATCH (n:L) WHERE n.k > $b RETURN n.row ORDER BY n.row",
                    ),
                    (
                        ">=",
                        "MATCH (n:L) WHERE n.k >= $b RETURN n.row ORDER BY n.row",
                    ),
                    (
                        ">",
                        "MATCH (n:L) WHERE $b < n.k RETURN n.row ORDER BY n.row",
                    ),
                ] {
                    let want: Vec<i64> = sample
                        .iter()
                        .enumerate()
                        .filter(|(_, (_, a))| truth(&session, language, op, a, b) == Some(true))
                        .map(|(row, _)| i64::try_from(row).unwrap())
                        .collect();
                    let found = run(&session, language, form, params.clone()).map(|rows| {
                        rows.iter()
                            .map(|row| match row[0] {
                                Value::Int64(at) => at,
                                ref other => panic!("{other:?}"),
                            })
                            .collect::<Vec<_>>()
                    });
                    if found.as_ref() != Ok(&want) {
                        wrong.push(format!(
                            "{language:?}, index {indexed}: {form} with {b_name}: {found:?}, want {want:?}"
                        ));
                    }
                }
            }
        }
    }
    let shown: Vec<&str> = wrong.iter().take(60).map(String::as_str).collect();
    assert!(
        wrong.is_empty(),
        "{} wrong, the first of them:\n{}",
        wrong.len(),
        shown.join("\n")
    );
}

/// Two values `=` holds of are one group and one distinct row; two it does not hold of are two, in a
/// join too.
#[test]
fn grouping_distinct_and_joins_follow_equality() {
    let sample = sample();
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for (a_name, a) in &sample {
            for (b_name, b) in &sample {
                if a.is_null() || b.is_null() {
                    continue;
                }
                let equal = truth(&session, language, "=", a, b) == Some(true);
                let want = if equal { 1 } else { 2 };
                for form in [
                    "UNWIND [$a, $b] AS x RETURN DISTINCT x",
                    "UNWIND [$a, $b] AS x RETURN x, count(*) AS c",
                ] {
                    let found = run(&session, language, form, pair(a, b)).map(|rows| rows.len());
                    if found != Ok(want) {
                        wrong.push(format!(
                            "{language:?}: {form} with {a_name}, {b_name}: {found:?}, want {want}"
                        ));
                    }
                }
            }
        }
    }
    // A join on the stored values: a row per pair `=` holds of.
    for (a_name, a) in &sample {
        for (b_name, b) in &sample {
            if a.is_null() || b.is_null() {
                continue;
            }
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            run(
                &session,
                Language::Gql,
                "INSERT (:A {k: $a}), (:B {k: $b})",
                pair(a, b),
            )
            .unwrap();
            let want = usize::from(truth(&session, Language::Gql, "=", a, b) == Some(true));
            for language in [Language::Gql, Language::Cypher] {
                let found = run(
                    &session,
                    language,
                    "MATCH (x:A), (y:B) WHERE x.k = y.k RETURN x.k",
                    HashMap::new(),
                )
                .map(|rows| rows.len());
                if found != Ok(want) {
                    wrong.push(format!(
                        "{language:?}: a join of {a_name} and {b_name}: {found:?}, want {want}"
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

/// `min` and `max` and `ORDER BY` keep their order over mixed values, which is the order GQL's `<` now
/// follows across kinds.
#[test]
fn gql_orders_across_kinds_as_order_by_does() {
    let sample = sample();
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut wrong = Vec::new();
    for (a_name, a) in &sample {
        for (b_name, b) in &sample {
            if a.is_null() || b.is_null() || outcome(&session, Language::Gql, "=", a, b) == "true" {
                continue;
            }
            let first = run(
                &session,
                Language::Gql,
                "UNWIND [$a, $b] AS x RETURN x ORDER BY x LIMIT 1",
                pair(a, b),
            )
            .unwrap()[0][0]
                .clone();
            let less = truth(&session, Language::Gql, "<", a, b);
            let greater = truth(&session, Language::Gql, ">", a, b);
            // Within a family `<` may be null where ORDER BY still orders (NaN, durations, a list
            // whose deciding pair holds a null): only a definite answer is checked.
            if let (Some(less), Some(greater)) = (less, greater)
                && (less && first != *a || greater && first != *b)
            {
                wrong.push(format!(
                    "{a_name} < {b_name} is {less}, ORDER BY puts {first:?} first"
                ));
            }
            let min = run(
                &session,
                Language::Gql,
                "UNWIND [$a, $b] AS x RETURN min(x) AS m",
                pair(a, b),
            )
            .unwrap()[0][0]
                .clone();
            if min != first {
                wrong.push(format!(
                    "min of {a_name} and {b_name} is {min:?}, ORDER BY puts {first:?} first"
                ));
            }
            let last = run(
                &session,
                Language::Gql,
                "UNWIND [$a, $b] AS x RETURN x ORDER BY x DESC LIMIT 1",
                pair(a, b),
            )
            .unwrap()[0][0]
                .clone();
            let max = run(
                &session,
                Language::Gql,
                "UNWIND [$a, $b] AS x RETURN max(x) AS m",
                pair(a, b),
            )
            .unwrap()[0][0]
                .clone();
            if max != last {
                wrong.push(format!(
                    "max of {a_name} and {b_name} is {max:?}, ORDER BY DESC puts {last:?} first"
                ));
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

/// Inside a list and a map, `NULLIF` and a set operation's key: each holds as `=` of the two values.
#[test]
fn nested_values_nullif_and_set_operations_follow_equality() {
    let sample = sample();
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut wrong = Vec::new();
    for (a_name, a) in &sample {
        for (b_name, b) in &sample {
            for language in [Language::Gql, Language::Cypher] {
                let equal = outcome(&session, language, "=", a, b);
                // A null element is its own question (`[null] = [null]`), outside the rule of kinds.
                let forms: &[&str] = if a.is_null() || b.is_null() {
                    &[]
                } else {
                    &["RETURN [$a] = [$b] AS r", "RETURN {x: $a} = {x: $b} AS r"]
                };
                for form in forms {
                    let found =
                        run(&session, language, form, pair(a, b)).map(|rows| match &rows[0][0] {
                            Value::Bool(flag) => flag.to_string(),
                            Value::Null => "null".to_string(),
                            other => format!("{other:?}"),
                        });
                    if found.as_ref().ok() != Some(&equal) {
                        wrong.push(format!("{language:?}: {form} with {a_name}, {b_name}: {found:?}, `=` gives {equal}"));
                    }
                }
                if a.is_null() || b.is_null() {
                    continue;
                }
                let want = if equal == "true" { 1 } else { 2 };
                let found = run(
                    &session,
                    language,
                    "RETURN $a AS x UNION RETURN $b AS x",
                    pair(a, b),
                )
                .map(|rows| rows.len());
                if found != Ok(want) {
                    wrong.push(format!(
                        "{language:?}: UNION of {a_name} and {b_name}: {found:?}, want {want}"
                    ));
                }
            }
            let equal = outcome(&session, Language::Gql, "=", a, b);
            if a.is_null() {
                continue;
            }
            let found = run(
                &session,
                Language::Gql,
                "RETURN NULLIF($a, $b) IS NULL AS r",
                pair(a, b),
            )
            .map(|rows| rows[0][0].clone());
            if found != Ok(Value::Bool(equal == "true")) {
                wrong.push(format!(
                    "NULLIF({a_name}, {b_name}): {found:?}, `=` gives {equal}"
                ));
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

/// A stored value against another stored value, with the comparison of the two values.
#[test]
fn a_property_against_a_property_answers_as_the_comparison() {
    let sample = sample();
    let mut wrong = Vec::new();
    for (a_name, a) in &sample {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for (b_name, b) in &sample {
            let _ = b_name;
            run(
                &session,
                Language::Gql,
                "INSERT (:A {k: $a}), (:B {k: $b})",
                pair(a, b),
            )
            .unwrap();
        }
        for language in [Language::Gql, Language::Cypher] {
            for op in OPS {
                let want = sample
                    .iter()
                    .filter(|(_, b)| truth(&session, language, op, a, b) == Some(true))
                    .count();
                let query = format!("MATCH (x:A), (y:B) WHERE x.k {op} y.k RETURN count(*) AS c");
                let found =
                    run(&session, language, &query, HashMap::new()).map(|rows| rows[0][0].clone());
                // Every `A` holds `a`: the pairs are `B`'s values times the `A` nodes.
                let a_nodes = i64::try_from(sample.len()).unwrap();
                let want = i64::try_from(want).unwrap() * a_nodes;
                if found != Ok(Value::Int64(want)) {
                    wrong.push(format!(
                        "{language:?}: {a_name} {op} each: {found:?}, want {want}"
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

/// How the key `k` is indexed while `MERGE` and the UNIQUE rule look for its holders.
#[derive(Clone, Copy, Debug)]
enum KeyIndex {
    /// No index: the label is read.
    None,
    /// A property index on `k` (every label).
    Property,
    /// An index on `k` of the label (`CREATE INDEX ... FOR (n:L) ON (n.k)`).
    Label,
}

const KEY_INDEXES: [KeyIndex; 3] = [KeyIndex::None, KeyIndex::Property, KeyIndex::Label];

fn index_key(db: &GrafeoDB, session: &Session, index: KeyIndex, labels: &[&str]) {
    match index {
        KeyIndex::None => {}
        KeyIndex::Property => db.create_property_index("k").unwrap(),
        KeyIndex::Label => {
            for label in labels {
                run(
                    session,
                    Language::Gql,
                    &format!("CREATE INDEX k_of_{label} FOR (n:{label}) ON (n.k)"),
                    HashMap::new(),
                )
                .unwrap();
            }
        }
    }
}

/// Whether a UNIQUE rule takes `a` and `b` for one key: the grouping form (22.12), where two NaNs
/// are one key though `NaN = NaN` is false, else `=` (`equal`). The values here hold NaN only as
/// a float of their own.
fn grouping_same(a: &Value, b: &Value, equal: bool) -> bool {
    match (a, b) {
        (Value::Float64(x), Value::Float64(y)) if x.is_nan() && y.is_nan() => true,
        _ => equal,
    }
}

/// `MERGE` of `a` then `b`, in GQL and in Cypher, makes one node where `=` holds of them and two
/// where it does not (the equality form); a UNIQUE rule with `a` held refuses `b` exactly where the
/// grouping form takes them for one key (two NaNs are, though `NaN = NaN` is false). With the key
/// indexed as `index` before either write.
fn merge_and_unique(pairs: &[(&str, Value, &str, Value)], index: KeyIndex) -> Vec<String> {
    let mut wrong = Vec::new();
    for (a_name, a, b_name, b) in pairs {
        if a.is_null() || b.is_null() {
            continue;
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let equal = outcome(&session, Language::Gql, "=", a, b) == "true";
        index_key(&db, &session, index, &["M", "C"]);
        for (language, merge) in [
            (Language::Gql, "MERGE (n:M {k: $a}) RETURN n.k"),
            (Language::Cypher, "MERGE (n:C {k: $a}) RETURN n.k"),
        ] {
            run(&session, language, merge, pair(a, b)).unwrap();
            run(&session, language, &merge.replace("$a", "$b"), pair(a, b)).unwrap();
        }
        for label in ["M", "C"] {
            let count = run(
                &session,
                Language::Gql,
                &format!("MATCH (n:{label}) RETURN count(n) AS c"),
                HashMap::new(),
            )
            .unwrap()[0][0]
                .clone();
            let want = Value::Int64(if equal { 1 } else { 2 });
            if count != want {
                wrong.push(format!(
                    "{index:?}: MERGE {label} of {a_name} then {b_name}: {count:?}, want {want:?}"
                ));
            }
        }
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        index_key(&db, &session, index, &["U"]);
        run(
            &session,
            Language::Gql,
            "CREATE CONSTRAINT one_k FOR (n:U) ON (n.k) UNIQUE",
            HashMap::new(),
        )
        .unwrap();
        run(&session, Language::Gql, "INSERT (:U {k: $a})", pair(a, b)).unwrap();
        let second = run(&session, Language::Gql, "INSERT (:U {k: $b})", pair(a, b));
        let same = grouping_same(a, b, equal);
        if second.is_ok() == same {
            wrong.push(format!(
                "{index:?}: UNIQUE with {a_name} then {b_name}: {second:?}, one key: {same}"
            ));
        }
    }
    wrong
}

/// `MERGE` finds a node whose key `=` holds of and makes one where it does not; a UNIQUE rule refuses
/// a second value `=` holds of and takes one it does not. One value of each family against each.
#[test]
fn merge_and_unique_keys_follow_equality() {
    let sample = sample();
    let pairs: Vec<_> = sample
        .iter()
        .flat_map(|(a_name, a)| {
            sample
                .iter()
                .map(|(b_name, b)| (*a_name, a.clone(), *b_name, b.clone()))
        })
        .collect();
    let wrong = merge_and_unique(&pairs, KeyIndex::None);
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// The same with and without an index, over every pair within a family, where the values `=` holds
/// of across kinds are (`1` and `1.0`, `0.0` and `-0.0`, `[1]` and `[1.0]`, two zoned datetimes of
/// one instant): an index answers as the label does, so a `MERGE` of `1.0` finds the node of `1`
/// and makes no second one.
#[test]
fn merge_and_unique_keys_follow_equality_with_an_index() {
    let values = values();
    let pairs: Vec<_> = values
        .iter()
        .flat_map(|(a_name, a_family, _, a)| {
            values
                .iter()
                .filter(move |(_, b_family, _, _)| b_family == a_family)
                .map(|(b_name, _, _, b)| (*a_name, a.clone(), *b_name, b.clone()))
        })
        .collect();
    let mut wrong = Vec::new();
    for index in KEY_INDEXES {
        wrong.extend(merge_and_unique(&pairs, index));
    }
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A value changed from a number to text in an open transaction is read as text by its own reader,
/// through the key's column and an index, and as the number by another until the commit.
#[test]
fn a_change_of_kind_in_a_transaction_is_read_by_kind() {
    for indexed in [false, true] {
        let db = GrafeoDB::new_in_memory();
        let mut writer = db.session();
        run(
            &writer,
            Language::Gql,
            "INSERT (:L {row: 0, k: 50}), (:L {row: 1, k: 7})",
            HashMap::new(),
        )
        .unwrap();
        if indexed {
            run(
                &writer,
                Language::Gql,
                "CREATE INDEX by_k FOR (n:L) ON (n.k)",
                HashMap::new(),
            )
            .unwrap();
        }
        writer.begin_transaction().unwrap();
        run(
            &writer,
            Language::Gql,
            "MATCH (n:L {row: 0}) SET n.k = '50'",
            HashMap::new(),
        )
        .unwrap();
        let rows = |session: &Session, language: Language, query: &str| -> Vec<Vec<Value>> {
            run(session, language, query, HashMap::new()).unwrap()
        };
        let row = |at: i64| vec![vec![Value::Int64(at)]];
        let none: Vec<Vec<Value>> = Vec::new();
        for (language, query, own, other) in [
            (
                Language::Gql,
                "MATCH (n:L) WHERE n.k = 50 RETURN n.row",
                none.clone(),
                row(0),
            ),
            (
                Language::Gql,
                "MATCH (n:L {k: '50'}) RETURN n.row",
                row(0),
                none.clone(),
            ),
            (
                Language::Gql,
                "MATCH (n:L) WHERE n.k > 40 RETURN n.row",
                none.clone(),
                row(0),
            ),
            // In GQL text orders before every number.
            (
                Language::Gql,
                "MATCH (n:L) WHERE n.k < 5 RETURN n.row",
                row(0),
                none.clone(),
            ),
            (
                Language::Cypher,
                "MATCH (n:L) WHERE n.k < 5 RETURN n.row",
                none.clone(),
                none.clone(),
            ),
            (
                Language::Cypher,
                "MATCH (n:L) WHERE n.k = 50 RETURN n.row",
                none.clone(),
                row(0),
            ),
        ] {
            assert_eq!(
                rows(&writer, language, query),
                own,
                "index {indexed}, {language:?}, own: {query}"
            );
            assert_eq!(
                rows(&db.session(), language, query),
                other,
                "index {indexed}, {language:?}, another: {query}"
            );
        }
        writer.commit().unwrap();
        assert_eq!(
            rows(
                &db.session(),
                Language::Gql,
                "MATCH (n:L {k: '50'}) RETURN n.row"
            ),
            row(0),
            "index {indexed}, committed"
        );
    }
}

/// Every form above once more after a reopen, where the values are read from cold chunks.
#[test]
fn every_form_answers_as_the_comparison_reopened() {
    let sample = sample();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("kinds.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = db.session();
        for (row, (_, value)) in sample.iter().enumerate() {
            let props = HashMap::from([
                ("row".to_string(), Value::Int64(i64::try_from(row).unwrap())),
                ("v".to_string(), value.clone()),
            ]);
            run(
                &session,
                Language::Gql,
                "INSERT (:L {row: $row, k: $v})",
                props,
            )
            .unwrap();
        }
        drop(session);
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    let mut wrong = Vec::new();
    for language in [Language::Gql, Language::Cypher] {
        for (b_name, b) in &sample {
            for op in OPS {
                let want: Vec<i64> = sample
                    .iter()
                    .enumerate()
                    .filter(|(_, (_, a))| truth(&session, language, op, a, b) == Some(true))
                    .map(|(row, _)| i64::try_from(row).unwrap())
                    .collect();
                let query = format!("MATCH (n:L) WHERE n.k {op} $b RETURN n.row ORDER BY n.row");
                let params = HashMap::from([("b".to_string(), b.clone())]);
                let found = run(&session, language, &query, params).map(|rows| {
                    rows.iter()
                        .map(|row| match row[0] {
                            Value::Int64(at) => at,
                            ref other => panic!("{other:?}"),
                        })
                        .collect::<Vec<_>>()
                });
                if found.as_ref() != Ok(&want) {
                    wrong.push(format!(
                        "{language:?}: {query} with {b_name}: {found:?}, want {want:?}"
                    ));
                }
            }
        }
    }
    drop(session);
    db.close().unwrap();
    assert!(
        wrong.is_empty(),
        "{} wrong:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// SPARQL compares terms by its own rules, which the change leaves alone: `"5"` and `5` stay two
/// terms in a `FILTER`, a `DISTINCT` and a join.
#[test]
fn sparql_keeps_text_and_numbers_apart() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute_sparql("INSERT DATA { <urn:a> <urn:p> \"5\" . <urn:b> <urn:p> 5 . <urn:c> <urn:q> \"5\" . <urn:d> <urn:q> 5 . <urn:e> <urn:p> 5.0 }")
        .unwrap();
    let count = |query: &str| session.execute_sparql(query).unwrap().row_count();
    assert_eq!(
        count("SELECT DISTINCT ?o WHERE { ?s <urn:p> ?o }"),
        3,
        "\"5\", 5 and 5.0 are three terms"
    );
    assert_eq!(
        count("SELECT ?s WHERE { ?s <urn:p> ?o FILTER(?o = 5) }"),
        2,
        "5 and 5.0 by value, not the text"
    );
    assert_eq!(
        count("SELECT ?s WHERE { ?s <urn:p> ?o FILTER(?o = \"5\") }"),
        1,
        "the text alone"
    );
    assert_eq!(
        count("SELECT ?s WHERE { ?s <urn:p> ?o FILTER(?o > 4) }"),
        2,
        "the numbers alone"
    );
    assert_eq!(
        count("SELECT ?x ?y WHERE { ?x <urn:p> ?o . ?y <urn:q> ?o }"),
        2,
        "a join pairs a term with the same term only"
    );
}
