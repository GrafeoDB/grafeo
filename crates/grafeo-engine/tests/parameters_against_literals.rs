//! Class test 6: a parameter answers as the same value written as a literal, in every position an
//! expression can hold it, in GQL and Cypher (class-tests design, 2026-11-05).
//!
//! The positions are keyed by the `LogicalExpression` catalogue (`tests/catalogue/logical_expression.txt`):
//! each variant has templates that put the value where that expression holds it, or "not reachable"
//! with the reason. Clause positions (LIMIT, ORDER BY, UNWIND, a grouping key, an aggregate's argument)
//! follow, then the grouping keys built from a parameter. Each template runs once with the literal
//! spelled in and once with `$p`: both fail, or both give the same rows.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main parameters_against_literals::
//! ```

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use grafeo_common::types::{Date, Decimal, Duration, PropertyKey, Time, Value, ZonedDatetime};
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

const EXPRESSIONS: &str = include_str!("../../../tests/catalogue/logical_expression.txt");

#[derive(Clone, Copy, Debug, PartialEq)]
enum Language {
    Gql,
    Cypher,
}

const BOTH: &[Language] = &[Language::Gql, Language::Cypher];
const CYPHER: &[Language] = &[Language::Cypher];

fn list(items: Vec<Value>) -> Value {
    Value::List(items.into())
}

fn map(pairs: &[(&str, Value)]) -> Value {
    Value::Map(Arc::new(
        pairs
            .iter()
            .map(|(key, value)| (PropertyKey::new(*key), value.clone()))
            .collect::<BTreeMap<_, _>>(),
    ))
}

/// A value and its literal in GQL and in Cypher; `None` where the language has no literal for it.
struct Spelled {
    value: Value,
    gql: Option<&'static str>,
    cypher: Option<&'static str>,
}

fn spelled(value: Value, gql: Option<&'static str>, cypher: Option<&'static str>) -> Spelled {
    Spelled { value, gql, cypher }
}

fn values() -> Vec<Spelled> {
    vec![
        spelled(Value::Null, Some("null"), Some("null")),
        spelled(Value::Bool(true), Some("true"), Some("true")),
        spelled(Value::Bool(false), Some("false"), Some("false")),
        spelled(Value::Int64(1), Some("1"), Some("1")),
        spelled(Value::Int64(0), Some("0"), Some("0")),
        spelled(Value::Int64(-1), Some("-1"), Some("-1")),
        spelled(Value::Float64(1.5), Some("1.5e0"), Some("1.5")),
        spelled(Value::Float64(2.0), Some("2.0e0"), Some("2.0")),
        // Cypher has no decimal literal.
        spelled(
            Value::Decimal(Decimal::parse("1.5").unwrap()),
            Some("1.5"),
            None,
        ),
        spelled(Value::from("a"), Some("'a'"), Some("'a'")),
        spelled(Value::from(""), Some("''"), Some("''")),
        spelled(Value::from("1"), Some("'1'"), Some("'1'")),
        spelled(
            Value::Date(Date::parse("2024-01-01").unwrap()),
            Some("DATE '2024-01-01'"),
            Some("date('2024-01-01')"),
        ),
        spelled(
            Value::Time(Time::parse("10:00:00").unwrap()),
            Some("TIME '10:00:00'"),
            Some("localtime('10:00:00')"),
        ),
        spelled(
            Value::Time(Time::parse("10:00:00+01:00").unwrap()),
            Some("TIME '10:00:00+01:00'"),
            Some("time('10:00:00+01:00')"),
        ),
        spelled(
            Value::ZonedDatetime(ZonedDatetime::parse("2024-01-01T10:00:00+01:00").unwrap()),
            Some("ZONED DATETIME '2024-01-01T10:00:00+01:00'"),
            Some("datetime('2024-01-01T10:00:00+01:00')"),
        ),
        spelled(
            Value::Duration(Duration::parse("P1D").unwrap()),
            Some("DURATION 'P1D'"),
            Some("duration('P1D')"),
        ),
        spelled(
            list(vec![Value::Int64(1), Value::Int64(2)]),
            Some("[1, 2]"),
            Some("[1, 2]"),
        ),
        spelled(list(vec![]), Some("[]"), Some("[]")),
        spelled(
            list(vec![Value::from("a"), Value::Null]),
            Some("['a', null]"),
            Some("['a', null]"),
        ),
        spelled(
            map(&[("x", Value::Int64(1))]),
            Some("{x: 1}"),
            Some("{x: 1}"),
        ),
        spelled(
            Value::Vector(vec![1.0_f32, 2.0].into()),
            Some("vector([1.0e0, 2.0e0])"),
            Some("vector([1.0, 2.0])"),
        ),
    ]
}

/// Where a `LogicalExpression` variant holds a value: templates with `@` for it, or why none can.
enum Position {
    Templates(Vec<(&'static str, &'static [Language])>),
    NotReachable(&'static str),
}

fn position(variant: &str) -> Position {
    use Position::{NotReachable, Templates};
    match variant {
        "Literal" => Templates(vec![("RETURN @ AS r", BOTH)]),
        "Parameter" => NotReachable("the parameter itself: every template here reads one"),
        "Variable" => Templates(vec![("WITH @ AS v RETURN v AS r", BOTH)]),
        "Property" => Templates(vec![
            (
                "MATCH (n:P) WHERE n.k = @ RETURN n.name AS r ORDER BY r",
                BOTH,
            ),
            ("MATCH (n:P {k: @}) RETURN n.name AS r ORDER BY r", BOTH),
            (
                "MATCH (n) WHERE n.k = @ RETURN n.name AS r ORDER BY r",
                BOTH,
            ),
        ]),
        "Binary" => Templates(vec![
            ("RETURN @ = @ AS r", BOTH),
            ("RETURN @ <> 1 AS r", BOTH),
            ("RETURN @ + 1 AS r", BOTH),
            ("RETURN @ * 2 AS r", BOTH),
            (
                "MATCH (n:P) WHERE n.k > @ RETURN n.name AS r ORDER BY r",
                BOTH,
            ),
            (
                "MATCH (n:P) WHERE n.k IN [@, 3] RETURN n.name AS r ORDER BY r",
                BOTH,
            ),
            (
                "MATCH (n:P) WHERE @ IN n.l RETURN n.name AS r ORDER BY r",
                BOTH,
            ),
            (
                "MATCH (n:P) WHERE n.s = @ OR n.k = @ RETURN n.name AS r ORDER BY r",
                BOTH,
            ),
        ]),
        "Unary" => Templates(vec![
            ("RETURN @ IS NULL AS r", BOTH),
            ("RETURN -(@) AS r", BOTH),
            ("RETURN NOT (@) AS r", BOTH),
        ]),
        "FunctionCall" => Templates(vec![
            ("RETURN toString(@) AS r", BOTH),
            ("RETURN size(@) AS r", BOTH),
            ("RETURN coalesce(@, 7) AS r", BOTH),
            ("RETURN range(0, @) AS r", BOTH),
        ]),
        "SparqlCall" => NotReachable("SPARQL's functions: SPARQL has no parameters"),
        "List" => Templates(vec![
            ("RETURN [@] AS r", BOTH),
            ("RETURN [@, @] AS r", BOTH),
        ]),
        "Map" => Templates(vec![("RETURN {v: @} AS r", BOTH)]),
        "IndexAccess" => Templates(vec![
            ("RETURN [1, 2, 3][@] AS r", BOTH),
            ("RETURN [@][0] AS r", BOTH),
        ]),
        "MapAccess" => Templates(vec![("RETURN {v: @}.v AS r", BOTH)]),
        "SliceAccess" => Templates(vec![("RETURN [1, 2, 3][0..@] AS r", BOTH)]),
        "Case" => Templates(vec![
            (
                "MATCH (n:P) RETURN n.name AS r, CASE n.k WHEN @ THEN 'yes' ELSE 'no' END AS c ORDER BY r",
                BOTH,
            ),
            ("RETURN CASE WHEN @ IS NULL THEN 0 ELSE @ END AS r", BOTH),
        ]),
        "Labels" | "Type" | "Id" => NotReachable("it reads an element, which no literal spells"),
        "ListComprehension" => Templates(vec![("RETURN [x IN [1, 2] | [x, @]] AS r", BOTH)]),
        "ListPredicate" => Templates(vec![("RETURN any(x IN [1, 2] WHERE x = @) AS r", BOTH)]),
        "ExistsSubquery" => Templates(vec![(
            "MATCH (n:P) WHERE EXISTS { MATCH (m:Q) WHERE m.k = @ } RETURN n.name AS r ORDER BY r",
            BOTH,
        )]),
        "CountSubquery" => Templates(vec![(
            "MATCH (n:P) RETURN n.name AS r, COUNT { MATCH (m:Q) WHERE m.k = @ } AS c ORDER BY r",
            BOTH,
        )]),
        "ValueSubquery" => Templates(vec![(
            "RETURN VALUE { MATCH (m:Q) WHERE m.k = @ RETURN m.name } AS r",
            &[Language::Gql],
        )]),
        "MapProjection" => Templates(vec![("MATCH (n:Q) RETURN n {.name, v: @} AS r", CYPHER)]),
        "Reduce" => Templates(vec![("RETURN reduce(s = @, x IN [1, 2] | s) AS r", BOTH)]),
        "PatternComprehension" => Templates(vec![(
            "MATCH (n:P) RETURN n.name AS r, [(m:Q) WHERE m.k = @ | m.name] AS ms ORDER BY r",
            CYPHER,
        )]),
        other => panic!("the catalogue names {other}: give it a position here"),
    }
}

/// Positions in clauses, which no expression variant names.
const CLAUSES: &[&str] = &[
    "MATCH (n:P) RETURN n.name AS r ORDER BY r LIMIT @",
    "MATCH (n:P) RETURN n.name AS r ORDER BY r SKIP @",
    "MATCH (n:P) RETURN n.name AS r ORDER BY n.k * @, r",
    "UNWIND @ AS x RETURN x AS r",
    "MATCH (n:P) WITH n, @ AS v WHERE n.k > v RETURN n.name AS r ORDER BY r",
    "MATCH (n:P) RETURN count(n) AS c, @ AS r",
    "UNWIND [1, 2] AS i RETURN @ AS x, count(*) AS c",
    "UNWIND [1, 2] AS i RETURN collect(@) AS r",
    "UNWIND [1, 2] AS i RETURN count(@) AS r",
];

const DATA: &str = "INSERT (:P {name: 'Alix', k: 1, s: 'a', l: [1, 2]}), \
     (:P {name: 'Gus', k: 2, s: 'b', l: [3]}), (:P {name: 'Mia', k: 3, s: 'c'}), \
     (:Q {name: 'Vincent', k: 1})";

fn catalogue_names() -> Vec<&'static str> {
    EXPRESSIONS
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

fn outcome(
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

/// The template with the value as a literal, and with it as `$p`: what differs.
fn compare(
    session: &Session,
    template: &str,
    languages: &[Language],
    wrong: &mut Vec<String>,
) -> usize {
    let mut pairs = 0;
    for &language in languages {
        for value in values() {
            let literal = match language {
                Language::Gql => value.gql,
                Language::Cypher => value.cypher,
            };
            let Some(literal) = literal else { continue };
            let spelled = template.replace('@', literal);
            let parametric = template.replace('@', "$p");
            let a = outcome(session, language, &spelled, HashMap::new());
            let b = outcome(
                session,
                language,
                &parametric,
                HashMap::from([("p".to_string(), value.value.clone())]),
            );
            pairs += 1;
            let same = match (&a, &b) {
                (Ok(x), Ok(y)) => x == y,
                (Err(_), Err(_)) => true,
                _ => false,
            };
            if !same {
                wrong.push(format!(
                    "{language:?}: {template} with {literal}: literal {a:?}, parameter {b:?}"
                ));
            }
        }
    }
    pairs
}

#[test]
fn every_expression_variant_has_a_position() {
    for name in catalogue_names() {
        match position(name) {
            Position::Templates(templates) => assert!(!templates.is_empty(), "{name}"),
            Position::NotReachable(reason) => assert!(!reason.is_empty(), "{name}"),
        }
    }
}

#[test]
fn a_parameter_answers_as_its_literal_in_every_expression() {
    let db = GrafeoDB::new_in_memory();
    db.execute(DATA).unwrap();
    let session = db.session();
    let mut wrong = Vec::new();
    let mut pairs = 0;
    for name in catalogue_names() {
        if let Position::Templates(templates) = position(name) {
            for (template, languages) in templates {
                pairs += compare(&session, template, languages, &mut wrong);
            }
        }
    }
    for template in CLAUSES {
        pairs += compare(&session, template, BOTH, &mut wrong);
    }
    assert!(pairs > 1_000, "{pairs} pairs");
    assert!(
        wrong.is_empty(),
        "{} of {pairs} differ:\n{}",
        wrong.len(),
        wrong.join("\n")
    );
}

/// A grouping key built from a parameter groups as the same key written as a literal (2026-11-06;
/// it failed as an undefined variable).
#[test]
fn a_grouping_key_built_from_a_parameter_groups_as_its_literal() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let mut wrong = Vec::new();
    for (parametric, params, literal) in [
        (
            "UNWIND [1, 2] AS i RETURN $m.k AS x, count(*) AS c",
            vec![("m", map(&[("k", Value::Int64(1))]))],
            "UNWIND [1, 2] AS i RETURN {k: 1}.k AS x, count(*) AS c",
        ),
        (
            "UNWIND [1, 2] AS i RETURN $vals[0] AS x, count(*) AS c",
            vec![("vals", list(vec![Value::Int64(5), Value::Int64(6)]))],
            "UNWIND [1, 2] AS i RETURN [5, 6][0] AS x, count(*) AS c",
        ),
        (
            "UNWIND [1, 2] AS i RETURN $p + 1 AS x, count(*) AS c",
            vec![("p", Value::Int64(1))],
            "UNWIND [1, 2] AS i RETURN 1 + 1 AS x, count(*) AS c",
        ),
        (
            "UNWIND [1, 2] AS i RETURN size($l) AS x, count(*) AS c",
            vec![("l", list(vec![Value::Int64(1), Value::Int64(2)]))],
            "UNWIND [1, 2] AS i RETURN size([1, 2]) AS x, count(*) AS c",
        ),
        (
            "UNWIND [1, 2] AS i RETURN $vals[i - 1] AS x, count(*) AS c ORDER BY x",
            vec![("vals", list(vec![Value::Int64(5), Value::Int64(6)]))],
            "UNWIND [1, 2] AS i RETURN [5, 6][i - 1] AS x, count(*) AS c ORDER BY x",
        ),
    ] {
        for language in [Language::Gql, Language::Cypher] {
            let params: HashMap<String, Value> = params
                .iter()
                .map(|(name, value)| ((*name).to_string(), value.clone()))
                .collect();
            let a = outcome(&session, language, literal, HashMap::new());
            let b = outcome(&session, language, parametric, params);
            if a.is_err() || a != b {
                wrong.push(format!(
                    "{language:?}: {parametric}: {b:?}, the literal {literal}: {a:?}"
                ));
            }
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
