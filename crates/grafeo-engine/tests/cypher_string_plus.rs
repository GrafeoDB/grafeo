//! `+` with a string in Cypher concatenates both ways (GA04 step b, the ruling of 2026-11-06, after
//! openCypher): `1 + 'a'` is `'1a'`, `'a' + 1.5` is `'a1.5'`, and the number's text is the text of
//! `toString()`, so the two can never disagree. A string with a list is list concatenation, an entity
//! is an error and a null gives null. GQL refuses `+` of a string and any value but a string or a
//! null (22G12, the user's decisions of 2026-10-30 and 2026-11-06: ISO's `+` takes numbers, or a
//! datetime with a duration), on either side. 0.6.1 gave null in Cypher. Booleans, temporals and
//! durations stay as they are in Cypher: the openCypher TCK's string concatenation feature
//! (`String6.feature`) holds no scenario that pins them. The class cases at the end run every kind
//! both ways, from parameters and from a stored property.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main cypher_string_plus::
//! ```

use std::collections::HashMap;

use grafeo_common::types::{Decimal, Value};
use grafeo_common::utils::gqlstatus::GqlStatus;
use grafeo_engine::GrafeoDB;
use grafeo_engine::session::Session;

fn cypher(session: &Session, query: &str, params: HashMap<String, Value>) -> Value {
    session
        .execute_cypher_with_params(query, params)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()[0][0]
        .clone()
}

fn text(value: &str) -> Value {
    Value::from(value)
}

#[test]
fn a_number_and_a_string_concatenate_both_ways() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (expression, want) in [
        ("1 + 'a'", text("1a")),
        ("'a' + 1", text("a1")),
        ("'a' + 1.5", text("a1.5")),
        ("1.5 + 'a'", text("1.5a")),
        ("-3 + 'a'", text("-3a")),
        ("'a' + -3", text("a-3")),
        ("'a' + 0", text("a0")),
        ("'' + 7", text("7")),
        ("7 + ''", text("7")),
        ("'a' + 1 + 2", text("a12")),
        ("1 + 2 + 'a'", text("3a")),
        ("'a' + (1 + 2)", text("a3")),
        ("'Alix' + 'Gus'", text("AlixGus")),
    ] {
        let found = cypher(
            &session,
            &format!("RETURN {expression} AS v"),
            HashMap::new(),
        );
        assert_eq!(found, want, "{expression}");
    }
}

/// A float's text in `+` is `toString()`'s, for every shape of float: whole, fractional, large,
/// small, negative, signed zero, NaN and the infinities, from a literal, a parameter and a property.
#[test]
fn a_float_takes_the_text_of_to_string() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let floats = [
        1.0,
        -1.0,
        0.1,
        2.5,
        1.0e20,
        1.0e-7,
        123_456_789.125,
        -0.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
    ];
    for x in floats {
        let params = HashMap::from([("x".to_string(), Value::Float64(x))]);
        let row = session
            .execute_cypher_with_params(
                "RETURN $x + 'a' AS l, toString($x) + 'a' AS lt, 'a' + $x AS r, 'a' + toString($x) AS rt",
                params,
            )
            .unwrap_or_else(|error| panic!("{x}: {error}"))
            .rows()[0]
            .clone();
        assert_eq!(row[0], row[1], "{x} + 'a'");
        assert_eq!(row[2], row[3], "'a' + {x}");
        assert!(matches!(row[0], Value::String(_)), "{x}: {:?}", row[0]);
    }
    // A stored float, and integers alike.
    session
        .execute_cypher("CREATE (:F {f: 2.0, i: 42, n: -7})")
        .unwrap();
    let row = session
        .execute_cypher(
            "MATCH (x:F) RETURN x.f + 'a' = toString(x.f) + 'a' AS f, \
             'a' + x.i = 'a' + toString(x.i) AS i, x.n + '' = toString(x.n) AS n",
        )
        .unwrap()
        .rows()[0]
        .clone();
    assert_eq!(
        row,
        vec![Value::Bool(true), Value::Bool(true), Value::Bool(true)]
    );
    // The literal 1.0 too: `1.0 + 'a'` and `toString(1.0) + 'a'` are one text.
    assert_eq!(
        cypher(
            &session,
            "RETURN 1.0 + 'a' = toString(1.0) + 'a' AS v",
            HashMap::new()
        ),
        Value::Bool(true)
    );
}

/// A string with a list is list concatenation (openCypher, `List4.feature`), not text.
#[test]
fn a_string_and_a_list_concatenate_as_lists() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (expression, want) in [
        (
            "'a' + [1]",
            Value::List(vec![text("a"), Value::Int64(1)].into()),
        ),
        (
            "[1] + 'a'",
            Value::List(vec![Value::Int64(1), text("a")].into()),
        ),
        ("'a' + []", Value::List(vec![text("a")].into())),
        (
            "['Alix'] + 'Gus'",
            Value::List(vec![text("Alix"), text("Gus")].into()),
        ),
    ] {
        let found = cypher(
            &session,
            &format!("RETURN {expression} AS v"),
            HashMap::new(),
        );
        assert_eq!(found, want, "{expression}");
    }
}

#[test]
fn a_null_gives_null() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for expression in ["null + 'a'", "'a' + null", "null + 1 + 'a'", "'a' + $p"] {
        let params = HashMap::from([("p".to_string(), Value::Null)]);
        let found = cypher(&session, &format!("RETURN {expression} AS v"), params);
        assert_eq!(found, Value::Null, "{expression}");
    }
    // A missing property reads null.
    session.execute_cypher("CREATE (:M {name: 'Mia'})").unwrap();
    assert_eq!(
        cypher(
            &session,
            "MATCH (m:M) RETURN m.age + 'a' AS v",
            HashMap::new()
        ),
        Value::Null
    );
}

/// A node, an edge or a path with a string is an error, whichever side it stands on.
#[test]
fn an_entity_with_a_string_is_an_error() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute_cypher("CREATE (:P {name: 'Vincent'})-[:K]->(:P {name: 'Jules'})")
        .unwrap();
    for expression in [
        "n + 'a'", "'a' + n", "r + 'a'", "'a' + r", "p + 'a'", "'a' + p",
    ] {
        let query =
            format!("MATCH p = (n:P {{name: 'Vincent'}})-[r:K]->() RETURN {expression} AS v");
        let error = session
            .execute_cypher(&query)
            .expect_err(&format!("{expression} is not text"));
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            "22G12",
            "{expression}: {error}"
        );
    }
}

/// GQL keeps its rule: `+` of a string and a number raises 22G12, both ways, for a literal, a
/// parameter and a property.
#[test]
fn gql_refuses_a_string_and_a_number() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("INSERT (:F {f: 2.5, s: 'a'})").unwrap();
    for query in [
        "RETURN 1 + 'a' AS v",
        "RETURN 'a' + 1 AS v",
        "RETURN 'a' + 1.5 AS v",
        "RETURN $n + 'a' AS v",
        "MATCH (x:F) RETURN x.f + x.s AS v",
        "MATCH (x:F) RETURN x.s + x.f AS v",
    ] {
        let params = HashMap::from([("n".to_string(), Value::Int64(1))]);
        let error = session
            .execute_with_params(query, params)
            .expect_err(&format!("{query} has no value in GQL"));
        assert_eq!(
            GqlStatus::from(&error).as_str(),
            "22G12",
            "{query}: {error}"
        );
    }
}

/// One stored graph, both languages: Cypher's text, GQL's error.
#[test]
fn one_graph_answers_each_language_by_its_rule() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:F {i: 3, s: 'x'}), (:F {i: 4, s: 'y'})")
        .unwrap();
    let mut texts: Vec<String> = session
        .execute_cypher("MATCH (n:F) RETURN n.s + n.i AS v")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].as_str().unwrap().to_string())
        .collect();
    texts.sort();
    assert_eq!(texts, ["x3", "y4"]);
    let error = session
        .execute("MATCH (n:F) RETURN n.s + n.i AS v")
        .expect_err("GQL refuses");
    assert_eq!(GqlStatus::from(&error).as_str(), "22G12", "{error}");
}

// ---------------------------------------------------------------------------
// Every kind beside a string
// ---------------------------------------------------------------------------

/// The `Value` variants, from the catalogue that grafeo-common's exhaustive match holds to the
/// enum (class test 0): a variant added to `Value` fails that crate's tests until it is named in
/// the catalogue, and then [`probe`] here until it has a row.
const CATALOGUE: &str = include_str!("../../../tests/catalogue/value.txt");

fn catalogue_names() -> Vec<&'static str> {
    CATALOGUE
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

/// What a variant brings beside a string: a value to probe, a case of its own, or no way into an
/// expression.
enum Probe {
    Value(Value),
    OwnCase(&'static str),
    NotReachable(&'static str),
}

/// The row of each catalogue variant; the temporal values as their literals give them.
fn probe(db: &GrafeoDB, name: &str) -> Probe {
    let literal = |text: &str| {
        Probe::Value(
            db.execute(&format!("RETURN {text} AS v"))
                .unwrap_or_else(|error| panic!("{text}: {error}"))
                .rows()[0][0]
                .clone(),
        )
    };
    match name {
        "Null" => Probe::OwnCase("a null gives null: `a_null_gives_null` and the GQL class test"),
        "Bool" => Probe::Value(Value::Bool(true)),
        "Int64" => Probe::Value(Value::Int64(7)),
        "Float64" => Probe::Value(Value::Float64(2.5)),
        "Decimal" => Probe::Value(Value::Decimal(Decimal::parse("1.5").unwrap())),
        "String" => Probe::OwnCase("two strings concatenate: the module's cases"),
        "Bytes" => Probe::Value(Value::Bytes(vec![1u8, 2].into())),
        "Timestamp" => literal("DATETIME '2024-01-02T10:00:00'"),
        "Date" => literal("DATE '2024-01-02'"),
        "Time" => literal("TIME '10:00:00'"),
        "Duration" => literal("DURATION 'P1D'"),
        "ZonedDatetime" => literal("ZONED DATETIME '2024-01-02T10:00:00+01:00'"),
        "List" => Probe::Value(Value::List(vec![Value::Int64(1)].into())),
        "Map" => literal("{k: 1}"),
        "Vector" => Probe::Value(Value::Vector(vec![1.0f32, 2.0].into())),
        "Path" => Probe::OwnCase(
            "a path, with a node and an edge: `an_entity_beside_a_string_is_refused_in_both_languages`",
        ),
        "GCounter" | "OnCounter" => Probe::NotReachable(
            "a counter lives inside a CRDT property and reaches no expression as a value",
        ),
        "RdfTerm" => Probe::NotReachable("an RDF term reaches no LPG expression"),
        other => panic!("`{other}` is new in the catalogue: give it a probe or a reason here"),
    }
}

/// The catalogue's variants that have a value to probe, by name.
fn kinds(db: &GrafeoDB) -> Vec<(&'static str, Value)> {
    catalogue_names()
        .into_iter()
        .filter_map(|name| match probe(db, name) {
            Probe::Value(value) => Some((name, value)),
            Probe::OwnCase(_) | Probe::NotReachable(_) => None,
        })
        .collect()
}

/// Every catalogue variant has a row, and each one without a probe says why.
#[test]
fn every_variant_of_the_catalogue_has_a_row() {
    let db = GrafeoDB::new_in_memory();
    let names = catalogue_names();
    assert!(
        names.contains(&"Bool") && names.contains(&"Path"),
        "{names:?}"
    );
    for name in names {
        match probe(&db, name) {
            Probe::Value(_) => {}
            Probe::OwnCase(reason) | Probe::NotReachable(reason) => {
                assert!(!reason.is_empty(), "{name}: a reason");
            }
        }
    }
    assert_eq!(kinds(&db).len(), 13, "the probed variants");
}

/// `+` of the string `'x'` and `value` in `language`: `[parameters, string first; parameters,
/// string last; stored, string first; stored, string last]`, an error as its GQLSTATUS.
fn plus_both_ways(db: &GrafeoDB, language: &str, value: &Value) -> Vec<Result<Value, String>> {
    db.execute("MATCH (n:Probe) DETACH DELETE n").unwrap();
    db.execute_with_params(
        "INSERT (:Probe {s: 'x', p: $x})",
        HashMap::from([("x".to_string(), value.clone())]),
    )
    .unwrap();
    let params = HashMap::from([
        ("s".to_string(), text("x")),
        ("x".to_string(), value.clone()),
    ]);
    let run = |query: &str, params: Option<HashMap<String, Value>>| {
        db.execute_language(query, language, params)
            .map(|result| result.rows()[0][0].clone())
            .map_err(|error| GqlStatus::from(&error).as_str().to_string())
    };
    vec![
        run("RETURN $s + $x AS v", Some(params.clone())),
        run("RETURN $x + $s AS v", Some(params)),
        run("MATCH (n:Probe) RETURN n.s + n.p AS v", None),
        run("MATCH (n:Probe) RETURN n.p + n.s AS v", None),
    ]
}

/// GQL: a string and any value but a string or a null raise 22G12, either way round, from a
/// parameter and from a property; a null gives null.
#[test]
fn gql_refuses_a_string_beside_every_other_kind() {
    let db = GrafeoDB::new_in_memory();
    for (kind, value) in kinds(&db) {
        assert_eq!(
            plus_both_ways(&db, "gql", &value),
            vec![Err("22G12".to_string()); 4],
            "{kind}"
        );
    }
    assert_eq!(
        plus_both_ways(&db, "gql", &Value::Null)[..2],
        [Ok(Value::Null), Ok(Value::Null)]
    );
}

/// Cypher's answer for each kind, string first and string last. A number becomes its
/// `toString()` text and a list joins (the ruling); the rest is today's answer, which no TCK
/// scenario pins: a boolean joins only after the string, and a temporal, a duration, a map, bytes
/// and a vector give null.
#[test]
fn cypher_answers_each_kind_as_it_does_today() {
    let db = GrafeoDB::new_in_memory();
    let null = || Ok(Value::Null);
    let joined = |s: &str| Ok(text(s));
    for (kind, value) in kinds(&db) {
        let (first, last) = match kind {
            "Bool" => (joined("xtrue"), null()),
            "Int64" => (joined("x7"), joined("7x")),
            "Float64" => (joined("x2.5"), joined("2.5x")),
            "Decimal" => (joined("x1.5"), joined("1.5x")),
            "List" => (
                Ok(Value::List(vec![text("x"), Value::Int64(1)].into())),
                Ok(Value::List(vec![Value::Int64(1), text("x")].into())),
            ),
            _ => (null(), null()),
        };
        assert_eq!(
            plus_both_ways(&db, "cypher", &value),
            vec![first.clone(), last.clone(), first, last],
            "{kind}"
        );
    }
}

/// A node, an edge and a path beside a string raise 22G12 in both languages, either way round.
#[test]
fn an_entity_beside_a_string_is_refused_in_both_languages() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:E {s: 'x'})-[:R]->(:E {s: 'y'})")
        .unwrap();
    for language in ["gql", "cypher"] {
        for expression in [
            "n.s + n", "n + n.s", "n.s + r", "r + n.s", "n.s + p", "p + n.s",
        ] {
            let query = format!("MATCH p = (n:E)-[r:R]->() RETURN {expression} AS v");
            let error = db
                .execute_language(&query, language, None)
                .expect_err(&format!("{language}: {expression}"));
            assert_eq!(
                GqlStatus::from(&error).as_str(),
                "22G12",
                "{language}: {expression}: {error}"
            );
        }
    }
}
