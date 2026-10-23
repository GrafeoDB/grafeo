//! A statement with values is optimized once with its parameters in, and
//! each call fills a copy of that prepared plan with its own values
//! (DuckDB's prepared statements): every call answers as the statement with
//! its values written as literals answers, in GQL, Cypher and SQL/PGQ; no
//! cached plan ever holds a value; a change of the indexes
//! prepares it again; and a statement whose binding depends on a value
//! (a WITH or RETURN item that is a parameter alone) is bound with its
//! values on every call.
//!
//! cargo test -p grafeo-engine --all-features --test prepared_plans

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;
use grafeo_engine::query::{CacheKey, QueryLanguage};

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

/// People with a name, an age and a city, and who knows whom.
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (alix:Person {name: 'Alix', age: 30, city: 'Amsterdam'}), \
                (gus:Person {name: 'Gus', age: 25, city: 'Berlin'}), \
                (vincent:Person {name: 'Vincent', age: 41, city: 'Paris'}), \
                (jules:Person {name: 'Jules', age: 41, city: 'Prague'}), \
                (mia:Person {name: 'Mia', age: 25, city: 'Amsterdam'}), \
                (alix)-[:KNOWS]->(gus), (alix)-[:KNOWS]->(mia), \
                (gus)-[:KNOWS]->(vincent), (mia)-[:KNOWS]->(jules)",
    )
    .unwrap();
    db
}

/// The rows of `result`, sorted, so that two runs compare whatever their
/// order.
fn sorted(rows: &[Vec<Value>]) -> Vec<Vec<Value>> {
    let mut rows = rows.to_vec();
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

/// `template` with each `$name` written as the literal of its value (a
/// float as a conversion: a number with a point is a decimal in GQL), in
/// GQL, Cypher and SQL/PGQ syntax alike.
fn with_literals(template: &str, values: &HashMap<String, Value>) -> String {
    let mut names: Vec<&String> = values.keys().collect();
    // The longest name first, so `$ab` is not taken for `$a`.
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut text = template.to_string();
    for name in names {
        let literal = match &values[name] {
            Value::String(text) => format!("'{text}'"),
            Value::Int64(number) => number.to_string(),
            Value::Float64(number) => format!("toFloat({number:?})"),
            Value::Bool(flag) => flag.to_string(),
            other => panic!("no literal for {other:?}"),
        };
        text = text.replace(&format!("${name}"), &literal);
    }
    text
}

/// The prepared-plan hits and misses so far.
fn lookups(db: &GrafeoDB) -> (u64, u64) {
    let stats = db.query_cache().stats();
    (stats.optimized_hits, stats.optimized_misses)
}

/// The statements and the values each call takes: lookups through an index
/// and without, a pattern, a relationship MERGE, an aggregate, SKIP and
/// LIMIT, a value read in the RETURN.
const GQL: [&str; 6] = [
    "MATCH (p:Person {name: $name}) RETURN p.age",
    "MATCH (p:Person) WHERE p.age > $least AND p.city <> $city RETURN p.name",
    "MATCH (a:Person {name: $name})-[:KNOWS]->(b:Person) WHERE b.age >= $least RETURN b.name",
    "MATCH (p:Person) WHERE p.age = $age RETURN p.city, count(p) AS n",
    "MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP $skip LIMIT $limit",
    "MATCH (p:Person {city: $city}) RETURN p.name, p.age + $years",
];

fn calls() -> Vec<HashMap<String, Value>> {
    let names = ["Alix", "Gus", "Mia", "Nobody", "Alix"];
    let cities = ["Amsterdam", "Berlin", "Paris", "Prague", "Amsterdam"];
    let ages = [25, 30, 41, 99, 25];
    (0..5)
        .map(|index| {
            params(&[
                ("name", Value::from(names[index])),
                ("city", Value::from(cities[index])),
                ("least", Value::Int64(ages[index] - 5)),
                ("age", Value::Int64(ages[index])),
                ("skip", Value::Int64(i64::try_from(index % 3).unwrap())),
                ("limit", Value::Int64(i64::try_from(index % 4 + 1).unwrap())),
                ("years", Value::Int64(i64::try_from(index * 10).unwrap())),
            ])
        })
        .collect()
}

/// Only the values a statement names: a call gives exactly its parameters.
fn named(statement: &str, values: &HashMap<String, Value>) -> HashMap<String, Value> {
    values
        .iter()
        .filter(|(name, _)| statement.contains(&format!("${name}")))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect()
}

/// Each GQL statement, called with five sets of values in turn, answers as
/// the statement with those values written as literals; from the second
/// call on its prepared plan is found.
#[test]
fn gql_calls_answer_as_their_literal_statements() {
    let db = people();
    let session = db.session();
    for statement in GQL {
        for (index, values) in calls().into_iter().enumerate() {
            let values = named(statement, &values);
            let expected = session.execute(&with_literals(statement, &values)).unwrap();
            let (hits, _) = lookups(&db);
            let got = session
                .execute_with_params(statement, values.clone())
                .unwrap();
            assert_eq!(
                sorted(got.rows()),
                sorted(expected.rows()),
                "`{statement}` with {values:?}"
            );
            if index > 0 {
                assert_eq!(lookups(&db).0, hits + 1, "`{statement}`: the prepared plan");
            }
        }
    }
}

/// Cypher alike, with its own MERGE: the relationship is made once, and
/// the next call of the same values finds it.
#[cfg(feature = "cypher")]
#[test]
fn cypher_calls_answer_as_their_literal_statements() {
    let db = people();
    let session = db.session();
    let statements = [
        "MATCH (p:Person {name: $name}) RETURN p.age",
        "MATCH (p:Person) WHERE p.age > $least AND p.city <> $city RETURN p.name",
        "MATCH (p:Person) RETURN p.name ORDER BY p.name SKIP $skip LIMIT $limit",
        "MATCH (p:Person {city: $city}) RETURN p.name, p.age + $years",
    ];
    for statement in statements {
        for values in calls() {
            let values = named(statement, &values);
            let expected = session
                .execute_cypher(&with_literals(statement, &values))
                .unwrap();
            let got = session
                .execute_cypher_with_params(statement, values.clone())
                .unwrap();
            assert_eq!(
                sorted(got.rows()),
                sorted(expected.rows()),
                "`{statement}` with {values:?}"
            );
        }
    }

    let merge = "MATCH (a:Person {name: $a}), (b:Person {name: $b}) \
                 MERGE (a)-[r:LIKES]->(b) RETURN a.name, b.name";
    let likes = |db: &GrafeoDB| {
        db.execute("MATCH (:Person)-[r:LIKES]->(:Person) RETURN count(r)")
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    for (a, b, total) in [
        ("Alix", "Gus", 1),
        ("Gus", "Mia", 2),
        ("Alix", "Gus", 2),
        ("Mia", "Alix", 3),
        ("Gus", "Mia", 3),
    ] {
        let rows = session
            .execute_cypher_with_params(
                merge,
                params(&[("a", Value::from(a)), ("b", Value::from(b))]),
            )
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(
            rows,
            [vec![Value::from(a), Value::from(b)]],
            "{a} likes {b}"
        );
        assert_eq!(likes(&db), Value::Int64(total), "after {a} likes {b}");
    }
}

/// SQL/PGQ alike, also with a HAVING that reads a value.
#[cfg(feature = "sql-pgq")]
#[test]
fn sql_pgq_calls_answer_as_their_literal_statements() {
    let db = people();
    let session = db.session();
    let statements = [
        "SELECT name FROM GRAPH_TABLE (MATCH (p:Person) WHERE p.age > $least \
         COLUMNS (p.name AS name))",
        "SELECT city, COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) \
         COLUMNS (p.city AS city)) GROUP BY city HAVING COUNT(*) > $skip",
        "SELECT b FROM GRAPH_TABLE (MATCH (a:Person)-[:KNOWS]->(x:Person) \
         WHERE a.name = $name COLUMNS (x.name AS b))",
    ];
    for statement in statements {
        for values in calls() {
            let values = named(statement, &values);
            let expected = session
                .execute_sql(&with_literals(statement, &values))
                .unwrap();
            let got = session
                .execute_sql_with_params(statement, values.clone())
                .unwrap();
            assert_eq!(
                sorted(got.rows()),
                sorted(expected.rows()),
                "`{statement}` with {values:?}"
            );
        }
    }
}

/// The cached plan holds the parameters, never a value of a call: the
/// prepared plan of each statement still names `$name` after calls with
/// several names.
#[test]
fn a_prepared_plan_holds_no_value() {
    let db = people();
    let session = db.session();
    let statement = "MATCH (a:Person {name: $name})-[:KNOWS]->(b:Person) RETURN b.name";
    for name in ["Alix", "Gus", "Mia"] {
        session
            .execute_with_params(statement, params(&[("name", Value::from(name))]))
            .unwrap();
    }
    let key = CacheKey::in_scope(statement, QueryLanguage::Gql, None, None).prepared();
    let plan = db
        .query_cache()
        .get_optimized(&key)
        .expect("the statement is prepared");
    let text = format!("{plan:?}");
    assert!(text.contains("Parameter(\"name\")"), "{text}");
    for name in ["Alix", "Gus", "Mia"] {
        assert!(
            !text.contains(name),
            "{name} is in the prepared plan: {text}"
        );
    }
}

/// A value of another type than the last call's answers as the statement
/// with that value as a literal does, a match or an error alike.
#[test]
fn a_value_of_another_type_answers_as_its_literal() {
    let db = people();
    let session = db.session();
    for statement in [
        "MATCH (p:Person) WHERE p.age = $v RETURN p.name",
        "MATCH (p:Person {name: 'Alix'}) RETURN p.age + $v",
    ] {
        for value in [
            Value::Int64(30),
            Value::from("30"),
            Value::Float64(30.0),
            Value::Bool(true),
            Value::Int64(25),
        ] {
            let values = params(&[("v", value.clone())]);
            let expected = session.execute(&with_literals(statement, &values));
            let got = session.execute_with_params(statement, values);
            match (got, expected) {
                (Ok(got), Ok(expected)) => assert_eq!(
                    sorted(got.rows()),
                    sorted(expected.rows()),
                    "`{statement}` with {value:?}"
                ),
                (Err(got), Err(expected)) => assert_eq!(
                    got.to_string(),
                    expected.to_string(),
                    "`{statement}` with {value:?}"
                ),
                (got, expected) => {
                    panic!("`{statement}` with {value:?}: {got:?} against {expected:?}")
                }
            }
        }
    }
}

/// An index created after a statement was prepared prepares it again, and
/// its plan seeks through the index from then on; a drop prepares it again
/// without one.
#[test]
fn an_index_made_after_the_statement_was_prepared_is_used() {
    let db = people();
    let session = db.session();
    let statement = "MATCH (p:Person {name: $name}) RETURN p.age";
    let profile = |name: &str| {
        session
            .execute_with_params(
                &format!("PROFILE {statement}"),
                params(&[("name", Value::from(name))]),
            )
            .unwrap()
            .rows()
            .iter()
            .map(|row| format!("{:?}", row[0]))
            .collect::<Vec<_>>()
            .join("\n")
    };
    let call = |name: &str| {
        session
            .execute_with_params(statement, params(&[("name", Value::from(name))]))
            .unwrap()
            .rows()
            .to_vec()
    };
    assert_eq!(call("Alix"), [vec![Value::Int64(30)]]);
    let without = profile("Alix");
    assert!(
        without.contains("Scan (p:Person)") && !without.contains("NodeList"),
        "no index yet:\n{without}"
    );

    db.create_property_index("name").unwrap();
    let (hits, misses) = lookups(&db);
    assert_eq!(call("Gus"), [vec![Value::Int64(25)]]);
    assert_eq!(lookups(&db), (hits, misses + 1), "prepared again");
    let with = profile("Gus");
    assert!(
        with.contains("NodeList (p.name"),
        "the index is used:\n{with}"
    );

    assert!(db.drop_property_index("name").unwrap());
    assert_eq!(call("Mia"), [vec![Value::Int64(25)]]);
    let dropped = profile("Mia");
    assert!(
        !dropped.contains("NodeList"),
        "the index is gone:\n{dropped}"
    );
}

/// A WITH or RETURN item that is a parameter alone binds by its value, as
/// its literal does (a variable bound to a number or a text cannot be a
/// node; one bound to a null or a list may be): such a statement is bound
/// with its values on every call, so after a call whose value binds, a call
/// whose value does not fails as the literal statement fails.
#[test]
fn a_projected_parameter_binds_by_its_value() {
    let db = people();
    let session = db.session();
    let statement = "MATCH (p:Person {name: 'Alix'}) WITH $x AS x MATCH (x) RETURN x";
    let literal =
        |x: &str| format!("MATCH (p:Person {{name: 'Alix'}}) WITH {x} AS x MATCH (x) RETURN x");
    for (value, text) in [
        (Value::Null, "null"),
        (Value::Int64(1), "1"),
        (Value::List(vec![Value::Int64(1)].into()), "[1]"),
        (Value::from("a"), "'a'"),
        (Value::Null, "null"),
    ] {
        let got = session
            .execute_with_params(statement, params(&[("x", value.clone())]))
            .map(|result| result.rows().to_vec())
            .map_err(|error| error.to_string());
        let expected = session
            .execute(&literal(text))
            .map(|result| result.rows().to_vec())
            .map_err(|error| error.to_string());
        assert_eq!(got, expected, "x = {text}");
    }
    // A RETURN of a parameter alone answers with each call's value.
    for x in [1, 2] {
        let rows = session
            .execute_with_params("RETURN $x AS x", params(&[("x", Value::Int64(x))]))
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(rows, [vec![Value::Int64(x)]]);
    }
}

/// A parameter nobody gave fails as before, on the first call and on a
/// call after the statement was prepared.
#[test]
fn a_missing_parameter_fails_before_and_after_the_statement_is_prepared() {
    let db = people();
    let session = db.session();
    let statement = "MATCH (p:Person) WHERE p.name = $name AND p.age > $least RETURN p.name";
    let first = session
        .execute_with_params(statement, params(&[("name", Value::from("Alix"))]))
        .unwrap_err()
        .to_string();
    assert!(first.contains("Missing parameter: $least"), "{first}");
    session
        .execute_with_params(
            statement,
            params(&[("name", Value::from("Alix")), ("least", Value::Int64(1))]),
        )
        .unwrap();
    let after = session
        .execute_with_params(statement, params(&[("least", Value::Int64(1))]))
        .unwrap_err()
        .to_string();
    assert!(after.contains("Missing parameter: $name"), "{after}");
}

/// A plan's operators without its run's figures (rows and times), to
/// compare two plans of one statement.
fn operators(profile: &str) -> String {
    profile
        .lines()
        .filter(|line| !line.starts_with("Total time"))
        .map(|line| line.split("  rows=").next().unwrap_or(line).trim_end())
        .collect::<Vec<_>>()
        .join("\n")
}

/// A restore replaces the graphs, indexes and schemas a cached plan was made
/// for, so every plan is made again after it: the next call of a statement
/// misses the cache and runs the plan a database restored from the same
/// snapshot makes, whether the snapshot adds an index the old data lacked or
/// lacks one it had, and whatever schema it brings.
#[test]
fn a_restore_makes_the_cached_plans_again() {
    let statement = "MATCH (p:Person {name: $name}) RETURN p.age";
    let indexed = people();
    indexed.create_property_index("name").unwrap();
    let with_index = indexed.export_snapshot().unwrap();
    let without_index = people().export_snapshot().unwrap();

    let profile_of = |db: &GrafeoDB| {
        let rows = db
            .session()
            .execute_with_params(
                &format!("PROFILE {statement}"),
                params(&[("name", Value::from("Gus"))]),
            )
            .unwrap()
            .rows()
            .to_vec();
        match &rows[0][0] {
            Value::String(text) => operators(text),
            other => panic!("a plan, not {other:?}"),
        }
    };
    let fresh = |snapshot: &[u8]| {
        let db = GrafeoDB::new_in_memory();
        db.restore_snapshot(snapshot).unwrap();
        profile_of(&db)
    };
    assert_ne!(
        fresh(&with_index),
        fresh(&without_index),
        "the index changes the plan"
    );

    let db = GrafeoDB::new_in_memory();
    db.restore_snapshot(&without_index).unwrap();
    let session = db.session();
    let call = || {
        session
            .execute_with_params(statement, params(&[("name", Value::from("Gus"))]))
            .unwrap()
            .rows()
            .to_vec()
    };
    for (what, snapshot) in [
        ("adds an index", &with_index),
        ("lacks the index", &without_index),
        ("adds it again", &with_index),
    ] {
        call();
        call();
        db.restore_snapshot(snapshot).unwrap();
        let (hits, misses) = lookups(&db);
        assert_eq!(
            call(),
            [vec![Value::Int64(25)]],
            "after a restore that {what}"
        );
        assert_eq!(
            lookups(&db),
            (hits, misses + 1),
            "made again after a restore that {what}"
        );
        assert_eq!(
            profile_of(&db),
            fresh(snapshot),
            "the plan after a restore that {what}"
        );
    }

    // A schema the restore brings, changes or takes away: a plan cached in
    // it is made again, and each call reads what the restored schema holds.
    let schema_with = |rows: i64| {
        let db = GrafeoDB::new_in_memory();
        let s = db.session();
        s.execute("CREATE SCHEMA reports").unwrap();
        s.execute("SESSION SET SCHEMA reports").unwrap();
        for q in 0..rows {
            s.execute_with_params("INSERT (:Row {q: $q})", params(&[("q", Value::Int64(q))]))
                .unwrap();
        }
        db.export_snapshot().unwrap()
    };
    let (one, three) = (schema_with(1), schema_with(3));
    let db = GrafeoDB::new_in_memory();
    db.restore_snapshot(&one).unwrap();
    let session = db.session();
    session.execute("SESSION SET SCHEMA reports").unwrap();
    let count = |least: i64| {
        session
            .execute_with_params(
                "MATCH (r:Row) WHERE r.q >= $least RETURN count(r)",
                params(&[("least", Value::Int64(least))]),
            )
            .unwrap()
            .rows()[0][0]
            .clone()
    };
    assert_eq!(count(0), Value::Int64(1));
    assert_eq!(count(0), Value::Int64(1));
    db.restore_snapshot(&three).unwrap();
    let (hits, misses) = lookups(&db);
    assert_eq!(count(1), Value::Int64(2), "the restored schema's rows");
    assert_eq!(
        lookups(&db),
        (hits, misses + 1),
        "made again after the restore"
    );
}
