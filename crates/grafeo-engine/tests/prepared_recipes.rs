//! A prepared statement plans its physical operators once: each later call opens the cached
//! recipe with its own values and view (per-call physical planning, part 1).
//!
//! The oracle is today's path: the same statement with its value written as a literal, which
//! has no prepared plan and is planned in full. In a build with `--cfg grafeo_work_counters`,
//! every call with an integer or a string key also asserts that it opened the recipe and planned
//! nothing (`PlanBuilt` 0).

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// IS1's shape: a person a parameter pins, the city they live in.
const IS1: &str = "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City) \
                   RETURN n.firstName AS firstName, c.id AS cityId";

/// IS3's shape without its sort: a person's friends and when they met.
const IS3: &str = "MATCH (n:Person {id: $personId})-[k:KNOWS]-(friend:Person) \
                   RETURN friend.id AS friendId, k.creationDate AS since";

/// Alix (1), Gus (2), Vincent (3, no city) and Mia ('m', a string id) with their cities and
/// friendships.
fn people(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(
        "INSERT (alix:Person {id: 1, firstName: 'Alix'})-[:LIVES_IN]->(:City {id: 10}), \
                (gus:Person {id: 2, firstName: 'Gus'})-[:LIVES_IN]->(:City {id: 20}), \
                (vincent:Person {id: 3, firstName: 'Vincent'}), \
                (mia:Person:Admin {id: 'm', firstName: 'Mia'})-[:LIVES_IN]->(:City {id: 30}), \
                (alix)-[:KNOWS {creationDate: 5}]->(gus), (gus)-[:KNOWS {creationDate: 7}]->(vincent), \
                (mia)-[:KNOWS {creationDate: 9}]->(alix)",
    )
    .unwrap();
    db
}

fn with(id: Value) -> HashMap<String, Value> {
    HashMap::from([("personId".to_string(), id)])
}

/// `value` written as a GQL literal.
fn literal(value: &Value) -> String {
    match value {
        Value::Int64(int) => int.to_string(),
        Value::Float64(float) => format!("{float:?}"),
        Value::String(text) => format!("'{text}'"),
        Value::Null => "NULL".to_string(),
        Value::Bool(flag) => flag.to_string().to_uppercase(),
        Value::List(items) => format!(
            "[{}]",
            items.iter().map(literal).collect::<Vec<_>>().join(", ")
        ),
        Value::Date(date) => format!("DATE '{date}'"),
        other => panic!("no literal for {other:?}"),
    }
}

fn sorted(result: grafeo_engine::database::QueryResult) -> Vec<String> {
    let mut rows: Vec<String> = result.rows().iter().map(|row| format!("{row:?}")).collect();
    rows.sort();
    rows
}

fn run(
    db: &GrafeoDB,
    cypher: bool,
    query: &str,
    values: HashMap<String, Value>,
) -> Result<Vec<String>, String> {
    let result = if cypher {
        db.execute_cypher_with_params(query, values)
    } else {
        db.execute_with_params(query, values)
    };
    result.map(sorted).map_err(|error| error.to_string())
}

/// `query` with its key `id` as a parameter. With work counters, an integer or a string key
/// must open the recipe and plan nothing.
fn on(db: &GrafeoDB, cypher: bool, query: &str, id: &Value) -> Result<Vec<String>, String> {
    #[cfg(grafeo_work_counters)]
    let _ = grafeo_common::work::take();
    let answer = run(db, cypher, query, with(id.clone()));
    #[cfg(grafeo_work_counters)]
    if matches!(id, Value::Int64(_) | Value::String(_)) {
        let work = grafeo_common::work::take();
        assert_eq!(
            work.get(grafeo_common::work::Work::PlanBuilt),
            0,
            "{query} with {id:?} opened no recipe: {}",
            work.describe()
        );
    }
    answer
}

/// Today's path: `query` with `id` written as a literal, planned in full.
fn today(db: &GrafeoDB, cypher: bool, query: &str, id: &Value) -> Result<Vec<String>, String> {
    let text = query.replace("$personId", &literal(id));
    run(db, cypher, &text, HashMap::new())
}

/// The recipe's answers equal today's for `ids`, after a first call that caches the plan.
fn same_as_today(db: &GrafeoDB, cypher: bool, query: &str, ids: &[Value]) {
    let _ = run(db, cypher, query, with(Value::Int64(1)));
    for id in ids {
        assert_eq!(
            on(db, cypher, query, id),
            today(db, cypher, query, id),
            "{query} with {id:?}, cypher {cypher}"
        );
    }
}

fn keys() -> Vec<Value> {
    vec![
        Value::Int64(1),
        Value::Int64(2),
        Value::Int64(3),
        Value::Int64(4),
        Value::from("m"),
        Value::from("nobody"),
    ]
}

/// No call of a covered statement plans it: its first call makes the recipe, which it and
/// every later call open, so the planner builds no physical plan.
#[cfg(grafeo_work_counters)]
#[test]
fn a_prepared_statement_plans_once() {
    use grafeo_common::work::Work;
    let db = people(Config::in_memory());
    for (call, id) in [1, 2, 3, 1].into_iter().enumerate() {
        let _ = grafeo_common::work::take();
        let rows = run(&db, true, IS1, with(Value::Int64(id)));
        let work = grafeo_common::work::take();
        assert_eq!(
            work.get(Work::PlanBuilt),
            0,
            "call {call}: {}",
            work.describe()
        );
        assert_eq!(
            rows,
            today(&db, true, IS1, &Value::Int64(id)),
            "call {call}"
        );
    }
}

/// IS1's and IS3's shapes give today's rows for integer and string keys, one nobody has, in
/// GQL and Cypher, without an index and with one.
#[test]
fn the_rows_are_todays() {
    let db = people(Config::in_memory());
    for indexed in [false, true] {
        if indexed {
            db.create_property_index("id").unwrap();
        }
        for cypher in [false, true] {
            for query in [IS1, IS3] {
                same_as_today(&db, cypher, query, &keys());
            }
        }
    }
}

/// Each call's lookup reads the store as that call sees it: a person created after the first
/// call is found by the next, one deleted is not, and a transaction finds its own new person.
#[test]
fn each_call_reads_its_own_view() {
    let db = people(Config::in_memory());
    let city_of = |id: i64| on(&db, true, IS1, &Value::Int64(id)).unwrap();
    assert_eq!(city_of(9), Vec::<String>::new());
    db.execute("INSERT (:Person {id: 9, firstName: 'Mia'})-[:LIVES_IN]->(:City {id: 90})")
        .unwrap();
    assert_eq!(city_of(9).len(), 1, "created after the first call");
    db.execute("MATCH (n:Person {id: 9}) DETACH DELETE n")
        .unwrap();
    assert_eq!(city_of(9), Vec::<String>::new(), "deleted before this call");

    let mut session = db.session();
    session
        .execute_cypher_with_params(IS1, with(Value::Int64(1)))
        .unwrap();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {id: 11, firstName: 'Jules'})-[:LIVES_IN]->(:City {id: 110})")
        .unwrap();
    let own = session
        .execute_cypher_with_params(IS1, with(Value::Int64(11)))
        .unwrap();
    assert_eq!(
        own.row_count(),
        1,
        "the transaction's own uncommitted person"
    );
    session.rollback().unwrap();
    assert_eq!(city_of(11), Vec::<String>::new(), "rolled back");
}

/// A call that leaves the parameter out fails as with the cache off, after calls that gave it.
#[test]
fn a_missing_parameter_fails_as_today() {
    let db = people(Config::in_memory());
    let off = people(Config::in_memory().with_plan_cache_limit(0));
    for _ in 0..2 {
        on(&db, true, IS1, &Value::Int64(1)).unwrap();
    }
    let missing = HashMap::from([("other".to_string(), Value::Int64(1))]);
    let answer = run(&db, true, IS1, missing.clone());
    assert!(answer.is_err(), "{answer:?}");
    assert_eq!(answer, run(&off, true, IS1, missing));
}

/// A call with a value nested deeper than a stored one may be, even an unused one, is refused
/// as today, after calls that opened the recipe.
#[test]
fn a_too_deep_value_is_refused_as_today() {
    let db = people(Config::in_memory());
    let off = people(Config::in_memory().with_plan_cache_limit(0));
    for _ in 0..2 {
        on(&db, false, IS1, &Value::Int64(1)).unwrap();
    }
    let mut deep = Value::Int64(0);
    for _ in 0..200 {
        deep = Value::List(vec![deep].into());
    }
    let values = HashMap::from([
        ("personId".to_string(), Value::Int64(1)),
        ("unused".to_string(), deep),
    ]);
    let answer = run(&db, false, IS1, values.clone());
    assert!(answer.is_err(), "{answer:?}");
    assert_eq!(answer, run(&off, false, IS1, values));
}

/// Each kind of key value answers as today, call after call on one cached plan: an integer, 1.0
/// against the integer key 1, a string, NULL (no lookup, no rows), a boolean, a list and a date.
#[test]
fn a_value_of_another_kind_answers_as_today() {
    let db = people(Config::in_memory());
    let kinds = [
        Value::Int64(1),
        Value::Float64(1.0),
        Value::from("m"),
        Value::Null,
        Value::Bool(true),
        Value::List(vec![Value::Int64(1)].into()),
        Value::Date(grafeo_common::types::Date::from_ymd(2024, 1, 2).unwrap()),
        Value::Int64(2),
    ];
    same_as_today(&db, false, IS1, &kinds);
}

/// A zoned datetime key answers as with the cache off: its call is planned in full, as the
/// recipe opens only for integer and string keys.
#[test]
fn a_zoned_datetime_key_answers_as_today() {
    let db = people(Config::in_memory());
    let off = people(Config::in_memory().with_plan_cache_limit(0));
    let zoned = off
        .execute("RETURN ZONED_DATETIME('2024-01-02T03:04:05+01:00') AS z")
        .unwrap()
        .rows()[0][0]
        .clone();
    on(&db, false, IS1, &Value::Int64(1)).unwrap();
    assert_eq!(
        run(&db, false, IS1, with(zoned.clone())),
        run(&off, false, IS1, with(zoned))
    );
}

/// A label check over the pinned start (`WHERE n:Admin`) joins the key in one filter, which no
/// recipe covers: the call is planned as today, with today's rows.
#[test]
fn a_label_check_over_the_pinned_start_gives_todays_rows() {
    let db = people(Config::in_memory());
    let query = "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City) WHERE n:Admin                  RETURN n.firstName AS firstName, c.id AS cityId";
    for cypher in [false, true] {
        for id in keys() {
            assert_eq!(
                run(&db, cypher, query, with(id.clone())),
                today(&db, cypher, query, &id),
                "{id:?}, cypher {cypher}"
            );
        }
    }
}

/// An index created between two calls is read by the next call, and one dropped is not: the
/// rows and the rows read are those of today's path on the same database.
#[test]
fn an_index_created_or_dropped_between_calls_is_followed() {
    let db = people(Config::in_memory());
    let reads = |query: &str, values: HashMap<String, Value>| {
        let result = db.execute_cypher_with_params(query, values).unwrap();
        result.rows_scanned().unwrap_or(0)
    };
    let both = |label: &str| {
        let id = Value::Int64(2);
        assert_eq!(
            on(&db, true, IS1, &id),
            today(&db, true, IS1, &id),
            "{label}"
        );
        assert_eq!(
            reads(IS1, with(id.clone())),
            reads(&IS1.replace("$personId", "2"), HashMap::new()),
            "{label}: rows read"
        );
    };
    both("no index");
    db.create_property_index("id").unwrap();
    both("index created");
    assert!(db.drop_property_index("id").unwrap());
    both("index dropped");
}

/// Two sessions on two threads open one cached recipe at once, each with its own values: every
/// call's rows are today's.
#[test]
fn two_sessions_share_one_recipe() {
    let db = std::sync::Arc::new(people(Config::in_memory()));
    let expected: Vec<Result<Vec<String>, String>> = (1..=3)
        .map(|id| today(&db, true, IS1, &Value::Int64(id)))
        .collect();
    let threads: Vec<_> = (0..2)
        .map(|thread| {
            let db = std::sync::Arc::clone(&db);
            let expected = expected.clone();
            std::thread::spawn(move || {
                let session = db.session();
                for call in 0..200_usize {
                    let id = (call + thread) % 3;
                    let result = session
                        .execute_cypher_with_params(
                            IS1,
                            with(Value::Int64(i64::try_from(id).unwrap() + 1)),
                        )
                        .map(sorted)
                        .map_err(|error| error.to_string());
                    assert_eq!(result, expected[id], "thread {thread} call {call}");
                }
            })
        })
        .collect();
    for thread in threads {
        thread.join().unwrap();
    }
}
