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

/// IS3 with its sort: a person's friends, the most recent friendship first.
const IS3_ORDERED: &str = "MATCH (n:Person {id: $personId})-[k:KNOWS]-(friend:Person) RETURN friend.id AS friendId, k.creationDate AS since ORDER BY since DESC, friendId ASC";

/// IS2's shape: a person's latest messages, as one top-k.
const IS2: &str = "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS messageId, m.creationDate AS created ORDER BY created DESC, messageId ASC LIMIT 3";

/// The ordered shapes: IS3 sorted, IS2 at several limits, a sort by a property the RETURN
/// drops, SKIP with LIMIT and alone, and a LIMIT with no sort. Every sort key leaves no ties.
fn ordered_shapes() -> Vec<String> {
    let dropped = "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS messageId ORDER BY m.creationDate DESC";
    vec![
        IS3_ORDERED.to_string(),
        IS2.to_string(),
        IS2.replace("LIMIT 3", "LIMIT 10"),
        IS2.replace("LIMIT 3", "LIMIT 1"),
        IS2.replace("LIMIT 3", "LIMIT 0"),
        IS2.replace("LIMIT 3", "SKIP 1 LIMIT 2"),
        IS2.replace("LIMIT 3", "SKIP 2"),
        dropped.to_string(),
        format!("{dropped} LIMIT 2"),
        "MATCH (:Person {id: $personId})<-[:HAS_CREATOR]-(m:Message) RETURN m.id AS messageId LIMIT 2"
            .to_string(),
    ]
}

/// Alix (1), Gus (2), Vincent (3, no city) and Mia ('m', a string id) with their cities,
/// friendships and messages (Alix five, Gus two, each at its own time).
fn people(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    db.execute(
        "INSERT (alix:Person {id: 1, firstName: 'Alix'})-[:LIVES_IN]->(:City {id: 10}), \
                (gus:Person {id: 2, firstName: 'Gus'})-[:LIVES_IN]->(:City {id: 20}), \
                (vincent:Person {id: 3, firstName: 'Vincent'}), \
                (mia:Person:Admin {id: 'm', firstName: 'Mia'})-[:LIVES_IN]->(:City {id: 30}), \
                (alix)-[:KNOWS {creationDate: 5}]->(gus), (gus)-[:KNOWS {creationDate: 7}]->(vincent), \
                (mia)-[:KNOWS {creationDate: 9}]->(alix), (:Message {id: 101, creationDate: 14})-[:HAS_CREATOR]->(alix), (:Message {id: 102, creationDate: 12})-[:HAS_CREATOR]->(alix), (:Message {id: 103, creationDate: 15})-[:HAS_CREATOR]->(alix), (:Message {id: 104, creationDate: 11})-[:HAS_CREATOR]->(alix), (:Message {id: 105, creationDate: 13})-[:HAS_CREATOR]->(alix), (:Message {id: 201, creationDate: 21})-[:HAS_CREATOR]->(gus), (:Message {id: 202, creationDate: 22})-[:HAS_CREATOR]->(gus)",
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

/// The rows of `query` with `values`, in the order the statement gives them.
fn run_ordered(
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
    result
        .map(|result| result.rows().iter().map(|row| format!("{row:?}")).collect())
        .map_err(|error| error.to_string())
}

/// Today's path for `query` with `values`: each value written as a literal, planned in full,
/// its rows in the statement's order.
fn today_ordered(
    db: &GrafeoDB,
    cypher: bool,
    query: &str,
    values: &HashMap<String, Value>,
) -> Result<Vec<String>, String> {
    let mut names: Vec<&String> = values.keys().collect();
    // The longest name first, so `$n` does not replace the start of `$name`.
    names.sort_by_key(|name| std::cmp::Reverse(name.len()));
    let mut text = query.to_string();
    for name in names {
        text = text.replace(&format!("${name}"), &literal(&values[name]));
    }
    run_ordered(db, cypher, &text, HashMap::new())
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
    for query in [IS1, IS2] {
        for (call, id) in [1, 2, 3, 1].into_iter().enumerate() {
            let _ = grafeo_common::work::take();
            let rows = run(&db, true, query, with(Value::Int64(id)));
            let work = grafeo_common::work::take();
            assert_eq!(
                work.get(Work::PlanBuilt),
                0,
                "{query} call {call}: {}",
                work.describe()
            );
            assert_eq!(
                rows,
                today(&db, true, query, &Value::Int64(id)),
                "{query} call {call}"
            );
        }
    }
}

/// The ordered shapes give today's rows in today's order, in GQL and Cypher, call after call
/// on one cached plan.
#[test]
fn the_ordered_rows_are_todays() {
    let db = people(Config::in_memory());
    for cypher in [false, true] {
        for query in ordered_shapes() {
            let _ = run_ordered(&db, cypher, &query, with(Value::Int64(2)));
            for id in keys() {
                #[cfg(grafeo_work_counters)]
                let _ = grafeo_common::work::take();
                let rows = run_ordered(&db, cypher, &query, with(id.clone()));
                #[cfg(grafeo_work_counters)]
                if matches!(id, Value::Int64(_) | Value::String(_)) {
                    let work = grafeo_common::work::take();
                    assert_eq!(
                        work.get(grafeo_common::work::Work::PlanBuilt),
                        0,
                        "{query} with {id:?}: {}",
                        work.describe()
                    );
                }
                assert_eq!(
                    rows,
                    today_ordered(&db, cypher, &query, &with(id.clone())),
                    "{query} with {id:?}, cypher {cypher}"
                );
            }
        }
    }
}

/// Many distinct literal LIMIT counts, each its own text and so its own cache entry, keep the
/// plan cache within its byte bound, and each call gives the rows of its literal text.
#[test]
fn distinct_literal_limits_keep_the_cache_within_its_bound() {
    const LIMIT: usize = 64 << 10;
    let db = people(Config::in_memory().with_plan_cache_limit(LIMIT));
    for count in 0..400 {
        let query = IS2.replace("LIMIT 3", &format!("LIMIT {count}"));
        assert_eq!(
            run_ordered(&db, true, &query, with(Value::Int64(1))),
            today_ordered(&db, true, &query, &with(Value::Int64(1))),
            "LIMIT {count}"
        );
        let caches = db.memory_usage().caches;
        let held = caches.parsed_plan_cache_bytes + caches.optimized_plan_cache_bytes;
        assert!(held <= LIMIT, "LIMIT {count}: {held} bytes held");
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
    let query = "MATCH (n:Person {id: $personId})-[:LIVES_IN]->(c:City) WHERE n:Admin RETURN n.firstName AS firstName, c.id AS cityId";
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

/// IS2's shape with its count a parameter.
fn is2_counted(count: &str) -> String {
    IS2.replace("LIMIT 3", count)
}

/// The values of a call of Alix's messages with `counts`.
fn alix_with(counts: &[(&str, Value)]) -> HashMap<String, Value> {
    let mut values = with(Value::Int64(1));
    values.extend(
        counts
            .iter()
            .map(|(name, value)| ((*name).to_string(), value.clone())),
    );
    values
}

/// Each call of one prepared statement takes its own count: a top-k, then a plain limit at 0,
/// then a top-k again, and a count past the sort's input; SKIP with LIMIT, and SKIP alone. Each
/// call's rows are those of the statement with its counts written as literals, and no call after
/// the first plans.
#[test]
fn a_count_parameter_follows_each_call() {
    let int = |count: i64| Value::Int64(count);
    let cases: Vec<(String, Vec<Vec<(&str, Value)>>)> = vec![
        (
            is2_counted("LIMIT $n"),
            [3, 0, 1, 50, 0, 5, 0]
                .into_iter()
                .map(|n| vec![("n", int(n))])
                .collect(),
        ),
        (
            is2_counted("SKIP $s LIMIT $n"),
            [(1, 2), (0, 0), (2, 50), (10, 1), (0, 3)]
                .into_iter()
                .map(|(s, n)| vec![("s", int(s)), ("n", int(n))])
                .collect(),
        ),
        (
            is2_counted("SKIP $s"),
            [0, 1, 4, 10]
                .into_iter()
                .map(|s| vec![("s", int(s))])
                .collect(),
        ),
    ];
    let db = people(Config::in_memory());
    for cypher in [false, true] {
        for (query, calls) in &cases {
            for (call, counts) in calls.iter().enumerate() {
                let values = alix_with(counts);
                #[cfg(grafeo_work_counters)]
                let _ = grafeo_common::work::take();
                let rows = run_ordered(&db, cypher, query, values.clone());
                #[cfg(grafeo_work_counters)]
                if call > 0 {
                    let work = grafeo_common::work::take();
                    assert_eq!(
                        work.get(grafeo_common::work::Work::PlanBuilt),
                        0,
                        "{query} call {call}: {}",
                        work.describe()
                    );
                }
                assert_eq!(
                    rows,
                    today_ordered(&db, cypher, query, &values),
                    "{query} with {counts:?}, cypher {cypher}, call {call}"
                );
            }
        }
    }
}

/// A count that is negative, a float, a string or NULL, or that is missing, fails with the
/// error today's path gives, after calls that opened the recipe with good counts.
#[test]
fn a_bad_count_fails_as_today() {
    let db = people(Config::in_memory());
    let query = is2_counted("LIMIT $n");
    for cypher in [false, true] {
        for _ in 0..2 {
            run_ordered(&db, cypher, &query, alix_with(&[("n", Value::Int64(2))])).unwrap();
        }
        let answers: Vec<Result<Vec<String>, String>> = [
            Some(Value::Int64(-1)),
            Some(Value::Float64(1.5)),
            Some(Value::from("3")),
            Some(Value::Null),
            None,
        ]
        .into_iter()
        .map(|value| {
            let values = match value {
                Some(value) => alix_with(&[("n", value)]),
                None => alix_with(&[]),
            };
            run_ordered(&db, cypher, &query, values)
        })
        .collect();
        // Today's errors, observed while a count parameter still planned each call in full.
        let expected: Vec<Result<Vec<String>, String>> = [
            "GRAFEO-Q006: negative limit value: SKIP/LIMIT must be non-negative, got -1",
            "GRAFEO-Q006: invalid value type: SKIP/LIMIT must be an integer, got 1.5",
            "GRAFEO-Q006: invalid value type: SKIP/LIMIT must be an integer, got \"3\"",
            "GRAFEO-Q006: invalid value type: SKIP/LIMIT must be an integer, got NULL",
            "GRAFEO-Q002: semantic error: Missing parameter for SKIP/LIMIT: $n",
        ]
        .into_iter()
        .map(|text| Err(text.to_string()))
        .collect();
        assert_eq!(answers, expected, "cypher {cypher}");
    }
}

/// A bad count fails with the count's error, as on today's path, on a graph dropped after calls
/// that opened the recipe. The oracle is the same call of a statement no recipe covers
/// (`ORDER BY abs(...)`), which takes today's path.
#[test]
fn a_bad_count_on_a_dropped_graph_fails_as_today() {
    let answers: Vec<String> = [
        is2_counted("LIMIT $n"),
        is2_counted("LIMIT $n").replace("ORDER BY created DESC", "ORDER BY abs(created) DESC"),
    ]
    .iter()
    .map(|query| {
        let db = GrafeoDB::new_in_memory();
        db.execute("CREATE GRAPH g").unwrap();
        let session = db.session();
        session.execute("USE GRAPH g").unwrap();
        session
            .execute(
                "INSERT (:Message {id: 101, creationDate: 1})-[:HAS_CREATOR]->(:Person {id: 1})",
            )
            .unwrap();
        for _ in 0..2 {
            session
                .execute_with_params(query, alix_with(&[("n", Value::Int64(2))]))
                .unwrap();
        }
        db.execute("DROP GRAPH g").unwrap();
        session
            .execute_with_params(query, alix_with(&[("n", Value::Int64(-1))]))
            .map(|_| ())
            .unwrap_err()
            .to_string()
    })
    .collect();
    assert_eq!(answers[0], answers[1], "the recipe's call against today's");
    assert_eq!(
        answers[1],
        "GRAFEO-Q006: negative limit value: SKIP/LIMIT must be non-negative, got -1"
    );
}

/// Aggregates the recipe builds by their route: the edge count over a pinned start (also
/// grouped), the aggregate over the rows (of a property), the two-hop count by the middle nodes,
/// a label's count, and a range's fold with a literal bound.
fn covered_aggregates() -> Vec<&'static str> {
    vec![
        "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) RETURN count(f) AS friends",
        "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) \
         RETURN f.firstName AS name, count(*) AS n",
        "MATCH (n:Person {id: $personId})-[:KNOWS]-(f:Person) \
         RETURN count(f.firstName) AS named, max(f.firstName) AS last",
        "MATCH (a:Person)-[:KNOWS]->(m:Person)-[:KNOWS]->(b:Person) RETURN count(*) AS paths",
        "MATCH (n:Person) RETURN count(n) AS people",
        "MATCH (m:Message) WHERE m.creationDate > 12 \
         RETURN count(*) AS later, avg(m.creationDate) AS mean",
    ]
}

/// Each covered aggregate gives today's rows for every key, in GQL and Cypher, and (with work
/// counters) no call plans. With factorized execution off the rows are today's too; there a
/// count over a chain runs over its rows, and Cypher's check that the chain's edges differ is a
/// filter no recipe covers, so that statement alone plans each call.
#[test]
fn aggregates_give_todays_rows() {
    let db = people(Config::in_memory());
    for cypher in [false, true] {
        for query in covered_aggregates() {
            same_as_today(&db, cypher, query, &keys());
        }
    }
    let plain = people(Config::in_memory().without_factorized_execution());
    for cypher in [false, true] {
        for query in covered_aggregates() {
            // Cypher's chain of two hops checks that its edges differ, a filter no recipe covers.
            let opens = !(cypher && query.contains("(m:Person)-[:KNOWS]->(b:Person)"));
            for id in keys() {
                #[cfg(grafeo_work_counters)]
                let _ = grafeo_common::work::take();
                let rows = run(&plain, cypher, query, with(id.clone()));
                #[cfg(grafeo_work_counters)]
                if matches!(id, Value::Int64(_) | Value::String(_)) {
                    let built =
                        grafeo_common::work::take().get(grafeo_common::work::Work::PlanBuilt);
                    assert_eq!(built == 0, opens, "{query} with {id:?}, cypher {cypher}");
                }
                assert_eq!(
                    rows,
                    today(&plain, cypher, query, &id),
                    "{query} with {id:?}, cypher {cypher}, factorized off, opens {opens}"
                );
            }
        }
    }
}

/// A range fold whose bound is a parameter is planned per call, with today's rows.
#[test]
fn a_range_fold_over_a_parameter_gives_todays_rows() {
    let db = people(Config::in_memory());
    let query = "MATCH (m:Message) WHERE m.creationDate > $personId \
                 RETURN count(*) AS later, avg(m.creationDate) AS mean";
    for cypher in [false, true] {
        for bound in [0, 12, 14, 30] {
            let bound = Value::Int64(bound);
            assert_eq!(
                run(&db, cypher, query, with(bound.clone())),
                today(&db, cypher, query, &bound),
                "bound {bound:?}, cypher {cypher}"
            );
        }
    }
}

/// A second session opens the recipe the first one made, with its own values: each call's
/// rows are today's.
#[test]
fn a_second_session_opens_the_same_aggregate_recipe() {
    let db = people(Config::in_memory());
    let query = covered_aggregates()[0];
    let first = db.session();
    let second = db.session();
    for (call, id) in [1, 2, 3, 1, 2].into_iter().enumerate() {
        let session = if call % 2 == 0 { &first } else { &second };
        let rows = session
            .execute_with_params(query, with(Value::Int64(id)))
            .map(sorted)
            .map_err(|error| error.to_string());
        assert_eq!(
            rows,
            today(&db, false, query, &Value::Int64(id)),
            "call {call}"
        );
    }
}

/// A range fold over a temporal literal follows each session's time zone: two sessions in
/// other zones, one without a zone, and one that changes its zone between calls, each give the
/// rows of their own run
/// of the statement without values. The bound is the filter's only condition, so it is a range
/// fold's: over a zoned or a local datetime the session's temporal rule orders the values, so the
/// call is planned per call (with work counters: it plans); over a date the recipe opens.
#[test]
fn a_range_fold_over_a_temporal_literal_follows_each_sessions_time_zone() {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Event {day: DATE '2024-01-01', at: DATETIME '2024-01-02T10:00:00'}), (:Event {day: DATE '2024-01-02', at: DATETIME '2024-01-02T12:00:00'}), (:Event {day: DATE '2024-01-03', at: ZONED DATETIME '2024-01-02T11:00:00+02:00'})",
    )
    .unwrap();
    let in_zone = |zone: &str| {
        let session = db.session();
        session
            .execute(&format!("SESSION SET TIME ZONE '{zone}'"))
            .unwrap();
        session
    };
    let rows = |session: &grafeo_engine::session::Session, query: &str, values| {
        let result = if values {
            session.execute_with_params(query, with(Value::Int64(0)))
        } else {
            session.execute(query)
        };
        result.map(sorted).map_err(|error| error.to_string())
    };
    let amsterdam = in_zone("+01:00");
    let tokyo = in_zone("+09:00");
    // A session without a time zone reads datetimes by the default rule, where the planner folds
    // them; the recipe still leaves them to each call.
    let plain = db.session();
    for (bound, opens) in [
        ("e.at > ZONED DATETIME '2024-01-02T10:30:00+01:00'", false),
        ("e.at > DATETIME '2024-01-02T10:30:00'", false),
        ("e.day > DATE '2024-01-01'", true),
    ] {
        let query = format!("MATCH (e:Event) WHERE {bound} RETURN count(*) AS later");
        let calls = [&amsterdam, &tokyo, &plain, &amsterdam];
        for (call, session) in calls.into_iter().enumerate() {
            #[cfg(grafeo_work_counters)]
            let _ = grafeo_common::work::take();
            let prepared = rows(session, &query, true);
            #[cfg(grafeo_work_counters)]
            if call > 0 {
                let built = grafeo_common::work::take().get(grafeo_common::work::Work::PlanBuilt);
                assert_eq!(built == 0, opens, "{bound} call {call}: planned {built}");
            }
            assert_eq!(
                prepared,
                rows(session, &query, false),
                "{bound} call {call}, opens {opens}"
            );
        }
        amsterdam.execute("SESSION SET TIME ZONE '-05:00'").unwrap();
        assert_eq!(
            rows(&amsterdam, &query, true),
            rows(&amsterdam, &query, false),
            "{bound} after a zone change"
        );
        amsterdam.execute("SESSION SET TIME ZONE '+01:00'").unwrap();
    }
}

/// A SQL/PGQ range fold, which no recipe covers, gives the rows of its statement without values.
#[cfg(feature = "sql-pgq")]
#[test]
fn a_sql_range_fold_gives_todays_rows() {
    let db = people(Config::in_memory());
    let query = "SELECT count(*) AS later FROM GRAPH_TABLE (MATCH (m:Message) WHERE m.creationDate > 12 COLUMNS (m.id AS id))";
    for id in keys() {
        let prepared = db
            .execute_sql_with_params(query, with(id.clone()))
            .map(sorted)
            .map_err(|error| error.to_string());
        let plain = db
            .execute_sql(query)
            .map(sorted)
            .map_err(|error| error.to_string());
        assert_eq!(prepared, plain, "{id:?}");
    }
}
