//! A parameterized statement is checked, tracked and planned exactly like the
//! same statement with literal values (#526): schema and constraints, write
//! conflicts per graph, EXPLAIN and PROFILE (#460), and a cached plan never
//! keeps one call's values.

#![cfg(feature = "gql")]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

fn params(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
    pairs
        .iter()
        .map(|(name, value)| ((*name).to_string(), value.clone()))
        .collect()
}

fn count(db: &GrafeoDB, label: &str) -> Value {
    db.execute(&format!("MATCH (n:{label}) RETURN count(n)"))
        .unwrap()
        .rows()[0][0]
        .clone()
}

/// A write that uses a parameter nobody supplied fails before it writes:
/// it used to store the text "$e". Through `execute` (no parameter map at
/// all), an empty map, Cypher and GraphQL (a declared variable without a
/// default).
#[test]
fn an_unsupplied_parameter_fails_before_writing() {
    let db = GrafeoDB::new_in_memory();
    for result in [
        db.execute("INSERT (:P {e: $e})"),
        db.execute_with_params("INSERT (:P {e: $e})", HashMap::new()),
        #[cfg(feature = "cypher")]
        db.execute_cypher("CREATE (:P {e: $e})"),
        #[cfg(feature = "graphql")]
        db.execute_graphql("mutation ($e: String) { createP(e: $e) { e } }"),
    ] {
        let error = result.unwrap_err().to_string();
        assert!(error.contains("Missing parameter: $e"), "{error}");
    }
    assert_eq!(count(&db, "P"), Value::Int64(0));

    // EXPLAIN shows the plan without the values; PROFILE runs it, so it fails.
    db.execute("EXPLAIN INSERT (:P {e: $e})").unwrap();
    let error = db
        .execute("PROFILE INSERT (:P {e: $e})")
        .unwrap_err()
        .to_string();
    assert!(error.contains("Missing parameter: $e"), "{error}");
    assert_eq!(count(&db, "P"), Value::Int64(0));
}

#[test]
fn constraints_hold_for_parameterized_writes() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    db.execute("CREATE CONSTRAINT person_name FOR (n:Person) ON (n.name) NOT NULL")
        .unwrap();
    db.execute("INSERT (:Person {name: 'Alix', email: 'alix@example.org'})")
        .unwrap();
    let email = |value: &str| params(&[("email", Value::from(value))]);

    let err = db
        .execute_with_params(
            "INSERT (:Person {name: 'Gus', email: $email})",
            email("alix@example.org"),
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    let err = db
        .execute_with_params("INSERT (:Person {email: $email})", email("gus@example.org"))
        .unwrap_err();
    assert!(err.to_string().contains("name"), "{err}");
    db.execute_with_params(
        "INSERT (:Person {name: 'Gus', email: $email})",
        email("gus@example.org"),
    )
    .unwrap();
    let err = db
        .execute_with_params(
            "MATCH (n:Person {name: 'Gus'}) SET n.email = $email",
            email("alix@example.org"),
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    assert_eq!(count(&db, "Person"), Value::Int64(2));
}

#[test]
fn the_size_limit_holds_for_parameterized_writes() {
    let db = GrafeoDB::with_config(Config::in_memory().with_max_property_size(64)).unwrap();
    db.execute("INSERT (:Person {bio: 'short'})").unwrap();
    let err = db
        .execute_with_params(
            "MATCH (n:Person) SET n.bio = $bio",
            params(&[("bio", Value::from("x".repeat(1000)))]),
        )
        .unwrap_err();
    assert!(err.to_string().contains("exceeds maximum size"), "{err}");
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_parameters_are_checked_too() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE CONSTRAINT person_email FOR (n:Person) ON (n.email) UNIQUE")
        .unwrap();
    db.execute("INSERT (:Person {email: 'alix@example.org'})")
        .unwrap();
    let err = db
        .execute_cypher_with_params(
            "CREATE (:Person {email: $email})",
            params(&[("email", Value::from("alix@example.org"))]),
        )
        .unwrap_err();
    assert!(err.to_string().contains("UNIQUE"), "{err}");
    assert_eq!(count(&db, "Person"), Value::Int64(1));
}

/// A parameterized write in a named graph is recorded as a write to that
/// graph: it conflicts with another transaction's write to the same node, and
/// not with the node of the same number in another graph.
#[test]
fn parameterized_writes_are_tracked_per_graph() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE GRAPH extraction").unwrap();
    db.execute("CREATE GRAPH model").unwrap();
    for graph in ["extraction", "model"] {
        db.graph(graph)
            .unwrap()
            .execute("INSERT (:Doc {id: 'd1'})")
            .unwrap();
    }
    let set = "MATCH (n:Doc {id: $id}) SET n.v = $v";
    let args = |v: i64| params(&[("id", Value::from("d1")), ("v", Value::Int64(v))]);

    let mut first = db.graph("model").unwrap().session().unwrap();
    first.begin_transaction().unwrap();
    first
        .execute("MATCH (n:Doc {id: 'd1'}) SET n.v = 1")
        .unwrap();

    let mut same_graph = db.graph("model").unwrap().session().unwrap();
    same_graph.begin_transaction().unwrap();
    assert!(
        same_graph.execute_with_params(set, args(2)).is_err(),
        "the same node in the same graph conflicts"
    );
    same_graph.rollback().unwrap();

    let mut other_graph = db.graph("extraction").unwrap().session().unwrap();
    other_graph.begin_transaction().unwrap();
    other_graph.execute_with_params(set, args(3)).unwrap();
    other_graph.commit().unwrap();
    first.commit().unwrap();
}

#[test]
fn a_cached_plan_does_not_keep_the_values() {
    let db = GrafeoDB::new_in_memory();
    db.create_property_index("id").unwrap();
    db.execute("UNWIND range(0, 9) AS i INSERT (:Doc {id: 'd' + toString(i), n: i})")
        .unwrap();
    let query = "MATCH (n:Doc {id: $id}) RETURN n.n";
    for (id, n) in [("d3", 3), ("d7", 7), ("d3", 3), ("missing", -1)] {
        let rows = db
            .execute_with_params(query, params(&[("id", Value::from(id))]))
            .unwrap()
            .rows()
            .to_vec();
        if n < 0 {
            assert!(rows.is_empty(), "{id}: {rows:?}");
        } else {
            assert_eq!(rows, [vec![Value::Int64(n)]], "{id}");
        }
    }
}

/// An empty map fills in nothing: a statement without parameters reuses its
/// optimized plan like the same call without a map, and one that names a
/// parameter still fails as missing it.
#[test]
fn an_empty_parameter_map_uses_the_cached_plan() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let query = "MATCH (p:Person) RETURN p.name";
    let hits = || db.query_cache().stats().optimized_hits;
    let before = hits();
    for _ in 0..3 {
        let rows = db
            .execute_with_params(query, HashMap::new())
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(rows, [vec![Value::from("Alix")]]);
    }
    assert_eq!(
        hits() - before,
        2,
        "the second and third call reuse the plan"
    );

    let error = db
        .execute_with_params(
            "MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            HashMap::new(),
        )
        .unwrap_err();
    assert!(error.to_string().contains("$name"), "{error}");
}

/// A statement without parameters looks its optimized plan up once per call:
/// the first call misses and caches it, every later call hits it, in GQL and
/// in Cypher alike.
#[test]
fn a_statement_without_parameters_looks_its_plan_up_once_per_call() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let session = db.session();
    let lookups = || {
        let stats = db.query_cache().stats();
        (stats.optimized_hits, stats.optimized_misses)
    };

    let (hits, misses) = lookups();
    for _ in 0..3 {
        let rows = session
            .execute("MATCH (p:Person) RETURN p.name")
            .unwrap()
            .rows()
            .to_vec();
        assert_eq!(rows, [vec![Value::from("Alix")]]);
    }
    assert_eq!(lookups(), (hits + 2, misses + 1), "GQL");

    #[cfg(feature = "cypher")]
    {
        let (hits, misses) = lookups();
        for _ in 0..3 {
            let rows = session
                .execute_cypher("MATCH (p:Person) RETURN p.name")
                .unwrap()
                .rows()
                .to_vec();
            assert_eq!(rows, [vec![Value::from("Alix")]]);
        }
        assert_eq!(lookups(), (hits + 2, misses + 1), "Cypher");
    }

    #[cfg(feature = "sql-pgq")]
    {
        let (hits, misses) = lookups();
        for _ in 0..3 {
            let rows = session
                .execute_sql(
                    "SELECT * FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.name AS name))",
                )
                .unwrap()
                .rows()
                .to_vec();
            assert_eq!(rows, [vec![Value::from("Alix")]]);
        }
        assert_eq!(lookups(), (hits + 2, misses + 1), "SQL/PGQ");
    }
}

#[test]
fn explain_and_profile_take_parameters() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let name = || params(&[("name", Value::from("Alix"))]);

    let profile = db
        .execute_with_params(
            "PROFILE MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            name(),
        )
        .unwrap();
    assert_eq!(profile.columns, ["profile"]);
    let explain = db
        .execute_with_params(
            "EXPLAIN MATCH (p:Person) WHERE p.name = $name RETURN p.name",
            name(),
        )
        .unwrap();
    assert_ne!(explain.columns, ["p.name"], "{:?}", explain.columns);
    #[cfg(feature = "cypher")]
    {
        let profile = db
            .execute_cypher_with_params(
                "PROFILE MATCH (p:Person) WHERE p.name = $name RETURN p.name",
                name(),
            )
            .unwrap();
        assert_eq!(profile.columns, ["profile"]);
    }
}

/// Dotted access reads keys of a map parameter, in GQL and Cypher.
#[test]
fn dotted_access_reads_a_map_parameter() {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::PropertyKey;

    let db = GrafeoDB::new_in_memory();
    let route = Value::Map(Arc::new(BTreeMap::from([(
        PropertyKey::new("to"),
        Value::from("Prague"),
    )])));
    let meta = Value::Map(Arc::new(BTreeMap::from([(
        PropertyKey::new("route"),
        route,
    )])));
    let result = db
        .execute_with_params(
            "RETURN $meta.route.to AS to",
            params(&[("meta", meta.clone())]),
        )
        .unwrap();
    assert_eq!(result.rows(), [[Value::from("Prague")]]);
    #[cfg(feature = "cypher")]
    {
        let result = db
            .session()
            .execute_cypher_with_params("RETURN $meta.route.to AS to", params(&[("meta", meta)]))
            .unwrap();
        assert_eq!(result.rows(), [[Value::from("Prague")]]);
    }
}

/// A list, map or bytes parameter returned as it is comes back as given
/// (#574): it came back as `''`. Directly, through WITH, through UNWIND,
/// nested, and in Cypher.
#[test]
fn list_and_map_parameters_are_returned_as_given() {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use grafeo_common::types::PropertyKey;

    let list = Value::List(vec![Value::Int64(1), Value::Int64(2)].into());
    let map = Value::Map(Arc::new(BTreeMap::from([
        (PropertyKey::new("city"), Value::from("Amsterdam")),
        (PropertyKey::new("tags"), list.clone()),
    ])));
    let rows = Value::List(vec![map.clone(), map.clone()].into());
    let bytes = Value::Bytes(vec![0x47, 0x00, 0xff].into());

    let db = GrafeoDB::new_in_memory();
    for (query, value, expected) in [
        ("RETURN $x AS v", &list, vec![list.clone()]),
        ("RETURN $x AS v", &map, vec![map.clone()]),
        ("RETURN $x AS v", &bytes, vec![bytes.clone()]),
        (
            "UNWIND [0] AS i WITH $x AS w RETURN w AS v",
            &list,
            vec![list.clone()],
        ),
        (
            "UNWIND [0] AS i WITH $x AS w RETURN w AS v",
            &map,
            vec![map.clone()],
        ),
        (
            "UNWIND $x AS v RETURN v",
            &rows,
            vec![map.clone(), map.clone()],
        ),
        (
            "RETURN [$x] AS v",
            &map,
            vec![Value::List(vec![map.clone()].into())],
        ),
    ] {
        let result = db
            .execute_with_params(query, params(&[("x", value.clone())]))
            .unwrap();
        let got: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
        assert_eq!(got, expected, "{query} with {value:?}");
        #[cfg(feature = "cypher")]
        {
            let result = db
                .session()
                .execute_cypher_with_params(query, params(&[("x", value.clone())]))
                .unwrap();
            let got: Vec<Value> = result.rows().iter().map(|row| row[0].clone()).collect();
            assert_eq!(got, expected, "Cypher: {query} with {value:?}");
        }
    }
}

/// Dotted access on a node that a function returns reads the node's
/// property; on a value that is neither a map, a node nor an edge it is an
/// error that says what to do, never a silent null.
#[test]
fn dotted_access_on_a_node_expression_reads_the_node() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:City {name: 'Amsterdam'})-[:ROAD]->(:City {name: 'Berlin'})")
        .unwrap();
    let result = db
        .execute("MATCH (:City)-[r:ROAD]->(:City) RETURN startNode(r).name, endNode(r).name")
        .unwrap();
    assert_eq!(
        result.rows(),
        [vec![Value::from("Amsterdam"), Value::from("Berlin")]]
    );
    let err = db
        .execute("MATCH (c:City) RETURN toUpper(c.name).name")
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("is not a map value, a node or an edge"),
        "{err}"
    );
    assert!(
        err.contains("read .name of a node, an edge or a map value"),
        "{err}"
    );
}

/// The people of three cities, for the grouping cases: Amsterdam three,
/// Berlin two, Paris one; `x` and `y` with `y = 2x`.
fn cities() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (name, city, x) in [
        ("Alix", "Amsterdam", 1),
        ("Gus", "Amsterdam", 2),
        ("Vincent", "Amsterdam", 3),
        ("Jules", "Berlin", 1),
        ("Mia", "Berlin", 2),
        ("Butch", "Paris", 1),
    ] {
        db.execute_with_params(
            "INSERT (:P {name: $name, city: $city, x: $x, y: $y})",
            params(&[
                ("name", Value::from(name)),
                ("city", Value::from(city)),
                ("x", Value::Int64(x)),
                ("y", Value::Int64(2 * x)),
            ]),
        )
        .unwrap();
    }
    db
}

/// A parameter in a HAVING condition is filled in: in GQL and SQL/PGQ it
/// reached the planner unresolved ("Unresolved parameter"); each call reads
/// its own value.
#[test]
fn a_having_condition_takes_parameters() {
    let db = cities();
    for (least, expected) in [(1, vec!["Amsterdam", "Berlin"]), (2, vec!["Amsterdam"])] {
        let rows = db
            .execute_with_params(
                "MATCH (p:P) RETURN p.city AS city GROUP BY city HAVING count(*) > $least \
                 ORDER BY city",
                params(&[("least", Value::Int64(least))]),
            )
            .unwrap()
            .rows()
            .to_vec();
        let expected: Vec<Vec<Value>> = expected.iter().map(|c| vec![Value::from(*c)]).collect();
        assert_eq!(rows, expected, "GQL, more than {least}");
        #[cfg(feature = "sql-pgq")]
        {
            let rows = db
                .execute_sql_with_params(
                    "SELECT city FROM GRAPH_TABLE (MATCH (p:P) COLUMNS (p.city AS city)) \
                     GROUP BY city HAVING COUNT(*) > $least ORDER BY city",
                    params(&[("least", Value::Int64(least))]),
                )
                .unwrap()
                .rows()
                .to_vec();
            assert_eq!(rows, expected, "SQL/PGQ, more than {least}");
        }
    }
}

/// A parameter in the second argument of a binary set function
/// (`covar_pop(y, x)`) is filled in: it reached the planner unresolved.
#[cfg(feature = "cypher")]
#[test]
fn the_second_argument_of_a_binary_set_function_takes_parameters() {
    let db = cities();
    // Amsterdam's x are 1, 2, 3 and y = 2x: covar_pop(y, x) is 4/3, and
    // covar_pop(y, k * x) is k times that.
    for k in [1_i64, 3] {
        let rows = db
            .execute_cypher_with_params(
                "MATCH (p:P {city: 'Amsterdam'}) RETURN covar_pop(p.y, p.x * $k)",
                params(&[("k", Value::Int64(k))]),
            )
            .unwrap()
            .rows()
            .to_vec();
        let Value::Float64(covariance) = rows[0][0] else {
            panic!("a covariance, not {rows:?}");
        };
        let expected = 4.0 / 3.0 * k as f64;
        assert!(
            (covariance - expected).abs() < 1e-9,
            "k = {k}: {covariance}"
        );
    }
}

/// A parameter in the WHERE of a pattern comprehension is filled in, with
/// each call's value.
#[cfg(feature = "cypher")]
#[test]
fn the_where_of_a_pattern_comprehension_takes_parameters() {
    let db = GrafeoDB::new_in_memory();
    db.execute_cypher(
        "CREATE (a:P {name: 'Alix'}), (a)-[:KNOWS]->(:P {name: 'Gus', age: 25}), \
         (a)-[:KNOWS]->(:P {name: 'Mia', age: 40})",
    )
    .unwrap();
    for (least, expected) in [(30, vec!["Mia"]), (20, vec!["Gus", "Mia"])] {
        let rows = db
            .execute_cypher_with_params(
                "MATCH (a:P {name: 'Alix'}) \
                 RETURN [(a)-[:KNOWS]->(b) WHERE b.age > $least | b.name] AS names",
                params(&[("least", Value::Int64(least))]),
            )
            .unwrap()
            .rows()
            .to_vec();
        let Value::List(names) = &rows[0][0] else {
            panic!("a list, not {rows:?}");
        };
        let mut names: Vec<Value> = names.to_vec();
        names.sort_by_key(|name| name.to_string());
        let expected: Vec<Value> = expected.iter().map(|n| Value::from(*n)).collect();
        assert_eq!(names, expected, "older than {least}");
    }
}

/// A streamed statement with a parameter nobody supplied fails as `execute`
/// does, and leaves nothing in the plan cache that a later call would run:
/// it cached the plan with the parameter unresolved, so the next `execute`
/// of the same text failed with an internal error.
#[test]
fn a_streamed_statement_fails_on_a_missing_parameter_as_execute_does() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:P {x: 1})").unwrap();
    let session = db.session();
    let query = "MATCH (n:P) WHERE n.x = $x RETURN n.x";
    let streamed = session
        .execute_streaming(query)
        .err()
        .expect("a missing parameter fails")
        .to_string();
    assert!(streamed.contains("Missing parameter: $x"), "{streamed}");
    let executed = session.execute(query).unwrap_err().to_string();
    assert!(executed.contains("Missing parameter: $x"), "{executed}");
    let rows = session
        .execute_with_params(query, params(&[("x", Value::Int64(1))]))
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(rows, [vec![Value::Int64(1)]]);
}

/// The hits and misses of the optimized-plan cache so far.
fn plan_lookups(db: &GrafeoDB) -> (u64, u64) {
    let stats = db.query_cache().stats();
    (stats.optimized_hits, stats.optimized_misses)
}

/// An index created or dropped through the API, by the database or by a
/// session, makes every cached plan again: the optimizer's choices depend on
/// the indexes (an expand starts at the end an index seeks).
#[test]
fn an_index_change_through_the_api_makes_the_plans_again() {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})")
        .unwrap();
    let session = db.session();
    let query = "MATCH (a:Person)-[:KNOWS]->(b:Person {name: 'Gus'}) RETURN a.name";
    let run = || {
        let rows = session.execute(query).unwrap().rows().to_vec();
        assert_eq!(rows, [vec![Value::from("Alix")]]);
    };
    run();
    let (hits, misses) = plan_lookups(&db);
    run();
    assert_eq!(plan_lookups(&db), (hits + 1, misses), "cached");
    let changes: [(&str, Box<dyn Fn()>); 4] = [
        (
            "the database creates",
            Box::new(|| db.create_property_index("name").unwrap()),
        ),
        (
            "the database drops",
            Box::new(|| assert!(db.drop_property_index("name").unwrap())),
        ),
        (
            "a session creates",
            Box::new(|| session.create_property_index("name").unwrap()),
        ),
        (
            "a session drops",
            Box::new(|| assert!(session.drop_property_index("name").unwrap())),
        ),
    ];
    for (what, change) in changes {
        change();
        let (hits, misses) = plan_lookups(&db);
        run();
        assert_eq!(
            plan_lookups(&db),
            (hits, misses + 1),
            "after {what} an index"
        );
    }
}

/// The plan cache keeps a plan per current schema: the same text in another
/// schema is planned for that schema's graphs, and each schema's plan is
/// found again.
#[test]
fn a_plan_is_cached_per_schema() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("CREATE SCHEMA reports").unwrap();
    session.execute("INSERT (:Row {q: 1})").unwrap();
    let query = "MATCH (r:Row) RETURN count(r)";
    let count = || session.execute(query).unwrap().rows()[0][0].clone();
    assert_eq!(count(), Value::Int64(1));
    session.execute("SESSION SET SCHEMA reports").unwrap();
    let (hits, misses) = plan_lookups(&db);
    assert_eq!(count(), Value::Int64(0), "the schema's own default graph");
    assert_eq!(
        plan_lookups(&db),
        (hits, misses + 1),
        "planned for the schema"
    );
    assert_eq!(count(), Value::Int64(0));
    assert_eq!(plan_lookups(&db), (hits + 1, misses + 1), "found again");
    // The session command is looked up too (a miss: it is never cached).
    session.execute("SESSION RESET SCHEMA").unwrap();
    let (hits, misses) = plan_lookups(&db);
    assert_eq!(count(), Value::Int64(1));
    assert_eq!(
        plan_lookups(&db),
        (hits + 1, misses),
        "the default schema's plan"
    );
}

/// Two statements that differ only inside a string, a quoted name or a
/// comment are two statements: the plan cache kept one plan for both when
/// it collapsed the whitespace of the whole text, so `RETURN 'a  b'` after
/// `RETURN 'a b'` returned "a b". Each pair runs in both orders, through
/// GQL and Cypher, with and without values.
#[test]
fn texts_that_differ_inside_a_literal_are_planned_apart() {
    let pairs: [(&str, &str, Value, Value); 4] = [
        (
            "RETURN 'a b' AS s",
            "RETURN 'a  b' AS s",
            Value::from("a b"),
            Value::from("a  b"),
        ),
        (
            "RETURN \"x\ty\" AS s",
            "RETURN \"x y\" AS s",
            Value::from("x\ty"),
            Value::from("x y"),
        ),
        (
            "MATCH (p:P) RETURN p.`a b` AS v",
            "MATCH (p:P) RETURN p.`a  b` AS v",
            Value::Int64(1),
            Value::Int64(2),
        ),
        (
            "RETURN 'a b' AS s /* don't */",
            "RETURN 'a  b' AS s /* don't */",
            Value::from("a b"),
            Value::from("a  b"),
        ),
    ];
    for (first, second, first_value, second_value) in pairs {
        for (a, b, a_value, b_value) in [
            (first, second, &first_value, &second_value),
            (second, first, &second_value, &first_value),
        ] {
            let db = GrafeoDB::new_in_memory();
            db.execute("INSERT (:P {`a  b`: 2, `a b`: 1})").unwrap();
            for (query, value) in [(a, a_value), (b, b_value), (a, a_value)] {
                let gql = db.execute(query).unwrap().rows().to_vec();
                assert_eq!(gql, [vec![value.clone()]], "GQL `{query}`");
                let with_values = db
                    .execute_with_params(query, params(&[("unused", Value::Int64(0))]))
                    .unwrap()
                    .rows()
                    .to_vec();
                assert_eq!(
                    with_values,
                    [vec![value.clone()]],
                    "GQL with values `{query}`"
                );
                #[cfg(feature = "cypher")]
                {
                    let cypher = db.execute_cypher(query).unwrap().rows().to_vec();
                    assert_eq!(cypher, [vec![value.clone()]], "Cypher `{query}`");
                }
            }
        }
    }
}
