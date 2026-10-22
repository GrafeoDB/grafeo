//! Values and types in queries: where `ORDER BY` puts nulls (#571), comparisons
//! of zoned datetimes (#584), the values a typed property takes (#568), closed
//! graph types (#567) and calls of functions that do not exist (#570).
//!
//! Queries quoted from Microsoft Fabric's GQL documentation
//! (<https://learn.microsoft.com/en-us/fabric/graph/>) are MIT, Copyright (c)
//! Microsoft Corporation.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test values_and_types
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use grafeo_common::types::{Timestamp, Value};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// The values of the first column, in row order.
fn column(result: &QueryResult) -> Vec<Value> {
    result.rows().iter().map(|row| row[0].clone()).collect()
}

fn ints(values: &[Option<i64>]) -> Vec<Value> {
    values
        .iter()
        .map(|value| value.map_or(Value::Null, Value::Int64))
        .collect()
}

// === #571: where ORDER BY puts nulls ===

/// Alix ranks 19, Gus 3, Vincent has no rank.
fn ranked() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for insert in [
        "INSERT (:Person {name: 'Alix', rank: 19})",
        "INSERT (:Person {name: 'Gus', rank: 3})",
        "INSERT (:Person {name: 'Vincent'})",
    ] {
        db.execute(insert).unwrap();
    }
    db
}

/// ISO/IEC 39075:2024 leaves the null ordering of a sort key without
/// `NULLS FIRST` or `NULLS LAST` to the implementation (as SQL does, 10.10
/// <sort specification list> of ISO/IEC 9075-2). Grafeo's GQL puts nulls last
/// in both directions, as Microsoft Fabric's GQL does.
#[test]
fn gql_order_by_puts_nulls_last_ascending_and_descending() {
    let db = ranked();
    let query = |order: &str| {
        column(
            &db.execute(&format!(
                "MATCH (n:Person) RETURN n.rank AS rank ORDER BY rank {order}"
            ))
            .unwrap(),
        )
    };
    assert_eq!(query("ASC"), ints(&[Some(3), Some(19), None]), "ASC");
    assert_eq!(query("DESC"), ints(&[Some(19), Some(3), None]), "DESC");
    assert_eq!(query(""), ints(&[Some(3), Some(19), None]), "no direction");
    assert_eq!(
        query("DESC NULLS FIRST"),
        ints(&[None, Some(19), Some(3)]),
        "an explicit NULLS FIRST still wins"
    );
    assert_eq!(
        query("ASC NULLS FIRST"),
        ints(&[None, Some(3), Some(19)]),
        "an explicit NULLS FIRST still wins"
    );
}

/// The default holds for every way a GQL query sorts: a key that is not
/// returned, a top-K under LIMIT, the keys of an aggregation, and a sort on
/// several keys.
#[test]
fn gql_nulls_last_holds_for_hidden_keys_limits_and_aggregates() {
    let db = ranked();
    let names = |query: &str| column(&db.execute(query).unwrap());
    assert_eq!(
        names("MATCH (n:Person) RETURN n.name AS name ORDER BY n.rank DESC"),
        vec![
            Value::from("Alix"),
            Value::from("Gus"),
            Value::from("Vincent")
        ],
        "a sort key that is not returned"
    );
    assert_eq!(
        names("MATCH (n:Person) RETURN n.rank AS rank ORDER BY rank DESC LIMIT 2"),
        ints(&[Some(19), Some(3)]),
        "a top-K keeps the largest values, not the null"
    );
    assert_eq!(
        names("MATCH (n:Person) RETURN n.rank AS rank, count(*) AS people ORDER BY rank DESC"),
        ints(&[Some(19), Some(3), None]),
        "the keys of an aggregation"
    );
    assert_eq!(
        names("MATCH (n:Person) RETURN n.name AS name ORDER BY n.rank DESC, name DESC"),
        vec![
            Value::from("Alix"),
            Value::from("Gus"),
            Value::from("Vincent")
        ],
        "several keys"
    );
}

/// Cypher keeps openCypher's order: null is the largest value, so it comes
/// last ascending and first descending.
#[test]
fn cypher_order_by_keeps_nulls_as_the_largest_value() {
    let db = ranked();
    let session = db.session();
    let query = |order: &str| {
        column(
            &session
                .execute_cypher(&format!(
                    "MATCH (n:Person) RETURN n.rank AS rank ORDER BY rank {order}"
                ))
                .unwrap(),
        )
    };
    assert_eq!(query("ASC"), ints(&[Some(3), Some(19), None]), "ASC");
    assert_eq!(query("DESC"), ints(&[None, Some(19), Some(3)]), "DESC");
}

// === #584: zoned datetimes compare as instants ===

/// The single value a query returns.
fn single(db: &GrafeoDB, query: &str) -> Value {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    assert_eq!(result.rows().len(), 1, "{query} returns one row");
    result.rows()[0][0].clone()
}

/// ISO/IEC 39075:2024 19.3 <comparison predicate>: zoned datetimes compare by
/// the instant they denote, so every ordering operator agrees with `=` and
/// with `ORDER BY`. 2022-01-01T00:00+01:00 is 2021-12-31T23:00Z, before
/// 2021-12-31T23:30Z although its date is later.
#[test]
fn zoned_datetimes_compare_as_instants_with_every_operator() {
    let db = GrafeoDB::new_in_memory();
    let earlier = "ZONED DATETIME '2022-01-01T00:00:00+01:00'";
    let later = "ZONED_DATETIME('2021-12-31T23:30:00Z')";
    for (op, expected) in [
        ("<", true),
        ("<=", true),
        (">", false),
        (">=", false),
        ("=", false),
        ("<>", true),
    ] {
        assert_eq!(
            single(&db, &format!("RETURN {earlier} {op} {later} AS v")),
            Value::Bool(expected),
            "{earlier} {op} {later}"
        );
        assert_eq!(
            single(&db, &format!("RETURN {later} {op} {earlier} AS v")),
            Value::Bool(match op {
                "<" | "<=" => !expected,
                ">" | ">=" => !expected,
                _ => expected,
            }),
            "{later} {op} {earlier}"
        );
    }
    let same = "ZONED DATETIME '2021-12-31T23:00:00Z'";
    for (op, expected) in [("<", false), ("<=", true), (">=", true), ("=", true)] {
        assert_eq!(
            single(&db, &format!("RETURN {earlier} {op} {same} AS v")),
            Value::Bool(expected),
            "the same instant at another offset: {op}"
        );
    }
}

/// A timestamp compares with a zoned datetime as an instant in UTC, never as
/// a silent false or null.
#[test]
fn a_timestamp_and_a_zoned_datetime_compare_as_instants() {
    let db = GrafeoDB::new_in_memory();
    let timestamp = "datetime('2024-03-15T10:30:00Z')";
    let same = "ZONED DATETIME '2024-03-15T11:30:00+01:00'";
    let after = "ZONED DATETIME '2024-03-15T11:31:00+01:00'";
    for (left, op, right, expected) in [
        (timestamp, "=", same, true),
        (same, "=", timestamp, true),
        (timestamp, "<>", same, false),
        (timestamp, "<", after, true),
        (after, ">", timestamp, true),
        (timestamp, ">=", same, true),
        (timestamp, ">", after, false),
    ] {
        assert_eq!(
            single(&db, &format!("RETURN {left} {op} {right} AS v")),
            Value::Bool(expected),
            "{left} {op} {right}"
        );
    }
}

/// Events at zoned instants and one at a timestamp, all after 2020 but
/// Gus's.
fn events() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for insert in [
        "INSERT (:Event {name: 'Alix', at: ZONED DATETIME '2024-03-15T10:30:00+01:00'})",
        "INSERT (:Event {name: 'Gus', at: ZONED DATETIME '2019-03-15T10:30:00Z'})",
        "INSERT (:Event {name: 'Mia', at: datetime('2024-03-15T10:30:00Z')})",
        "INSERT (:Event {name: 'Jules', at: ZONED DATETIME '2020-01-01T00:30:00+01:00'})",
    ] {
        db.execute(insert).unwrap();
    }
    db
}

fn sorted_names(db: &GrafeoDB, query: &str) -> Vec<String> {
    let result = db
        .execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"));
    let mut names: Vec<String> = result
        .rows()
        .iter()
        .map(|row| row[0].as_str().unwrap().to_string())
        .collect();
    names.sort();
    names
}

/// A filter on stored zoned values keeps the rows after the instant, through
/// every way a filter runs: `WHERE` and `FILTER`, the literal on either side,
/// a range of two bounds, and a property index. Jules at 00:30+01:00 is
/// 23:30Z, before 2020.
#[test]
fn filters_on_stored_zoned_datetimes_keep_the_rows_after_the_instant() {
    let db = events();
    let after_2020 = vec!["Alix".to_string(), "Mia".to_string()];
    for query in [
        "MATCH (n:Event) WHERE n.at > ZONED DATETIME '2020-01-01T00:00:00Z' RETURN n.name",
        "MATCH (n:Event) WHERE n.at >= ZONED_DATETIME('2020-01-01T00:00:00Z') RETURN n.name",
        "MATCH (n:Event) WHERE ZONED_DATETIME('2020-01-01T00:00:00Z') < n.at RETURN n.name",
        "MATCH (n:Event) FILTER n.at > ZONED DATETIME '2020-01-01T00:00:00Z' RETURN n.name",
        "MATCH (n:Event) WHERE n.at > ZONED DATETIME '2020-01-01T00:00:00Z' \
         AND n.at < ZONED DATETIME '2030-01-01T00:00:00Z' RETURN n.name",
        "MATCH (n:Event) WHERE n.at > datetime('2020-01-01T00:00:00Z') RETURN n.name",
    ] {
        assert_eq!(sorted_names(&db, query), after_2020, "{query}");
    }
    db.execute("CREATE INDEX event_at FOR (e:Event) ON (e.at)")
        .unwrap();
    assert_eq!(
        sorted_names(
            &db,
            "MATCH (n:Event) WHERE n.at > ZONED DATETIME '2020-01-01T00:00:00Z' RETURN n.name"
        ),
        after_2020,
        "with a property index on the property"
    );
}

/// `min` and `max` of zoned datetimes are the earliest and the latest
/// instant (they returned the first value).
#[test]
fn min_and_max_of_zoned_datetimes_are_the_earliest_and_latest_instants() {
    let db = events();
    let result = db
        .execute(
            "MATCH (n:Event) WHERE n.name <> 'Mia' \
             RETURN min(n.at) AS earliest, max(n.at) AS latest",
        )
        .unwrap();
    assert_eq!(
        result.rows()[0][0].to_string(),
        "2019-03-15T10:30:00Z",
        "min"
    );
    assert_eq!(
        result.rows()[0][1].to_string(),
        "2024-03-15T10:30:00+01:00",
        "max"
    );
}

// === #568 and #584: the value a typed property takes ===

/// A shop whose customers spend a `FLOAT64` and buy at a `FLOAT64` price.
fn shop() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "CREATE NODE TYPE Customer (customerId STRING NOT NULL, totalSpend FLOAT64, \
         scores LIST<FLOAT64>)",
    )
    .unwrap();
    db.execute("CREATE NODE TYPE Product (sku STRING NOT NULL)")
        .unwrap();
    db.execute("CREATE EDGE TYPE buys (price FLOAT64)").unwrap();
    db
}

/// The `totalSpend` of the customer with `id`.
fn spend(db: &GrafeoDB, id: &str) -> Value {
    single(
        db,
        &format!("MATCH (c:Customer {{customerId: '{id}'}}) RETURN c.totalSpend"),
    )
}

/// ISO/IEC 39075:2024 allows an implicit conversion between numeric types on
/// assignment, as SQL's store assignment does: an `INT64` written to a
/// `FLOAT64` property is stored as a `FLOAT64`. JavaScript sends every whole
/// number as an integer, so a typed schema took no ordinary price.
#[test]
fn an_integer_written_to_a_float64_property_is_stored_as_a_float() {
    let db = shop();
    db.execute("INSERT (:Customer {customerId: 'a', totalSpend: 3})")
        .unwrap();
    assert_eq!(spend(&db, "a"), Value::Float64(3.0), "INSERT");

    db.execute_with_params(
        "INSERT (:Customer {customerId: 'b', totalSpend: $s})",
        [("s".to_string(), Value::Int64(19))].into_iter().collect(),
    )
    .unwrap();
    assert_eq!(spend(&db, "b"), Value::Float64(19.0), "a parameter");

    db.execute("MATCH (c:Customer {customerId: 'a'}) SET c.totalSpend = 88")
        .unwrap();
    assert_eq!(spend(&db, "a"), Value::Float64(88.0), "SET n.p");
    db.execute("MATCH (c:Customer {customerId: 'a'}) SET c += {totalSpend: 3}")
        .unwrap();
    assert_eq!(spend(&db, "a"), Value::Float64(3.0), "SET n += {{...}}");
    db.execute("MATCH (c:Customer {customerId: 'a'}) SET c = {customerId: 'a', totalSpend: 19}")
        .unwrap();
    assert_eq!(spend(&db, "a"), Value::Float64(19.0), "SET n = {{...}}");
    db.execute("MERGE (c:Customer {customerId: 'c'}) ON CREATE SET c.totalSpend = 3")
        .unwrap();
    assert_eq!(
        spend(&db, "c"),
        Value::Float64(3.0),
        "MERGE ... ON CREATE SET"
    );

    db.execute("INSERT (:Customer {customerId: 'd', scores: [3, 19.5]})")
        .unwrap();
    assert_eq!(
        single(&db, "MATCH (c:Customer {customerId: 'd'}) RETURN c.scores"),
        Value::List(vec![Value::Float64(3.0), Value::Float64(19.5)].into()),
        "the items of a LIST<FLOAT64>"
    );

    db.execute(
        "MATCH (c:Customer {customerId: 'a'}) \
         INSERT (c)-[:buys {price: 88}]->(:Product {sku: 'p1'})",
    )
    .unwrap();
    assert_eq!(
        single(&db, "MATCH ()-[b:buys]->() RETURN b.price"),
        Value::Float64(88.0),
        "an edge property"
    );
    db.execute("MATCH ()-[b:buys]->() SET b.price = 3").unwrap();
    assert_eq!(
        single(&db, "MATCH ()-[b:buys]->() RETURN b.price"),
        Value::Float64(3.0),
        "SET on an edge property"
    );
}

/// The direct API converts the same way.
#[test]
fn the_direct_api_stores_an_integer_in_a_float64_property_as_a_float() {
    let db = shop();
    let alix = db
        .create_node_with_props(
            &["Customer"],
            [
                ("customerId", Value::from("alix")),
                ("totalSpend", Value::Int64(3)),
            ],
        )
        .unwrap();
    assert_eq!(
        spend(&db, "alix"),
        Value::Float64(3.0),
        "create_node_with_props"
    );
    db.set_node_property(alix, "totalSpend", Value::Int64(19))
        .unwrap();
    assert_eq!(
        spend(&db, "alix"),
        Value::Float64(19.0),
        "set_node_property"
    );
    let product = db
        .create_node_with_props(&["Product"], [("sku", Value::from("p1"))])
        .unwrap();
    let buys = db
        .create_edge_with_props(alix, product, "buys", [("price", Value::Int64(88))])
        .unwrap();
    assert_eq!(
        db.get_edge(buys).unwrap().get_property("price").cloned(),
        Some(Value::Float64(88.0)),
        "create_edge_with_props"
    );
}

/// An integer that has no exact `FLOAT64` value (beyond 2^53) is refused,
/// not rounded; 2^53 itself converts.
#[test]
fn an_integer_without_an_exact_float_is_refused_for_a_float64_property() {
    let db = shop();
    db.execute("INSERT (:Customer {customerId: 'a', totalSpend: 9007199254740992})")
        .unwrap();
    assert_eq!(
        spend(&db, "a"),
        Value::Float64(9_007_199_254_740_992.0),
        "2^53"
    );
    for value in [
        "9007199254740993",
        "9223372036854775807",
        "-9007199254740993",
    ] {
        let error = db
            .execute(&format!(
                "INSERT (:Customer {{customerId: 'b', totalSpend: {value}}})"
            ))
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("totalSpend") && error.contains("exact"),
            "{value}: {error}"
        );
    }
    assert_eq!(
        single(&db, "MATCH (c:Customer) RETURN count(c)"),
        Value::Int64(1),
        "nothing refused was written"
    );
}

/// A float is not narrowed to an `INT64` property.
#[test]
fn a_float_written_to_an_int64_property_is_still_refused() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE Stop (minutes INT64)").unwrap();
    assert!(db.execute("INSERT (:Stop {minutes: 3.5})").is_err());
    assert!(db.execute("INSERT (:Stop {minutes: 3.0})").is_err());
}

/// A zoned datetime written to a `DATETIME` property is stored as its
/// instant, so the property compares with zoned values (#584). A date is
/// refused with the property in the message.
#[test]
fn a_zoned_datetime_written_to_a_datetime_property_is_stored_as_its_instant() {
    let db = GrafeoDB::new_in_memory();
    db.execute("CREATE NODE TYPE Trip (departs DATETIME)")
        .unwrap();
    db.execute("INSERT (:Trip {departs: CAST('2024-03-15T10:30:00Z' AS DATETIME)})")
        .unwrap();
    db.execute("INSERT (:Trip {departs: ZONED_DATETIME('2024-03-15T10:30:00+01:00')})")
        .unwrap();
    let result = db
        .execute("MATCH (t:Trip) RETURN t.departs ORDER BY t.departs")
        .unwrap();
    assert_eq!(
        column(&result),
        vec![
            Value::Timestamp(Timestamp::from_micros(1_710_495_000_000_000)),
            Value::Timestamp(Timestamp::from_micros(1_710_498_600_000_000)),
        ],
        "09:30Z and 10:30Z, both timestamps"
    );
    assert_eq!(
        single(
            &db,
            "MATCH (t:Trip) WHERE t.departs > ZONED_DATETIME('2020-01-01T00:00:00Z') \
             RETURN count(t)"
        ),
        Value::Int64(2),
        "a DATETIME property compares with a zoned value"
    );
    let error = db
        .execute("INSERT (:Trip {departs: date('2024-03-15')})")
        .unwrap_err()
        .to_string();
    assert!(error.contains("departs"), "{error}");
}

// === #567: a closed graph type holds only what it declares ===

/// The shop of #567 as a closed graph type `shop`, graph `g` typed by it,
/// and graph `loose` typed by an open graph type over the same node types.
fn closed_shop() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for statement in [
        "CREATE NODE TYPE Customer (customerId STRING NOT NULL, totalSpend FLOAT64)",
        "CREATE NODE TYPE Product (sku STRING NOT NULL)",
        "CREATE EDGE TYPE buys CONNECTING (Customer) TO (Product)",
        "CREATE GRAPH TYPE shop (NODE TYPE Customer, NODE TYPE Product, EDGE TYPE buys)",
        "CREATE GRAPH g TYPED shop",
        "CREATE GRAPH TYPE market {node_types: [Customer, Product], edge_types: [buys], open: true}",
        "CREATE GRAPH loose TYPED market",
    ] {
        db.execute(statement)
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    db
}

/// Runs `statement` in graph `graph` and returns its error, failing when it
/// succeeds.
fn refused(db: &GrafeoDB, graph: &str, statement: &str) -> String {
    let session = db.session();
    session.use_graph(graph);
    match session.execute(statement) {
        Ok(_) => panic!("{statement} in {graph} succeeded"),
        Err(error) => error.to_string(),
    }
}

fn run_in(db: &GrafeoDB, graph: &str, statement: &str) {
    let session = db.session();
    session.use_graph(graph);
    session
        .execute(statement)
        .unwrap_or_else(|error| panic!("{statement} in {graph}: {error}"));
}

fn count_in(db: &GrafeoDB, graph: &str, query: &str) -> Value {
    let session = db.session();
    session.use_graph(graph);
    session.execute(query).unwrap().rows()[0][0].clone()
}

/// ISO/IEC 39075:2024 4.13 (graph types): a graph of a closed graph type
/// holds only nodes and edges its node and edge types describe. A label that
/// is not one of the graph type's node types is refused, with the label and
/// the graph type in the message.
#[test]
fn a_closed_graph_type_refuses_a_label_it_does_not_declare() {
    let db = closed_shop();
    for statement in [
        "INSERT (:Spaceship {id: 'x'})",
        "INSERT (:Customer:Spaceship {customerId: 'c1'})",
        "INSERT ()",
    ] {
        let error = refused(&db, "g", statement);
        assert!(error.contains("shop"), "{statement}: {error}");
        if statement.contains("Spaceship") {
            assert!(error.contains("Spaceship"), "{statement}: {error}");
        }
    }
    run_in(&db, "g", "INSERT (:Customer {customerId: 'c1'})");
    let error = refused(&db, "g", "MATCH (c:Customer) SET c:Spaceship");
    assert!(error.contains("Spaceship"), "SET n:Label: {error}");
    assert_eq!(
        count_in(&db, "g", "MATCH (n) RETURN count(n)"),
        Value::Int64(1),
        "only the declared node was written"
    );
    assert_eq!(
        count_in(&db, "g", "MATCH (n:Spaceship) RETURN count(n)"),
        Value::Int64(0),
        "the refused label was not added"
    );
}

/// A property a node type of the closed graph type does not declare is
/// refused, through INSERT and every form of SET, with the property, the
/// node type and the graph type in the message.
#[test]
fn a_closed_graph_type_refuses_a_property_its_node_type_does_not_declare() {
    let db = closed_shop();
    let error = refused(
        &db,
        "g",
        "INSERT (:Customer {customerId: 'c5', shoeSize: 44})",
    );
    for name in ["shoeSize", "Customer", "shop"] {
        assert!(error.contains(name), "INSERT names {name}: {error}");
    }
    run_in(
        &db,
        "g",
        "INSERT (:Customer {customerId: 'c5', totalSpend: 19})",
    );
    for statement in [
        "MATCH (c:Customer) SET c.shoeSize = 44",
        "MATCH (c:Customer) SET c += {shoeSize: 44}",
        "MATCH (c:Customer) SET c = {customerId: 'c5', shoeSize: 44}",
    ] {
        let error = refused(&db, "g", statement);
        assert!(error.contains("shoeSize"), "{statement}: {error}");
    }
    assert_eq!(
        count_in(
            &db,
            "g",
            "MATCH (c:Customer) WHERE c.shoeSize IS NOT NULL RETURN count(c)"
        ),
        Value::Int64(0),
        "no undeclared property was written"
    );
    run_in(&db, "g", "MATCH (c:Customer) SET c.totalSpend = 88");
    run_in(&db, "g", "MATCH (c:Customer) REMOVE c.totalSpend");
}

/// An edge of a type the closed graph type does not declare, or with a
/// property its edge type does not declare, is refused.
#[test]
fn a_closed_graph_type_refuses_undeclared_edge_types_and_edge_properties() {
    let db = closed_shop();
    run_in(
        &db,
        "g",
        "INSERT (:Customer {customerId: 'c1'})-[:buys]->(:Product {sku: 'p1'})",
    );
    let error = refused(
        &db,
        "g",
        "MATCH (c:Customer), (p:Product) INSERT (c)-[:likes]->(p)",
    );
    assert!(
        error.contains("likes") && error.contains("shop"),
        "an undeclared edge type: {error}"
    );
    let error = refused(
        &db,
        "g",
        "MATCH (c:Customer), (p:Product) INSERT (c)-[:buys {quantity: 3}]->(p)",
    );
    assert!(
        error.contains("quantity") && error.contains("buys"),
        "an undeclared edge property: {error}"
    );
    let error = refused(&db, "g", "MATCH ()-[b:buys]->() SET b.quantity = 3");
    assert!(error.contains("quantity"), "SET on an edge: {error}");
    assert_eq!(
        count_in(&db, "g", "MATCH ()-[r]->() RETURN count(r)"),
        Value::Int64(1),
        "only the declared edge was written"
    );
}

/// A graph handle and the direct API check the same rules as a session in
/// the graph.
#[test]
fn a_graph_handle_checks_the_closed_graph_type_too() {
    let db = closed_shop();
    let g = db.graph("g").unwrap();
    assert!(
        g.execute("INSERT (:Spaceship {id: 'x'})").is_err(),
        "a query"
    );
    assert!(
        g.execute("INSERT (:Customer {customerId: 'c5', shoeSize: 44})")
            .is_err(),
        "a query with an undeclared property"
    );
    assert!(
        g.create_node_with_props(&["Spaceship"], [("id", Value::from("x"))])
            .is_err(),
        "create_node_with_props with an undeclared label"
    );
    assert!(
        g.create_node_with_props(
            &["Customer"],
            [
                ("customerId", Value::from("c5")),
                ("shoeSize", Value::Int64(44)),
            ],
        )
        .is_err(),
        "create_node_with_props with an undeclared property"
    );
    let alix = g
        .create_node_with_props(&["Customer"], [("customerId", Value::from("alix"))])
        .unwrap();
    assert!(
        g.set_node_property(alix, "shoeSize", Value::Int64(44))
            .is_err(),
        "set_node_property"
    );
    assert_eq!(
        g.execute("MATCH (n) RETURN count(n)").unwrap().rows()[0][0],
        Value::Int64(1)
    );
}

/// An open graph type, a graph without a type and the default graph take
/// any label and property, even with the same node types defined.
#[test]
fn open_and_untyped_graphs_take_any_label_and_property() {
    let db = closed_shop();
    db.execute("CREATE GRAPH free").unwrap();
    for graph in ["loose", "free", "default"] {
        run_in(&db, graph, "INSERT (:Spaceship {id: 'x'})");
        run_in(
            &db,
            graph,
            "INSERT (:Customer {customerId: 'c5', shoeSize: 44})-[:likes {since: 2019}]->(:Product {sku: 'p'})",
        );
    }
}

// === #570: calls of functions that do not exist ===

/// Alix knows Gus, Gus knows Vincent.
fn chain() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute(
        "INSERT (:Person {name: 'Alix'})-[:knows]->(:Person {name: 'Gus'})\
         -[:knows]->(:Person {name: 'Vincent'})",
    )
    .unwrap();
    db
}

/// The error of `query` in `language`, failing when it succeeds.
fn error_of(db: &GrafeoDB, language: &str, query: &str) -> String {
    let session = db.session();
    let result = match language {
        "cypher" => session.execute_cypher(query),
        _ => session.execute(query),
    };
    match result {
        Ok(result) => panic!("[{language}] {query} succeeded: {:?}", result.rows()),
        Err(error) => error.to_string(),
    }
}

/// ISO/IEC 39075:2024 20.1 <value expression>: a function call names a
/// function the implementation defines, so a call of an unknown one is a
/// syntax or semantic error, not a null. It fails while the query is planned,
/// wherever the call is, with the name and a close one to try.
#[test]
fn a_call_of_an_unknown_function_fails_with_its_name() {
    let db = chain();
    for (language, query) in [
        ("gql", "RETURN nope(1) AS v"),
        ("cypher", "RETURN nope(1) AS v"),
        ("gql", "MATCH (n:Person) WHERE nope(n.name) RETURN n.name"),
        (
            "cypher",
            "MATCH (n:Person) WHERE nope(n.name) RETURN n.name",
        ),
        (
            "gql",
            "MATCH (n:Person) RETURN n.name ORDER BY nope(n.name)",
        ),
        ("gql", "MATCH (n:Person) RETURN sum(nope(n.name)) AS s"),
        ("gql", "MATCH (n:Person) RETURN count(*) AS c, nope(1) AS v"),
        ("gql", "MATCH (n:Person {name: 'Alix'}) SET n.e = nope(1)"),
        ("gql", "INSERT (:Person {name: nope(1)})"),
        ("cypher", "CREATE (:Person {name: nope(1)})"),
    ] {
        let error = error_of(&db, language, query);
        assert!(
            error.contains("nope") && error.to_lowercase().contains("unknown function"),
            "[{language}] {query}: {error}"
        );
    }
    let error = error_of(&db, "gql", "RETURN toUpperr('a') AS v");
    assert!(
        error.contains("toUpper"),
        "a close name is suggested: {error}"
    );
    // Nothing was written by the refused statements.
    assert_eq!(
        single(&db, "MATCH (n) RETURN count(n)"),
        Value::Int64(3),
        "the refused INSERT wrote nothing"
    );
    assert_eq!(
        single(
            &db,
            "MATCH (n:Person {name: 'Alix'}) RETURN n.e IS NULL AS unset"
        ),
        Value::Bool(true),
        "the refused SET wrote nothing"
    );
}

/// A call with a number of arguments the function does not take fails
/// like an unknown name, with the numbers it takes (it stored null:
/// `SET n.embedding = vector($vec, 384)`).
#[test]
fn a_call_with_the_wrong_number_of_arguments_fails_with_the_accepted_ones() {
    let db = chain();
    for (language, query, name, accepted) in [
        ("gql", "RETURN toUpper('a', 'b') AS v", "toUpper", "1"),
        ("cypher", "RETURN toUpper('a', 'b') AS v", "toUpper", "1"),
        (
            "gql",
            "RETURN substring('Amsterdam') AS v",
            "substring",
            "2 or 3",
        ),
        (
            "gql",
            "MATCH (n:Person {name: 'Alix'}) SET n.embedding = vector([3.0, 19.0], 88)",
            "vector",
            "1",
        ),
        ("cypher", "RETURN trim('a', 'b') AS v", "trim", "1 or 3"),
    ] {
        let error = error_of(&db, language, query);
        assert!(
            error.contains(name) && error.contains(accepted) && error.contains("argument"),
            "[{language}] {query}: {error}"
        );
    }
    assert_eq!(
        single(
            &db,
            "MATCH (n:Person {name: 'Alix'}) RETURN n.embedding IS NULL AS unset"
        ),
        Value::Bool(true),
        "the refused SET stored nothing"
    );
}

/// Known functions keep working, also in their other spellings and with
/// every number of arguments they take.
#[test]
fn known_functions_keep_working_with_every_arity_they_take() {
    let db = chain();
    for (query, expected) in [
        ("RETURN upper('ab') AS v", Value::from("AB")),
        ("RETURN toUpper('ab') AS v", Value::from("AB")),
        ("RETURN TOUPPER('ab') AS v", Value::from("AB")),
        (
            "RETURN substring('Amsterdam', 2) AS v",
            Value::from("sterdam"),
        ),
        (
            "RETURN substring('Amsterdam', 0, 3) AS v",
            Value::from("Ams"),
        ),
        ("RETURN trim('  a  ') AS v", Value::from("a")),
        ("RETURN coalesce(null, 3, 19) AS v", Value::Int64(3)),
        ("RETURN size(range(3, 19, 8)) AS v", Value::Int64(3)),
        ("RETURN date() IS NOT NULL AS v", Value::Bool(true)),
        ("RETURN pi() > 3 AS v", Value::Bool(true)),
    ] {
        assert_eq!(single(&db, query), expected, "{query}");
    }
}

/// GQL's `path_length(p)` (ISO/IEC 39075:2024 20.21 <length expression>,
/// `PATH_LENGTH`) is the number of edges of `p`, like `length(p)`, also for
/// `ANY SHORTEST` paths.
#[test]
fn path_length_is_the_number_of_edges_of_a_path() {
    let db = chain();
    for (language, query) in [
        (
            "gql",
            "MATCH p = (a:Person {name: 'Alix'})-[:knows]->{1,4}(b) \
             RETURN b.name AS b, path_length(p) AS hops ORDER BY hops",
        ),
        (
            "gql",
            "MATCH p = ANY SHORTEST (a:Person {name: 'Alix'})-[:knows]->{1,4}(b) \
             RETURN b.name AS b, path_length(p) AS hops ORDER BY hops",
        ),
        (
            "cypher",
            "MATCH p = (a:Person {name: 'Alix'})-[:knows*1..4]->(b) \
             RETURN b.name AS b, path_length(p) AS hops ORDER BY hops",
        ),
    ] {
        let session = db.session();
        let result = match language {
            "cypher" => session.execute_cypher(query),
            _ => session.execute(query),
        }
        .unwrap_or_else(|error| panic!("[{language}] {query}: {error}"));
        let rows: Vec<(Value, Value)> = result
            .rows()
            .iter()
            .map(|row| (row[0].clone(), row[1].clone()))
            .collect();
        assert_eq!(
            rows,
            vec![
                (Value::from("Gus"), Value::Int64(1)),
                (Value::from("Vincent"), Value::Int64(2)),
            ],
            "[{language}] {query}"
        );
    }
    assert_eq!(
        single(
            &db,
            "MATCH p = (:Person {name: 'Alix'})-[:knows]->(:Person) RETURN path_length(p) AS hops"
        ),
        Value::Int64(1),
        "a fixed-length path"
    );
}
