//! Node and edge types created inside a schema (`SESSION SET SCHEMA s1;
//! CREATE NODE TYPE Customer (...)`) check the nodes and edges written in that
//! schema: property types, `NOT NULL`, inherited properties and defaults, from
//! GQL, Cypher and the direct API (the database's, a session's and a graph
//! handle's). Such a type is registered as `s1/Customer`, and the checks used
//! to look it up by the bare label, so its nodes were never checked. Types of
//! another schema, the default one included, do not apply: as `SHOW NODE
//! TYPES` shows, a schema sees its own types only.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test types_in_a_schema
//! ```

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::Result;
use grafeo_engine::database::QueryResult;
use grafeo_engine::{GrafeoDB, Session};

/// Schemas `s1` and `s2`. In `s1`: node type `Customer (customerId STRING NOT
/// NULL, age INT64)` and edge type `Buys (amount INT64 NOT NULL)`; in `s2` a
/// `Customer (age STRING)` of its own.
fn two_schemas() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for statement in [
        "CREATE SCHEMA s1",
        "CREATE SCHEMA s2",
        "SESSION SET SCHEMA s1",
        "CREATE NODE TYPE Customer (customerId STRING NOT NULL, age INT64)",
        "CREATE EDGE TYPE Buys (amount INT64 NOT NULL)",
        "SESSION SET SCHEMA s2",
        "CREATE NODE TYPE Customer (age STRING)",
    ] {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("`{statement}` failed: {error}"));
    }
    db
}

/// A session working in `schema`.
fn session_in(db: &GrafeoDB, schema: &str) -> Session {
    let session = db.session();
    session
        .execute(&format!("SESSION SET SCHEMA {schema}"))
        .unwrap();
    session
}

/// Each statement, run by `run`, fails with a message that contains its
/// expected part.
fn assert_refused(run: impl Fn(&str) -> Result<QueryResult>, cases: &[(&str, &str)]) {
    for (statement, expected) in cases {
        match run(statement) {
            Ok(result) => panic!("`{statement}` succeeded with {:?}", result.rows()),
            Err(error) => assert!(
                error.to_string().contains(expected),
                "`{statement}`: expected an error with `{expected}`, got: {error}"
            ),
        }
    }
}

fn count(session: &Session, query: &str) -> Value {
    session.execute(query).unwrap().rows()[0][0].clone()
}

#[test]
fn gql_writes_in_the_schema_are_checked() {
    let db = two_schemas();
    let session = session_in(&db, "s1");
    assert_refused(
        |statement| session.execute(statement),
        &[
            ("INSERT (:Customer {age: 3})", "customerId"),
            ("INSERT (:Customer {customerId: 'c3', age: 'old'})", "age"),
            (
                "INSERT (:Customer {customerId: NULL, age: 3})",
                "customerId",
            ),
            ("INSERT (:Shop)-[:Buys {amount: 'x'}]->(:Item)", "amount"),
            ("INSERT (:Shop)-[:Buys]->(:Item)", "amount"),
        ],
    );
    session
        .execute("INSERT (:Customer {customerId: 'c3', age: 19})-[:Buys {amount: 88}]->(:Item)")
        .unwrap();
    assert_refused(
        |statement| session.execute(statement),
        &[
            ("MATCH (c:Customer) SET c.age = 'old'", "age"),
            ("MATCH (c:Customer) SET c.customerId = NULL", "customerId"),
            ("MATCH (c:Customer) REMOVE c.customerId", "customerId"),
            ("MATCH ()-[b:Buys]->() SET b.amount = 'x'", "amount"),
            ("MATCH ()-[b:Buys]->() SET b.amount = NULL", "amount"),
        ],
    );
    assert_eq!(
        session
            .execute("MATCH (c:Customer)-[b:Buys]->() RETURN c.customerId, c.age, b.amount")
            .unwrap()
            .rows(),
        [vec![
            Value::String("c3".into()),
            Value::Int64(19),
            Value::Int64(88)
        ]],
        "the refused statements wrote nothing"
    );
    assert_eq!(
        count(&session, "MATCH (n) RETURN count(n)"),
        Value::Int64(2)
    );
}

#[cfg(feature = "cypher")]
#[test]
fn cypher_writes_in_the_schema_are_checked() {
    let db = two_schemas();
    let session = session_in(&db, "s1");
    assert_refused(
        |statement| session.execute_cypher(statement),
        &[
            ("CREATE (:Customer {age: 3})", "customerId"),
            ("CREATE (:Customer {customerId: 'c3', age: 'old'})", "age"),
            ("MERGE (:Customer {customerId: 'c19', age: 'old'})", "age"),
            ("CREATE (:Shop)-[:Buys {amount: 'x'}]->(:Item)", "amount"),
            ("CREATE (:Shop)-[:Buys]->(:Item)", "amount"),
        ],
    );
    session
        .execute_cypher(
            "CREATE (:Customer {customerId: 'c3', age: 19})-[:Buys {amount: 88}]->(:Item)",
        )
        .unwrap();
    assert_refused(
        |statement| session.execute_cypher(statement),
        &[
            ("MATCH (c:Customer) SET c.age = 'old'", "age"),
            ("MATCH (c:Customer) REMOVE c.customerId", "customerId"),
            ("MATCH ()-[b:Buys]->() SET b.amount = 'x'", "amount"),
        ],
    );
    assert_eq!(
        count(&session, "MATCH (n) RETURN count(n)"),
        Value::Int64(2)
    );
}

/// The database's direct API in its current schema, a session's direct API
/// in its schema (also in a transaction), and a graph handle on a graph of
/// the schema.
#[test]
fn direct_writes_in_the_schema_are_checked() {
    let db = two_schemas();
    db.set_current_schema(Some("s1")).unwrap();
    let missing = db.create_node_with_props(&["Customer"], [("age", Value::Int64(3))]);
    assert!(
        missing.is_err_and(|error| error.to_string().contains("customerId")),
        "a Customer without customerId"
    );
    let wrong = db.create_node_with_props(
        &["Customer"],
        [
            ("customerId", Value::String("c3".into())),
            ("age", Value::String("old".into())),
        ],
    );
    assert!(
        wrong.is_err_and(|error| error.to_string().contains("age")),
        "a Customer with a string age"
    );
    let customer = db
        .create_node_with_props(&["Customer"], [("customerId", Value::String("c3".into()))])
        .unwrap();
    let item = db.create_node(&["Item"]).unwrap();
    assert!(
        db.set_node_property(customer, "age", Value::String("old".into()))
            .is_err_and(|error| error.to_string().contains("age"))
    );
    assert!(
        db.create_edge_with_props(
            customer,
            item,
            "Buys",
            [("amount", Value::String("x".into()))]
        )
        .is_err_and(|error| error.to_string().contains("amount"))
    );
    assert!(
        db.create_edge(customer, item, "Buys")
            .is_err_and(|error| error.to_string().contains("amount"))
    );

    // A session in s1, outside and inside a transaction.
    let mut session = session_in(&db, "s1");
    assert!(
        session
            .create_node_with_props(&["Customer"], [("age", Value::Int64(3))])
            .is_err_and(|error| error.to_string().contains("customerId"))
    );
    session.begin_transaction().unwrap();
    assert!(
        session
            .create_node_with_props(&["Customer"], [("age", Value::Int64(3))])
            .is_err_and(|error| error.to_string().contains("customerId"))
    );
    session.rollback().unwrap();

    // A graph handle on a graph of s1.
    session_in(&db, "s1").execute("CREATE GRAPH g").unwrap();
    let graph = db.graph_in(Some("s1"), "g").unwrap();
    assert!(
        graph
            .create_node_with_props(&["Customer"], [("age", Value::Int64(3))])
            .is_err_and(|error| error.to_string().contains("customerId"))
    );
    let in_graph: NodeId = graph
        .create_node_with_props(&["Customer"], [("customerId", Value::String("c19".into()))])
        .unwrap();
    assert!(
        graph
            .set_node_property(in_graph, "age", Value::String("old".into()))
            .is_err_and(|error| error.to_string().contains("age"))
    );
}

/// A Customer in s2 follows s2's type, one in no schema no type at all.
#[test]
fn types_of_another_schema_do_not_apply() {
    let db = two_schemas();
    let s2 = session_in(&db, "s2");
    s2.execute("INSERT (:Customer {age: 'old'})").unwrap();
    assert_refused(
        |statement| s2.execute(statement),
        &[("INSERT (:Customer {age: 3})", "age")],
    );
    let unscoped = db.session();
    unscoped.execute("INSERT (:Customer {age: 'old'})").unwrap();
    db.create_node_with_props(&["Customer"], [("age", Value::String("old".into()))])
        .unwrap();
    // A type of the default schema does not check writes in s1.
    unscoped
        .execute("CREATE NODE TYPE Person (name STRING NOT NULL)")
        .unwrap();
    assert_refused(
        |statement| unscoped.execute(statement),
        &[("INSERT (:Person {age: 3})", "name")],
    );
    session_in(&db, "s1")
        .execute("INSERT (:Person {age: 3})")
        .unwrap();
    // s1's edge type does not check edges in s2.
    s2.execute("INSERT (:Shop)-[:Buys {amount: 'x'}]->(:Item)")
        .unwrap();
}

/// A type in a schema inherits from its parent in the same schema, and gives
/// its default to a property a new node leaves out.
#[test]
fn inherited_properties_and_defaults_in_a_schema() {
    let db = two_schemas();
    let session = session_in(&db, "s1");
    for statement in [
        "CREATE NODE TYPE Base (code STRING NOT NULL)",
        "CREATE NODE TYPE Child EXTENDS Base (x INT64)",
        "CREATE NODE TYPE Ticket (id INT64, status STRING NOT NULL DEFAULT 'new')",
    ] {
        session.execute(statement).unwrap();
    }
    assert_refused(
        |statement| session.execute(statement),
        &[
            ("INSERT (:Child {x: 3})", "code"),
            ("INSERT (:Child {code: 3, x: 3})", "code"),
        ],
    );
    session
        .execute("INSERT (:Child {code: 'c', x: 3})")
        .unwrap();
    session.execute("INSERT (:Ticket {id: 19})").unwrap();
    assert_eq!(
        count(&session, "MATCH (t:Ticket) RETURN t.status"),
        Value::String("new".into())
    );
}

/// A type in a schema converts the values it declares as a type outside one
/// does (an integer written to a `FLOAT64` property is stored as a float,
/// #568) and gives an edge the defaults of its edge type, from GQL and from a
/// graph handle on a graph of the schema.
#[test]
fn conversions_and_edge_defaults_in_a_schema() {
    let db = two_schemas();
    let session = session_in(&db, "s1");
    for statement in [
        "CREATE NODE TYPE Shop (name STRING, revenue FLOAT64)",
        "CREATE EDGE TYPE Ships (km FLOAT64, lanes INT64 DEFAULT 3)",
        "INSERT (:Shop {name: 'Amsterdam', revenue: 19})-[:Ships {km: 88}]->(:Shop {name: 'Berlin'})",
    ] {
        session
            .execute(statement)
            .unwrap_or_else(|error| panic!("`{statement}` failed: {error}"));
    }
    let rows = session
        .execute("MATCH (a:Shop)-[s:Ships]->(:Shop) RETURN a.revenue, s.km, s.lanes")
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(
        rows,
        vec![vec![
            Value::Float64(19.0),
            Value::Float64(88.0),
            Value::Int64(3)
        ]],
        "the integers stored as floats, and the default lanes"
    );

    session.execute("CREATE GRAPH g").unwrap();
    let graph = db.graph_in(Some("s1"), "g").unwrap();
    let paris = graph
        .create_node_with_props(&["Shop"], [("revenue", Value::Int64(3))])
        .unwrap();
    let prague = graph.create_node(&["Shop"]).unwrap();
    let ships = graph
        .create_edge_with_props(paris, prague, "Ships", [("km", Value::Int64(19))])
        .unwrap();
    let node = graph.get_node(paris).unwrap().unwrap();
    assert_eq!(node.get_property("revenue"), Some(&Value::Float64(3.0)));
    let edge = graph.get_edge(ships).unwrap().unwrap();
    assert_eq!(edge.get_property("km"), Some(&Value::Float64(19.0)));
    assert_eq!(
        edge.get_property("lanes"),
        Some(&Value::Int64(3)),
        "the edge type's default"
    );
}

/// A closed graph type in a schema refuses a property its node type does
/// not declare (#567), as one outside a schema does.
#[test]
fn a_closed_graph_type_in_a_schema_refuses_undeclared_properties() {
    let db = two_schemas();
    let session = session_in(&db, "s1");
    for statement in [
        "CREATE GRAPH TYPE depot (NODE TYPE Crate (code STRING, weight FLOAT64))",
        "CREATE GRAPH g TYPED depot",
        "SESSION SET GRAPH g",
    ] {
        session.execute(statement).unwrap();
    }
    assert_refused(
        |statement| session.execute(statement),
        &[
            ("INSERT (:Crate {code: 'c3', colour: 'red'})", "colour"),
            ("INSERT (:Pallet {code: 'p3'})", "Pallet"),
        ],
    );
    session
        .execute("INSERT (:Crate {code: 'c19', weight: 88})")
        .unwrap();
    assert_eq!(
        count(&session, "MATCH (c:Crate) RETURN c.weight"),
        Value::Float64(88.0)
    );
}

/// The node types a graph type declares inline in a schema check the nodes
/// of a graph typed by it.
#[test]
fn inline_node_types_of_a_graph_type_in_a_schema() {
    let db = two_schemas();
    let session = session_in(&db, "s1");
    for statement in [
        "CREATE GRAPH TYPE shop (NODE TYPE Product (sku STRING NOT NULL, price INT64))",
        "CREATE GRAPH g TYPED shop",
        "SESSION SET GRAPH g",
    ] {
        session.execute(statement).unwrap();
    }
    assert_refused(
        |statement| session.execute(statement),
        &[
            ("INSERT (:Product {price: 3})", "sku"),
            ("INSERT (:Product {sku: 'p3', price: 'cheap'})", "price"),
        ],
    );
    session
        .execute("INSERT (:Product {sku: 'p3', price: 19})")
        .unwrap();
    assert_eq!(
        count(&session, "MATCH (p:Product) RETURN count(p)"),
        Value::Int64(1)
    );
}
