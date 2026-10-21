//! SQL/PGQ `CREATE PROPERTY GRAPH` declares what it names: a node type per
//! node table (its columns typed properties, its `PRIMARY KEY` the type's
//! key), an edge type per edge table between the node types its two
//! `REFERENCES` columns name, a closed graph type of them and an empty graph
//! of that type. It reported success and created nothing. `DROP PROPERTY
//! GRAPH` takes them away again, and `GRAPH_TABLE (g ...)` reads the graph g.

#![cfg(all(feature = "sql-pgq", feature = "gql", feature = "lpg"))]

use grafeo_common::utils::GqlStatus;
use grafeo_engine::{GrafeoDB, Session};

const SHOP: &str = "CREATE PROPERTY GRAPH Shop \
    NODE TABLES ( \
        Item (id BIGINT PRIMARY KEY, name VARCHAR(40), price DECIMAL(6, 2), made DATE), \
        Box (code VARCHAR PRIMARY KEY) \
    ) \
    EDGE TABLES ( \
        Holds (item BIGINT REFERENCES Item(id), box VARCHAR REFERENCES Box(code), qty INT) \
    )";

fn sql(session: &Session, query: &str) -> grafeo_common::utils::error::Result<Vec<Vec<String>>> {
    session.execute_sql(query).map(|result| {
        result
            .rows()
            .iter()
            .map(|row| row.iter().map(text).collect())
            .collect()
    })
}

/// A value as text, a string without its quotes.
fn text(value: &grafeo_common::types::Value) -> String {
    match value {
        grafeo_common::types::Value::String(text) => text.to_string(),
        other => format!("{other}"),
    }
}

fn status(error: &grafeo_common::utils::error::Error) -> String {
    GqlStatus::from(error).as_str().to_string()
}

/// A session on the database with the property graph `Shop`, using it.
fn shop(db: &GrafeoDB) -> Session {
    let session = db.session();
    let created = sql(&session, SHOP).unwrap();
    assert_eq!(
        created,
        [["Property graph 'Shop' created (2 node tables, 1 edge tables)"]]
    );
    session.execute("USE GRAPH Shop").unwrap();
    session
}

#[test]
fn the_tables_become_typed_node_and_edge_types() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session
        .execute(
            "INSERT (:Item {id: 1, name: 'Lamp', price: 19.95, made: DATE '2024-03-01'})\
             -[:Holds {qty: 3}]->(:Box {code: 'A1'})",
        )
        .unwrap();
    let rows = session
        .execute("MATCH (i:Item)-[h:Holds]->(b:Box) RETURN i.name, i.price, h.qty, b.code")
        .unwrap();
    assert_eq!(rows.rows().len(), 1);

    for (query, what) in [
        (
            "INSERT (:Item {id: 2, name: 3})",
            "a string column holds a number",
        ),
        (
            "INSERT (:Item {id: 3, price: 'cheap'})",
            "a decimal column holds a string",
        ),
        (
            "INSERT (:Item {id: 4, price: 123456.0})",
            "more digits than DECIMAL(6, 2)",
        ),
        (
            "INSERT (:Item {id: 5, made: 'yesterday'})",
            "a date column holds a string",
        ),
        (
            "INSERT (:Shelf {id: 6})",
            "a label the graph type does not have",
        ),
    ] {
        assert!(session.execute(query).is_err(), "{what}: {query}");
    }
}

#[test]
fn the_primary_key_is_unique_and_not_null() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session
        .execute("INSERT (:Item {id: 1, name: 'Lamp'})")
        .unwrap();
    assert!(
        session
            .execute("INSERT (:Item {id: 1, name: 'Chair'})")
            .is_err(),
        "a second Item with the key 1"
    );
    assert!(
        session.execute("INSERT (:Item {name: 'Desk'})").is_err(),
        "an Item without its key"
    );
    let count = session
        .execute("MATCH (i:Item) RETURN count(i) AS c")
        .unwrap();
    assert_eq!(format!("{}", count.rows()[0][0]), "1");
}

#[test]
fn an_edge_connects_the_tables_its_references_name() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session
        .execute("INSERT (:Item {id: 1})-[:Holds]->(:Box {code: 'A1'})")
        .unwrap();
    assert!(
        session
            .execute("INSERT (:Box {code: 'B2'})-[:Holds]->(:Item {id: 2})")
            .is_err(),
        "an edge from a Box to an Item, against its references"
    );
}

#[test]
fn graph_table_reads_the_graph_it_names() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("INSERT (:Item {name: 'In the default graph'})")
        .unwrap();
    sql(&session, SHOP).unwrap();
    let shop = db.session();
    shop.execute("USE GRAPH Shop").unwrap();
    shop.execute("INSERT (:Item {id: 1, name: 'Lamp'})")
        .unwrap();

    let names = |query: &str| sql(&session, query).unwrap();
    assert_eq!(
        names("SELECT * FROM GRAPH_TABLE (Shop MATCH (i:Item) COLUMNS (i.name AS name))"),
        [["Lamp"]]
    );
    assert_eq!(
        names("SELECT * FROM GRAPH_TABLE (MATCH (i:Item) COLUMNS (i.name AS name))"),
        [["In the default graph"]]
    );
    let error = sql(
        &session,
        "SELECT * FROM GRAPH_TABLE (Nowhere MATCH (i:Item) COLUMNS (i.name AS name))",
    )
    .unwrap_err();
    assert_eq!(status(&error), "42002", "{error}");
    assert!(
        error.to_string().contains("Graph 'Nowhere' does not exist"),
        "{error}"
    );
}

/// A set operation reads one graph: both sides name the same one, or the
/// statement is refused.
#[test]
fn a_set_operation_reads_one_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session
        .execute("INSERT (:Item {id: 1, name: 'Lamp'})")
        .unwrap();
    let both = sql(
        &session,
        "SELECT * FROM GRAPH_TABLE (Shop MATCH (i:Item) COLUMNS (i.name AS name))          UNION SELECT * FROM GRAPH_TABLE (Shop MATCH (i:Item) COLUMNS (i.name AS name))",
    )
    .unwrap();
    assert_eq!(both, [["Lamp"]]);
    let error = sql(
        &session,
        "SELECT * FROM GRAPH_TABLE (Shop MATCH (i:Item) COLUMNS (i.name AS name))          UNION SELECT * FROM GRAPH_TABLE (MATCH (i:Item) COLUMNS (i.name AS name))",
    )
    .unwrap_err();
    assert!(
        error
            .to_string()
            .contains("a set operation reads one graph"),
        "{error}"
    );
}

#[test]
fn an_existing_property_graph_is_refused_kept_or_replaced() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session
        .execute("INSERT (:Item {id: 1, name: 'Lamp'})")
        .unwrap();

    let error = sql(&session, SHOP).unwrap_err();
    assert_eq!(status(&error), "42000", "{error}");
    assert!(
        error.to_string().contains("Graph 'Shop' already exists"),
        "{error}"
    );

    let kept = sql(
        &session,
        &SHOP.replacen(
            "CREATE PROPERTY GRAPH",
            "CREATE PROPERTY GRAPH IF NOT EXISTS",
            1,
        ),
    )
    .unwrap();
    assert_eq!(kept, [["No change"]]);
    let count = session
        .execute("MATCH (i:Item) RETURN count(i) AS c")
        .unwrap();
    assert_eq!(
        format!("{}", count.rows()[0][0]),
        "1",
        "IF NOT EXISTS kept the graph"
    );

    // OR REPLACE declares the new tables and an empty graph: Box and Holds
    // go, Item has other columns.
    let replaced = sql(
        &session,
        "CREATE OR REPLACE PROPERTY GRAPH Shop NODE TABLES (Item (sku VARCHAR PRIMARY KEY))",
    )
    .unwrap();
    assert_eq!(
        replaced,
        [["Property graph 'Shop' created (1 node tables, 0 edge tables)"]]
    );
    session.execute("USE GRAPH Shop").unwrap();
    let count = session
        .execute("MATCH (i:Item) RETURN count(i) AS c")
        .unwrap();
    assert_eq!(
        format!("{}", count.rows()[0][0]),
        "0",
        "the replaced graph is empty"
    );
    session.execute("INSERT (:Item {sku: 'L-1'})").unwrap();
    assert!(
        session.execute("INSERT (:Box {code: 'A1'})").is_err(),
        "Box went"
    );
    assert!(
        session.execute("INSERT (:Item {id: 1})").is_err(),
        "Item has no id now"
    );
}

#[test]
fn or_replace_and_drop_refuse_a_graph_type_another_graph_has() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session.execute("CREATE GRAPH Copy TYPED Shop").unwrap();
    for query in [
        SHOP.replacen(
            "CREATE PROPERTY GRAPH",
            "CREATE OR REPLACE PROPERTY GRAPH",
            1,
        ),
        "DROP PROPERTY GRAPH Shop".to_string(),
    ] {
        let error = sql(&session, &query).unwrap_err();
        assert!(
            error.to_string().contains("is the type of graph"),
            "{query}: {error}"
        );
    }
    session.execute("DROP GRAPH Copy").unwrap();
    sql(&session, "DROP PROPERTY GRAPH Shop").unwrap();
}

#[test]
fn drop_takes_the_graph_its_type_and_its_element_types() {
    let db = GrafeoDB::new_in_memory();
    let session = shop(&db);
    session.execute("INSERT (:Item {id: 1})").unwrap();
    assert_eq!(
        sql(&session, "DROP PROPERTY GRAPH Shop").unwrap(),
        [["Property graph 'Shop' dropped"]]
    );
    assert!(
        session.execute("USE GRAPH Shop").is_err(),
        "the graph is gone"
    );
    // The types are gone too: the same names can be declared again.
    session.execute("CREATE NODE TYPE Item (id INT64)").unwrap();
    session
        .execute("CREATE GRAPH TYPE Shop (NODE TYPE Item)")
        .unwrap();

    assert_eq!(
        sql(&session, "DROP PROPERTY GRAPH IF EXISTS Nowhere").unwrap(),
        [["No change"]]
    );
    let error = sql(&session, "DROP PROPERTY GRAPH Nowhere").unwrap_err();
    assert_eq!(status(&error), "42002", "{error}");
}

#[test]
fn an_element_type_of_another_schema_object_is_not_taken_over() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .execute("CREATE NODE TYPE Box (code STRING)")
        .unwrap();
    let error = sql(&session, SHOP).unwrap_err();
    assert!(
        error.to_string().contains("Node type 'Box' already exists"),
        "{error}"
    );
    assert!(
        session.execute("USE GRAPH Shop").is_err(),
        "nothing was created"
    );
}

#[test]
fn a_definition_the_tables_cannot_make_is_refused() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (query, says, code) in [
        (
            "CREATE PROPERTY GRAPH G NODE TABLES (A (id BIGINT)) EDGE TABLES (E (a BIGINT REFERENCES A(id)))",
            "needs two columns that reference node tables",
            "42000",
        ),
        (
            "CREATE PROPERTY GRAPH G NODE TABLES (A (id BIGINT)) EDGE TABLES (E (a BIGINT REFERENCES A(id), b BIGINT REFERENCES B(id)))",
            "which is no node table",
            "42002",
        ),
        (
            "CREATE PROPERTY GRAPH G NODE TABLES (A (id BIGINT)) EDGE TABLES (E (a BIGINT REFERENCES A(nope), b BIGINT REFERENCES A(id)))",
            "does not have",
            "42002",
        ),
        (
            "CREATE PROPERTY GRAPH G NODE TABLES (A (id BIGINT, id VARCHAR))",
            "Column 'id' is defined twice",
            "42000",
        ),
        (
            "CREATE PROPERTY GRAPH G NODE TABLES (A (id BIGINT)) EDGE TABLES (E (k BIGINT PRIMARY KEY, a BIGINT REFERENCES A(id), b BIGINT REFERENCES A(id)))",
            "PRIMARY KEY on an edge table",
            "42000",
        ),
        (
            "CREATE OR REPLACE PROPERTY GRAPH IF NOT EXISTS G NODE TABLES (A (id BIGINT))",
            "OR REPLACE or IF NOT EXISTS",
            "42001",
        ),
    ] {
        let error = sql(&session, query).unwrap_err();
        assert!(error.to_string().contains(says), "{query}: {error}");
        assert_eq!(status(&error), code, "{query}: {error}");
    }
    assert!(
        session.execute("USE GRAPH G").is_err(),
        "no graph was created"
    );
}

#[test]
fn a_read_only_transaction_refuses_the_ddl() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session.execute("START TRANSACTION READ ONLY").unwrap();
    let error = sql(&session, SHOP).unwrap_err();
    assert_eq!(status(&error), "25G03", "{error}");
    session.execute("ROLLBACK").unwrap();
}

/// The declarations are logged: a reopened database has the graph, its types
/// and its key.
#[cfg(feature = "wal")]
#[test]
fn a_property_graph_survives_a_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shop.grafeo");
    {
        let db = GrafeoDB::open(&path).unwrap();
        let session = shop(&db);
        session
            .execute("INSERT (:Item {id: 1, name: 'Lamp'})")
            .unwrap();
        db.close().unwrap();
    }
    let db = GrafeoDB::open(&path).unwrap();
    let session = db.session();
    session.execute("USE GRAPH Shop").unwrap();
    assert!(
        session.execute("INSERT (:Item {id: 1})").is_err(),
        "the key holds"
    );
    assert!(
        session.execute("INSERT (:Item {id: 2, name: 7})").is_err(),
        "the types hold"
    );
    session
        .execute("INSERT (:Item {id: 2, name: 'Desk'})")
        .unwrap();
    db.close().unwrap();
}
