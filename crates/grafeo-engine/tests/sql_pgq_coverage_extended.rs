//! SQL/PGQ tests requiring Rust APIs (not expressible as .gtest files).
//!
//! Covers:
//! - Parameterized queries (`execute_sql_with_params`)
//! - Transaction integration (BEGIN/COMMIT/ROLLBACK + SQL/PGQ)
//!
//! All other SQL/PGQ tests live in `tests/spec/lpg/sql_pgq/*.gtest`.
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features sql-pgq --test sql_pgq_coverage_extended
//! ```

#![cfg(feature = "sql-pgq")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixture
// ============================================================================

fn create_rich_network() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let alix = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Alix".into())),
                ("age", Value::Int64(30)),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();
    let gus = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Gus".into())),
                ("age", Value::Int64(25)),
                ("city", Value::String("Berlin".into())),
            ],
        )
        .unwrap();
    let harm = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Harm".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();
    let vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("age", Value::Int64(28)),
                ("city", Value::String("Paris".into())),
            ],
        )
        .unwrap();
    let mia = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Mia".into())),
                ("age", Value::Int64(32)),
                ("city", Value::String("Berlin".into())),
            ],
        )
        .unwrap();

    let e1 = session.create_edge(alix, gus, "KNOWS").unwrap();
    db.set_edge_property(e1, "since", Value::Int64(2020))
        .unwrap();
    let e2 = session.create_edge(alix, harm, "KNOWS").unwrap();
    db.set_edge_property(e2, "since", Value::Int64(2018))
        .unwrap();
    let e3 = session.create_edge(gus, harm, "KNOWS").unwrap();
    db.set_edge_property(e3, "since", Value::Int64(2021))
        .unwrap();
    let e4 = session.create_edge(vincent, mia, "KNOWS").unwrap();
    db.set_edge_property(e4, "since", Value::Int64(2019))
        .unwrap();
    let e5 = session.create_edge(alix, vincent, "FOLLOWS").unwrap();
    db.set_edge_property(e5, "since", Value::Int64(2022))
        .unwrap();
    let e6 = session.create_edge(gus, alix, "FOLLOWS").unwrap();
    db.set_edge_property(e6, "since", Value::Int64(2023))
        .unwrap();

    db
}

// ============================================================================
// Parameterized Queries
// ============================================================================

#[test]
fn test_param_string_in_inner_where() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert("name".to_string(), Value::String("Alix".into()));

    let result = session
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                WHERE n.name = $name
                COLUMNS (n.name AS name, n.age AS age)
            )",
            params,
        )
        .unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(30));
}

#[test]
fn test_param_integer_in_inner_where() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert("min_age".to_string(), Value::Int64(30));

    let result = session
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                WHERE n.age >= $min_age
                COLUMNS (n.name AS name, n.age AS age)
            ) ORDER BY name",
            params,
        )
        .unwrap();

    assert_eq!(result.row_count(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[1][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[2][0], Value::String("Mia".into()));
}

#[test]
fn test_param_multiple_in_inner_where() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert("min_age".to_string(), Value::Int64(26));
    params.insert("max_age".to_string(), Value::Int64(31));

    let result = session
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                WHERE n.age >= $min_age AND n.age <= $max_age
                COLUMNS (n.name AS name, n.age AS age)
            ) ORDER BY name",
            params,
        )
        .unwrap();

    assert_eq!(result.row_count(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into()));
}

#[test]
fn test_param_in_columns_expression() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert("bonus".to_string(), Value::Int64(100));

    let result = session
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                WHERE n.name = 'Gus'
                COLUMNS (n.name AS name, n.age + $bonus AS boosted_age)
            )",
            params,
        )
        .unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][1], Value::Int64(125), "25 + 100 = 125");
}

#[test]
fn test_param_with_in_operator() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert(
        "cities".to_string(),
        Value::List(
            vec![
                Value::String("Amsterdam".into()),
                Value::String("Paris".into()),
            ]
            .into(),
        ),
    );

    let result = session
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                WHERE n.city IN $cities
                COLUMNS (n.name AS name, n.city AS city)
            ) ORDER BY name",
            params,
        )
        .unwrap();

    assert_eq!(result.row_count(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Alix".into()));
    assert_eq!(result.rows()[1][0], Value::String("Harm".into()));
    assert_eq!(result.rows()[2][0], Value::String("Vincent".into()));
}

#[test]
fn test_param_in_outer_sql_where() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert("city".to_string(), Value::String("Berlin".into()));

    let result = session
        .execute_sql_with_params(
            "SELECT * FROM GRAPH_TABLE (
                MATCH (n:Person)
                COLUMNS (n.name AS name, n.city AS city)
            ) AS g WHERE g.city = $city ORDER BY g.name",
            params,
        )
        .unwrap();

    assert_eq!(result.row_count(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Gus".into()));
    assert_eq!(result.rows()[1][0], Value::String("Mia".into()));
}

#[test]
fn test_param_missing_parameter_error() {
    let db = create_rich_network();
    let session = db.session();

    let params = std::collections::HashMap::new();

    let result = session.execute_sql_with_params(
        "SELECT * FROM GRAPH_TABLE (
            MATCH (n:Person)
            WHERE n.name = $name
            COLUMNS (n.name AS name)
        )",
        params,
    );

    assert!(result.is_err(), "Should fail when parameter is missing");
}

#[test]
fn test_param_wrong_type_comparison() {
    let db = create_rich_network();
    let session = db.session();

    let mut params = std::collections::HashMap::new();
    params.insert("age".to_string(), Value::String("thirty".into()));

    let result = session.execute_sql_with_params(
        "SELECT * FROM GRAPH_TABLE (
            MATCH (n:Person)
            WHERE n.age = $age
            COLUMNS (n.name AS name)
        )",
        params,
    );

    match result {
        Ok(qr) => assert_eq!(
            qr.row_count(),
            0,
            "String 'thirty' should not match any integer age"
        ),
        Err(_) => {} // type-mismatch error is also acceptable
    }
}

// ============================================================================
// Transaction Integration
// ============================================================================

#[test]
fn test_tx_commit_makes_data_visible_to_sql_pgq() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();
    session.commit().unwrap();

    let result = session
        .execute_sql("SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))")
        .unwrap();

    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_tx_rollback_hides_data_from_sql_pgq() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session
        .execute("INSERT (:Person {name: 'Alix', age: 30})")
        .unwrap();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Gus', age: 25})")
        .unwrap();
    session.rollback().unwrap();

    let result = session
        .execute_sql("SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))")
        .unwrap();

    assert_eq!(result.row_count(), 1);
    assert_eq!(result.rows()[0][0].as_str(), Some("Alix"));
}

#[test]
fn test_tx_sql_pgq_sees_data_in_same_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();

    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Person {name: 'Mia', city: 'Berlin'})")
        .unwrap();
    session
        .execute("INSERT (:Person {name: 'Butch', city: 'Amsterdam'})")
        .unwrap();

    let result = session
        .execute_sql("SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))")
        .unwrap();

    assert_eq!(result.row_count(), 2);
    session.commit().unwrap();
}

#[test]
fn test_tx_isolation_uncommitted_not_visible() {
    let db = GrafeoDB::new_in_memory();

    let setup = db.session();
    setup
        .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
        .unwrap();

    let mut session1 = db.session();
    session1.begin_transaction().unwrap();
    session1
        .execute("INSERT (:Person {name: 'Django'})")
        .unwrap();

    let session2 = db.session();
    let result = session2
        .execute_sql("SELECT * FROM GRAPH_TABLE (MATCH (n:Person) COLUMNS (n.name AS name))")
        .unwrap();

    assert_eq!(
        result.row_count(),
        1,
        "session2 should not see uncommitted data"
    );
    assert_eq!(result.rows()[0][0].as_str(), Some("Alix"));

    session1.rollback().unwrap();
}
