//! A CHECK constraint means what its predicate means in a query: a node
//! satisfies `CHECK (p)` exactly when `MATCH (n) WHERE p RETURN n`, with each
//! property name of `p` read from `n`, returns it. Comparisons of dates and
//! times, `1 = 1.0` and unknown (null) results all agree with the query.
//!
//! CHECK constraints have no DDL; the catalog API declares them (and a
//! `.grafeo` file keeps them).
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test check_constraints
//! ```

#![cfg(feature = "gql")]

use std::sync::Arc;

use grafeo_common::types::Value;
use grafeo_core::execution::operators::ConstraintValidator;
use grafeo_engine::GrafeoDB;
use grafeo_engine::catalog::{
    Catalog, CatalogConstraintValidator, CatalogError, NodeTypeDefinition, TypeConstraint,
};

/// Nodes to check (the properties of an INSERT), and predicates, each written
/// as a CHECK reads it and as a WHERE about the node `n` reads it.
const NODES: &[&str] = &[
    "{begins: date('2024-03-19'), ends: date('2024-03-22'), seats: 3, price: 19.0}",
    "{begins: date('2024-03-22'), ends: date('2024-03-19'), seats: 1, price: 1.5}",
    "{begins: date('2024-03-19'), seats: 88, opens: time('08:30:00'), closes: time('19:00:00')}",
    "{ends: date('2024-03-22'), price: 3, name: 'Prague'}",
    "{seats: 19, name: 'Berlin', sent: datetime('2024-03-19T08:30:00')}",
];

const PREDICATES: &[(&str, &str)] = &[
    ("begins < ends", "n.begins < n.ends"),
    (
        "begins <= ends AND seats > 1",
        "n.begins <= n.ends AND n.seats > 1",
    ),
    ("NOT (begins > ends)", "NOT (n.begins > n.ends)"),
    ("opens < closes", "n.opens < n.closes"),
    ("seats = 3.0", "n.seats = 3.0"),
    ("price = 19", "n.price = 19"),
    ("price <> 3", "n.price <> 3"),
    ("seats IN (1, 19, 88)", "n.seats IN [1, 19, 88]"),
    ("seats NOT IN (3, NULL)", "NOT (n.seats IN [3, NULL])"),
    ("seats BETWEEN 3 AND 19", "n.seats >= 3 AND n.seats <= 19"),
    ("seats * price >= 57", "n.seats * n.price >= 57"),
    (
        "name IS NULL OR name <> 'Paris'",
        "n.name IS NULL OR n.name <> 'Paris'",
    ),
    ("NOT (name = 'Berlin')", "NOT (n.name = 'Berlin')"),
    ("sent IS NOT NULL", "n.sent IS NOT NULL"),
    (
        "ends >= begins OR seats > 19",
        "n.ends >= n.begins OR n.seats > 19",
    ),
];

/// Whether the only `Probe` node of `db` passes the WHERE `predicate`.
fn query_keeps(db: &GrafeoDB, predicate: &str) -> bool {
    let query = format!("MATCH (n:Probe) WHERE {predicate} RETURN count(n)");
    match db.execute(&query).unwrap().rows()[0][0] {
        Value::Int64(count) => count == 1,
        ref other => panic!("{query}: {other:?}"),
    }
}

/// The properties of the only `Probe` node of `db`.
fn properties(db: &GrafeoDB) -> Vec<(String, Value)> {
    let result = db.execute("MATCH (n:Probe) RETURN properties(n)").unwrap();
    let Value::Map(map) = &result.rows()[0][0] else {
        panic!("properties: {:?}", result.rows());
    };
    map.iter()
        .map(|(key, value)| (key.as_str().to_string(), value.clone()))
        .collect()
}

/// A validator whose catalog has the node type `Probe` with `CHECK (check)`.
fn validator(check: &str) -> CatalogConstraintValidator {
    let catalog = Catalog::with_schema();
    catalog
        .register_node_type(NodeTypeDefinition {
            name: "Probe".to_string(),
            properties: Vec::new(),
            constraints: vec![TypeConstraint::Check {
                name: Some("probe_check".to_string()),
                expression: check.to_string(),
            }],
            parent_types: Vec::new(),
            key_labels: Vec::new(),
        })
        .unwrap();
    CatalogConstraintValidator::new(Arc::new(catalog))
}

#[test]
fn a_check_holds_for_the_nodes_its_predicate_keeps_in_a_query() {
    let mut outcomes = [0_usize; 2];
    for node in NODES {
        let db = GrafeoDB::new_in_memory();
        db.execute(&format!("INSERT (:Probe {node})")).unwrap();
        let properties = properties(&db);
        for (check, predicate) in PREDICATES {
            let kept = query_keeps(&db, predicate);
            outcomes[usize::from(kept)] += 1;
            let checked =
                validator(check).validate_node_complete(&["Probe".to_string()], &properties);
            assert_eq!(
                checked.is_ok(),
                kept,
                "CHECK ({check}) on {node}: {checked:?}, while WHERE {predicate} keeps it: {kept}"
            );
            // A predicate that is not true (also unknown) is a violation,
            // not an error of the expression.
            if let Err(error) = checked {
                assert!(
                    error
                        .to_string()
                        .contains("CHECK constraint 'probe_check' violated on :Probe"),
                    "CHECK ({check}) on {node}: {error}"
                );
            }
        }
    }
    assert!(
        outcomes[0] >= 20 && outcomes[1] >= 20,
        "the cases keep and drop nodes alike: {outcomes:?}"
    );
}

/// An expression that is not one is refused when the constraint is added,
/// not by every later write.
#[test]
fn a_check_that_does_not_parse_is_refused_when_added() {
    let catalog = Catalog::with_schema();
    for expression in ["seats >", "seats = 'open", "(seats > 3", "seats ? 3"] {
        let error = catalog
            .add_constraint_to_type(
                "Probe",
                TypeConstraint::Check {
                    name: None,
                    expression: expression.to_string(),
                },
            )
            .unwrap_err();
        assert!(
            matches!(&error, CatalogError::InvalidCheck { .. }),
            "{expression}: {error}"
        );
        assert!(error.to_string().contains(expression), "{error}");
    }
    assert!(
        catalog.get_node_type("Probe").is_none(),
        "a refused CHECK declared the type"
    );
}
