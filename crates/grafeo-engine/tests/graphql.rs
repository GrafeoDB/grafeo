//! GraphQL LPG Integration Tests
//!
//! Verifies end-to-end query execution through the full pipeline:
//! Parse → Translate → Bind → Optimize → Plan → Execute
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features graphql --test graphql
//! ```

#![cfg(feature = "graphql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Test Fixtures
// ============================================================================

/// Creates a social network graph for testing.
///
/// Structure:
/// - Alix (Person, age: 30, city: "Amsterdam") -KNOWS-> Gus (Person, age: 25, city: "Berlin")
/// - Alix -KNOWS-> Vincent (Person, age: 35, city: "Paris")
/// - Gus -KNOWS-> Vincent
/// - Alix -WORKS_AT-> Acme (Company, revenue: 1000000)
fn create_social_network() -> GrafeoDB {
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
    let vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Paris".into())),
            ],
        )
        .unwrap();
    let acme = session
        .create_node_with_props(
            &["Company"],
            [
                ("name", Value::String("Acme".into())),
                ("revenue", Value::Int64(1_000_000)),
            ],
        )
        .unwrap();

    session.create_edge(alix, gus, "KNOWS").unwrap();
    session.create_edge(alix, vincent, "KNOWS").unwrap();
    session.create_edge(gus, vincent, "KNOWS").unwrap();
    session.create_edge(alix, acme, "WORKS_AT").unwrap();

    db
}

// ============================================================================
// Basic Queries: Node Scans
// ============================================================================

#[test]
fn test_query_all_nodes_by_label() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 3, "Should find 3 Person nodes");
}

#[test]
fn test_query_different_label() {
    let db = create_social_network();
    let result = db.execute_graphql("{ company { name } }").unwrap();
    assert_eq!(result.row_count(), 1, "Should find 1 Company node");
}

#[test]
fn test_query_multiple_scalar_fields() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person { name age city } }").unwrap();
    assert_eq!(result.row_count(), 3);
    assert!(
        result.columns.len() >= 3,
        "Should return name, age, and city columns"
    );
}

#[test]
fn test_query_nonexistent_label() {
    let db = create_social_network();
    let result = db.execute_graphql("{ movie { title } }").unwrap();
    assert_eq!(result.row_count(), 0, "No Movie nodes exist");
}

// ============================================================================
// Named Queries
// ============================================================================

#[test]
fn test_named_query() {
    let db = create_social_network();
    let result = db
        .execute_graphql("query GetPeople { person { name } }")
        .unwrap();
    assert_eq!(result.row_count(), 3);
}

// ============================================================================
// Anonymous (shorthand) Queries
// ============================================================================

#[test]
fn test_anonymous_query() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 3);
}

// ============================================================================
// Field Aliases
// ============================================================================

#[test]
fn test_field_alias() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person { userName: name } }").unwrap();
    assert_eq!(result.row_count(), 3);

    let alias_col = result.columns.iter().find(|c| c == &"userName");
    assert!(alias_col.is_some(), "Should have aliased column 'userName'");
}

#[test]
fn test_multiple_aliases() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person { n: name a: age } }").unwrap();
    assert_eq!(result.row_count(), 3);
}

// ============================================================================
// Direct Argument Filtering (Equality)
// ============================================================================

#[test]
fn test_filter_by_equality() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(name: "Alix") { name age } }"#)
        .unwrap();
    assert_eq!(
        result.row_count(),
        1,
        "Should find exactly 1 person named Alix"
    );
}

#[test]
fn test_filter_by_integer() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person(age: 25) { name } }").unwrap();
    assert_eq!(result.row_count(), 1, "Only Gus has age 25");
}

#[test]
fn test_filter_no_match() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(name: "Django") { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 0);
}

#[test]
fn test_filter_multiple_arguments() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(name: "Alix", age: 30) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1);
}

// ============================================================================
// Direct Argument Range Filters (Operator Suffixes)
// ============================================================================

#[test]
fn test_filter_gt() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(age_gt: 28) { name } }")
        .unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "Alix (30) and Vincent (35) have age > 28"
    );
}

#[test]
fn test_filter_gte() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(age_gte: 30) { name } }")
        .unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "Alix (30) and Vincent (35) have age >= 30"
    );
}

#[test]
fn test_filter_lt() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(age_lt: 30) { name } }")
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Gus (25) has age < 30");
}

#[test]
fn test_filter_lte() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(age_lte: 25) { name } }")
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Gus (25) has age <= 25");
}

#[test]
fn test_filter_ne() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(city_ne: "Amsterdam") { name } }"#)
        .unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "Gus (Berlin) and Vincent (Paris) are not in Amsterdam"
    );
}

#[test]
fn test_filter_contains() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(city_contains: "er") { name } }"#)
        .unwrap();
    // "Amsterdam" and "Berlin" both contain "er"
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_filter_starts_with() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(city_starts_with: "Am") { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Amsterdam starts with 'Am'");
}

#[test]
fn test_filter_ends_with() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(city_ends_with: "is") { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Paris ends with 'is'");
}

#[test]
fn test_filter_compound_range() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(age_gt: 20, age_lt: 35) { name } }")
        .unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "Gus (25) and Alix (30) have 20 < age < 35"
    );
}

// ============================================================================
// Where Clause Filtering
// ============================================================================

#[test]
fn test_where_gt() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { age_gt: 30 }) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Vincent (35) has age > 30");
}

#[test]
fn test_where_contains() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { name_contains: "li" }) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Alix contains 'li'");
}

#[test]
fn test_where_ne() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { city_ne: "Paris" }) { name city } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 2, "Alix and Gus are not in Paris");
}

#[test]
fn test_where_multiple_conditions() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { age_gte: 25, age_lte: 30 }) { name } }"#)
        .unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "Gus (25) and Alix (30) are in range [25, 30]"
    );
}

#[test]
fn test_where_starts_with() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { name_starts_with: "V" }) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Vincent starts with 'V'");
}

#[test]
fn test_where_ends_with() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { name_ends_with: "us" }) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 1, "Only Gus ends with 'us'");
}

#[test]
fn test_where_in() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { city_in: ["Amsterdam", "Paris"] }) { name } }"#)
        .unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "Alix (Amsterdam) and Vincent (Paris)"
    );
}

// ============================================================================
// Pagination: first, skip, limit, offset
// ============================================================================

#[test]
fn test_pagination_first() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person(first: 2) { name } }").unwrap();
    assert_eq!(result.row_count(), 2, "first: 2 should limit to 2 results");
}

#[test]
fn test_pagination_skip() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person(skip: 1) { name } }").unwrap();
    assert_eq!(result.row_count(), 2, "skip 1 of 3 should yield 2");
}

#[test]
fn test_pagination_first_and_skip() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(first: 1, skip: 1) { name } }")
        .unwrap();
    assert_eq!(result.row_count(), 1, "skip 1, take 1 of 3 should yield 1");
}

#[test]
fn test_pagination_limit_alias() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person(limit: 2) { name } }").unwrap();
    assert_eq!(
        result.row_count(),
        2,
        "limit: 2 should work the same as first: 2"
    );
}

#[test]
fn test_pagination_offset_alias() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person(offset: 2) { name } }")
        .unwrap();
    assert_eq!(result.row_count(), 1, "offset 2 of 3 should yield 1");
}

// ============================================================================
// Ordering: orderBy
// ============================================================================

#[test]
fn test_order_by_asc() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(orderBy: { age: ASC }) { name age } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 3);

    let names: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert_eq!(names, vec!["Gus", "Alix", "Vincent"]);
}

#[test]
fn test_order_by_desc() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(orderBy: { age: DESC }) { name age } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 3);

    let names: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert_eq!(names, vec!["Vincent", "Alix", "Gus"]);
}

#[test]
fn test_order_by_with_pagination() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(orderBy: { age: ASC }, first: 2) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 2);

    let names: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert_eq!(names, vec!["Gus", "Alix"]);
}

#[test]
fn test_order_by_multiple_fields() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(orderBy: { city: ASC }) { name city } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 3);

    let cities: Vec<&str> = result.rows().iter().filter_map(|r| r[1].as_str()).collect();
    assert_eq!(cities, vec!["Amsterdam", "Berlin", "Paris"]);
}

// ============================================================================
// Combined Filters, Ordering, and Pagination
// ============================================================================

#[test]
fn test_filter_order_limit() {
    let db = create_social_network();
    let result = db
        .execute_graphql(
            r#"{ person(age_gte: 25, orderBy: { name: ASC }, first: 2) { name age } }"#,
        )
        .unwrap();
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_where_with_pagination() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(where: { age_gt: 20 }, first: 2) { name } }"#)
        .unwrap();
    assert_eq!(result.row_count(), 2);
}

// ============================================================================
// Nested Relationship Queries (Expand)
// ============================================================================

#[test]
fn test_nested_relationship() {
    let db = create_social_network();
    let result = db
        .execute_graphql("{ person { name knows { name } } }")
        .unwrap();
    // Returns rows for each person-knows pair
    assert!(
        result.row_count() >= 1,
        "Should return results for nested relationship traversal"
    );
}

#[test]
fn test_nested_relationship_with_filter() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person(name: "Alix") { name knows { name } } }"#)
        .unwrap();
    assert!(
        result.row_count() >= 1,
        "Alix should have KNOWS relationships"
    );
}

#[test]
fn test_nested_with_filter_on_child() {
    let db = create_social_network();
    let result = db
        .execute_graphql(r#"{ person { name knows(age_gt: 30) { name } } }"#)
        .unwrap();
    assert!(
        result.row_count() >= 1,
        "Should filter nested results by age > 30"
    );
}

// ============================================================================
// Create Mutations
// ============================================================================

#[test]
fn test_create_mutation() {
    let db = GrafeoDB::new_in_memory();
    db.execute_graphql(r#"mutation { createPerson(name: "Jules", age: 28) { name } }"#)
        .unwrap();

    let result = db.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 1, "Should have 1 Person after create");

    let names: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(names.contains(&"Jules"));
}

#[test]
fn test_create_multiple_nodes() {
    let db = GrafeoDB::new_in_memory();
    db.execute_graphql(r#"mutation { createPerson(name: "Alix", age: 30) { name } }"#)
        .unwrap();
    db.execute_graphql(r#"mutation { createPerson(name: "Gus", age: 25) { name } }"#)
        .unwrap();

    let result = db.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 2);
}

#[test]
fn test_create_mutation_capitalizes_label() {
    let db = GrafeoDB::new_in_memory();
    // "createMovie" should create nodes with label "Movie"
    db.execute_graphql(r#"mutation { createMovie(title: "Pulp Fiction") { title } }"#)
        .unwrap();

    let result = db.execute_graphql("{ movie { title } }").unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_create_mutation_with_multiple_properties() {
    let db = GrafeoDB::new_in_memory();
    db.execute_graphql(
        r#"mutation { createPerson(name: "Mia", age: 28, city: "Prague") { name age city } }"#,
    )
    .unwrap();

    let result = db.execute_graphql("{ person { name age city } }").unwrap();
    assert_eq!(result.row_count(), 1);
}

// ============================================================================
// Update Mutations
// ============================================================================

#[test]
fn test_update_mutation_by_id() {
    let db = create_social_network();

    // Get Alix's node to find properties, then update
    let result = db
        .execute_graphql(r#"mutation { updatePerson(name: "Alix", city: "Prague") { name city } }"#)
        .unwrap();
    assert!(
        result.row_count() >= 1,
        "Update should return the updated node"
    );
}

#[test]
fn test_update_mutation_without_selection_set() {
    let db = create_social_network();
    let result = db.execute_graphql(r#"mutation { updatePerson(name: "Gus", city: "Barcelona") }"#);
    assert!(
        result.is_ok(),
        "Update without selection set should work: {:?}",
        result.err()
    );
}

#[test]
fn test_update_mutation_requires_two_arguments() {
    let db = create_social_network();
    let result = db.execute_graphql(r#"mutation { updatePerson(name: "Alix") { name } }"#);
    assert!(
        result.is_err(),
        "Update with only 1 argument should fail (need filter + property)"
    );
}

// ============================================================================
// Delete Mutations
// ============================================================================

#[test]
fn test_delete_mutation() {
    let db = create_social_network();
    db.execute_graphql(r#"mutation { deleteCompany(name: "Acme") }"#)
        .unwrap();

    let result = db.execute_graphql("{ company { name } }").unwrap();
    assert_eq!(result.row_count(), 0, "Acme should be deleted");
}

#[test]
fn test_delete_mutation_by_property() {
    let db = create_social_network();
    db.execute_graphql(r#"mutation { deletePerson(name: "Vincent") }"#)
        .unwrap();

    let result = db.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 2, "Should have 2 Person nodes left");
}

#[test]
fn test_delete_mutation_requires_filter() {
    let db = create_social_network();
    let result = db.execute_graphql("mutation { deletePerson }");
    assert!(
        result.is_err(),
        "Delete without filter arguments should fail"
    );
}

// ============================================================================
// Mutation Error Cases
// ============================================================================

#[test]
fn test_unknown_mutation_prefix() {
    let db = create_social_network();
    let result = db.execute_graphql(r#"mutation { mergePerson(name: "Alix") { name } }"#);
    assert!(
        result.is_err(),
        "Unknown mutation prefix should fail (only create/update/delete)"
    );
}

#[test]
fn test_subscription_not_supported() {
    let db = create_social_network();
    let result = db.execute_graphql("subscription { personCreated { name } }");
    assert!(result.is_err(), "Subscriptions should not be supported");
}

// ============================================================================
// Query Then Mutate (End-to-End Workflows)
// ============================================================================

#[test]
fn test_create_then_query() {
    let db = GrafeoDB::new_in_memory();

    // Create nodes
    db.execute_graphql(r#"mutation { createPerson(name: "Alix", age: 30) { name } }"#)
        .unwrap();
    db.execute_graphql(r#"mutation { createPerson(name: "Gus", age: 25) { name } }"#)
        .unwrap();

    // Query with filter
    let result = db
        .execute_graphql("{ person(age_gt: 28) { name } }")
        .unwrap();
    assert_eq!(result.row_count(), 1);

    let names: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(names.contains(&"Alix"));
}

#[test]
fn test_create_update_query() {
    let db = GrafeoDB::new_in_memory();

    // Create
    db.execute_graphql(r#"mutation { createPerson(name: "Butch", age: 40) { name } }"#)
        .unwrap();

    // Update
    db.execute_graphql(r#"mutation { updatePerson(name: "Butch", age: 41) { name } }"#)
        .unwrap();

    // Verify
    let result = db.execute_graphql("{ person(age: 41) { name } }").unwrap();
    assert_eq!(result.row_count(), 1);
}

#[test]
fn test_create_delete_query() {
    let db = GrafeoDB::new_in_memory();

    // Create two nodes
    db.execute_graphql(r#"mutation { createPerson(name: "Shosanna", age: 26) { name } }"#)
        .unwrap();
    db.execute_graphql(r#"mutation { createPerson(name: "Hans", age: 50) { name } }"#)
        .unwrap();

    // Delete one
    db.execute_graphql(r#"mutation { deletePerson(name: "Hans") }"#)
        .unwrap();

    // Verify
    let result = db.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 1);

    let names: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(names.contains(&"Shosanna"));
}

// ============================================================================
// Parameterized Queries
// ============================================================================

#[test]
fn test_with_params_api_exists() {
    // GraphQL $variable references are parsed but not yet wired to Parameter
    // expressions in the logical plan, so they resolve to Null during translation.
    // This test verifies the API is callable without panicking.
    let db = create_social_network();
    let mut params = std::collections::HashMap::new();
    params.insert("name".to_string(), Value::String("Alix".into()));

    let result = db.execute_graphql_with_params(r#"{ person { name age } }"#, params);
    assert!(result.is_ok(), "execute_graphql_with_params should succeed");
}

// ============================================================================
// Fragments
// ============================================================================

#[test]
fn test_inline_fragment() {
    let db = create_social_network();
    let result = db
        .execute_graphql(
            r#"
            query {
                person {
                    ... on Person {
                        name
                    }
                }
            }
        "#,
        )
        .unwrap();
    assert!(
        result.row_count() >= 1,
        "Inline fragment should return Person fields"
    );
}

// ============================================================================
// Syntax Errors
// ============================================================================

#[test]
fn test_syntax_error_unclosed_brace() {
    let db = create_social_network();
    let result = db.execute_graphql("{ person { name }");
    assert!(result.is_err(), "Unclosed brace should fail");
}

#[test]
fn test_syntax_error_missing_field_name() {
    let db = create_social_network();
    let result = db.execute_graphql("{ { name } }");
    assert!(result.is_err(), "Missing field name should fail");
}

#[test]
fn test_syntax_error_empty_query() {
    let db = create_social_network();
    let result = db.execute_graphql("");
    assert!(result.is_err(), "Empty query should fail");
}

// ============================================================================
// Session-Level Execution
// ============================================================================

#[test]
fn test_session_execute_graphql() {
    let db = create_social_network();
    let session = db.session();

    let result = session.execute_graphql("{ person { name } }").unwrap();
    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_session_execute_graphql_with_params() {
    let db = create_social_network();
    let session = db.session();

    let params = std::collections::HashMap::new();
    let result = session.execute_graphql_with_params("{ person { name } }", params);
    assert!(
        result.is_ok(),
        "Session-level execute_graphql_with_params should work"
    );
    assert_eq!(result.unwrap().row_count(), 3);
}

// ============================================================================
// execute_language Dispatch
// ============================================================================

#[test]
fn test_execute_language_graphql() {
    let db = create_social_network();
    let result = db
        .execute_language("{ person { name } }", "graphql", None)
        .unwrap();
    assert_eq!(result.row_count(), 3);
}
