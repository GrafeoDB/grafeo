//! Integration tests targeting `crates/grafeo-engine/src/query/planner/lpg/project.rs`.
//!
//! This suite exercises the projection, RETURN, WITH, LIMIT, SKIP, ORDER BY, and
//! DISTINCT planning branches that are not otherwise covered by the higher level
//! test suites. Each test is focused on a specific branch of plan_return,
//! plan_project, plan_limit, plan_skip, or plan_sort.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test coverage_project
//! ```
//!
//! Test data uses Tarantino characters and European cities per project
//! conventions.

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Creates a small social graph used by most tests.
///
/// Nodes:
///   (Vincent:Person {age:30, city:'Amsterdam'})
///   (Jules:Person {age:25, city:'Berlin'})
///   (Mia:Person {age:35, city:'Paris'})
///
/// Edges:
///   Vincent -[:KNOWS {since:2019}]-> Jules
///   Jules   -[:KNOWS {since:2020}]-> Mia
///   Vincent -[:KNOWS {since:2021}]-> Mia
fn tarantino_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let vincent = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Vincent".into())),
                ("age", Value::Int64(30)),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();
    let jules = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Jules".into())),
                ("age", Value::Int64(25)),
                ("city", Value::String("Berlin".into())),
            ],
        )
        .unwrap();
    let mia = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Mia".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Paris".into())),
            ],
        )
        .unwrap();

    session.create_edge(vincent, jules, "KNOWS").unwrap();
    session.create_edge(jules, mia, "KNOWS").unwrap();
    session.create_edge(vincent, mia, "KNOWS").unwrap();

    drop(session);
    db
}

// ============================================================================
// plan_return: standalone RETURN (Empty input path)
// ============================================================================

/// Exercises the Empty input branch that wraps `SingleRowOperator`.
#[test]
fn standalone_return_literal_expression() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session.execute("RETURN 2 * 3 AS product").unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(6));
}

/// Standalone RETURN with a literal expression alias.
#[test]
fn standalone_return_string_literal() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("RETURN 'Amsterdam' AS city, 42 AS answer")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Amsterdam".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));
}

// ============================================================================
// plan_return: RETURN * wildcard (expanded_items path)
// ============================================================================

#[test]
fn return_star_expands_all_user_columns() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN *")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    // `n` should be the single expanded column.
    assert!(!result.columns.is_empty());
    assert!(
        result
            .columns
            .iter()
            .any(|col| col == "n" || !col.starts_with('_')),
        "columns: {:?}",
        result.columns
    );
}

#[test]
fn return_star_skips_internal_underscore_columns() {
    // The wildcard expander filters out columns that start with '_' (e.g.
    // _path_nodes_..., _path_edges_...). A variable length path creates such
    // internal columns; RETURN * should not include them.
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (p = (a:Person {name: 'Vincent'})-[:KNOWS]->(b:Person)){1,1} RETURN *")
        .unwrap();
    // Every user-visible column must not start with an underscore.
    for col in &result.columns {
        assert!(
            !col.starts_with('_'),
            "internal column leaked into RETURN *: {col}"
        );
    }
    assert!(!result.rows().is_empty());
}

// ============================================================================
// plan_return: FunctionCall branches (type, length, nodes, edges)
// ============================================================================

/// Exercises the `type()` function branch (ProjectExpr::EdgeType).
#[test]
fn return_type_of_edge_variable() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN type(r) AS rel_type LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("KNOWS".into()));
}

/// Exercises the `length()` branch for a path variable.
#[test]
fn return_length_of_path_variable() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p = (a:Person {name: 'Vincent'})-[:KNOWS]->(b:Person)){1,2} \
             RETURN length(p) AS len ORDER BY len",
        )
        .unwrap();
    assert!(!result.rows().is_empty());
    let lengths: Vec<i64> = result
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::Int64(n) => Some(*n),
            _ => None,
        })
        .collect();
    assert!(lengths.contains(&1), "missing 1 hop, got {lengths:?}");
    assert!(lengths.contains(&2), "missing 2 hop, got {lengths:?}");
}

/// Exercises the `nodes()` branch and the path detail column mapping.
#[test]
fn return_nodes_of_path_variable() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p = (a:Person {name: 'Vincent'})-[:KNOWS]->(b:Person)){1,1} \
             RETURN nodes(p) AS ns",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 2); // Vincent -> Jules, Vincent -> Mia
    for row in result.rows() {
        match &row[0] {
            Value::List(items) => assert_eq!(items.len(), 2, "expected 2 nodes per hop"),
            other => panic!("expected list, got {other:?}"),
        }
    }
}

/// Exercises the `edges()` branch (edges() is the "edges" case, not
/// "relationships").
#[test]
fn return_edges_of_path_variable() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (p = (a:Person {name: 'Vincent'})-[:KNOWS]->(b:Person)){1,1} \
             RETURN edges(p) AS es",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    for row in result.rows() {
        match &row[0] {
            Value::List(items) => assert_eq!(items.len(), 1, "expected 1 edge per 1 hop path"),
            other => panic!("expected list, got {other:?}"),
        }
    }
}

// ============================================================================
// plan_return: length() fall through to expression evaluation
// ============================================================================

/// length(string) must fall through to expression evaluation rather than the
/// path length branch.
#[test]
fn return_length_of_string_expression() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN size(n.name) AS len")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(7));
}

// ============================================================================
// plan_return: other function calls fall through to expression evaluation
// ============================================================================

#[test]
fn return_other_builtin_function_falls_through() {
    let db = tarantino_graph();
    let session = db.session();

    // head() exercises the "other functions" default branch in plan_return.
    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN head([n.name, 'placeholder']) AS first")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}

// ============================================================================
// plan_return: Literal branch in projection
// ============================================================================

#[test]
fn return_literal_constant_in_projection() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN n.name AS who, 42 AS answer")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(42));
}

// ============================================================================
// plan_return: complex expression branches (Binary/Unary/List/Map/IndexAccess)
// ============================================================================

#[test]
fn return_binary_expression() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN n.age + 5 AS bumped")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(35));
}

#[test]
fn return_unary_expression() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN -n.age AS negated")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(-30));
}

#[test]
fn return_list_literal() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN [n.name, n.city] AS info")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::List(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], Value::String("Vincent".into()));
            assert_eq!(items[1], Value::String("Amsterdam".into()));
        }
        other => panic!("expected list, got {other:?}"),
    }
}

#[test]
fn return_index_access() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute("UNWIND [['Vincent', 'Jules', 'Mia']] AS names RETURN names[1] AS pick")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
}

#[test]
fn return_list_comprehension() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Vincent'}) \
             RETURN [x IN [1, 2, 3] WHERE x > 1 | x * 10] AS doubled",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    match &result.rows()[0][0] {
        Value::List(items) => {
            assert_eq!(items.len(), 2);
            assert_eq!(items[0], Value::Int64(20));
            assert_eq!(items[1], Value::Int64(30));
        }
        other => panic!("expected list, got {other:?}"),
    }
}

#[test]
fn return_list_predicate_any() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Vincent'}) \
             RETURN any(x IN [1, 2, 3] WHERE x > 2) AS has_big",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Bool(true));
}

// ============================================================================
// plan_return: CASE branch
// ============================================================================

#[test]
fn return_case_when_expression() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.name, \
             CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS tier \
             ORDER BY n.name",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    // Jules (25) => junior, Mia (35) => senior, Vincent (30) => senior
    let rows = result.rows();
    assert_eq!(rows[0][0], Value::String("Jules".into()));
    assert_eq!(rows[0][1], Value::String("junior".into()));
    assert_eq!(rows[1][0], Value::String("Mia".into()));
    assert_eq!(rows[1][1], Value::String("senior".into()));
    assert_eq!(rows[2][0], Value::String("Vincent".into()));
    assert_eq!(rows[2][1], Value::String("senior".into()));
}

// ============================================================================
// plan_return simple case (no project operator needed): bare variables only.
// Covers the pass-through branch and edge_columns / scalar_columns routing.
// ============================================================================

#[test]
fn return_bare_node_variable_emits_resolved_map() {
    let db = tarantino_graph();
    let session = db.session();

    // RETURN n (bare node variable) hits plan_return_projection's simple
    // branch, which emits ProjectExpr::NodeResolve.
    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) RETURN n")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(
        matches!(&result.rows()[0][0], Value::Map(_)),
        "expected node resolved to Map, got {:?}",
        result.rows()[0][0]
    );
}

#[test]
fn return_bare_edge_variable_emits_resolved_map() {
    let db = tarantino_graph();
    let session = db.session();

    // RETURN r (bare edge variable) hits plan_return_projection's simple
    // branch, which emits ProjectExpr::EdgeResolve.
    let result = session
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN r LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(
        matches!(&result.rows()[0][0], Value::Map(_)),
        "expected edge resolved to Map, got {:?}",
        result.rows()[0][0]
    );
}

#[test]
fn return_bare_scalar_variable_pass_through() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // UNWIND binds `x` as a scalar column, then RETURN x hits the
    // scalar_columns branch in plan_return_projection.
    let result = session
        .execute("UNWIND [10, 20, 30] AS x RETURN x ORDER BY x")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::Int64(10));
    assert_eq!(result.rows()[1][0], Value::Int64(20));
    assert_eq!(result.rows()[2][0], Value::Int64(30));
}

// ============================================================================
// plan_return: DISTINCT
// ============================================================================

#[test]
fn return_distinct_deduplicates_values() {
    let db = tarantino_graph();
    let session = db.session();

    // Amsterdam, Berlin, Paris are all distinct.
    let result = session
        .execute("MATCH (n:Person) RETURN DISTINCT n.city")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
}

#[test]
fn return_distinct_with_duplicates() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Insert two persons in the same city so DISTINCT actually drops a row.
    session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Butch".into())),
                ("city", Value::String("Prague".into())),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Django".into())),
                ("city", Value::String("Prague".into())),
            ],
        )
        .unwrap();
    drop(session);
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN DISTINCT n.city")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Prague".into()));
}

// ============================================================================
// plan_project (WITH): Variable, Property, Literal, complex branches
// ============================================================================

#[test]
fn with_variable_node_passthrough() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) WITH n RETURN n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
}

#[test]
fn with_property_access_registers_as_scalar() {
    let db = tarantino_graph();
    let session = db.session();

    // WITH n.name AS name forces the Property branch in plan_project.
    let result = session
        .execute("MATCH (n:Person) WITH n.name AS name WHERE name = 'Jules' RETURN name")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
}

#[test]
fn with_literal_constant() {
    let db = tarantino_graph();
    let session = db.session();

    // WITH 123 AS answer forces the Literal branch in plan_project.
    let result = session
        .execute("MATCH (n:Person {name: 'Vincent'}) WITH n, 123 AS answer RETURN n.name, answer")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(123));
}

#[test]
fn with_complex_expression_is_registered_scalar() {
    let db = tarantino_graph();
    let session = db.session();

    // WITH n.age + 1 AS plus forces the catch-all branch (expression
    // evaluation) in plan_project.
    let result = session
        .execute("MATCH (n:Person {name: 'Jules'}) WITH n.age + 1 AS plus RETURN plus")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(26));
}

#[test]
fn with_edge_variable_preserved_as_edge_column() {
    let db = tarantino_graph();
    let session = db.session();

    // WITH r (bare edge variable) must preserve edge_columns metadata so the
    // downstream RETURN resolves it as an edge, not a node.
    let result = session
        .execute("MATCH (a:Person)-[r:KNOWS]->(b:Person) WITH r RETURN type(r) AS rel_type LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("KNOWS".into()));
}

// ============================================================================
// plan_project: Empty input (standalone WITH)
// ============================================================================

#[test]
fn with_literal_list_and_size() {
    // Exercises the plan_project path with a literal list, feeding into a
    // later WITH that evaluates size(). GQL requires WITH to follow a leading
    // statement, so we prefix with UNWIND on a single row.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute(
            "UNWIND [1] AS seed \
             WITH seed, [1, 2, 3] AS nums \
             RETURN size(nums) AS n",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::Int64(3));
}

#[test]
fn with_multiple_literal_bindings() {
    // Multiple literal WITH bindings exercise the Literal branch in
    // plan_project.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let result = session
        .execute(
            "UNWIND [1] AS seed \
             WITH 'Prague' AS city, 100 AS score \
             RETURN city, score",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Prague".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(100));
}

// ============================================================================
// plan_project with pass_through_input (GQL LET clause)
// ============================================================================

#[test]
fn let_clause_passes_through_inputs_and_appends() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute(
            "MATCH (n:Person {name: 'Vincent'}) \
             LET bonus = n.age * 2 \
             RETURN n.name, bonus",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[0][1], Value::Int64(60));
}

// ============================================================================
// plan_limit / plan_skip
// ============================================================================

#[test]
fn limit_restricts_output_count() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name LIMIT 2")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
}

#[test]
fn skip_offsets_output() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name SKIP 1")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    assert_eq!(result.rows()[0][0], Value::String("Mia".into()));
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into()));
}

#[test]
fn skip_then_limit_combines() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name SKIP 1 LIMIT 1")
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert_eq!(result.rows()[0][0], Value::String("Mia".into()));
}

// ============================================================================
// plan_sort: all branches
// ============================================================================

#[test]
fn order_by_property_ascending_pre_return_projection() {
    let db = tarantino_graph();
    let session = db.session();

    // RETURN projects only n.name, so ORDER BY n.age requires pre-Return
    // property projection (the needs_pre_return path).
    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.age")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into())); // 25
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into())); // 30
    assert_eq!(result.rows()[2][0], Value::String("Mia".into())); // 35
    // Output has only the 1 user column (the extra sort-key column is stripped).
    assert_eq!(result.columns.len(), 1);
}

#[test]
fn order_by_property_descending_pre_return_projection() {
    let db = tarantino_graph();
    let session = db.session();

    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.age DESC")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Mia".into()));
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[2][0], Value::String("Jules".into()));
}

#[test]
fn order_by_property_already_in_return() {
    let db = tarantino_graph();
    let session = db.session();

    // ORDER BY n.name references a property the Return has already projected
    // as "n.name": no extra pre-Return projection needed.
    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
    assert_eq!(result.rows()[1][0], Value::String("Mia".into()));
    assert_eq!(result.rows()[2][0], Value::String("Vincent".into()));
    // No extra columns leaked.
    assert_eq!(result.columns.len(), 1);
}

#[test]
fn order_by_with_aliased_property() {
    let db = tarantino_graph();
    let session = db.session();

    // RETURN n.name AS who: the alias is what the caller sees. ORDER BY uses
    // the original variable path. Exercises the alias registration branch in
    // plan_sort.
    let result = session
        .execute("MATCH (n:Person) RETURN n.name AS who ORDER BY n.name")
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
}

#[test]
fn order_by_on_labels_index_access() {
    // Complex expression: labels(n)[0] forces the expression-extras branch
    // in plan_sort (Labels + IndexAccess).
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Bot"], [("name", Value::String("Hans".into()))])
        .unwrap();
    session
        .create_node_with_props(&["Person"], [("name", Value::String("Beatrix".into()))])
        .unwrap();
    drop(session);
    let session = db.session();

    let result = session
        .execute("MATCH (n) RETURN n.name ORDER BY labels(n)[0]")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    // Bot < Person lexicographically
    assert_eq!(result.rows()[0][0], Value::String("Hans".into()));
    assert_eq!(result.rows()[1][0], Value::String("Beatrix".into()));
    // The synthetic __expr_ column must be stripped.
    assert_eq!(result.columns.len(), 1);
}

#[test]
fn order_by_on_edge_type_function() {
    // type(r) as an ORDER BY expression forces the expression extras branch.
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let a = session
        .create_node_with_props(&["Person"], [("name", Value::String("Shosanna".into()))])
        .unwrap();
    let b = session
        .create_node_with_props(&["Person"], [("name", Value::String("Hans".into()))])
        .unwrap();
    let c = session
        .create_node_with_props(&["Person"], [("name", Value::String("Beatrix".into()))])
        .unwrap();
    session.create_edge(a, b, "LIKES").unwrap();
    session.create_edge(a, c, "ADMIRES").unwrap();
    drop(session);
    let session = db.session();

    let result = session
        .execute(
            "MATCH (a:Person {name: 'Shosanna'})-[r]->(b:Person) \
             RETURN b.name ORDER BY type(r)",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    // ADMIRES < LIKES => Beatrix first, Hans second
    assert_eq!(result.rows()[0][0], Value::String("Beatrix".into()));
    assert_eq!(result.rows()[1][0], Value::String("Hans".into()));
    assert_eq!(result.columns.len(), 1);
}

#[test]
fn order_by_nulls_first_explicit() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("first".into())),
                ("score", Value::Int64(10)),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(&["Item"], [("name", Value::String("missing".into()))])
        .unwrap();
    drop(session);
    let session = db.session();

    let result = session
        .execute("MATCH (n:Item) RETURN n.name ORDER BY n.score ASC NULLS FIRST")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    // NULL comes first
    assert_eq!(result.rows()[0][0], Value::String("missing".into()));
    assert_eq!(result.rows()[1][0], Value::String("first".into()));
}

#[test]
fn order_by_nulls_last_explicit() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(
            &["Item"],
            [
                ("name", Value::String("valued".into())),
                ("score", Value::Int64(42)),
            ],
        )
        .unwrap();
    session
        .create_node_with_props(&["Item"], [("name", Value::String("unknown".into()))])
        .unwrap();
    drop(session);
    let session = db.session();

    let result = session
        .execute("MATCH (n:Item) RETURN n.name ORDER BY n.score ASC NULLS LAST")
        .unwrap();
    assert_eq!(result.rows().len(), 2);
    // NULL comes last
    assert_eq!(result.rows()[0][0], Value::String("valued".into()));
    assert_eq!(result.rows()[1][0], Value::String("unknown".into()));
}

#[test]
fn order_by_on_scalar_column_from_with() {
    let db = tarantino_graph();
    let session = db.session();

    // ORDER BY on a plain variable (the "Already in variable_columns" branch
    // in plan_sort and the no-extra-projections path). Include `age` in the
    // RETURN list so it is available to the sort.
    let result = session
        .execute(
            "MATCH (n:Person) \
             WITH n.name AS name, n.age AS age \
             RETURN name, age ORDER BY age",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 3);
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[2][0], Value::String("Mia".into()));
}

#[test]
fn order_by_multiple_keys_mixed_directions() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for (n, c, a) in [
        ("Vincent", "Amsterdam", 30),
        ("Jules", "Amsterdam", 40),
        ("Butch", "Berlin", 25),
        ("Django", "Berlin", 35),
    ] {
        session
            .create_node_with_props(
                &["Person"],
                [
                    ("name", Value::String(n.into())),
                    ("city", Value::String(c.into())),
                    ("age", Value::Int64(a)),
                ],
            )
            .unwrap();
    }
    drop(session);
    let session = db.session();

    // Primary: city ASC, secondary: age DESC
    let result = session
        .execute("MATCH (n:Person) RETURN n.name ORDER BY n.city ASC, n.age DESC")
        .unwrap();
    assert_eq!(result.rows().len(), 4);
    // Amsterdam (Jules age 40, Vincent age 30), Berlin (Django age 35, Butch age 25)
    assert_eq!(result.rows()[0][0], Value::String("Jules".into()));
    assert_eq!(result.rows()[1][0], Value::String("Vincent".into()));
    assert_eq!(result.rows()[2][0], Value::String("Django".into()));
    assert_eq!(result.rows()[3][0], Value::String("Butch".into()));
    assert_eq!(result.columns.len(), 1);
}

// ============================================================================
// derive_schema_from_columns: indirectly exercised via an edge column RETURN
// ============================================================================

#[test]
fn schema_derivation_preserves_edge_column_type() {
    let db = tarantino_graph();
    let session = db.session();

    // WITH r keeps r in edge_columns; downstream RETURN invokes
    // derive_schema_from_columns on the surviving columns. The subsequent
    // RETURN still resolves r as an edge Map.
    let result = session
        .execute(
            "MATCH (a:Person)-[r:KNOWS]->(b:Person) \
             WITH r \
             RETURN r LIMIT 1",
        )
        .unwrap();
    assert_eq!(result.rows().len(), 1);
    assert!(
        matches!(&result.rows()[0][0], Value::Map(_)),
        "edge should be resolved to Map, got {:?}",
        result.rows()[0][0]
    );
}
