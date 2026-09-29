//! SQL/PGQ CALL procedure coverage tests.
//!
//! Tests CALL procedure execution via SQL/PGQ syntax, covering:
//! - Catalog introspection (labels, relationshipTypes, propertyKeys)
//! - Graph algorithms (pagerank, degree_centrality, connected_components, etc.)
//! - YIELD, WHERE, ORDER BY, LIMIT clauses on procedure results
//! - Error cases (unknown procedure, invalid YIELD columns)
//! - GQL/SQL parity for CALL statements
//!
//! Run with:
//! ```bash
//! cargo test -p grafeo-engine --features "sql-pgq,algos" --test sql_pgq_call_coverage
//! ```

#![cfg(all(feature = "sql-pgq", feature = "algos"))]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Test Fixture
// ============================================================================

/// Creates a social network with 5 Person nodes and 6 edges (4 KNOWS + 2 FOLLOWS).
fn create_call_test_graph() -> GrafeoDB {
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
    let jules = session
        .create_node_with_props(
            &["Person"],
            [
                ("name", Value::String("Jules".into())),
                ("age", Value::Int64(35)),
                ("city", Value::String("Amsterdam".into())),
            ],
        )
        .unwrap();

    session.create_edge(alix, gus, "KNOWS").unwrap();
    session.create_edge(alix, vincent, "KNOWS").unwrap();
    session.create_edge(gus, vincent, "KNOWS").unwrap();
    session.create_edge(vincent, mia, "KNOWS").unwrap();
    session.create_edge(alix, jules, "FOLLOWS").unwrap();
    session.create_edge(jules, gus, "FOLLOWS").unwrap();

    db
}

// ============================================================================
// grafeo.procedures() -- Procedure Listing
// ============================================================================

#[test]
fn test_call_procedures_list_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.procedures()").unwrap();

    assert_eq!(result.columns[0], "name");
    assert_eq!(result.columns[1], "description");
    assert!(
        result.row_count() >= 22,
        "Expected at least 22 procedures, got {}",
        result.row_count()
    );
}

#[test]
fn test_call_procedures_list_yield_name() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.procedures() YIELD name")
        .unwrap();

    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0], "name");
    assert!(result.row_count() >= 22);
}

#[test]
fn test_call_procedures_list_yield_alias() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.procedures() YIELD name AS proc_name, description AS info")
        .unwrap();

    assert_eq!(result.columns[0], "proc_name");
    assert_eq!(result.columns[1], "info");
}

// ============================================================================
// Catalog Introspection: Labels
// ============================================================================

#[test]
fn test_call_labels_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.labels()").unwrap();

    assert_eq!(result.columns[0], "label");
    let labels: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(labels.contains(&"Person"));
}

#[test]
fn test_call_labels_yield_alias() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.labels() YIELD label AS node_label")
        .unwrap();

    assert_eq!(result.columns[0], "node_label");
}

#[test]
fn test_call_db_labels() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL db.labels()").unwrap();
    assert_eq!(result.columns[0], "label");
    assert!(result.row_count() >= 1);
}

#[test]
fn test_call_labels_empty_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.labels()").unwrap();
    assert_eq!(result.row_count(), 0);
}

#[test]
fn test_call_labels_multi_label_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(
            &["Person", "Employee"],
            [("name", Value::String("Django".into()))],
        )
        .unwrap();
    session
        .create_node_with_props(
            &["Company"],
            [("name", Value::String("GrafeoDB Inc".into()))],
        )
        .unwrap();

    let result = session.execute_sql("CALL grafeo.labels()").unwrap();
    let labels: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(labels.contains(&"Person"));
    assert!(labels.contains(&"Employee"));
    assert!(labels.contains(&"Company"));
}

// ============================================================================
// Catalog Introspection: Relationship Types
// ============================================================================

#[test]
fn test_call_relationship_types_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.relationshipTypes()")
        .unwrap();

    assert_eq!(result.columns[0], "relationshipType");
    let types: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(types.contains(&"KNOWS"));
    assert!(types.contains(&"FOLLOWS"));
}

#[test]
fn test_call_relationship_types_yield_alias() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.relationshipTypes() YIELD relationshipType AS edge_type")
        .unwrap();

    assert_eq!(result.columns[0], "edge_type");
}

#[test]
fn test_call_relationship_types_empty_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.relationshipTypes()")
        .unwrap();
    assert_eq!(result.row_count(), 0);
}

// ============================================================================
// Catalog Introspection: Property Keys
// ============================================================================

#[test]
fn test_call_property_keys_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.propertyKeys()").unwrap();

    assert_eq!(result.columns[0], "propertyKey");
    let keys: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    assert!(keys.contains(&"name"));
    assert!(keys.contains(&"age"));
    assert!(keys.contains(&"city"));
}

#[test]
fn test_call_property_keys_yield_alias() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.propertyKeys() YIELD propertyKey AS key")
        .unwrap();

    assert_eq!(result.columns[0], "key");
}

#[test]
fn test_call_property_keys_empty_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.propertyKeys()").unwrap();
    assert_eq!(result.row_count(), 0);
}

// ============================================================================
// PageRank
// ============================================================================

#[test]
fn test_call_pagerank_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.pagerank()").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "score");
    assert_eq!(result.row_count(), 5);
}

#[test]
fn test_call_pagerank_yield_score() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD score")
        .unwrap();

    assert_eq!(result.columns.len(), 1);
    assert_eq!(result.columns[0], "score");
    for row in result.rows() {
        if let Value::Float64(s) = &row[0] {
            assert!(*s > 0.0, "PageRank score should be positive");
        }
    }
}

#[test]
fn test_call_pagerank_yield_alias() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD node_id AS id, score AS rank")
        .unwrap();

    assert_eq!(result.columns[0], "id");
    assert_eq!(result.columns[1], "rank");
}

#[test]
fn test_call_pagerank_yield_where() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD node_id, score WHERE score > 999.0")
        .unwrap();

    assert_eq!(result.row_count(), 0, "No score should exceed 999.0");
}

#[test]
fn test_call_pagerank_yield_order_by_desc() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD node_id, score ORDER BY score DESC")
        .unwrap();

    let scores: Vec<f64> = result
        .rows()
        .iter()
        .map(|r| match &r[1] {
            Value::Float64(f) => *f,
            _ => 0.0,
        })
        .collect();
    for i in 1..scores.len() {
        assert!(
            scores[i - 1] >= scores[i],
            "Scores should be DESC: {scores:?}"
        );
    }
}

#[test]
fn test_call_pagerank_yield_limit() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD node_id, score ORDER BY score DESC LIMIT 3")
        .unwrap();

    assert_eq!(result.row_count(), 3);
}

#[test]
fn test_call_pagerank_yield_where_order_limit() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD node_id, score WHERE score > 0.0 ORDER BY score DESC LIMIT 2")
        .unwrap();

    assert!(result.row_count() <= 2);
}

#[test]
fn test_call_pagerank_empty_graph() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.pagerank()").unwrap();
    assert_eq!(result.row_count(), 0);
}

#[test]
fn test_call_pagerank_single_node() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    session
        .create_node_with_props(&["Isolated"], [("name", Value::String("Alix".into()))])
        .unwrap();
    let result = session.execute_sql("CALL grafeo.pagerank()").unwrap();
    assert_eq!(result.row_count(), 1);
}

// ============================================================================
// Degree Centrality
// ============================================================================

#[test]
fn test_call_degree_centrality_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.degree_centrality()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "in_degree");
    assert_eq!(result.columns[2], "out_degree");
    assert_eq!(result.columns[3], "total_degree");
    assert_eq!(result.row_count(), 5);
}

#[test]
fn test_call_degree_centrality_yield_subset() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.degree_centrality() YIELD node_id, total_degree")
        .unwrap();

    assert_eq!(result.columns.len(), 2);
}

#[test]
fn test_call_degree_centrality_yield_order_limit() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.degree_centrality() YIELD node_id, total_degree ORDER BY total_degree DESC LIMIT 1")
        .unwrap();

    assert_eq!(result.row_count(), 1);
}

// ============================================================================
// Betweenness / Closeness Centrality
// ============================================================================

#[test]
fn test_call_betweenness_centrality_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.betweenness_centrality()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "centrality");
    assert_eq!(result.row_count(), 5);
}

#[test]
fn test_call_closeness_centrality_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.closeness_centrality()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "centrality");
    assert_eq!(result.row_count(), 5);
}

// ============================================================================
// Connected Components
// ============================================================================

#[test]
fn test_call_connected_components_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.connected_components()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "component_id");
    assert_eq!(result.row_count(), 5);

    // All nodes should be in the same component
    let first_cid = &result.rows()[0][1];
    for row in result.rows() {
        assert_eq!(&row[1], first_cid, "All nodes should share a component");
    }
}

#[test]
fn test_call_connected_components_disconnected() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    let alix = session
        .create_node_with_props(&["Person"], [("name", Value::String("Alix".into()))])
        .unwrap();
    let gus = session
        .create_node_with_props(&["Person"], [("name", Value::String("Gus".into()))])
        .unwrap();
    session
        .create_node_with_props(&["Person"], [("name", Value::String("Vincent".into()))])
        .unwrap();
    session.create_edge(alix, gus, "KNOWS").unwrap();

    let result = session
        .execute_sql("CALL grafeo.connected_components()")
        .unwrap();

    assert_eq!(result.row_count(), 3);
    // Count distinct component IDs: connected pair + isolated node = 2 components
    let mut cids: Vec<String> = result
        .rows()
        .iter()
        .map(|r| format!("{:?}", r[1]))
        .collect();
    cids.sort();
    cids.dedup();
    assert_eq!(
        cids.len(),
        2,
        "Should have 2 distinct components: connected pair + isolated node"
    );
}

#[test]
fn test_call_strongly_connected_components_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.strongly_connected_components()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "component_id");
    assert_eq!(result.row_count(), 5);
}

// ============================================================================
// Traversal: BFS, DFS
// ============================================================================

#[test]
fn test_call_bfs_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.bfs(0)").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "depth");
    assert!(result.row_count() >= 1);
}

#[test]
fn test_call_dfs_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.dfs(0)").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "depth");
    assert!(result.row_count() >= 1);
}

// ============================================================================
// Shortest Path: Dijkstra, SSSP, Bellman-Ford, Floyd-Warshall
// ============================================================================

#[test]
fn test_call_dijkstra_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.dijkstra(0)").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "distance");
    assert!(result.row_count() >= 1);
}

#[test]
fn test_call_sssp_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    // sssp requires a {source: N} map parameter, but SQL/PGQ CALL doesn't support
    // map arguments in the same way as GQL. Test that it at least produces a meaningful error.
    let result = session.execute_sql("CALL grafeo.sssp(0)");
    assert!(
        result.is_err(),
        "sssp without named source param should error"
    );
}

#[test]
fn test_call_bellman_ford_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.bellman_ford(0)").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "distance");
    assert_eq!(result.columns[2], "has_negative_cycle");
}

#[test]
fn test_call_floyd_warshall_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.floyd_warshall()").unwrap();

    assert_eq!(result.columns[0], "source");
    assert_eq!(result.columns[1], "target");
    assert_eq!(result.columns[2], "distance");
    assert!(result.row_count() >= 1);
}

// ============================================================================
// Community Detection
// ============================================================================

#[test]
fn test_call_label_propagation_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.label_propagation()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "community_id");
    assert_eq!(result.row_count(), 5);
}

#[test]
fn test_call_louvain_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.louvain()").unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "community_id");
    assert_eq!(result.columns[2], "modularity");
    assert_eq!(result.row_count(), 5);
}

// ============================================================================
// Clustering, Topological Sort, Structural Analysis
// ============================================================================

#[test]
fn test_call_clustering_coefficient_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.clustering_coefficient()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "coefficient");
    assert_eq!(result.columns[2], "triangle_count");
    assert_eq!(result.row_count(), 5);

    for row in result.rows() {
        if let Value::Float64(coeff) = &row[1] {
            assert!(
                (0.0..=1.0).contains(coeff),
                "Coefficient {coeff} out of [0.0, 1.0]"
            );
        }
    }
}

#[test]
fn test_call_topological_sort_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.topological_sort()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
    assert_eq!(result.columns[1], "order");
}

#[test]
fn test_call_articulation_points_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.articulation_points()")
        .unwrap();

    assert_eq!(result.columns[0], "node_id");
}

#[test]
fn test_call_bridges_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.bridges()").unwrap();

    assert_eq!(result.columns[0], "source");
    assert_eq!(result.columns[1], "target");
}

#[test]
fn test_call_kcore_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.kcore()").unwrap();

    assert_eq!(result.columns[0], "node_id");
    // kcore returns "value" (core number) and "max_core"
    assert!(
        result.columns.len() >= 2,
        "kcore should return at least 2 columns"
    );
    assert_eq!(result.row_count(), 5);
}

// ============================================================================
// MST: Kruskal, Prim
// ============================================================================

#[test]
fn test_call_kruskal_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.kruskal()").unwrap();

    assert_eq!(result.columns[0], "source");
    assert_eq!(result.columns[1], "target");
    assert_eq!(result.columns[2], "weight");
}

#[test]
fn test_call_prim_basic() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.prim()").unwrap();

    assert_eq!(result.columns[0], "source");
    assert_eq!(result.columns[1], "target");
    assert_eq!(result.columns[2], "weight");
}

// ============================================================================
// Multiple Algorithms on Same Session
// ============================================================================

#[test]
fn test_call_multiple_algorithms_same_session() {
    let db = create_call_test_graph();
    let session = db.session();

    let pr = session.execute_sql("CALL grafeo.pagerank()").unwrap();
    let cc = session
        .execute_sql("CALL grafeo.connected_components()")
        .unwrap();
    let dc = session
        .execute_sql("CALL grafeo.degree_centrality()")
        .unwrap();
    let bc = session
        .execute_sql("CALL grafeo.betweenness_centrality()")
        .unwrap();

    assert_eq!(pr.row_count(), 5);
    assert_eq!(cc.row_count(), 5);
    assert_eq!(dc.row_count(), 5);
    assert_eq!(bc.row_count(), 5);
}

// ============================================================================
// GQL/SQL Parity
// ============================================================================

#[test]
fn test_call_parity_with_gql() {
    let db = create_call_test_graph();
    let session = db.session();

    let gql_result = session.execute("CALL grafeo.pagerank()").unwrap();
    let sql_result = session.execute_sql("CALL grafeo.pagerank()").unwrap();

    assert_eq!(gql_result.columns, sql_result.columns);
    assert_eq!(gql_result.row_count(), sql_result.row_count());
}

#[test]
fn test_call_parity_labels() {
    let db = create_call_test_graph();
    let session = db.session();

    let gql_result = session.execute("CALL grafeo.labels()").unwrap();
    let sql_result = session.execute_sql("CALL grafeo.labels()").unwrap();

    assert_eq!(gql_result.columns, sql_result.columns);
    assert_eq!(gql_result.row_count(), sql_result.row_count());
}

// ============================================================================
// Error Cases
// ============================================================================

#[test]
fn test_call_unknown_procedure() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.nonexistent()");
    assert!(result.is_err(), "Unknown procedure should return an error");
}

#[test]
fn test_call_yield_nonexistent_column() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session.execute_sql("CALL grafeo.pagerank() YIELD nonexistent_column");
    assert!(result.is_err(), "YIELD of nonexistent column should fail");
}

// ============================================================================
// Top-N Pattern
// ============================================================================

#[test]
fn test_call_pagerank_top_1() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.pagerank() YIELD node_id, score ORDER BY score DESC LIMIT 1")
        .unwrap();

    assert_eq!(result.row_count(), 1);

    // Verify this is actually the max score
    let all = session.execute_sql("CALL grafeo.pagerank()").unwrap();
    let max_score = all
        .rows()
        .iter()
        .map(|r| match &r[1] {
            Value::Float64(f) => *f,
            _ => f64::NEG_INFINITY,
        })
        .fold(f64::NEG_INFINITY, f64::max);

    if let Value::Float64(top) = &result.rows()[0][1] {
        assert!(
            (*top - max_score).abs() < 1e-10,
            "Top score {top} should equal max {max_score}"
        );
    }
}

// ============================================================================
// Catalog with ORDER BY and LIMIT
// ============================================================================

#[test]
fn test_call_relationship_types_yield_order() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql(
            "CALL grafeo.relationshipTypes() YIELD relationshipType ORDER BY relationshipType ASC",
        )
        .unwrap();

    let types: Vec<&str> = result.rows().iter().filter_map(|r| r[0].as_str()).collect();
    for i in 1..types.len() {
        assert!(types[i - 1] <= types[i], "Should be ASC order: {types:?}");
    }
}

#[test]
fn test_call_property_keys_yield_limit() {
    let db = create_call_test_graph();
    let session = db.session();
    let result = session
        .execute_sql("CALL grafeo.propertyKeys() YIELD propertyKey LIMIT 2")
        .unwrap();

    assert!(result.row_count() <= 2);
}
