//! Targeted coverage tests for the LPG planner (project.rs, filter.rs, mod.rs).
//!
//! Exercises uncovered branches: RETURN projection (type/length/nodes/edges
//! dispatch, CASE), sort with augmented projection and column stripping,
//! Top-K rewrite negative cases, zone-map early exits, compound filters with
//! remaining predicates, BETWEEN range extraction, correlated EXISTS with
//! ParameterScan, text/vector scan planning paths, and resolve_vector_literal.

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

// ============================================================================
// Fixtures
// ============================================================================

/// Social graph: five Persons (Alix, Gus, Vincent, Jules, Mia) in European
/// cities with KNOWS and FOLLOWS edges. Ages span 22..40 for range tests.
fn social_graph() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.session()
        .execute(
            "CREATE (alix:Person {name: 'Alix', age: 30, city: 'Amsterdam'}),
                    (gus:Person {name: 'Gus', age: 25, city: 'Berlin'}),
                    (vincent:Person {name: 'Vincent', age: 40, city: 'Paris'}),
                    (jules:Person {name: 'Jules', age: 35, city: 'Amsterdam'}),
                    (mia:Person {name: 'Mia', age: 22, city: 'Prague'}),
                    (alix)-[:KNOWS {since: 2020}]->(gus),
                    (gus)-[:KNOWS {since: 2021}]->(vincent),
                    (vincent)-[:KNOWS {since: 2019}]->(jules),
                    (alix)-[:FOLLOWS {weight: 1.5}]->(jules),
                    (jules)-[:FOLLOWS {weight: 2.0}]->(mia)",
        )
        .unwrap();
    db
}

/// Two Doc nodes with 3D embeddings. When `with_index` is set, creates a
/// vector index with the given metric so pushdown paths fire.
#[cfg(feature = "vector-index")]
fn vector_graph(metric: &str, with_index: bool) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    for (title, vec) in [
        ("near", vec![0.9f32, 0.1, 0.0]),
        ("far", vec![0.0f32, 1.0, 0.0]),
    ] {
        let n = db.create_node(&["Doc"]).unwrap();
        db.set_node_property(n, "title", Value::String(title.into()))
            .unwrap();
        db.set_node_property(n, "embedding", Value::Vector(vec.into()))
            .unwrap();
    }
    if with_index {
        db.create_vector_index("Doc", "embedding", Some(3), Some(metric), None, None, None)
            .unwrap();
    }
    db
}

/// Collects all string values from the first column of a QueryResult.
fn strings_col0(result: &grafeo_engine::database::QueryResult) -> Vec<String> {
    result
        .rows()
        .iter()
        .filter_map(|r| match &r[0] {
            Value::String(s) => Some(s.to_string()),
            _ => None,
        })
        .collect()
}

// ============================================================================
// project.rs: RETURN function dispatch (type/length/nodes/edges, CASE)
// ============================================================================

/// type(r) dispatch (project.rs line 139-160).
#[test]
fn test_project_type_function() {
    let session = social_graph().session();
    let r = session
        .execute("MATCH ()-[r:KNOWS]->() RETURN type(r) AS t")
        .unwrap();
    assert!(!r.rows().is_empty());
    for row in r.rows() {
        assert_eq!(row[0], Value::String("KNOWS".into()));
    }
}

/// length(p) dispatch over a `_path_length_` column (project.rs line 162-194).
#[test]
fn test_project_length_function() {
    let session = social_graph().session();
    let r = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..3]->(b:Person) \
             RETURN length(p) AS len ORDER BY len",
        )
        .unwrap();
    assert!(!r.rows().is_empty());
    for row in r.rows() {
        match &row[0] {
            Value::Int64(n) => assert!((1..=3).contains(n)),
            other => panic!("expected Int64 length, got {other:?}"),
        }
    }
}

/// nodes()/edges() dispatch over `_path_nodes_` / `_path_edges_` (line 196-230).
#[test]
fn test_project_nodes_and_edges_functions() {
    let session = social_graph().session();
    let r = session
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[:KNOWS*1..2]->(b:Person) \
             RETURN nodes(p) AS ns, edges(p) AS es",
        )
        .unwrap();
    assert!(!r.rows().is_empty());
    for row in r.rows() {
        let (nodes, edges) = match (&row[0], &row[1]) {
            (Value::List(a), Value::List(b)) => (a, b),
            other => panic!("expected (List, List), got {other:?}"),
        };
        assert_eq!(nodes.len(), edges.len() + 1);
    }
}

/// CASE expression arm in plan_return_projection (line 241).
#[test]
fn test_project_case_expression_ok() {
    let session = social_graph().session();
    let r = session
        .execute(
            "MATCH (n:Person) \
             RETURN n.name AS name, CASE WHEN n.age >= 30 THEN 'senior' ELSE 'junior' END AS bucket \
             ORDER BY name",
        )
        .unwrap();
    assert!(!r.rows().is_empty());
    for row in r.rows() {
        match &row[1] {
            Value::String(s) => {
                let t: &str = s;
                assert!(t == "senior" || t == "junior");
            }
            other => panic!("expected String bucket, got {other:?}"),
        }
    }
}

/// ORDER BY references a property not in RETURN: augmented Return projection
/// (line 600-657) and extra-column stripping after Sort (line 853-870).
#[test]
fn test_sort_by_property_not_in_return() {
    let session = social_graph().session();
    let r = session
        .execute(
            "MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(b:Person) \
             RETURN b.name AS name ORDER BY b.age",
        )
        .unwrap();
    assert_eq!(r.column_count(), 1, "extra age column must be stripped");
    assert_eq!(r.rows().len(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Gus".into()));
}

/// ORDER BY references a complex expression (labels(n)[0]) not in RETURN:
/// augmented return + expr-extra stripping (expr_extra_count branch).
#[test]
fn test_sort_by_complex_expression_not_in_return() {
    let session = social_graph().session();
    let r = session
        .execute("MATCH (n:Person) RETURN n.name AS name ORDER BY labels(n)[0], n.name")
        .unwrap();
    assert_eq!(r.column_count(), 1);
    assert_eq!(
        strings_col0(&r),
        vec!["Alix", "Gus", "Jules", "Mia", "Vincent"]
    );
}

// ============================================================================
// project.rs: Top-K negative cases (try_topk_rewrite returns Ok(None))
// ============================================================================

/// Wrong sort direction for a similarity metric: no Top-K, fallback sort runs.
#[cfg(feature = "vector-index")]
#[test]
fn test_topk_negative_wrong_direction() {
    let session = vector_graph("cosine", true).session();
    let r = session
        .execute(
            "MATCH (d:Doc) RETURN d.title \
             ORDER BY cosine_similarity(d.embedding, [0.9, 0.1, 0.0]) ASC LIMIT 1",
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("far".into()));
}

/// No vector index: try_vector_topk returns Ok(None); brute-force fallback.
#[cfg(feature = "vector-index")]
#[test]
fn test_topk_negative_no_index() {
    let session = vector_graph("cosine", false).session();
    let r = session
        .execute(
            "MATCH (d:Doc) RETURN d.title \
             ORDER BY cosine_similarity(d.embedding, [0.9, 0.1, 0.0]) DESC LIMIT 1",
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("near".into()));
}

/// Score function references a variable other than the scan variable: no rewrite.
#[cfg(feature = "vector-index")]
#[test]
fn test_topk_negative_wrong_variable() {
    let db = GrafeoDB::new_in_memory();
    let d = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(d, "title", Value::String("doc1".into()))
        .unwrap();
    db.set_node_property(d, "embedding", Value::Vector(vec![0.9f32, 0.1, 0.0].into()))
        .unwrap();
    let o = db.create_node(&["Other"]).unwrap();
    db.set_node_property(o, "embedding", Value::Vector(vec![0.5f32, 0.5, 0.0].into()))
        .unwrap();
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let session = db.session();
    // ORDER BY references o.embedding but we scan d:Doc. try_vector_topk
    // bails out because the Property variable doesn't match the scan variable.
    let r = session
        .execute(
            "MATCH (d:Doc), (o:Other) RETURN d.title \
             ORDER BY cosine_similarity(o.embedding, [0.5, 0.5, 0.0]) DESC LIMIT 1",
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("doc1".into()));
}

// ============================================================================
// filter.rs: Zone-map early exit (line 57-65)
// ============================================================================

/// Impossible literal forces EmptyOperator via the zone-map short-circuit.
#[test]
fn test_zone_map_negative_early_exit() {
    let session = social_graph().session();
    let r = session
        .execute("MATCH (n:Person) WHERE n.age = 999999 RETURN n.name")
        .unwrap();
    assert_eq!(r.row_count(), 0);
}

// ============================================================================
// filter.rs: Compound filter with remaining predicate (line 905-926)
// ============================================================================

/// Equality pushed down by property index, range kept as residual FilterOperator.
#[test]
fn test_compound_filter_with_remaining_predicate() {
    let db = social_graph();
    db.create_property_index("name");
    let r = db
        .session()
        .execute("MATCH (n:Person) WHERE n.name = 'Alix' AND n.age > 25 RETURN n.name")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

/// Same compound path but the residual range excludes the equality match.
#[test]
fn test_compound_filter_remaining_predicate_filters_out() {
    let db = social_graph();
    db.create_property_index("name");
    let r = db
        .session()
        .execute("MATCH (n:Person) WHERE n.name = 'Alix' AND n.age > 35 RETURN n.name")
        .unwrap();
    assert_eq!(r.row_count(), 0);
}

// ============================================================================
// filter.rs: BETWEEN / dual-bound range extraction (line 1097-1137)
// ============================================================================

/// Inclusive bounds (Ge + Le) hit extract_between_predicate.
#[test]
fn test_between_range_pattern() {
    let session = social_graph().session();
    let r = session
        .execute("MATCH (n:Person) WHERE n.age >= 25 AND n.age <= 35 RETURN n.name ORDER BY n.name")
        .unwrap();
    assert_eq!(strings_col0(&r), vec!["Alix", "Gus", "Jules"]);
}

/// Exclusive bounds (Gt + Lt) also route through extract_between_predicate.
#[test]
fn test_between_range_exclusive() {
    let session = social_graph().session();
    let r = session
        .execute("MATCH (n:Person) WHERE n.age > 25 AND n.age < 35 RETURN n.name")
        .unwrap();
    assert_eq!(r.row_count(), 1);
    assert_eq!(r.rows()[0][0], Value::String("Alix".into()));
}

// ============================================================================
// filter.rs: Correlated EXISTS with ParameterScan (line 302-320, 400-468)
// ============================================================================

/// Translator emits ParameterScan inside EXISTS; extract_parameter_scan_vars
/// triggers plan_correlated_exists (ApplyOperator with EXISTS mode).
#[test]
fn test_correlated_exists_subquery() {
    let session = social_graph().session();
    let r = session
        .execute(
            "MATCH (n:Person) \
             WHERE EXISTS { MATCH (n)-[:KNOWS]->() } \
             RETURN n.name ORDER BY n.name",
        )
        .unwrap();
    assert_eq!(strings_col0(&r), vec!["Alix", "Gus", "Vincent"]);
}

// ============================================================================
// filter_hybrid.rs: Vector predicate extraction (with and without index)
// ============================================================================

/// Vector pushdown: with an index, filter becomes a VectorScan.
#[cfg(feature = "vector-index")]
#[test]
fn test_vector_predicate_extraction_with_index() {
    let session = vector_graph("euclidean", true).session();
    let r = session
        .execute(
            "MATCH (d:Doc) WHERE euclidean_distance(d.embedding, [0.9, 0.1, 0.0]) < 0.5 \
             RETURN d.title",
        )
        .unwrap();
    assert_eq!(strings_col0(&r), vec!["near"]);
}

/// No vector index: filter falls through to per-row brute-force evaluation.
#[cfg(feature = "vector-index")]
#[test]
fn test_vector_predicate_extraction_no_index() {
    let session = vector_graph("euclidean", false).session();
    let r = session
        .execute(
            "MATCH (d:Doc) WHERE euclidean_distance(d.embedding, [0.9, 0.1, 0.0]) < 0.5 \
             RETURN d.title",
        )
        .unwrap();
    assert_eq!(strings_col0(&r), vec!["near"]);
}

// ============================================================================
// mod.rs: plan_text_scan threshold variant and score projection
// ============================================================================

/// text_score(..) > threshold triggers TextScanOperator::with_threshold
/// (mod.rs line 784-791) and projects the score column (line 802-805).
#[cfg(feature = "text-index")]
#[test]
fn test_plan_text_scan_with_threshold() {
    let db = GrafeoDB::new_in_memory();
    for (title, body) in [
        ("Rust Internals", "rust memory safety and transactions"),
        ("Graph Databases", "property graphs and cypher queries"),
        ("ML Systems", "attention mechanisms in neural networks"),
    ] {
        let n = db.create_node(&["Article"]).unwrap();
        db.set_node_property(n, "title", Value::String(title.into()))
            .unwrap();
        db.set_node_property(n, "body", Value::String(body.into()))
            .unwrap();
    }
    db.create_text_index("Article", "body").unwrap();

    let r = db
        .session()
        .execute(
            "MATCH (d:Article) WHERE text_score(d.body, 'rust') > 0.5 \
             RETURN d.title, text_score(d.body, 'rust') AS score",
        )
        .unwrap();
    assert_eq!(r.column_count(), 2);
    assert!(strings_col0(&r).contains(&"Rust Internals".to_string()));
    for row in r.rows() {
        match &row[1] {
            Value::Float64(s) => assert!(*s > 0.5),
            other => panic!("expected Float64 score, got {other:?}"),
        }
    }
}

// ============================================================================
// mod.rs: plan_vector_scan with_min_similarity / with_max_distance
// ============================================================================

/// cosine_similarity > threshold => with_min_similarity (mod.rs line 879).
#[cfg(feature = "vector-index")]
#[test]
fn test_plan_vector_scan_min_similarity() {
    let session = vector_graph("cosine", true).session();
    let r = session
        .execute(
            "MATCH (d:Doc) WHERE cosine_similarity(d.embedding, [0.9, 0.1, 0.0]) > 0.5 \
             RETURN d.title",
        )
        .unwrap();
    assert_eq!(strings_col0(&r), vec!["near"]);
}

/// euclidean_distance < threshold => with_max_distance (mod.rs line 882).
#[cfg(feature = "vector-index")]
#[test]
fn test_plan_vector_scan_max_distance() {
    let session = vector_graph("euclidean", true).session();
    let r = session
        .execute(
            "MATCH (d:Doc) WHERE euclidean_distance(d.embedding, [0.9, 0.1, 0.0]) < 0.5 \
             RETURN d.title",
        )
        .unwrap();
    assert_eq!(strings_col0(&r), vec!["near"]);
}

// ============================================================================
// mod.rs: resolve_vector_literal (line 903-941)
// ============================================================================

/// Float literal list and Int literal list: both go through the
/// `LogicalExpression::List` branch, coercing Int64 to f32.
#[cfg(feature = "vector-index")]
#[test]
fn test_resolve_vector_literal_from_numeric_list() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(a, "title", Value::String("target".into()))
        .unwrap();
    db.set_node_property(a, "embedding", Value::Vector(vec![1.0f32, 2.0, 3.0].into()))
        .unwrap();
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let session = db.session();
    let r = session
        .execute(
            "MATCH (d:Doc) WHERE cosine_similarity(d.embedding, [1.0, 2.0, 3.0]) > 0.99 \
             RETURN d.title",
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);

    // Int literals path: parser emits Literal(Int64) inside the List.
    let r = session
        .execute(
            "MATCH (d:Doc) WHERE cosine_similarity(d.embedding, [1, 2, 3]) > 0.99 \
             RETURN d.title",
        )
        .unwrap();
    assert_eq!(r.row_count(), 1);
}

/// Non-literal argument: pushdown cannot fire. The planner falls through to
/// per-row evaluation and must neither panic nor silently drop rows.
#[cfg(feature = "vector-index")]
#[test]
fn test_resolve_vector_literal_non_literal_falls_through() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(a, "title", Value::String("target".into()))
        .unwrap();
    db.set_node_property(a, "embedding", Value::Vector(vec![1.0f32, 2.0, 3.0].into()))
        .unwrap();
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let result = db
        .session()
        .execute(
            "MATCH (d:Doc) WHERE cosine_similarity(d.embedding, d.embedding) > 0.99 \
             RETURN d.title",
        )
        .expect("non-literal query vector must fall through to per-row evaluation");
    assert_eq!(result.row_count(), 1);
}

// ============================================================================
// mod.rs: plan_map_collect (line 731-753)
// ============================================================================

/// `collect({k: ..., v: ...})` with a grouping column lowers to MapCollectOp
/// when the binder recognises the pattern. The query must plan successfully;
/// any planner error is a regression we want to surface.
#[test]
fn test_plan_map_collect_via_collect_map() {
    let rs = social_graph()
        .session()
        .execute(
            "MATCH (n:Person) \
             RETURN n.city AS city, collect({name: n.name, age: n.age}) AS people \
             ORDER BY city",
        )
        .expect("collect-map query must plan and execute");
    assert!(rs.column_count() >= 2);
}

// ============================================================================
// mod.rs: plan_horizontal_aggregate edge entity kind (line 704)
// ============================================================================

/// sum(r.weight) over FOLLOWS edges in a variable-length path forces the Edge
/// branch of plan_horizontal_aggregate when the binder emits a
/// HorizontalAggregateOp. No ORDER BY: referencing the aggregate alias in
/// ORDER BY is a separate planner concern unrelated to this coverage target.
#[test]
fn test_plan_horizontal_aggregate_edge() {
    let rs = social_graph()
        .session()
        .execute(
            "MATCH p = (a:Person {name: 'Alix'})-[r:FOLLOWS*1..2]->(b:Person) \
             RETURN sum(r.weight)",
        )
        .expect("horizontal-aggregate edge query must plan and execute");
    assert!(rs.row_count() >= 1);
}

// ============================================================================
// Regression: score column reuse must not cross-contaminate different queries
// (project.rs find_projected_score / vector_score_column_name).
// ============================================================================

/// Two cosine_similarity calls with DIFFERENT query vectors on the same
/// property must not share a score column. Before the fix, the RETURN'd
/// `other` value was read from the filter's score column (computed against
/// [1,0,0]) instead of being recomputed against [0,1,0].
#[cfg(feature = "vector-index")]
#[test]
fn test_score_reuse_isolates_different_query_vectors() {
    let db = GrafeoDB::new_in_memory();
    let a = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(a, "title", Value::String("x-aligned".into()))
        .unwrap();
    db.set_node_property(a, "embedding", Value::Vector(vec![1.0f32, 0.0, 0.0].into()))
        .unwrap();
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let r = db
        .session()
        .execute(
            "MATCH (d:Doc) \
             WHERE cosine_similarity(d.embedding, [1.0, 0.0, 0.0]) > 0.99 \
             RETURN d.title, cosine_similarity(d.embedding, [0.0, 1.0, 0.0]) AS other",
        )
        .expect("query must plan and execute");
    assert_eq!(r.row_count(), 1);
    // The filter matched [1,0,0] (sim=1.0); `other` is against [0,1,0] (sim=0.0).
    // If the planner incorrectly reused the filter's score column, `other`
    // would read 1.0 instead of 0.0.
    let row = &r.rows()[0];
    let other: f64 = match &row[1] {
        Value::Float64(f) => *f,
        other => panic!("expected Float64, got {:?}", other),
    };
    assert!(
        other.abs() < 0.01,
        "second call with orthogonal query should be ~0, got {}",
        other,
    );
}

// ============================================================================
// Regression: VectorScan must use the requested metric, not the index's, when
// they differ (mod.rs plan_vector_scan falls back to brute-force).
// ============================================================================

/// A cosine-built index queried with `euclidean_distance` must produce
/// euclidean-correct results. Before the fix, plan_vector_scan still routed
/// through the HNSW index (ranked by cosine) and only rescaled threshold
/// comparisons, returning the wrong neighbors.
#[cfg(feature = "vector-index")]
#[test]
fn test_vector_scan_metric_mismatch_uses_brute_force() {
    let db = GrafeoDB::new_in_memory();
    // A is close to query in Euclidean space (distance ~1.41) but far in cosine.
    // B is far in Euclidean space (distance ~99) but cosine-identical to query.
    let a = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(a, "title", Value::String("near".into()))
        .unwrap();
    db.set_node_property(a, "embedding", Value::Vector(vec![1.0f32, 0.0, 0.0].into()))
        .unwrap();
    let b = db.create_node(&["Doc"]).unwrap();
    db.set_node_property(b, "title", Value::String("far".into()))
        .unwrap();
    db.set_node_property(
        b,
        "embedding",
        Value::Vector(vec![0.0f32, 100.0, 0.0].into()),
    )
    .unwrap();
    db.create_vector_index(
        "Doc",
        "embedding",
        Some(3),
        Some("cosine"),
        None,
        None,
        None,
    )
    .unwrap();

    let r = db
        .session()
        .execute(
            "MATCH (d:Doc) \
             WHERE euclidean_distance(d.embedding, [0.0, 1.0, 0.0]) < 10.0 \
             RETURN d.title",
        )
        .expect("query must plan and execute");
    let titles = strings_col0(&r);
    assert_eq!(
        titles,
        vec!["near"],
        "euclidean threshold must filter out B (dist=99); got {:?}",
        titles,
    );
}
