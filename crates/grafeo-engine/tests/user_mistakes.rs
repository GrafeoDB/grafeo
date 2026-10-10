//! A user mistake is reported as one (#588): an error code of the validation
//! or query families, never `GRAFEO-X001` (an internal error, which reads as
//! a bug in Grafeo), and a message that names no Rust method, as the
//! bindings spell their methods otherwise.

#![cfg(all(feature = "lpg", feature = "gql"))]

use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::GrafeoDB;

/// A paper with an embedding and an abstract, and no index on either.
fn papers() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("INSERT (:Paper {title: 'Graphs in Amsterdam', embedding: [0.3, 0.19, 0.88]})")
        .unwrap();
    db
}

/// Asserts that `error` has `code`, names `names`, and names no Rust method
/// (`create_vector_index()`, `Config::with_query_timeout()` and the like; a
/// GQL call such as `grafeo.procedures()` is fine).
fn assert_user_mistake(error: &Error, code: ErrorCode, names: &str, what: &str) {
    assert_eq!(error.error_code(), code, "{what}: {error}");
    let message = error.to_string();
    assert!(message.contains(names), "{what}: names {names}: {message}");
    let rust_method = message
        .split(|c: char| c.is_whitespace() || c == '`' || c == '\'')
        .any(|word| word.ends_with("()") && (word.contains('_') || word.contains("::")));
    assert!(!rust_method, "{what}: names no Rust method: {message}");
}

#[cfg(feature = "vector-index")]
#[test]
fn a_vector_search_without_a_vector_index_is_invalid_input() {
    let db = papers();
    let query = [0.3_f32, 0.19, 0.88];
    let cases = [
        (
            "vector_search",
            db.vector_search("Paper", "embedding", &query, 3, None, None)
                .map(|_| ()),
        ),
        (
            "batch_vector_search",
            db.batch_vector_search("Paper", "embedding", &[query.to_vec()], 3, None, None)
                .map(|_| ()),
        ),
        (
            "mmr_search",
            db.mmr_search("Paper", "embedding", &query, 3, None, None, None, None)
                .map(|_| ()),
        ),
    ];
    for (call, result) in cases {
        let error = result.expect_err(call);
        assert_user_mistake(&error, ErrorCode::InvalidInput, ":Paper(embedding)", call);
    }
}

#[cfg(all(feature = "vector-index", feature = "algos"))]
#[test]
fn a_vector_search_call_without_a_vector_index_is_invalid_input() {
    let db = papers();
    for query in [
        "CALL grafeo.search.vector('Paper', 'embedding', [0.3, 0.19, 0.88], 2)",
        "CALL grafeo.search.mmr('Paper', 'embedding', [0.3, 0.19, 0.88], 2, 3, 0.5)",
    ] {
        let error = db.execute(query).expect_err(query);
        assert_user_mistake(&error, ErrorCode::InvalidInput, ":Paper(embedding)", query);
    }
}

#[cfg(feature = "text-index")]
#[test]
fn a_text_search_without_a_text_index_is_invalid_input() {
    let db = papers();
    let error = db
        .text_search("Paper", "title", "Amsterdam", 3, None)
        .expect_err("text_search");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        ":Paper(title)",
        "text_search",
    );
}

#[cfg(all(feature = "text-index", feature = "algos"))]
#[test]
fn a_text_search_call_without_a_text_index_is_invalid_input() {
    let db = papers();
    let query = "CALL grafeo.search.text('Paper', 'title', 'Amsterdam', 2)";
    let error = db.execute(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::InvalidInput, ":Paper(title)", query);
}

/// An argument of another kind than the procedure takes is a semantic error,
/// as the query alone shows it; a list whose values are no numbers is
/// invalid input, as a parameter can hold it as well.
#[cfg(all(feature = "vector-index", feature = "algos"))]
#[test]
fn a_vector_search_call_with_a_query_that_is_no_vector_is_a_user_mistake() {
    let db = papers();
    for (query, code) in [
        (
            "CALL grafeo.search.vector('Paper', 'embedding', ['Alix', 'Gus'], 2)",
            ErrorCode::InvalidInput,
        ),
        (
            "CALL grafeo.search.vector('Paper', 'embedding', 'Amsterdam', 2)",
            ErrorCode::QuerySemantic,
        ),
    ] {
        let error = db.execute(query).expect_err(query);
        assert_user_mistake(&error, code, "'query'", query);
    }
}

#[cfg(feature = "vector-index")]
#[test]
fn an_unknown_quantization_is_invalid_input() {
    let db = papers();
    let error = db
        .create_vector_index(
            "Paper",
            "embedding",
            Some(3),
            Some("cosine"),
            None,
            None,
            Some("Mia"),
        )
        .expect_err("an unknown quantization");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        "'Mia'",
        "create_vector_index",
    );
}

#[cfg(feature = "embed")]
#[test]
fn an_unregistered_embedding_model_is_invalid_input() {
    let db = papers();
    let error = db
        .embed_text("Vincent", &["Amsterdam"])
        .expect_err("an unregistered model");
    assert_user_mistake(&error, ErrorCode::InvalidInput, "'Vincent'", "embed_text");
}

#[cfg(all(feature = "shacl", feature = "triple-store"))]
#[test]
fn a_missing_shacl_graph_is_invalid_input() {
    let db = papers();
    let error = db.validate_shacl("Butch").expect_err("no shapes graph");
    assert_user_mistake(&error, ErrorCode::InvalidInput, "'Butch'", "validate_shacl");
    let session = db.session();
    let error = session
        .validate_shacl_graph("Jules", "Butch")
        .expect_err("no data graph");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        "'Jules'",
        "validate_shacl_graph",
    );
}

#[cfg(feature = "grafeo-file")]
#[test]
fn a_read_only_open_of_a_missing_database_is_invalid_input() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("prague.grafeo");
    let error = GrafeoDB::open_read_only(&path)
        .map(|_| ())
        .expect_err("missing");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        "prague.grafeo",
        "open_read_only",
    );
    assert!(!path.exists(), "a read-only open creates nothing");
}

// ── What this database or build cannot run ──────────────────────────
//
// A statement this database or build cannot run is unsupported
// (`GRAFEO-Q004`); the tests without a feature run in the CI jobs that build
// the engine without it.

#[cfg(feature = "triple-store")]
#[test]
fn an_lpg_query_on_an_rdf_database_is_unsupported() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::config::GraphModel::Rdf),
    )
    .unwrap();
    let error = db
        .execute("MATCH (p:Paper) RETURN p")
        .expect_err("GQL on an RDF database");
    assert_user_mistake(&error, ErrorCode::QueryUnsupported, "RDF", "GQL");
}

#[cfg(not(feature = "vector-index"))]
#[test]
fn a_vector_index_in_a_build_without_vector_indexes_is_unsupported() {
    let db = papers();
    let query = "CREATE VECTOR INDEX papers ON :Paper(embedding) DIMENSION 3";
    let error = db.execute(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::QueryUnsupported, "vector-index", query);
}

#[cfg(not(feature = "text-index"))]
#[test]
fn a_text_index_in_a_build_without_text_indexes_is_unsupported() {
    let db = papers();
    let query = "CREATE INDEX titles FOR (p:Paper) ON (p.title) USING TEXT";
    let error = db.execute(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::QueryUnsupported, "text-index", query);
}

#[cfg(not(feature = "algos"))]
#[test]
fn a_procedure_call_in_a_build_without_procedures_is_unsupported() {
    let db = papers();
    let query = "CALL grafeo.pagerank()";
    let error = db.execute(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::QueryUnsupported, "algos", query);
}

// ── Timeouts ────────────────────────────────────────────────────────

/// A query past its timeout is a retryable query timeout (`GRAFEO-Q003`)
/// whose hint names no Rust method, and the metrics count it.
#[test]
fn a_query_past_its_timeout_is_a_retryable_timeout() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_query_timeout(std::time::Duration::from_nanos(1)),
    )
    .unwrap();
    for title in ["Graphs in Amsterdam", "Graphs in Berlin", "Graphs in Paris"] {
        db.create_node_with_props(
            &["Paper"],
            [(
                grafeo_common::types::PropertyKey::new("title"),
                grafeo_common::types::Value::from(title),
            )],
        )
        .unwrap();
    }
    let query = "MATCH (p:Paper) RETURN p.title AS title ORDER BY title";
    let error = db.execute(query).expect_err("past the timeout");
    assert_eq!(error.error_code(), ErrorCode::QueryTimeout, "{error}");
    assert!(error.error_code().is_retryable());
    let Error::Query(query_error) = &error else {
        panic!("a query error: {error:?}");
    };
    let hint = query_error.hint.as_deref().unwrap_or_default();
    assert!(
        !hint.contains("Config::") && !hint.contains("()"),
        "the hint names no Rust method: {hint}"
    );
    #[cfg(feature = "metrics")]
    assert_eq!(db.metrics().query_timeouts, 1, "the timeout is counted");
}

// ── Mistakes in the query text ──────────────────────────────────────

#[cfg(feature = "algos")]
#[test]
fn a_call_of_an_unknown_procedure_is_a_semantic_error() {
    let db = papers();
    let query = "CALL grafeo.vincent()";
    let error = db.execute(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::QuerySemantic, "'grafeo.vincent'", query);
}

#[cfg(feature = "algos")]
#[test]
fn a_yield_of_an_unknown_column_is_a_semantic_error() {
    let db = papers();
    let query = "CALL grafeo.procedures() YIELD mia";
    let error = db.execute(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::QuerySemantic, "'mia'", query);
}

#[cfg(feature = "algos")]
#[test]
fn a_call_with_another_number_of_arguments_is_a_semantic_error() {
    let db = papers();
    db.execute(
        "CREATE PROCEDURE greet(name STRING) RETURNS (greeting STRING) AS {          RETURN 'Hallo ' + $name AS greeting }",
    )
    .unwrap();
    for query in ["CALL greet()", "CALL greet('Alix', 'Gus')"] {
        let error = db.execute(query).expect_err(query);
        assert_user_mistake(&error, ErrorCode::QuerySemantic, "'greet'", query);
    }
}

/// A function called with another number of arguments than it takes is a
/// semantic error, wherever the planner or the binder finds it.
#[test]
fn functions_called_with_another_number_of_arguments_are_semantic_errors() {
    let db = papers();
    for query in [
        "MATCH (p:Paper)-[r]->(q) RETURN type()",
        "MATCH (p:Paper)-[r]->(q) RETURN type(r, r)",
        "MATCH path = (p:Paper)-[*]->(q) RETURN length()",
        "MATCH (p:Paper) RETURN labels()",
        "MATCH (p:Paper) RETURN id()",
    ] {
        let error = db.execute(query).expect_err(query);
        assert_eq!(
            error.error_code(),
            ErrorCode::QuerySemantic,
            "{query}: {error}"
        );
    }
}

// ── Calls that name what is not there ───────────────────────────────

#[test]
fn a_direct_write_to_a_missing_node_or_edge_names_it() {
    use grafeo_common::types::{EdgeId, NodeId, Value};

    let db = papers();
    let error = db
        .set_node_property(NodeId::new(999), "title", Value::from("Prague"))
        .unwrap_err();
    assert_user_mistake(&error, ErrorCode::NodeNotFound, "999", "set_node_property");
    let error = db
        .set_edge_property(EdgeId::new(999), "since", Value::Int64(3))
        .unwrap_err();
    assert_user_mistake(&error, ErrorCode::EdgeNotFound, "999", "set_edge_property");
    let alix = db.create_node(&["Person"]).unwrap();
    let error = db.create_edge(alix, NodeId::new(999), "KNOWS").unwrap_err();
    assert_user_mistake(&error, ErrorCode::NodeNotFound, "999", "create_edge");
}

// ── Statements that fail on the data ────────────────────────────────

#[test]
fn a_load_of_a_missing_file_is_a_query_execution_error() {
    let db = papers();
    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join("barcelona.csv")
        .to_string_lossy()
        .replace('\\', "/");
    let query =
        format!("LOAD DATA FROM '{path}' FORMAT CSV WITH HEADERS AS row RETURN row.name AS name");
    let error = db.execute(&query).expect_err(&query);
    assert_user_mistake(
        &error,
        ErrorCode::QueryExecution,
        "barcelona.csv",
        "LOAD DATA",
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_graph_mistakes_are_reported_by_kind() {
    let db = GrafeoDB::with_config(
        grafeo_engine::Config::in_memory().with_graph_model(grafeo_engine::config::GraphModel::Rdf),
    )
    .unwrap();
    db.execute_sparql("CREATE GRAPH <http://example.org/amsterdam>")
        .unwrap();
    let error = db
        .execute_sparql("CREATE GRAPH <http://example.org/amsterdam>")
        .expect_err("a graph that exists");
    assert_user_mistake(
        &error,
        ErrorCode::QueryExecution,
        "amsterdam",
        "CREATE GRAPH",
    );
    let error = db
        .execute_sparql("DROP GRAPH <http://example.org/berlin>")
        .expect_err("a graph that does not exist");
    assert_user_mistake(&error, ErrorCode::QueryExecution, "berlin", "DROP GRAPH");
    let query = "INSERT { GRAPH ?g { <http://example.org/alix> <http://example.org/knows> \
                 <http://example.org/gus> } } WHERE { BIND(<http://example.org/paris> AS ?g) }";
    let error = db.execute_sparql(query).expect_err(query);
    assert_user_mistake(&error, ErrorCode::QuerySemantic, "?g", query);
}

// ── Configuration and input files ───────────────────────────────────

#[test]
fn an_invalid_configuration_is_invalid_input() {
    let error = GrafeoDB::with_config(grafeo_engine::Config::in_memory().with_threads(0))
        .map(|_| ())
        .expect_err("zero threads");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        "threads",
        "with_threads(0)",
    );
}

#[test]
fn a_malformed_edge_list_is_invalid_input_naming_its_line() {
    let db = papers();
    for (data, names) in [("3 19\n88\n", "line 2"), ("3 19\nAlix 88\n", "'Alix'")] {
        let error = db.import_tsv_str(data, "KNOWS", true).expect_err(data);
        assert_user_mistake(&error, ErrorCode::InvalidInput, names, data);
    }
}

#[test]
fn a_malformed_matrix_market_file_is_invalid_input() {
    let db = papers();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("berlin.mtx");
    std::fs::write(&path, "3 19 88\n").unwrap();
    let error = db.import_mmio(&path, "KNOWS").expect_err("no header");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        "MatrixMarket",
        "import_mmio",
    );
}

#[test]
fn snapshot_input_that_cannot_be_read_says_so_by_kind() {
    let error = GrafeoDB::import_snapshot(&[])
        .map(|_| ())
        .expect_err("empty");
    assert_user_mistake(
        &error,
        ErrorCode::InvalidInput,
        "empty",
        "an empty snapshot",
    );
    let error = GrafeoDB::import_snapshot(&[19, 3, 88])
        .map(|_| ())
        .expect_err("another version");
    assert_user_mistake(&error, ErrorCode::InvalidInput, "19", "another version");
    // Outside input that does not decode (`GRAFEO-X002`, as FD9 has it).
    let error = GrafeoDB::import_snapshot(&[4, 255, 255, 255])
        .map(|_| ())
        .expect_err("no snapshot");
    assert_eq!(error.error_code(), ErrorCode::SerializationError, "{error}");
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
#[test]
fn a_backup_of_an_in_memory_database_is_unsupported() {
    let db = papers();
    let dir = tempfile::tempdir().unwrap();
    let error = db.backup_full(dir.path()).map(|_| ()).expect_err("full");
    assert_user_mistake(
        &error,
        ErrorCode::QueryUnsupported,
        "persistent",
        "backup_full",
    );
    let error = db
        .backup_incremental(dir.path())
        .map(|_| ())
        .expect_err("incremental");
    assert_user_mistake(
        &error,
        ErrorCode::QueryUnsupported,
        "WAL",
        "backup_incremental",
    );
}
