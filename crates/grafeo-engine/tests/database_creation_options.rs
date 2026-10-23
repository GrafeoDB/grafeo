//! Integration tests for database creation options (0.4.3).
//!
//! Verifies GraphModel, DurabilityMode, Config::validate(), query routing,
//! schema_constraints, and inspection API.

use grafeo_engine::{Config, ConfigError, DurabilityMode, GrafeoDB, GraphModel};

// --- GraphModel routing tests ---

#[test]
fn lpg_database_executes_gql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Alix'})").unwrap();
    let result = session.execute("MATCH (p:Person) RETURN p.name").unwrap();
    assert_eq!(result.rows().len(), 1);
}

#[cfg(feature = "triple-store")]
#[test]
fn rdf_database_rejects_gql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    let result = session.execute("MATCH (p:Person) RETURN p.name");
    assert!(result.is_err());
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("RDF database"),
        "Expected RDF error, got: {err_msg}"
    );
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn rdf_database_executes_sparql() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    // Simple SPARQL query - may return empty but should not error
    let result = session.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1");
    assert!(result.is_ok(), "SPARQL on RDF db should work: {result:?}");
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn lpg_database_allows_explicit_sparql() {
    // Explicit execute_sparql() works on any database (both stores are always initialized).
    // Only the generic execute() enforces graph model routing.
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let session = db.session();
    let result = session.execute_sparql("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1");
    assert!(
        result.is_ok(),
        "Explicit SPARQL should work on LPG db: {result:?}"
    );
}

#[cfg(feature = "cypher")]
#[test]
fn lpg_database_executes_cypher() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Lpg)).unwrap();
    let session = db.session();
    session.execute("INSERT (:Person {name: 'Gus'})").unwrap();
    let result = session.execute_cypher("MATCH (p:Person) RETURN p.name");
    assert!(result.is_ok());
}

#[cfg(all(feature = "cypher", feature = "triple-store"))]
#[test]
fn rdf_database_allows_explicit_cypher() {
    // Explicit execute_cypher() works on any database (both stores are always initialized).
    // Only the generic execute() enforces graph model routing.
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    let result = session.execute_cypher("MATCH (p:Person) RETURN p.name");
    assert!(
        result.is_ok(),
        "Explicit Cypher should work on RDF db: {result:?}"
    );
}

// --- Config::validate() tests ---

#[test]
fn validate_rejects_zero_memory_limit() {
    let config = Config::in_memory().with_memory_limit(0);
    assert_eq!(config.validate(), Err(ConfigError::ZeroMemoryLimit));
}

#[test]
fn validate_rejects_zero_threads() {
    let config = Config::in_memory().with_threads(0);
    assert_eq!(config.validate(), Err(ConfigError::ZeroThreads));
}

#[cfg(not(feature = "triple-store"))]
#[test]
fn validate_rejects_rdf_without_feature() {
    let config = Config::in_memory().with_graph_model(GraphModel::Rdf);
    assert_eq!(config.validate(), Err(ConfigError::RdfFeatureRequired));
}

#[test]
fn with_config_rejects_invalid_config() {
    let config = Config::in_memory().with_threads(0);
    let result = GrafeoDB::with_config(config);
    assert!(result.is_err());
}

// --- Inspection API tests ---

#[test]
fn graph_model_accessor_returns_lpg() {
    let db = GrafeoDB::new_in_memory();
    assert_eq!(db.graph_model(), GraphModel::Lpg);
}

#[cfg(feature = "triple-store")]
#[test]
fn graph_model_accessor_returns_rdf() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    assert_eq!(db.graph_model(), GraphModel::Rdf);
}

#[test]
fn memory_limit_accessor_returns_none_by_default() {
    let db = GrafeoDB::new_in_memory();
    // Default in-memory config has no explicit memory limit
    // (it gets set during buffer manager init, not in config)
    assert!(db.memory_limit().is_none());
}

#[test]
fn memory_limit_accessor_returns_configured_value() {
    let db =
        GrafeoDB::with_config(Config::in_memory().with_memory_limit(256 * 1024 * 1024)).unwrap();
    assert_eq!(db.memory_limit(), Some(256 * 1024 * 1024));
}

// --- DurabilityMode tests ---

#[test]
fn default_durability_is_batch() {
    let config = Config::default();
    assert_eq!(config.wal_durability(), DurabilityMode::default());
}

#[test]
fn config_with_sync_durability() {
    let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::Sync);
    assert_eq!(config.wal_durability(), DurabilityMode::Sync);
    assert!(config.validate().is_ok());
}

#[test]
fn config_with_nosync_durability() {
    let config = Config::persistent("/tmp/db").with_wal_durability(DurabilityMode::NoSync);
    assert_eq!(config.wal_durability(), DurabilityMode::NoSync);
    assert!(config.validate().is_ok());
}

// --- schema_constraints tests ---

#[test]
fn schema_constraints_default_is_false() {
    let config = Config::default();
    assert!(!config.schema_constraints());
}

#[test]
fn schema_constraints_can_be_enabled() {
    let config = Config::in_memory().with_schema_constraints();
    assert!(config.schema_constraints());
}

// --- CDC retention ---

/// The retention set with `Config::with_cdc_retention` bounds the change
/// history a database keeps: a garbage collection drops the oldest events
/// beyond it, while the default retention keeps them all.
#[cfg(all(feature = "cdc", feature = "gql"))]
#[test]
fn cdc_retention_from_the_config_bounds_the_history() {
    use grafeo_common::types::{EpochId, Value};
    use grafeo_engine::cdc::CdcRetentionConfig;

    let ages = [19, 88, 33, 38, 83];
    let history_after_gc = |config: Config| {
        let db = GrafeoDB::with_config(config.with_cdc()).unwrap();
        let session = db.session();
        session
            .execute("INSERT (:Person {name: 'Alix', age: 3})")
            .unwrap();
        for age in ages {
            session
                .execute(&format!("MATCH (p:Person) SET p.age = {age}"))
                .unwrap();
        }
        db.gc();
        db.changes_between(EpochId::new(0), EpochId::new(u64::MAX))
            .unwrap()
    };

    let kept_by_default = history_after_gc(Config::in_memory());
    assert_eq!(
        kept_by_default.len(),
        1 + ages.len(),
        "the default retention keeps the insert and every update"
    );

    let bounded = history_after_gc(
        Config::in_memory().with_cdc_retention(CdcRetentionConfig::unlimited().with_max_events(3)),
    );
    let kept_ages: Vec<Option<Value>> = bounded
        .iter()
        .map(|event| {
            event
                .after
                .as_ref()
                .and_then(|after| after.get("age").cloned())
        })
        .collect();
    assert_eq!(
        kept_ages,
        vec![
            Some(Value::Int64(33)),
            Some(Value::Int64(38)),
            Some(Value::Int64(83)),
        ],
        "a retention of 3 events keeps the 3 newest"
    );
}

// --- Session graph_model accessor ---

#[test]
fn session_reports_graph_model() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    assert_eq!(session.graph_model(), GraphModel::Lpg);
}

#[cfg(feature = "triple-store")]
#[test]
fn session_reports_rdf_graph_model() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    assert_eq!(session.graph_model(), GraphModel::Rdf);
}

// --- SPARQL FILTER regression tests ---

/// A plain literal is no number, though its text reads as one (#464): the
/// stored `"30"` is not the integer `30` in `FILTER(?age = 30)` and is in
/// `FILTER(?age != 30)` (literals of two types differ, SPARQL 1.1 17.3); the
/// integer literal `30` is.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn sparql_filter_compares_a_plain_literal_with_a_number_as_unequal() {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();
    session
        .execute_sparql(
            r#"INSERT DATA {
                <http://ex.org/alix> <http://ex.org/age> "30" .
                <http://ex.org/gus>   <http://ex.org/age> "25" .
                <http://ex.org/mia>   <http://ex.org/age> 30 .
            }"#,
        )
        .unwrap();
    let subjects = |filter: &str| -> Vec<String> {
        let mut found: Vec<String> = session
            .execute_sparql(&format!(
                "SELECT ?s WHERE {{ ?s <http://ex.org/age> ?age . FILTER({filter}) }}"
            ))
            .unwrap()
            .rows()
            .iter()
            .map(|row| row[0].to_string())
            .collect();
        found.sort();
        found
    };
    assert_eq!(subjects("?age = 30"), ["<http://ex.org/mia>"]);
    assert_eq!(
        subjects("?age != 30"),
        ["<http://ex.org/alix>", "<http://ex.org/gus>"]
    );
    assert_eq!(subjects("?age = \"30\""), ["<http://ex.org/alix>"]);
}
