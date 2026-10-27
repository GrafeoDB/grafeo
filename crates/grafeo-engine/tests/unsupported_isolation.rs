//! Serializable must be refused until queries record their reads (#671, #400).

use grafeo_common::utils::error::{Error, ErrorCode};
use grafeo_engine::GrafeoDB;
use grafeo_engine::transaction::IsolationLevel;

fn assert_unsupported(error: &Error) {
    assert_eq!(error.error_code(), ErrorCode::QueryUnsupported, "{error}");
    let message = error.to_string().to_lowercase();
    assert!(message.contains("serializable"), "{error}");
    assert!(message.contains("snapshot"), "{error}");
}

#[test]
fn rust_refuses_serializable_without_opening_a_transaction() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    let error = session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect_err("query reads are not tracked for Serializable");
    assert_unsupported(&error);
    assert!(!session.in_transaction());
    session.begin_transaction().unwrap();
    session.commit().unwrap();
    assert!(!session.in_transaction());
}

#[test]
fn rust_nested_refusal_preserves_the_outer_transaction_depth() {
    let db = GrafeoDB::new_in_memory();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let error = session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect_err("a nested begin cannot provide Serializable either");
    assert_unsupported(&error);
    assert!(session.in_transaction());
    session.commit().unwrap();
    assert!(
        !session.in_transaction(),
        "refusal must not add a savepoint"
    );
}

#[test]
fn rust_supported_levels_still_begin_and_commit() {
    let db = GrafeoDB::new_in_memory();
    for level in [
        IsolationLevel::ReadCommitted,
        IsolationLevel::SnapshotIsolation,
    ] {
        let mut session = db.session();
        session.begin_transaction_with_isolation(level).unwrap();
        assert!(session.in_transaction());
        session.commit().unwrap();
        assert!(!session.in_transaction());
    }
}

#[test]
#[cfg(feature = "triple-store")]
fn rdf_sessions_refuse_serializable_too() {
    use grafeo_engine::{Config, GraphModel};

    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let mut session = db.session();
    let error = session
        .begin_transaction_with_isolation(IsolationLevel::Serializable)
        .expect_err("RDF query reads are not tracked for Serializable");
    assert_unsupported(&error);
    assert!(!session.in_transaction());
    session.begin_transaction().unwrap();
    session.rollback().unwrap();
}

#[cfg(feature = "gql")]
mod gql {
    use super::*;
    use grafeo_common::types::Value;

    #[test]
    fn supported_start_modes_still_begin_and_commit() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        for level in [
            "READ COMMITTED",
            "SNAPSHOT",
            "SNAPSHOT ISOLATION",
            "REPEATABLE READ",
        ] {
            session
                .execute(&format!("START TRANSACTION ISOLATION LEVEL {level}"))
                .unwrap();
            assert!(session.in_transaction());
            session.execute("INSERT (:SupportedMode)").unwrap();
            session.execute("COMMIT").unwrap();
            assert!(!session.in_transaction());
        }
        assert_eq!(
            db.execute("MATCH (n:SupportedMode) RETURN count(n)")
                .unwrap()
                .rows(),
            vec![vec![Value::Int64(4)]]
        );
    }

    #[test]
    fn refuses_all_serializable_start_modes() {
        for query in [
            "START TRANSACTION ISOLATION LEVEL SERIALIZABLE",
            "START TRANSACTION READ ONLY ISOLATION LEVEL SERIALIZABLE",
            "START TRANSACTION READ WRITE ISOLATION LEVEL SERIALIZABLE",
        ] {
            let db = GrafeoDB::new_in_memory();
            let session = db.session();
            assert_unsupported(&session.execute(query).expect_err(query));
            assert!(!session.in_transaction(), "{query}");
            assert_unsupported(&db.execute(query).expect_err(query));
            session.execute("START TRANSACTION").unwrap();
            session.execute("ROLLBACK").unwrap();
        }
    }

    #[test]
    fn nested_refusal_preserves_writes_and_outer_commit() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("START TRANSACTION").unwrap();
        session.execute("INSERT (:Probe {value: 1})").unwrap();
        assert_unsupported(
            &session
                .execute("START TRANSACTION READ ONLY ISOLATION LEVEL SERIALIZABLE")
                .expect_err("refuse before adding a savepoint or changing read-only state"),
        );
        assert!(session.in_transaction());
        session.execute("INSERT (:Probe {value: 2})").unwrap();
        session.execute("COMMIT").unwrap();
        assert!(!session.in_transaction());
        assert_eq!(
            db.execute("MATCH (n:Probe) RETURN count(n)")
                .unwrap()
                .rows(),
            vec![vec![Value::Int64(2)]]
        );
    }

    #[test]
    fn nested_refusal_preserves_read_only_state_and_outer_rollback() {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        session.execute("START TRANSACTION READ ONLY").unwrap();
        assert_unsupported(
            &session
                .execute("START TRANSACTION READ WRITE ISOLATION LEVEL SERIALIZABLE")
                .expect_err("refuse without changing the outer read-only transaction"),
        );
        assert!(session.execute("INSERT (:Probe)").is_err());
        session.execute("ROLLBACK").unwrap();
        assert!(!session.in_transaction());
        session.execute("INSERT (:Probe)").unwrap();
    }
}
