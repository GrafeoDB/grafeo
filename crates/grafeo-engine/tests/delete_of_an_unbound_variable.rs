//! A `DELETE` of a variable that nothing binds fails with a semantic error
//! that names the variable, and deletes nothing. A GQL `DELETE w` or
//! `DETACH DELETE w` without a `MATCH` scanned the graph for `w` and deleted
//! every node (0.5.44 too), through every entry point, in a transaction and
//! in a stored procedure's body; in 0.5.44 so did one after `NEXT` from a
//! statement that does not pass `w` on. A Cypher `DELETE w` without input
//! already failed ("DELETE requires input"); it now names `w` as well.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test delete_of_an_unbound_variable
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]

use std::collections::HashMap;

use grafeo_common::types::Value;
use grafeo_common::utils::error::{Error, QueryErrorKind, Result};
use grafeo_engine::GrafeoDB;
use grafeo_engine::database::QueryResult;

/// Two `W` nodes without edges, so a `DELETE` without `DETACH` could delete
/// them too.
const SETUP: &str = "INSERT (:W {k: 3}), (:W {k: 19})";

/// The `k` of every node, in order: what is left of the graph.
fn graph(db: &GrafeoDB) -> Vec<Value> {
    db.execute("MATCH (n) RETURN n.k AS k ORDER BY k")
        .unwrap()
        .rows()
        .iter()
        .map(|row| row[0].clone())
        .collect()
}

/// Asserts that `result` is a semantic error whose message contains `names`.
fn assert_refused(result: Result<QueryResult>, names: &str, what: &str) {
    match result {
        Err(Error::Query(error)) => {
            assert_eq!(
                error.kind,
                QueryErrorKind::Semantic,
                "{what}: {}",
                error.message
            );
            assert!(
                error.message.contains(names),
                "{what}: the error names {names}: {}",
                error.message
            );
        }
        Err(other) => panic!("{what}: not a semantic error: {other}"),
        Ok(result) => panic!("{what} succeeded: {:?} {:?}", result.columns, result.rows()),
    }
}

/// Every GQL form, standalone and after `NEXT` from a statement that does
/// not pass `w` on (one without a result passes on one empty row).
#[test]
fn a_gql_delete_of_an_unbound_variable_deletes_nothing() {
    for query in [
        "DELETE w",
        "DETACH DELETE w",
        "NODETACH DELETE w",
        "DETACH DELETE w.friend",
        "INSERT (:V) NEXT DETACH DELETE w",
        "INSERT (:V) FINISH NEXT DETACH DELETE w",
        "MATCH (v:W {k: 3}) RETURN v NEXT DETACH DELETE w",
    ] {
        let db = GrafeoDB::new_in_memory();
        db.execute(SETUP).unwrap();
        assert_refused(db.execute(query), "'w'", &format!("`{query}`"));
        assert_eq!(
            graph(&db),
            [Value::Int64(3), Value::Int64(19)],
            "`{query}` deletes and writes nothing"
        );
    }
}

/// The entry points the bindings use, with parameters, and in a
/// transaction, which stays usable: its later write commits (`k` 88).
#[test]
fn every_entry_point_refuses_it() {
    type Run = fn(&GrafeoDB) -> Result<QueryResult>;
    let runs: [(&str, Run, &[i64]); 4] = [
        (
            "GrafeoDB::execute_language",
            |db| db.execute_language("DETACH DELETE w", "gql", None),
            &[3, 19],
        ),
        (
            "Session::execute_with_params",
            |db| {
                db.session()
                    .execute_with_params("DETACH DELETE w", HashMap::new())
            },
            &[3, 19],
        ),
        (
            "Session::execute_language with parameters",
            |db| {
                db.session()
                    .execute_language("DETACH DELETE w", "gql", Some(HashMap::new()))
            },
            &[3, 19],
        ),
        (
            "a transaction",
            |db| {
                let mut session = db.session();
                session.begin_transaction()?;
                let result = session.execute("DETACH DELETE w");
                session.execute("INSERT (:W {k: 88})")?;
                session.commit()?;
                result
            },
            &[3, 19, 88],
        ),
    ];
    for (entry, run, left) in runs {
        let db = GrafeoDB::new_in_memory();
        db.execute(SETUP).unwrap();
        assert_refused(run(&db), "'w'", entry);
        let left: Vec<Value> = left.iter().copied().map(Value::Int64).collect();
        assert_eq!(graph(&db), left, "{entry} deletes nothing");
    }
}

/// A stored procedure's body is bound when the call is planned, as a
/// statement is: its `DELETE w` is a semantic error naming `w` (it deleted
/// every node, then failed as an internal error), and deletes nothing.
#[test]
fn a_procedure_body_with_it_deletes_nothing() {
    let db = GrafeoDB::new_in_memory();
    db.execute(SETUP).unwrap();
    db.execute("CREATE PROCEDURE wipe() AS { DETACH DELETE w }")
        .unwrap();
    assert_refused(db.execute("CALL wipe()"), "'w'", "`CALL wipe()`");
    assert_eq!(
        graph(&db),
        [Value::Int64(3), Value::Int64(19)],
        "`CALL wipe()` deletes nothing"
    );
}

/// Cypher refused it already; it stays so, naming the variable.
#[test]
fn a_cypher_delete_without_input_deletes_nothing() {
    for query in ["DELETE w", "DETACH DELETE w"] {
        let db = GrafeoDB::new_in_memory();
        db.execute(SETUP).unwrap();
        assert_refused(db.execute_cypher(query), "'w'", &format!("`{query}`"));
        assert_eq!(
            graph(&db),
            [Value::Int64(3), Value::Int64(19)],
            "`{query}` deletes nothing"
        );
    }
}
