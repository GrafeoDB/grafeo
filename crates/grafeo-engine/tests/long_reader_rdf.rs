//! #673 part A for triple writes (#718 C1, design section 13): behind 2,000 SPARQL commits made
//! after an open transaction began, its delete of a triple one of them wrote fails at commit
//! with `GRAFEO-T001`, its delete of an untouched triple commits, and its insert of a triple
//! another commit inserted too commits (insert against insert does not conflict). Lands with
//! #718 C1 and part A's indexes; before C1, triple writes take no claims.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main long_reader_rdf::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_common::utils::error::ErrorCode;
use grafeo_engine::GrafeoDB;

const COMMITS: usize = 2_000;

fn q(body: &str) -> String {
    format!("PREFIX ex: <http://example.org/> {body}")
}

fn seeded() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute_sparql(&q(
        "INSERT DATA { ex:alix ex:knows ex:gus . ex:gus ex:knows ex:mia . \
         ex:mia ex:knows ex:vincent }",
    ))
    .unwrap();
    db
}

/// `COMMITS` commits of triples of their own.
fn unrelated_commits(db: &GrafeoDB) {
    for i in 0..COMMITS {
        db.execute_sparql(&q(&format!(
            "INSERT DATA {{ ex:n{i} ex:next ex:n{} }}",
            i + 1
        )))
        .unwrap();
    }
}

fn holds(db: &GrafeoDB, triple: &str) -> bool {
    db.execute_sparql(&q(&format!("ASK {{ {triple} }}")))
        .unwrap()
        .rows()[0][0]
        == grafeo_common::types::Value::Bool(true)
}

#[test]
fn triple_conflicts_hold_behind_2000_commits() {
    for (case, committed, own, conflict) in [
        (
            "delete against a committed delete",
            "DELETE DATA { ex:alix ex:knows ex:gus }",
            "DELETE DATA { ex:alix ex:knows ex:gus }",
            true,
        ),
        (
            "delete against a committed insert of the same triple",
            "INSERT DATA { ex:alix ex:likes ex:mia }",
            "DELETE DATA { ex:alix ex:likes ex:mia }",
            // Inserted by this transaction first, so its delete is a recorded write.
            true,
        ),
        (
            "delete of an untouched triple",
            "DELETE DATA { ex:gus ex:knows ex:mia }",
            "DELETE DATA { ex:mia ex:knows ex:vincent }",
            false,
        ),
        (
            "insert against a committed insert",
            "INSERT DATA { ex:vincent ex:knows ex:alix }",
            "INSERT DATA { ex:vincent ex:knows ex:alix }",
            false,
        ),
    ] {
        let db = seeded();
        let mut reader = db.session();
        reader.begin_transaction().unwrap();
        reader
            .execute_sparql(&q("SELECT ?s WHERE { ?s ex:knows ?o }"))
            .unwrap();
        if case.starts_with("delete against a committed insert") {
            reader
                .execute_sparql(&q("INSERT DATA { ex:alix ex:likes ex:mia }"))
                .unwrap();
        }
        unrelated_commits(&db);
        db.execute_sparql(&q(committed)).unwrap();
        unrelated_commits(&db);
        reader.execute_sparql(&q(own)).unwrap();
        let result = reader.commit();
        if conflict {
            let error = result.expect_err(case);
            assert_eq!(
                error.error_code(),
                ErrorCode::TransactionConflict,
                "{case}: {error}"
            );
        } else {
            result.unwrap_or_else(|error| panic!("{case}: {error}"));
            // What the commit wrote is there: its delete gone, its insert held.
            let triple = own
                .trim_start_matches("DELETE DATA {")
                .trim_start_matches("INSERT DATA {")
                .trim_end_matches('}');
            assert_eq!(holds(&db, triple), own.starts_with("INSERT"), "{case}");
        }
    }
}
