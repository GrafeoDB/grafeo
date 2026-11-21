//! Writes to named graphs, in a session transaction, in auto-commit and in a load into a new
//! named graph, leave the memory account at what the data holds, as writes to the default graph
//! do. Small named graphs and open transactions each hold a grant of about their own size, and a
//! write after them is admitted under the limit.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main rdf_named_graph_grants::
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use grafeo_engine::{Config, GrafeoDB, RdfFormat};

const MIB: usize = 1024 * 1024;

/// The commits each test makes.
const COMMITS: usize = 200;

fn database() -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_memory_limit(64 * MIB)).unwrap()
}

/// The bytes the database's buffer manager holds.
fn allocated(db: &GrafeoDB) -> usize {
    db.memory_usage().buffer_manager.allocated_bytes
}

/// Asserts that `COMMITS` writes of one triple each left the account within a small multiple of
/// what the triples hold.
fn assert_at_what_the_data_holds(db: &GrafeoDB, before: usize, what: &str) {
    let grown = allocated(db).saturating_sub(before);
    assert!(
        grown < 4 * MIB,
        "{what}: {COMMITS} one-triple commits left {grown} bytes in the account"
    );
}

/// A write and a read after the commits are still admitted under the limit.
fn assert_admitted(db: &GrafeoDB) {
    db.execute_sparql("INSERT DATA { GRAPH <urn:after> { <urn:butch> <urn:knows> \"after\" } }")
        .expect("a write after the commits is admitted");
    let result = db
        .execute_sparql("SELECT ?o WHERE { GRAPH <urn:after> { <urn:butch> <urn:knows> ?o } }")
        .expect("a read after the commits is admitted");
    assert_eq!(result.rows().len(), 1, "the write after the commits landed");
}

#[test]
fn session_commits_to_a_named_graph_leave_the_account_at_what_the_data_holds() {
    let db = database();
    let before = allocated(&db);
    let mut session = db.session();
    for k in 0..COMMITS {
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!(
                "INSERT DATA {{ GRAPH <urn:g> {{ <urn:alix> <urn:knows> \"{k}\" }} }}"
            ))
            .unwrap_or_else(|error| panic!("commit {k}: {error}"));
        session.commit().unwrap();
    }
    assert_at_what_the_data_holds(&db, before, "session commits");
    assert_admitted(&db);
}

#[test]
fn auto_commits_to_a_named_graph_leave_the_account_at_what_the_data_holds() {
    let db = database();
    let before = allocated(&db);
    for k in 0..COMMITS {
        db.execute_sparql(&format!(
            "INSERT DATA {{ GRAPH <urn:g> {{ <urn:gus> <urn:knows> \"{k}\" }} }}"
        ))
        .unwrap_or_else(|error| panic!("statement {k}: {error}"));
    }
    assert_at_what_the_data_holds(&db, before, "auto-commits");
    assert_admitted(&db);
}

#[test]
fn loads_into_new_named_graphs_leave_the_account_at_what_the_data_holds() {
    let db = database();
    let before = allocated(&db);
    for k in 0..COMMITS {
        db.load_rdf_text(
            &format!("<urn:mia> <urn:knows> \"{k}\" .\n"),
            RdfFormat::NTriples,
            Some(&format!("urn:graph{k}")),
        )
        .unwrap_or_else(|error| panic!("load {k}: {error}"));
    }
    assert_at_what_the_data_holds(&db, before, "loads into new named graphs");
    assert_admitted(&db);
}

#[test]
fn a_default_graph_commit_still_holds_what_it_wrote() {
    let db = database();
    let before = allocated(&db);
    for k in 0..COMMITS {
        db.execute_sparql(&format!(
            "INSERT DATA {{ <urn:vincent> <urn:knows> \"{k}\" }}"
        ))
        .unwrap();
    }
    let grown = allocated(&db) - before;
    assert!(
        grown > 0 && grown < 4 * MIB,
        "the default graph's triples are counted: {grown} bytes"
    );
    assert_admitted(&db);
}

#[test]
fn loads_into_one_new_named_graph_leave_the_account_at_what_the_data_holds() {
    let db = database();
    let before = allocated(&db);
    for k in 0..COMMITS {
        db.load_rdf_text(
            &format!("<urn:jules> <urn:knows> \"{k}\" .\n"),
            RdfFormat::NTriples,
            Some("urn:loaded"),
        )
        .unwrap_or_else(|error| panic!("load {k}: {error}"));
    }
    assert_at_what_the_data_holds(&db, before, "loads into one named graph");
    assert_admitted(&db);
}

/// Transactions open at once, each holding one inserted triple.
const OPEN_TRANSACTIONS: usize = 64;

#[test]
fn many_open_transactions_with_one_triple_each_leave_room() {
    let db = database();
    let before = allocated(&db);
    let mut sessions: Vec<_> = (0..OPEN_TRANSACTIONS).map(|_| db.session()).collect();
    for (k, session) in sessions.iter_mut().enumerate() {
        session.begin_transaction().unwrap();
        session
            .execute_sparql(&format!(
                "INSERT DATA {{ GRAPH <urn:open{k}> {{ <urn:hans> <urn:knows> \"{k}\" }} }}"
            ))
            .unwrap_or_else(|error| panic!("transaction {k}: {error}"));
    }
    let grown = allocated(&db).saturating_sub(before);
    assert!(
        grown < 16 * MIB,
        "{OPEN_TRANSACTIONS} open one-triple transactions hold {grown} bytes"
    );
    assert_admitted(&db);
    for session in &mut sessions {
        session.commit().unwrap();
    }
    assert_admitted(&db);
}
