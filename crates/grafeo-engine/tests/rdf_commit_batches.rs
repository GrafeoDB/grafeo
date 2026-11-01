//! An RDF commit applies each run of inserts into one graph as one batch
//! (0.6.1), and must keep what applying them one by one gave: the recorded
//! order (a delete or another graph ends a run), one change per triple that
//! changed the store, none for one it held or repeated, in the store, in the
//! change data capture events and in the log a crash replays.
//!
//! - insert, delete, insert of one triple in a transaction: it is there;
//! - a triple inserted twice: one event; one the store held: none;
//! - inserts into g1, g2, g1 with a delete in g2 between: the events in order;
//! - a transaction of 5,000+ inserts (the parallel index build) and one below
//!   4,096: the same store as single inserts, one event per new triple;
//! - after a crash, the reopened store equals the store the commits left.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_commit_batches
//! ```

#![cfg(all(feature = "sparql", feature = "triple-store"))]

use std::collections::BTreeSet;
use std::fmt::Write;

use grafeo_core::graph::rdf::{RdfStore, Term, Triple};
use grafeo_engine::GrafeoDB;

const PREFIX: &str = "PREFIX e: <http://ex.org/>";

/// Committed before the transactions: one triple in the default graph, one
/// in g2.
const SEED: &str = "INSERT DATA { e:held e:p \"held\" . GRAPH e:g2 { e:b0 e:p \"b0\" } }";

/// The transactions, each a list of statements.
const TRANSACTIONS: [&[&str]; 3] = [
    // Insert, delete, insert of one triple.
    &[
        "INSERT DATA { e:a e:p \"A\" }",
        "DELETE DATA { e:a e:p \"A\" }",
        "INSERT DATA { e:a e:p \"A\" }",
    ],
    // A triple twice, one the store holds, one new.
    &[
        "INSERT DATA { e:twice e:p \"T\" }",
        "INSERT DATA { e:twice e:p \"T\" . e:held e:p \"held\" . e:new e:p \"N\" }",
    ],
    // g1, g2, g1, then a delete in g2, then g1 again and g2 again.
    &[
        "INSERT DATA { GRAPH e:g1 { e:a1 e:p \"a1\" } GRAPH e:g2 { e:b1 e:p \"b1\" } \
         GRAPH e:g1 { e:a2 e:p \"a2\" } }",
        "DELETE DATA { GRAPH e:g2 { e:b0 e:p \"b0\" } }",
        "INSERT DATA { GRAPH e:g1 { e:a1 e:p \"a1\" } GRAPH e:g2 { e:b0 e:p \"b0\" } }",
    ],
];

fn run(db: &GrafeoDB, statements: &[&str]) {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    for statement in statements {
        session
            .execute_sparql(&format!("{PREFIX} {statement}"))
            .unwrap_or_else(|error| panic!("{statement}: {error}"));
    }
    session.commit().unwrap();
}

/// Every triple, in every graph, as `graph s p o`.
fn held(db: &GrafeoDB) -> BTreeSet<String> {
    let store = db.rdf_store();
    let mut held: BTreeSet<String> = store
        .triples()
        .iter()
        .map(|triple| format!("- {triple}"))
        .collect();
    for name in store.graph_names() {
        let graph = store.graph(&name).unwrap();
        held.extend(graph.triples().iter().map(|t| format!("{name} {t}")));
    }
    held
}

fn expected_after_all() -> BTreeSet<String> {
    let line = |graph: &str, subject: &str, object: &str| {
        format!("{graph} <http://ex.org/{subject}> <http://ex.org/p> \"{object}\" .")
    };
    [
        line("-", "held", "held"),
        line("-", "a", "A"),
        line("-", "twice", "T"),
        line("-", "new", "N"),
        line("http://ex.org/g1", "a1", "a1"),
        line("http://ex.org/g1", "a2", "a2"),
        line("http://ex.org/g2", "b1", "b1"),
        line("http://ex.org/g2", "b0", "b0"),
    ]
    .into_iter()
    .collect()
}

#[test]
fn the_commits_leave_what_single_applies_left() {
    let db = GrafeoDB::new_in_memory();
    run(&db, &[SEED]);
    for transaction in TRANSACTIONS {
        run(&db, transaction);
    }
    assert_eq!(held(&db), expected_after_all());
}

#[test]
fn insert_delete_insert_in_one_transaction_leaves_the_triple() {
    let db = GrafeoDB::new_in_memory();
    run(&db, TRANSACTIONS[0]);
    assert_eq!(db.rdf_store().len(), 1);
    // And delete, insert, delete leaves it out.
    run(
        &db,
        &[
            "DELETE DATA { e:a e:p \"A\" }",
            "INSERT DATA { e:a e:p \"A\" }",
            "DELETE DATA { e:a e:p \"A\" }",
        ],
    );
    assert_eq!(db.rdf_store().len(), 0);
}

/// The triples a transaction of `new` inserts writes: each new triple, the
/// held ones, then each new one again.
fn large_insert(new: usize) -> (String, Vec<Triple>, Vec<Triple>) {
    let iri = |name: String| Term::iri(format!("http://ex.org/{name}"));
    let fresh: Vec<Triple> = (0..new)
        .map(|i| {
            Triple::new(
                iri(format!("s{}", i % 997)),
                iri("p".into()),
                Term::literal(format!("o{i}")),
            )
        })
        .collect();
    let held: Vec<Triple> = (0..100)
        .map(|i| Triple::new(iri(format!("h{i}")), iri("p".into()), Term::literal("held")))
        .collect();
    let mut data = String::from("INSERT DATA {\n");
    for triple in fresh.iter().chain(&held).chain(fresh.iter().rev()) {
        writeln!(data, "{triple}").unwrap();
    }
    data.push('}');
    (data, fresh, held)
}

#[test]
fn a_large_transaction_leaves_the_store_single_inserts_leave() {
    for new in [3_000, 6_000] {
        let (data, fresh, held) = large_insert(new);
        let db = GrafeoDB::new_in_memory();
        db.rdf_store().batch_insert(held.clone());
        run(&db, &[&data]);
        let single = RdfStore::new();
        for triple in held.iter().chain(&fresh) {
            single.insert(triple.clone());
        }
        let names = |store: &RdfStore| -> BTreeSet<String> {
            store.triples().iter().map(ToString::to_string).collect()
        };
        assert_eq!(db.rdf_store().len(), new + 100);
        assert_eq!(names(db.rdf_store()), names(&single), "{new}");
        for subject in ["s0", "s500", "s996", "h7"] {
            let subject = Term::iri(format!("http://ex.org/{subject}"));
            assert_eq!(
                db.rdf_store().triples_with_subject(&subject).len(),
                single.triples_with_subject(&subject).len()
            );
        }
    }
}

#[cfg(feature = "cdc")]
mod events {
    use super::*;
    use grafeo_common::types::EpochId;
    use grafeo_engine::Config;

    fn cdc_db() -> GrafeoDB {
        GrafeoDB::with_config(Config::in_memory().with_cdc()).unwrap()
    }

    /// The triple events of the commits from here, in order, as
    /// `kind graph subject`.
    fn events_after(db: &GrafeoDB, seen: &mut usize) -> Vec<String> {
        let all = db
            .changes_between(EpochId::new(0), db.current_epoch())
            .unwrap();
        let new: Vec<String> = all[*seen..]
            .iter()
            .filter_map(|event| {
                let subject = event.triple_subject.as_deref()?;
                Some(format!(
                    "{:?} {} {}",
                    event.kind,
                    event.triple_graph.as_deref().unwrap_or("-"),
                    subject
                        .trim_start_matches("<http://ex.org/")
                        .trim_end_matches('>')
                ))
            })
            .collect();
        *seen = all.len();
        new
    }

    #[test]
    fn one_event_per_triple_that_changed_the_store_in_order() {
        let db = cdc_db();
        let mut seen = 0;
        run(&db, &[SEED]);
        assert_eq!(
            events_after(&db, &mut seen),
            ["Create - held", "Create http://ex.org/g2 b0"]
        );
        run(&db, TRANSACTIONS[0]);
        let first = events_after(&db, &mut seen);
        // Insert, delete, insert of one triple: the net change is one
        // insert; however the transaction records it, the events end with a
        // Create and never repeat a kind.
        assert_eq!(
            first.last().map(String::as_str),
            Some("Create - a"),
            "{first:?}"
        );
        assert!(first.windows(2).all(|pair| pair[0] != pair[1]), "{first:?}");
        run(&db, TRANSACTIONS[1]);
        assert_eq!(
            events_after(&db, &mut seen),
            ["Create - twice", "Create - new"],
            "twice once, the held triple not at all"
        );
        run(&db, TRANSACTIONS[2]);
        assert_eq!(
            events_after(&db, &mut seen),
            [
                "Create http://ex.org/g1 a1",
                "Create http://ex.org/g2 b1",
                "Create http://ex.org/g1 a2",
                "Delete http://ex.org/g2 b0",
                "Create http://ex.org/g2 b0",
            ]
        );
        assert_eq!(held(&db), expected_after_all());
    }

    #[test]
    fn a_large_transaction_reports_each_new_triple_once_in_order() {
        for new in [3_000, 6_000] {
            let (data, fresh, held) = large_insert(new);
            let db = cdc_db();
            db.rdf_store().batch_insert(held);
            let mut seen = db
                .changes_between(EpochId::new(0), db.current_epoch())
                .unwrap()
                .len();
            run(&db, &[&data]);
            let events = events_after(&db, &mut seen);
            let want: Vec<String> = fresh
                .iter()
                .map(|triple| {
                    let subject = triple.subject().to_string();
                    format!(
                        "Create - {}",
                        subject
                            .trim_start_matches("<http://ex.org/")
                            .trim_end_matches('>')
                    )
                })
                .collect();
            assert_eq!(events.len(), new);
            assert_eq!(events, want, "{new}");
        }
    }
}

#[cfg(all(feature = "wal", feature = "grafeo-file"))]
mod tests {
    use std::path::{Path, PathBuf};

    use grafeo_common::testing::child_process;
    use grafeo_engine::config::StorageFormat;
    use grafeo_engine::{Config, GrafeoDB};

    use super::{SEED, TRANSACTIONS, expected_after_all, held, large_insert, run};

    const PATH_VAR: &str = "GRAFEO_RDF_COMMIT_BATCHES_PATH";

    fn open(path: &Path) -> GrafeoDB {
        GrafeoDB::with_config(Config::persistent(path).with_storage_format(StorageFormat::Auto))
            .unwrap()
    }

    fn sidecar_wal(path: &Path) -> PathBuf {
        let mut sidecar = path.as_os_str().to_owned();
        sidecar.push(".wal");
        PathBuf::from(sidecar)
    }

    /// Child-process entry: the seed, the transactions and a large one, then
    /// an exit without closing, like a crash. A no-op when run directly.
    #[test]
    fn crash_child() {
        let Some(path) = std::env::var_os(PATH_VAR) else {
            return;
        };
        let db = open(Path::new(&path));
        run(&db, &[SEED]);
        for transaction in TRANSACTIONS {
            run(&db, transaction);
        }
        run(&db, &[&large_insert(5_000).0]);
        std::process::exit(0);
    }

    #[test]
    fn the_log_replays_what_the_commits_left() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batches.grafeo");
        let status = child_process::run(
            std::process::Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "tests::crash_child", "--nocapture"])
                .env(PATH_VAR, &path),
        )
        .unwrap();
        assert!(status.success(), "the child failed");
        assert!(sidecar_wal(&path).exists(), "the crash left the WAL");

        let want = {
            let db = GrafeoDB::new_in_memory();
            run(&db, &[SEED]);
            for transaction in TRANSACTIONS {
                run(&db, transaction);
            }
            run(&db, &[&large_insert(5_000).0]);
            held(&db)
        };
        assert!(want.is_superset(&expected_after_all()));
        let db = open(&path);
        assert_eq!(held(&db), want);
        db.close().unwrap();
    }
}
