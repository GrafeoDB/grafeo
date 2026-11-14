//! The RDF store's graph operations and bulk loads with writers on other threads (#718).
//!
//! - Clearing a named graph while other threads create and drop graphs ends: every writer takes
//!   the dataset's apply guard before the map of named graphs, so none waits on another.
//! - A bulk load with inserts on another thread leaves the memory account equal to its walk,
//!   and the ring it builds holds what the load left.
//!
//! The oracles: a deadline far above what the work takes, `account_mismatches()` (the account
//! against a walk of what the store holds), and the store's own triple count.
//!
//! ```bash
//! cargo test -p grafeo-core --all-features --test main rdf_store_concurrency::
//! ```

#![cfg(feature = "triple-store")]

use std::sync::{Arc, mpsc};
use std::time::Duration;

use grafeo_core::graph::rdf::{LatestUse, RdfStore, RdfView, Term, Triple};

fn triple(subject: &str) -> Triple {
    Triple::new(
        Term::iri(format!("http://ex.org/{subject}")),
        Term::iri("http://ex.org/p"),
        Term::literal(subject),
    )
}

/// What the store's account and its walk disagree on: nothing.
fn assert_account_holds(store: &RdfStore, what: &str) {
    assert_eq!(
        store.account_mismatches(),
        Vec::new(),
        "{what}: the account equals the walk"
    );
}

/// One thread clears a named graph over and over while another creates and drops graphs: both
/// end. A clear that held the map of named graphs while it waited for the apply guard, which a
/// create holds while it waits for the map, would leave both waiting.
#[test]
fn clearing_a_named_graph_while_graphs_come_and_go_finishes() {
    const ROUNDS: usize = 5_000;
    let store = Arc::new(RdfStore::new());
    let (done, finished) = mpsc::channel();
    let clearer = {
        let store = Arc::clone(&store);
        let done = done.clone();
        std::thread::spawn(move || {
            for i in 0..ROUNDS {
                store.insert_into(Some("g"), triple(&format!("alix{i}")));
                store.clear_graph(Some("g"));
            }
            done.send("the clears").unwrap();
        })
    };
    let maker = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for i in 0..ROUNDS {
                let name = format!("h{}", i % 8);
                store.graph_or_create(&name);
                store.drop_graph(&name);
                store.create_graph(&name);
                store.drop_graph(&name);
            }
            done.send("the creates and drops").unwrap();
        })
    };
    for _ in 0..2 {
        let ended = finished.recv_timeout(Duration::from_secs(60));
        assert!(
            ended.is_ok(),
            "a clear and the creates and drops on another thread waited for each other"
        );
    }
    clearer.join().unwrap();
    maker.join().unwrap();
    assert!(
        store
            .graph_at(&RdfView::latest(LatestUse::Test), "g")
            .is_some_and(|graph| graph.is_empty()),
        "g is there, cleared"
    );
    assert_account_holds(&store, "after the clears");
}

/// Bulk loads on one thread and inserts on another: the account equals its walk once both end
/// (an insert between a load's walk and the store of its count would be lost from it), and after
/// a load with no writer the ring holds what the load left.
#[test]
fn bulk_loads_with_inserts_on_another_thread_keep_the_account() {
    const LOADS: usize = 40;
    const LOADED: usize = 2_000;
    const INSERTS: usize = 20_000;
    let store = Arc::new(RdfStore::new());
    let loader = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for round in 0..LOADS {
                store.bulk_load((0..LOADED).map(|i| triple(&format!("gus{round}_{i}"))));
            }
        })
    };
    let inserter = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for i in 0..INSERTS {
                store.insert(triple(&format!("mia{i}")));
            }
        })
    };
    loader.join().unwrap();
    inserter.join().unwrap();
    assert_account_holds(&store, "after the loads and inserts");

    store.bulk_load((0..LOADED).map(|i| triple(&format!("vincent{i}"))));
    assert_eq!(store.len(), LOADED, "a load replaces what the store held");
    assert_account_holds(&store, "after a load alone");
    #[cfg(feature = "ring-index")]
    assert_eq!(
        store.ring_latest().map(|ring| ring.len()),
        Some(LOADED),
        "the ring holds what the load left"
    );
}
