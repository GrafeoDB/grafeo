//! The RDF statistics are kept across writes (0.6.1): `get_or_collect_statistics`
//! collects them again only once the triples inserted or removed since exceed
//! a tenth of the ones they counted, at least 64. A kept collection is the
//! same `Arc`; a new one counts the store as it is:
//!
//! - an empty, a small and a large store: kept up to the threshold, collected
//!   again one change past it;
//! - removals count, an insert and a removal both count, a removal of a
//!   triple that is not there does not; clear() collects again at once;
//! - a batch counts each triple it inserts, not its duplicates; a bulk load
//!   leaves statistics of what it loaded;
//! - queries are right with stale statistics: a predicate they do not list,
//!   most of a predicate's triples removed;
//! - the ring index (ring-index) still goes at the first change.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test rdf_statistics_drift
//! ```

#![cfg(feature = "triple-store")]

use std::sync::Arc;

use grafeo_core::graph::rdf::{RdfStore, STATISTICS_MIN_DRIFT, Term, Triple};
use grafeo_core::statistics::RdfStatistics;

fn triple(subject: usize, predicate: &str) -> Triple {
    Triple::new(
        Term::iri(format!("http://ex.org/s{subject}")),
        Term::iri(format!("http://ex.org/{predicate}")),
        Term::literal(format!("v{subject}")),
    )
}

/// A store with `n` triples of predicate `p`, and its statistics.
fn filled(n: usize) -> (RdfStore, Arc<RdfStatistics>) {
    let store = RdfStore::new();
    store.batch_insert((0..n).map(|i| triple(i, "p")));
    let statistics = store.get_or_collect_statistics();
    assert_eq!(statistics.total_triples, n as u64);
    (store, statistics)
}

fn kept(store: &RdfStore, before: &Arc<RdfStatistics>) -> bool {
    Arc::ptr_eq(before, &store.get_or_collect_statistics())
}

/// The threshold of a store of `n` triples: a tenth, at least 64.
fn threshold(n: usize) -> usize {
    (n / 10).max(STATISTICS_MIN_DRIFT)
}

#[test]
fn kept_up_to_the_threshold_and_collected_one_past_it() {
    for n in [0, 100, 2_000] {
        let (store, before) = filled(n);
        let limit = threshold(n);
        for i in 0..limit {
            assert!(store.insert(triple(n + i, "q")));
        }
        assert!(
            kept(&store, &before),
            "{n} triples: {limit} changes keep them"
        );
        assert!(store.insert(triple(n + limit, "q")));
        let after = store.get_or_collect_statistics();
        assert!(
            !Arc::ptr_eq(&before, &after),
            "{n} triples: one more collects"
        );
        assert_eq!(after.total_triples, (n + limit + 1) as u64);
        assert_eq!(
            after
                .get_predicate("<http://ex.org/q>")
                .map(|q| q.triple_count),
            Some((limit + 1) as u64)
        );
        assert!(kept(&store, &after), "{n} triples: kept again after");
    }
}

#[test]
fn removals_count_and_so_does_an_insert_then_a_removal() {
    let n = 2_000;
    let limit = threshold(n);
    let (store, before) = filled(n);
    for i in 0..limit {
        assert!(store.remove(&triple(i, "p")));
    }
    assert!(kept(&store, &before));
    assert!(store.remove(&triple(limit, "p")));
    let after = store.get_or_collect_statistics();
    assert_eq!(after.total_triples, (n - limit - 1) as u64);

    // Half inserts, half removals: the same number of triples, still enough
    // changes to collect again.
    let (store, before) = filled(n);
    for i in 0..=limit / 2 {
        assert!(store.insert(triple(n + i, "p")));
        assert!(store.remove(&triple(i, "p")));
    }
    let after = store.get_or_collect_statistics();
    assert!(!Arc::ptr_eq(&before, &after));
    assert_eq!(after.total_triples, n as u64);

    // A removal of a triple that is not there changes nothing.
    let (store, before) = filled(n);
    for i in 0..=limit {
        assert!(!store.remove(&triple(n + i, "absent")));
    }
    assert!(kept(&store, &before));
}

#[test]
fn clear_collects_again_at_once() {
    let (store, before) = filled(2_000);
    store.clear();
    let after = store.get_or_collect_statistics();
    assert!(!Arc::ptr_eq(&before, &after));
    assert_eq!(after.total_triples, 0);
}

/// A batch counts the triples it inserts: its duplicates and triples already
/// there do not count.
#[test]
fn a_batch_counts_each_triple_it_inserts() {
    let n = 100;
    let limit = threshold(n);
    let (store, before) = filled(n);
    let inserted = store.batch_insert(
        (0..limit)
            .map(|i| triple(n + i, "q"))
            .chain((0..10).map(|i| triple(n + i, "q")))
            .chain((0..10).map(|i| triple(i, "p"))),
    );
    assert_eq!(inserted, limit);
    assert!(kept(&store, &before), "{limit} new triples keep them");
    assert_eq!(store.batch_insert([triple(n + limit, "q")]), 1);
    assert_eq!(
        store.get_or_collect_statistics().total_triples,
        (n + limit + 1) as u64
    );
}

/// A bulk load replaces the store and leaves statistics of what it loaded:
/// the 1,000 changes before it no longer count, so 64 after it keep the
/// load's own statistics (300 triples, no q) rather than collect again.
#[test]
fn a_bulk_load_leaves_statistics_of_what_it_loaded() {
    let (store, _) = filled(10);
    for i in 0..1_000 {
        store.insert(triple(100 + i, "q"));
    }
    store.bulk_load((0..300).map(|i| triple(i, "r")));
    for i in 0..STATISTICS_MIN_DRIFT {
        assert!(store.insert(triple(1_000 + i, "r")));
    }
    let after = store.get_or_collect_statistics();
    assert_eq!(after.total_triples, 300, "the load's own statistics");
    assert!(after.get_predicate("<http://ex.org/q>").is_none());
    assert_eq!(
        after
            .get_predicate("<http://ex.org/r>")
            .map(|r| r.triple_count),
        Some(300)
    );
    assert!(kept(&store, &after));
    assert!(store.insert(triple(2_000, "r")));
    assert_eq!(
        store.get_or_collect_statistics().total_triples,
        300 + STATISTICS_MIN_DRIFT as u64 + 1,
        "one more change collects"
    );
}

/// The triples `<s{i}> <{predicate}> "{value}{i}"` for each `i` of `range`, as
/// SPARQL data.
#[cfg(feature = "sparql")]
fn data(range: std::ops::Range<usize>, predicate: &str, value: &str) -> String {
    use std::fmt::Write;
    range.fold(String::new(), |mut out, i| {
        write!(
            out,
            "<http://ex.org/s{i}> <http://ex.org/{predicate}> \"{value}{i}\" . "
        )
        .unwrap();
        out
    })
}

/// Queries with stale statistics: a predicate inserted after the collection
/// (the statistics do not list it), and most of a predicate's triples removed
/// (they still count them). The rows are those the triples give.
#[cfg(feature = "sparql")]
#[test]
fn queries_are_right_with_stale_statistics() {
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    let triples = data(0..1_000, "p", "v");
    db.execute_sparql(&format!("INSERT DATA {{ {triples} }}"))
        .unwrap();
    let count = |query: &str| db.execute_sparql(query).unwrap().row_count();
    assert_eq!(count("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }"), 1_000);

    // Ten new triples of a new predicate, on subjects 0 to 9.
    let triples = data(0..10, "q", "w");
    db.execute_sparql(&format!("INSERT DATA {{ {triples} }}"))
        .unwrap();
    let statistics = db.rdf_store().get_or_collect_statistics();
    assert!(
        statistics.get_predicate("<http://ex.org/q>").is_none(),
        "the statistics are stale: q is not in them"
    );
    assert_eq!(count("SELECT ?s WHERE { ?s <http://ex.org/q> ?o }"), 10);
    assert_eq!(
        count("SELECT ?s WHERE { ?s <http://ex.org/p> ?a . ?s <http://ex.org/q> ?b }"),
        10
    );
    assert_eq!(
        count("SELECT ?s WHERE { ?s <http://ex.org/q> ?b . ?s <http://ex.org/p> ?a }"),
        10
    );
    assert_eq!(
        count(
            "SELECT ?s ?b WHERE { ?s <http://ex.org/p> ?a OPTIONAL { ?s <http://ex.org/q> ?b } }"
        ),
        1_000
    );

    // 90 of p's triples removed: with the 10 inserts, 100 changes, at the
    // threshold of a tenth of 1,000: the statistics still count 1,000.
    let triples = data(10..100, "p", "v");
    db.execute_sparql(&format!("DELETE DATA {{ {triples} }}"))
        .unwrap();
    assert!(Arc::ptr_eq(
        &statistics,
        &db.rdf_store().get_or_collect_statistics()
    ));
    assert_eq!(count("SELECT ?s WHERE { ?s <http://ex.org/p> ?o }"), 910);
    assert_eq!(
        count("SELECT ?s WHERE { ?s <http://ex.org/p> ?a . ?s <http://ex.org/q> ?b }"),
        10
    );
}

/// The ring index answers queries: it goes at the first change, while the
/// statistics are kept.
#[cfg(feature = "ring-index")]
#[test]
fn the_ring_index_goes_at_the_first_change() {
    for change in ["insert", "remove", "batch"] {
        let store = RdfStore::new();
        store.bulk_load((0..500).map(|i| triple(i, "p")));
        assert!(store.ring().is_some(), "{change}");
        let before = store.get_or_collect_statistics();
        match change {
            "insert" => assert!(store.insert(triple(500, "p"))),
            "remove" => assert!(store.remove(&triple(0, "p"))),
            _ => assert_eq!(store.batch_insert([triple(500, "q")]), 1),
        }
        assert!(store.ring().is_none(), "{change}: the ring is stale");
        assert!(kept(&store, &before), "{change}: the statistics are kept");
    }
}
