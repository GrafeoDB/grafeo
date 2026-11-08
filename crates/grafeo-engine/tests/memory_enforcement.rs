//! The memory limit is enforced (#398, #751): a write that needs memory past it fails with
//! GRAFEO-S001 naming the bound, with the requested, used and limit numbers and the setting to
//! change, and the statement changes nothing (the store as before, the account equal to the
//! walk); earlier writes stay, and inside a transaction the failed statement is undone and the
//! transaction stays usable. Every way a write grows the store is refused the same way: a node
//! with each value kind, an edge, a label, a property, a merge, the bulk loads, an index build,
//! SPARQL and the RDF loads. Reads keep working, a thaw that cannot get a grant reads the cold
//! form, WAL replay never refuses, and a database over its limit opens with a warning and reads
//! (writes that need memory refused until the limit is raised).
//!
//! The per-query limit and the spill pairs are in `memory_query_limit`.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main memory_enforcement::
//! ```

#![cfg(all(feature = "gql", feature = "grafeo-file"))]
#![allow(
    deprecated,
    reason = "the oracle walks the raw store, which no session shows"
)]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_common::utils::error::Error;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;

/// `result` is the S001 refusal naming `bound` and `setting`.
fn assert_refused<T: std::fmt::Debug>(result: Result<T, Error>, bound: &str, setting: &str) {
    let error = result.expect_err("refused at the limit");
    let text = error.to_string();
    assert_eq!(error.error_code().as_str(), "GRAFEO-S001", "{text}");
    assert!(text.contains(bound), "names the {bound}: {text}");
    assert!(text.contains(setting), "names {setting}: {text}");
    assert!(
        text.chars().any(|c| c.is_ascii_digit()),
        "carries the numbers: {text}"
    );
}

/// The memory limit's refusal.
fn assert_over_the_limit<T: std::fmt::Debug>(result: Result<T, Error>) {
    assert_refused(result, "memory limit", "with_memory_limit");
}

/// The account against the walk: the report's store, indexes, versions and strings equal what a
/// walk over the store finds.
fn assert_account_matches_walk(db: &GrafeoDB, what: &str) {
    let (store, indexes, mvcc, strings) = db.store().unwrap().memory_breakdown();
    let usage = db.memory_usage();
    assert_eq!(usage.store.total_bytes, store.total_bytes, "{what}: store");
    assert_eq!(
        usage.indexes.total_bytes, indexes.total_bytes,
        "{what}: indexes"
    );
    assert_eq!(usage.mvcc.total_bytes, mvcc.total_bytes, "{what}: versions");
    assert_eq!(
        usage.string_pool.total_bytes, strings.total_bytes,
        "{what}: strings"
    );
}

fn count(db: &GrafeoDB, label: &str) -> usize {
    db.execute(&format!("MATCH (n:{label}) RETURN n"))
        .unwrap()
        .rows()
        .len()
}

/// About `kib` KiB of strings in one statement, on nodes labelled `label`.
fn insert_kib(db: &GrafeoDB, label: &str, kib: usize) -> Result<(), Error> {
    db.execute(&format!(
        "UNWIND range(1, {kib}) AS i INSERT (:{label} {{text: '{}'}})",
        "x".repeat(1_000)
    ))
    .map(|_| ())
}

/// Writes 1 MiB statements until one is refused; returns the refusal.
fn fill(db: &GrafeoDB) -> Error {
    for _ in 0..10_000 {
        if let Err(error) = insert_kib(db, "Fill", 1_024) {
            return error;
        }
    }
    panic!("the limit never bound");
}

fn limited(limit: usize) -> GrafeoDB {
    GrafeoDB::with_config(Config::in_memory().with_memory_limit(limit)).unwrap()
}

/// What the store holds, as rows that a refused statement must leave as they were.
fn everything(db: &GrafeoDB) -> Vec<String> {
    let mut rows: Vec<String> = db
        .execute("MATCH (n) RETURN labels(n) AS l, properties(n) AS p")
        .unwrap()
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect();
    rows.sort();
    rows
}

// ---------------------------------------------------------------------------
// #398: the limit holds and a refused write changes nothing
// ---------------------------------------------------------------------------

/// #398's own case: a 128 MiB store growing past its cap.
#[test]
fn a_store_growing_past_its_cap_is_refused_and_keeps_what_it_had() {
    let db = limited(128 * MIB);
    let refusal = fill(&db);
    assert_over_the_limit::<()>(Err(refusal));
    let usage = db.memory_usage();
    assert_eq!(usage.limit_bytes, 128 * MIB);
    let held = count(&db, "Fill");
    assert!(held > 0, "earlier writes stay");
    assert_over_the_limit(insert_kib(&db, "Fill", 1_024));
    assert_eq!(
        count(&db, "Fill"),
        held,
        "the refused statement changed nothing"
    );
    assert!(
        db.memory_usage().store.total_bytes <= 128 * MIB,
        "the account stays under the limit"
    );
    assert_account_matches_walk(&db, "after the refusal");
}

#[test]
fn inside_a_transaction_the_refused_statement_is_undone_and_the_transaction_goes_on() {
    let db = limited(64 * MIB);
    fill(&db);
    let held = count(&db, "Fill");
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("INSERT (:Small {name: 'Alix'})").unwrap();
    assert_over_the_limit(session.execute(&format!(
        "UNWIND range(1, 4096) AS i INSERT (:Big {{text: '{}'}})",
        "x".repeat(1_000)
    )));
    session.execute("INSERT (:Small {name: 'Gus'})").unwrap();
    session.commit().unwrap();
    assert_eq!(
        count(&db, "Small"),
        2,
        "the transaction went on and committed"
    );
    assert_eq!(count(&db, "Big"), 0, "the refused statement left nothing");
    assert_eq!(count(&db, "Fill"), held);
    assert_account_matches_walk(&db, "after the commit");

    // A rollback after a refused statement leaves nothing of the transaction.
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute("INSERT (:Small {name: 'Vincent'})")
        .unwrap();
    assert_over_the_limit(session.execute(&format!(
        "UNWIND range(1, 4096) AS i INSERT (:Big {{text: '{}'}})",
        "x".repeat(1_000)
    )));
    session.rollback().unwrap();
    assert_eq!(count(&db, "Small"), 2, "the rolled-back insert is gone");
    assert_eq!(
        count(&db, "Big"),
        0,
        "a rollback after a refusal leaves nothing"
    );
}

/// Every kind of write that grows a part is refused the same way, the store and the account as
/// before the statement. A delete that needs no memory still works.
#[test]
fn every_write_that_grows_a_part_is_refused_at_the_limit() {
    let big = "x".repeat(1_000);
    let forms: Vec<(&str, String)> = vec![
        (
            "a node with strings",
            format!("UNWIND range(1, 4096) AS i INSERT (:N {{t: '{big}'}})"),
        ),
        (
            "a node with lists",
            "UNWIND range(1, 4096) AS i INSERT (:N {t: range(1, 200)})".to_string(),
        ),
        (
            "a node with maps",
            format!("UNWIND range(1, 4096) AS i INSERT (:N {{t: {{a: '{big}', b: i}}}})"),
        ),
        (
            "a node with vectors",
            "UNWIND range(1, 8192) AS i INSERT (:N {t: vector([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, \
             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, \
             24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0])})"
                .to_string(),
        ),
        (
            "a node with nulls and NaN",
            "UNWIND range(1, 40000) AS i INSERT (:N {a: null, b: $nan, c: i})".to_string(),
        ),
        (
            "an edge insert",
            format!("UNWIND range(1, 4096) AS i INSERT (:E1)-[:R {{t: '{big}'}}]->(:E2)"),
        ),
        (
            "a property set",
            format!("MATCH (f:Fill) SET f.more = '{}'", "y".repeat(2_000)),
        ),
        (
            "a list property",
            "MATCH (f:Fill) SET f.items = range(1, 400)".to_string(),
        ),
        (
            "a label add",
            "MATCH (f:Fill) SET f:Tagged:Twice".to_string(),
        ),
        (
            "a merge",
            format!("UNWIND range(1, 4096) AS i MERGE (:M {{k: i, t: '{big}'}})"),
        ),
    ];
    for (what, statement) in forms {
        let db = limited(64 * MIB);
        fill(&db);
        let rows = everything(&db);
        let before = db.memory_usage().store.total_bytes;
        // The NaN comes as a parameter: GQL refuses `0.0 / 0.0` as a division by zero.
        let params = std::iter::once(("nan".to_string(), Value::Float64(f64::NAN))).collect();
        let result = db.execute_with_params(&statement, params);
        assert_over_the_limit(result);
        // The ops applied before the refusal may have grown capacity the undo keeps: the
        // account is no lower than before, and equals the walk.
        let after = db.memory_usage().store.total_bytes;
        assert!(
            after >= before,
            "{what}: the account {after} fell below {before}"
        );
        assert_eq!(everything(&db), rows, "{what}: the store is as before");
        assert_account_matches_walk(&db, what);
    }

    let db = limited(64 * MIB);
    fill(&db);
    db.execute("MATCH (f:Fill) WITH f LIMIT 10 DETACH DELETE f")
        .expect("a delete needs no memory");
}

/// The direct API writes are refused the same way.
#[test]
fn a_direct_write_is_refused_at_the_limit() {
    let db = limited(64 * MIB);
    fill(&db);
    let rows = everything(&db);
    let mut refused = None;
    for i in 0..100_000 {
        if let Err(error) = db.create_node_with_props(
            &["Direct"],
            [(
                grafeo_common::types::PropertyKey::new("t"),
                Value::from(format!("{i}-{}", "z".repeat(1_000))),
            )],
        ) {
            refused = Some(error);
            break;
        }
    }
    assert_over_the_limit::<()>(Err(refused.expect("the limit bound")));
    assert!(everything(&db).len() >= rows.len());
    assert_account_matches_walk(&db, "after a direct write");
}

/// The bulk loads are one write, refused whole: a CSV load through the API and through `CALL`.
#[test]
fn a_bulk_load_is_refused_whole_at_the_limit() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("people.csv");
    use std::fmt::Write;

    let mut csv = String::from("name,bio\n");
    for i in 0..20_000 {
        writeln!(csv, "person {i},{}", "b".repeat(1_000)).unwrap();
    }
    std::fs::write(&file, csv).unwrap();
    let path = file.to_string_lossy().replace('\\', "/");

    let db = limited(64 * MIB);
    fill(&db);
    assert_over_the_limit(db.load_nodes_csv(
        &file,
        &grafeo_engine::loader::NodeLoad {
            labels: vec!["Loaded".to_string()],
            ..grafeo_engine::loader::NodeLoad::default()
        },
    ));
    assert_eq!(count(&db, "Loaded"), 0, "the API load left nothing");
    assert_over_the_limit(db.execute(&format!(
        "CALL grafeo.load_nodes_csv({{path: '{path}', labels: ['Loaded']}}) YIELD rows RETURN rows"
    )));
    assert_eq!(count(&db, "Loaded"), 0, "the CALL load left nothing");
    assert_account_matches_walk(&db, "after the refused loads");
}

/// SPARQL `INSERT DATA`, `batch_insert_rdf` and `load_rdf` are refused at the statement, before
/// the commit; in a transaction the earlier statements still commit.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
#[test]
fn rdf_writes_are_refused_at_the_limit() {
    use grafeo_core::graph::rdf::{Term, Triple};

    let lines = |from: usize, to: usize| -> String {
        (from..to)
            .map(|i| {
                format!(
                    "<http://ex.org/s{i}> <http://ex.org/p> \"{}\" .",
                    "x".repeat(200)
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    let db = limited(64 * MIB);
    fill(&db);
    assert_over_the_limit(db.execute_sparql(&format!("INSERT DATA {{ {} }}", lines(0, 20_000))));
    assert_over_the_limit(db.batch_insert_rdf((0..20_000).map(|i| {
        Triple::new(
            Term::iri(format!("http://ex.org/b{i}")),
            Term::iri("http://ex.org/p"),
            Term::literal("y".repeat(200)),
        )
    })));
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("triples.nt");
    std::fs::write(&file, lines(20_000, 40_000)).unwrap();
    assert_over_the_limit(db.load_rdf(&file, None));
    let triples = |db: &GrafeoDB| db.memory_usage().rdf.triple_count;
    assert_eq!(triples(&db), 0, "the refused writes left no triple");

    let mut session = db.session();
    session.begin_transaction().unwrap();
    session
        .execute_sparql("INSERT DATA { <http://ex.org/alix> <http://ex.org/lives> \"Amsterdam\" }")
        .unwrap();
    assert_over_the_limit(
        session.execute_sparql(&format!("INSERT DATA {{ {} }}", lines(40_000, 60_000))),
    );
    session.commit().unwrap();
    assert_eq!(triples(&db), 1, "the earlier statement committed");
}

/// An index built over existing data past the limit is not created: property, text and vector.
#[test]
fn an_index_build_over_existing_data_is_refused_at_the_limit() {
    let db = limited(64 * MIB);
    db.execute(&format!(
        "UNWIND range(1, 20000) AS i INSERT (:Doc {{k: i, body: 'the canals of Amsterdam {}'}})",
        "w ".repeat(100)
    ))
    .unwrap();
    fill(&db);
    assert_over_the_limit(db.execute("CREATE INDEX by_k FOR (d:Doc) ON (d.k)"));
    #[cfg(feature = "text-index")]
    assert_over_the_limit(db.create_text_index("Doc", "body"));
    #[cfg(feature = "vector-index")]
    {
        db.execute("MATCH (d:Doc) WHERE d.k <= 5000 SET d.emb = vector([1.0, 0.5, 0.25, 0.125])")
            .ok();
        assert_over_the_limit(db.create_vector_index("Doc", "emb", None, None, None, None, None));
    }
    assert!(
        db.execute("SHOW INDEXES").unwrap().rows().is_empty(),
        "no index was created"
    );
}

/// Writers racing for the last step: some are refused, and the account equals the walk.
#[test]
fn writers_racing_for_the_last_step_keep_the_account() {
    let db = std::sync::Arc::new(limited(64 * MIB));
    fill(&db);
    let writers: Vec<_> = (0..4)
        .map(|w| {
            let db = std::sync::Arc::clone(&db);
            std::thread::spawn(move || {
                (0..20)
                    .map(|_| insert_kib(&db, &format!("W{w}"), 256).is_ok())
                    .filter(|written| *written)
                    .count()
            })
        })
        .collect();
    for writer in writers {
        writer.join().unwrap();
    }
    assert_over_the_limit(insert_kib(&db, "After", 1_024));
    assert_account_matches_walk(&db, "after the race");
}

/// Reads keep working at the limit: a scan, a lookup and a small sort.
#[test]
fn reads_keep_working_at_the_limit() {
    let db = limited(64 * MIB);
    fill(&db);
    let held = count(&db, "Fill");
    assert!(held > 0);
    let sorted = db
        .execute("MATCH (f:Fill) RETURN f.text AS t ORDER BY t LIMIT 3")
        .unwrap();
    assert_eq!(sorted.rows().len(), 3);
}

// ---------------------------------------------------------------------------
// Opens over the limit, replay and thaws
// ---------------------------------------------------------------------------

/// The variable naming the file a [`crashed_with_data_child`] writes.
const CHILD_PATH: &str = "GRAFEO_MEMORY_ENFORCEMENT_CHILD_PATH";

/// Writes 16 MiB of rows: long strings (Values chunks), a status from three (a Dict chunk) and
/// an integer.
fn write_data(db: &GrafeoDB) {
    for round in 0..16 {
        db.execute(&format!(
            "UNWIND range(1, 1024) AS i INSERT (:Fill {{text: '{}' || CAST(i AS STRING), \
             status: ['active', 'planned', 'retired'][i % 3], n: {round} * 1024 + i}})",
            "x".repeat(1_000)
        ))
        .unwrap();
    }
}

/// The child of a crashed file: writes the data and exits with the database open, so the WAL
/// holds the writes and nothing checkpoints them. A no-op when run directly.
#[test]
fn crashed_with_data_child() {
    let Some(path) = std::env::var_os(CHILD_PATH) else {
        return;
    };
    let db = GrafeoDB::open(Path::new(&path)).unwrap();
    write_data(&db);
    // A crash: no close, no destructors.
    std::process::exit(0);
}

/// A file with the data, closed cleanly or left as a crash leaves it (written by a child
/// process that exits with it open).
fn file_with_data(dir: &Path, close: bool) -> PathBuf {
    let path = dir.join("amsterdam.grafeo");
    if close {
        let db = GrafeoDB::open(&path).unwrap();
        write_data(&db);
        db.close().unwrap();
    } else {
        let status = grafeo_common::testing::child_process::run(
            grafeo_common::child_test!("crashed_with_data_child").env(CHILD_PATH, &path),
        )
        .unwrap();
        assert!(status.success(), "the child wrote the data");
        assert!(path.exists(), "the child created the file");
    }
    path
}

/// The open's warning that the database holds more than its limit.
fn over_the_limit_warning(db: &GrafeoDB) -> Option<(u64, u64)> {
    db.open_warnings().iter().find_map(|warning| match warning {
        grafeo_engine::OpenWarning::OverMemoryLimit { used, limit } => {
            assert_eq!(warning.code(), "MEMORY_LIMIT_PASSED");
            Some((*used, *limit))
        }
        _ => None,
    })
}

/// A database over its limit opens with the warning, reads, refuses writes that need memory,
/// and a higher limit lets them through again.
#[test]
fn a_database_over_its_limit_opens_and_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_with_data(dir.path(), true);
    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(4 * MIB)).unwrap();
    let (used, limit) = over_the_limit_warning(&db).expect("the open warns");
    assert!(
        used >= limit && limit == 4 * MIB as u64,
        "{used} of {limit}"
    );
    assert_eq!(count(&db, "Fill"), 16 * 1_024, "reads work over the limit");
    assert_over_the_limit(insert_kib(&db, "More", 1_024));
    db.close().unwrap();
    drop(db);
    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(512 * MIB)).unwrap();
    assert_eq!(
        over_the_limit_warning(&db),
        None,
        "under the limit, no warning"
    );
    insert_kib(&db, "More", 1_024).unwrap();
    db.close().unwrap();
}

/// WAL replay restores committed data whatever the limit: the open succeeds with the warning,
/// reads work, and the first write that needs memory is refused.
#[test]
fn replay_never_refuses() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_with_data(dir.path(), false);
    let db = GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(4 * MIB)).unwrap();
    assert!(over_the_limit_warning(&db).is_some(), "the open warns");
    assert_eq!(
        count(&db, "Fill"),
        16 * 1_024,
        "every committed row replayed"
    );
    assert_over_the_limit(insert_kib(&db, "More", 1_024));
    db.close().unwrap();
}

#[test]
fn a_read_only_open_over_the_limit_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_with_data(dir.path(), true);
    let db = GrafeoDB::with_config(Config::read_only(&path).with_memory_limit(4 * MIB)).unwrap();
    assert_eq!(count(&db, "Fill"), 16 * 1_024);
}

/// A read that would thaw a column without a grant for it reads the cold form: the same rows,
/// for a Dict chunk (the status) and a Values chunk (the text), by scan, lookup and point read.
#[test]
fn a_thaw_without_a_grant_reads_the_cold_form() {
    let dir = tempfile::tempdir().unwrap();
    let path = file_with_data(dir.path(), true);
    let open = |limit: usize| {
        GrafeoDB::with_config(Config::persistent(&path).with_memory_limit(limit)).unwrap()
    };
    let queries = [
        "MATCH (f:Fill) WHERE f.status = 'planned' RETURN count(f) AS c",
        "MATCH (f:Fill) WHERE f.text STARTS WITH 'x' RETURN count(f) AS c",
        "MATCH (f:Fill) WHERE f.n = 4097 RETURN f.status AS s, size(f.text) AS t",
    ];
    let answers = |db: &GrafeoDB| -> Vec<Vec<Vec<Value>>> {
        queries
            .iter()
            .map(|query| db.execute(query).unwrap().rows().to_vec())
            .collect()
    };
    let roomy = open(512 * MIB);
    let want = answers(&roomy);
    roomy.close().unwrap();
    drop(roomy);
    let tight = open(4 * MIB);
    assert_eq!(answers(&tight), want);
    tight.close().unwrap();
}
