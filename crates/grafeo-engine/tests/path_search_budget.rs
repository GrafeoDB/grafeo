//! A variable-length pattern whose search would hold more paths than a path
//! search may fails with an error that says what to do, instead of growing
//! until the process runs out of memory and aborts.
//!
//! Reported downstream (Deriva, 2026-10-09): `MATCH (repo:Repository)-[*]->(f:File)
//! WHERE repo.repoName = 'deriva' RETURN count(f)` on a code graph of about
//! 1,500 nodes (a CONTAINS tree, with IMPORTS and CALLS edges in cycles)
//! printed "memory allocation of 47244640256 bytes failed" and took the
//! server down: the expand gathered every walk of up to 101 edges from the
//! repository before it returned one. [`code_graph`] has that shape.
//!
//! The global allocator of this test binary counts the bytes held and
//! refuses a request that would hold more than [`CAP`]: a search that grew
//! without bound ends this process there with "memory allocation of ...
//! failed", never the machine. The tests measure what each query held at
//! most, so a search that stays within its budget but holds it twice over
//! fails too.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test path_search_budget
//! ```

#![cfg(all(feature = "lpg", feature = "gql", feature = "cypher"))]
#![expect(
    unsafe_code,
    reason = "GlobalAlloc is an unsafe trait; this forwards to System below a cap"
)]

use std::sync::{Mutex, MutexGuard, PoisonError};

use grafeo_common::types::{NodeId, Value};
use grafeo_common::utils::error::{Error, Result};
use grafeo_engine::database::QueryResult;
use grafeo_engine::{Config, GrafeoDB, Session};

/// The most this test binary holds: a request beyond it fails.
const CAP: usize = 3 * 1024 * 1024 * 1024;

/// The most a query that fails on its path budget may hold beyond what was
/// held before it: the search's budget (256 MiB), the old frontier while it
/// grows into the last of it, and the rest of the plan.
const QUERY_PEAK: usize = 768 * 1024 * 1024;

// --- Counting -----------------------------------------------------------------------

/// Counts the bytes held through the global allocator and the most held
/// since the last [`reset`](capped::reset), and refuses what would go over
/// [`CAP`].
mod capped {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CURRENT: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);

    /// Forwards every call to [`System`] while the bytes held stay within
    /// [`super::CAP`].
    pub struct Capped;

    /// Takes `by` more bytes, unless that would hold more than the cap.
    fn take(by: usize) -> bool {
        let taken = CURRENT.try_update(Ordering::Relaxed, Ordering::Relaxed, |now| {
            now.checked_add(by).filter(|&next| next <= super::CAP)
        });
        match taken {
            Ok(before) => {
                PEAK.fetch_max(before + by, Ordering::Relaxed);
                true
            }
            Err(_) => false,
        }
    }

    fn give_back(by: usize) {
        CURRENT.fetch_sub(by, Ordering::Relaxed);
    }

    // SAFETY: every method passes its arguments unchanged to `System`, which
    // meets the trait's contract, and returns what `System` returned, or
    // null (a failed allocation, which the contract allows) without calling
    // it. The counting touches atomics only and allocates nothing.
    unsafe impl GlobalAlloc for Capped {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if !take(layout.size()) {
                return std::ptr::null_mut();
            }
            // SAFETY: the caller meets `alloc`'s contract, which is `System`'s.
            let block = unsafe { System.alloc(layout) };
            if block.is_null() {
                give_back(layout.size());
            }
            block
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            if !take(layout.size()) {
                return std::ptr::null_mut();
            }
            // SAFETY: as in `alloc`.
            let block = unsafe { System.alloc_zeroed(layout) };
            if block.is_null() {
                give_back(layout.size());
            }
            block
        }

        unsafe fn dealloc(&self, block: *mut u8, layout: Layout) {
            // SAFETY: `block` was handed out by this allocator, so by
            // `System`, with `layout`.
            unsafe { System.dealloc(block, layout) };
            give_back(layout.size());
        }

        unsafe fn realloc(&self, block: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let growth = new_size.saturating_sub(layout.size());
            if !take(growth) {
                return std::ptr::null_mut();
            }
            // SAFETY: as in `dealloc`, and the caller meets `realloc`'s
            // contract for `new_size`.
            let moved = unsafe { System.realloc(block, layout, new_size) };
            if moved.is_null() {
                give_back(growth);
            } else if new_size < layout.size() {
                give_back(layout.size() - new_size);
            }
            moved
        }
    }

    /// The bytes held now.
    pub fn current() -> usize {
        CURRENT.load(Ordering::Relaxed)
    }

    /// The most bytes held since the last [`reset`].
    pub fn peak() -> usize {
        PEAK.load(Ordering::Relaxed)
    }

    /// Starts a measurement from what is held now.
    pub fn reset() {
        PEAK.store(current(), Ordering::Relaxed);
    }
}

#[global_allocator]
static CAPPED: capped::Capped = capped::Capped;

/// The counters are global: the tests that measure take turns.
static MEASURING: Mutex<()> = Mutex::new(());

fn measuring() -> MutexGuard<'static, ()> {
    MEASURING.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What `call` returned and the most it held beyond what was held before.
fn measure<T>(call: impl FnOnce() -> T) -> (T, usize) {
    let before = capped::current();
    capped::reset();
    let value = call();
    (value, capped::peak().saturating_sub(before))
}

// --- The graph ----------------------------------------------------------------------

/// A database with [`code_graph`] in it, and what the tests count on it.
struct CodeGraph {
    db: GrafeoDB,
    repo: NodeId,
    directories: Vec<NodeId>,
    files: Vec<NodeId>,
    /// Every edge, from and to, for the walks the tests count themselves.
    edges: Vec<(NodeId, NodeId)>,
}

/// A code graph of the reported shape: a repository (`repoName: 'deriva'`),
/// a CONTAINS tree of 84 directories three levels deep, 500 files in them and
/// 900 methods in the files, and edges in cycles: each file IMPORTS three
/// files, each method CALLS two methods and USES a file. 1,485 nodes, as the
/// reported graph, picked from a fixed pseudo-random sequence.
fn code_graph() -> CodeGraph {
    code_graph_in(GrafeoDB::new_in_memory())
}

/// [`code_graph`] in `db`, written in one transaction: a build with
/// `tiered-storage` gives every commit's epoch an arena of its own, so one
/// commit per node and edge would hold gigabytes before a query runs.
fn code_graph_in(db: GrafeoDB) -> CodeGraph {
    let mut session = db.session();
    session.begin_transaction().unwrap();
    let mut state: u64 = 0x0003_0019_0088;
    let mut pick = |bound: usize| {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        usize::try_from(state >> 33).unwrap() % bound
    };
    let mut edges = Vec::new();
    let mut edge = |session: &Session, from: NodeId, to: NodeId, edge_type: &str| {
        session.create_edge(from, to, edge_type).unwrap();
        edges.push((from, to));
    };

    let repo = session
        .create_node_with_props(&["Repository"], [("repoName", Value::from("deriva"))])
        .unwrap();
    let mut directories = Vec::new();
    let mut level = vec![repo];
    for _ in 0..3 {
        let mut next = Vec::new();
        for &parent in &level {
            for _ in 0..4 {
                let directory = session.create_node(&["Directory"]).unwrap();
                edge(&session, parent, directory, "CONTAINS");
                directories.push(directory);
                next.push(directory);
            }
        }
        level = next;
    }
    let mut files = Vec::new();
    for i in 0..500 {
        let file = session
            .create_node_with_props(&["File"], [("filePath", Value::from(format!("f{i}.py")))])
            .unwrap();
        edge(
            &session,
            directories[pick(directories.len())],
            file,
            "CONTAINS",
        );
        files.push(file);
    }
    let mut methods = Vec::new();
    for _ in 0..900 {
        let method = session.create_node(&["Method"]).unwrap();
        edge(&session, files[pick(files.len())], method, "CONTAINS");
        methods.push(method);
    }
    for &file in &files {
        for _ in 0..3 {
            edge(&session, file, files[pick(files.len())], "IMPORTS");
        }
    }
    for &method in &methods {
        for _ in 0..2 {
            edge(&session, method, methods[pick(methods.len())], "CALLS");
        }
        edge(&session, method, files[pick(files.len())], "USES");
    }
    session.commit().unwrap();
    drop(session);
    CodeGraph {
        db,
        repo,
        directories,
        files,
        edges,
    }
}

/// The number of walks of `min..=max` edges from `source` to a node of
/// `targets`, counted over `edges` layer by layer.
fn walks(
    edges: &[(NodeId, NodeId)],
    source: NodeId,
    targets: &[NodeId],
    min: usize,
    max: usize,
) -> i64 {
    use std::collections::HashMap;
    let mut ways: HashMap<NodeId, i64> = HashMap::from([(source, 1)]);
    let mut total = 0;
    for length in 1..=max {
        let mut next: HashMap<NodeId, i64> = HashMap::new();
        for &(from, to) in edges {
            if let Some(&count) = ways.get(&from) {
                *next.entry(to).or_insert(0) += count;
            }
        }
        ways = next;
        if length >= min {
            total += targets.iter().filter_map(|t| ways.get(t)).sum::<i64>();
        }
    }
    total
}

fn run(db: &GrafeoDB, language: &str, query: &str) -> Result<QueryResult> {
    match language {
        "cypher" => db.execute_cypher(query),
        _ => db.execute(query),
    }
}

/// The one integer `query` returns.
fn count(db: &GrafeoDB, language: &str, query: &str) -> i64 {
    let result = run(db, language, query).unwrap_or_else(|e| panic!("{language}: {query}: {e}"));
    let rows = result.rows();
    assert_eq!(rows.len(), 1, "{language}: {query}");
    rows[0][0]
        .as_int64()
        .unwrap_or_else(|| panic!("{language}: {query}: {:?}", rows[0][0]))
}

/// Asserts that `query` fails with the error of a path search over its
/// budget, which names what to do instead, and holds at most [`QUERY_PEAK`]
/// on the way.
fn assert_over_budget(db: &GrafeoDB, language: &str, query: &str) {
    let (result, peak) = measure(|| run(db, language, query));
    let error = match result {
        Ok(result) => panic!(
            "{language}: {query}: expected the path budget error, got {:?}",
            result.rows()
        ),
        Err(error) => error,
    };
    let message = error.to_string();
    assert!(
        matches!(&error, Error::Query(_)),
        "{language}: {query}: a query error, got {error:?}"
    );
    for advice in ["upper bound", "DISTINCT", "shortest"] {
        assert!(
            message.contains(advice),
            "{language}: {query}: the message names `{advice}`: {message}"
        );
    }
    assert!(
        peak <= QUERY_PEAK,
        "{language}: {query}: held {} MiB at most, more than {} MiB",
        peak >> 20,
        QUERY_PEAK >> 20
    );
}

#[test]
fn an_unbounded_pattern_over_cycles_fails_with_an_error_not_an_abort() {
    let _turn = measuring();
    let CodeGraph { db, .. } = code_graph();
    let filter = "WHERE repo.repoName = 'deriva'";
    for (language, query) in [
        // The reported query: walks of up to 101 edges, one row each
        (
            "cypher",
            format!("MATCH (repo:Repository)-[*]->(f:File) {filter} RETURN count(f)"),
        ),
        (
            "gql",
            format!("MATCH (repo:Repository)-[*]->(f:File) {filter} RETURN count(f)"),
        ),
        // A trail repeats no edge, and there are still too many
        (
            "gql",
            format!("MATCH TRAIL (repo:Repository)-[*]->(f:File) {filter} RETURN count(f)"),
        ),
        // A named path follows each path whole
        (
            "cypher",
            format!("MATCH p = (repo:Repository)-[*]->(f:File) {filter} RETURN count(p)"),
        ),
        (
            "gql",
            format!("MATCH p = (repo:Repository)-[*]->(f:File) {filter} RETURN count(p)"),
        ),
    ] {
        assert_over_budget(&db, language, &query);
    }
    // The database still answers
    assert_eq!(
        count(&db, "gql", "MATCH (f:File) RETURN count(f)"),
        500,
        "after the failed searches"
    );
}

#[test]
fn distinct_targets_and_exists_answer_on_the_same_graph() {
    let _turn = measuring();
    let CodeGraph { db, .. } = code_graph();
    let filter = "WHERE repo.repoName = 'deriva'";
    for language in ["gql", "cypher"] {
        // Every file is in the CONTAINS tree under the repository
        for query in [
            format!("MATCH (repo:Repository)-[*]->(f:File) {filter} RETURN count(DISTINCT f)"),
            format!(
                "MATCH (repo:Repository) {filter} MATCH (f:File) \
                 WHERE EXISTS {{ MATCH (repo)-[*]->(f) }} RETURN count(f)"
            ),
        ] {
            let (found, peak) = measure(|| count(&db, language, &query));
            assert_eq!(found, 500, "{language}: {query}");
            assert!(
                peak < 64 * 1024 * 1024,
                "{language}: {query}: a reachability search holds a few nodes, held {} MiB",
                peak >> 20
            );
        }
    }
    let query = format!(
        "MATCH (repo:Repository)-[*]->(f:File) {filter} RETURN DISTINCT f.filePath AS path"
    );
    for language in ["gql", "cypher"] {
        let rows = run(&db, language, &query).unwrap().rows().len();
        assert_eq!(rows, 500, "{language}: {query}");
    }
}

#[test]
fn a_bounded_pattern_over_the_same_graph_counts_every_walk() {
    let _turn = measuring();
    let graph = code_graph();
    let db = &graph.db;
    // The walks from the repository, which the search emits a chunk at a
    // time: up to eight edges are many chunks
    let mut most = 0;
    for (min, max) in [(1, 8), (0, 4), (5, 7), (3, 3)] {
        let expected = walks(&graph.edges, graph.repo, &graph.files, min, max);
        most = most.max(expected);
        let query = format!(
            "MATCH (repo:Repository)-[*{min}..{max}]->(f:File) \
             WHERE repo.repoName = 'deriva' RETURN count(f)"
        );
        assert_eq!(count(db, "gql", &query), expected, "{query}");
    }
    assert!(most > 8 * 2048, "the walks span many chunks: {most}");
    // From every directory, one search after the other
    assert_eq!(graph.directories.len(), 84);
    let expected: i64 = graph
        .directories
        .iter()
        .map(|&directory| walks(&graph.edges, directory, &graph.files, 1, 5))
        .sum();
    let query = "MATCH (d:Directory)-[*1..5]->(f:File) RETURN count(f)";
    assert_eq!(count(db, "gql", query), expected, "{query}");
}

/// A database with a memory limit gives each path search a quarter of it:
/// with 64 MiB the reported query fails at a 16 MiB budget, named in the
/// error, and holds far less on the way than the default 256 MiB.
#[test]
fn a_memory_limit_sets_the_path_search_budget() {
    let _turn = measuring();
    let config = Config::in_memory().with_memory_limit(64 * 1024 * 1024);
    let CodeGraph { db, .. } = code_graph_in(GrafeoDB::with_config(config).unwrap());
    let query =
        "MATCH (repo:Repository)-[*]->(f:File) WHERE repo.repoName = 'deriva' RETURN count(f)";
    for language in ["gql", "cypher"] {
        let (result, peak) = measure(|| run(&db, language, query));
        let message = match result {
            Ok(result) => panic!(
                "{language}: expected the path budget error, got {:?}",
                result.rows()
            ),
            Err(error) => error.to_string(),
        };
        assert!(
            message.contains("16 MiB"),
            "{language}: the budget is a quarter of 64 MiB: {message}"
        );
        assert!(
            peak <= 64 * 1024 * 1024,
            "{language}: held {} MiB at most, more than the 64 MiB memory limit",
            peak >> 20
        );
    }
}
