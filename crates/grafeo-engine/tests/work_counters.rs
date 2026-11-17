//! Work counters in `PROFILE` (#794): in a build with `--cfg grafeo_work_counters` each
//! operator's line names the work its own code counted; without it the text is as before.
//!
//! ```bash
//! RUSTFLAGS="--cfg grafeo_work_counters" CARGO_TARGET_DIR=C:/cargo-target/grafeo-work \
//!     cargo test -p grafeo-engine --all-features --test main work_counters::
//! ```

#![cfg(feature = "gql")]

use grafeo_common::types::Value;
use grafeo_engine::GrafeoDB;

/// Five `:N` nodes with `x` from 5 down to 1.
fn five() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("FOR i IN range(1, 5) INSERT (:N {x: 6 - i})")
        .unwrap();
    db
}

fn profile(db: &GrafeoDB, query: &str) -> String {
    match &db.execute(&format!("PROFILE {query}")).unwrap().rows()[0][0] {
        Value::String(text) => text.to_string(),
        other => panic!("PROFILE returns text, not {other:?}"),
    }
}

/// The sort builds one chunk of the five rows it returns, and its line says so; the scan below
/// it counts its own chunks on its own line, not on the sort's.
#[cfg(grafeo_work_counters)]
#[test]
fn profile_names_the_work_each_operator_did() {
    let text = profile(&five(), "MATCH (n:N) RETURN n.x ORDER BY n.x");
    let sort = text
        .lines()
        .find(|line| line.trim_start().starts_with("Sort"))
        .unwrap_or_else(|| panic!("a Sort line:\n{text}"));
    assert!(sort.contains("work:"), "{text}");
    assert!(sort.contains("chunks_built=1"), "{text}");
    assert!(sort.contains("rows_built=5"), "{text}");
}

/// The work of each line, without its times: what a second run must repeat.
#[cfg(grafeo_work_counters)]
fn work_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter(|line| line.contains("work: "))
        .map(|line| {
            let name = if line.starts_with("Total time") {
                "query"
            } else {
                line.split("  rows=").next().unwrap_or(line)
            };
            let work = line.split("work: ").nth(1).unwrap_or_default();
            format!("{name} | {work}")
        })
        .collect()
}

/// The counts are exact: the same statement on the same data counts the same work again, line by
/// line.
#[cfg(grafeo_work_counters)]
#[test]
fn a_second_run_counts_the_same_work() {
    let db = five();
    let query = "MATCH (n:N) WHERE n.x > 1 RETURN n.x ORDER BY n.x LIMIT 3";
    let first = work_lines(&profile(&db, query));
    assert!(!first.is_empty(), "something is counted");
    assert_eq!(work_lines(&profile(&db, query)), first);
}

/// Exact in the second state too: with an older transaction open across a committed update and
/// delete, so the reads go through stamped vectors, a second run counts the same work again.
#[cfg(grafeo_work_counters)]
#[test]
fn a_second_run_counts_the_same_work_with_an_older_snapshot_open() {
    let db = five();
    let mut older = db.session();
    older.begin_transaction().unwrap();
    older.execute("MATCH (n:N) RETURN n.x").unwrap();
    db.execute("MATCH (n:N {x: 2}) SET n.x = 20").unwrap();
    db.execute("MATCH (n:N {x: 4}) DETACH DELETE n").unwrap();
    let query = "MATCH (n:N) WHERE n.x > 1 RETURN n.x ORDER BY n.x LIMIT 3";
    let first = work_lines(&profile(&db, query));
    assert!(
        first.iter().any(|line| line.contains("stamped_reads=")),
        "the reads go through stamped vectors: {first:?}"
    );
    assert_eq!(work_lines(&profile(&db, query)), first);
    older.rollback().unwrap();
}

/// Without the cfg nothing is counted and `PROFILE` prints what it always did.
#[cfg(not(grafeo_work_counters))]
#[test]
fn without_the_cfg_profile_shows_no_work() {
    let text = profile(&five(), "MATCH (n:N) RETURN n.x ORDER BY n.x");
    assert!(!text.contains("work:"), "{text}");
    assert!(!text.contains("query work"), "{text}");
}

/// Five `:P` nodes (`x` 1 to 5, `g` alternating 1 and 0, `name` a string) and the chain
/// 1 -> 2 -> 3 of `:K` edges.
#[cfg(grafeo_work_counters)]
fn people() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    db.execute("FOR i IN range(1, 5) INSERT (:P {x: i, g: i % 2, name: 'p' + toString(i)})")
        .unwrap();
    db.execute("MATCH (a:P {x: 1}), (b:P {x: 2}) INSERT (a)-[:K]->(b)")
        .unwrap();
    db.execute("MATCH (a:P {x: 2}), (b:P {x: 3}) INSERT (a)-[:K]->(b)")
        .unwrap();
    db
}

/// The work of running `query` once on this thread, the statement's whole tally.
#[cfg(grafeo_work_counters)]
fn work_of(db: &GrafeoDB, query: &str) -> grafeo_common::work::WorkCounts {
    let _ = grafeo_common::work::take();
    db.execute(query).unwrap();
    grafeo_common::work::take()
}

/// A scan of five nodes reading one property: the label's five rows tested for visibility as one
/// word, then each node's row group looked up once, its visibility tested once, its value read
/// once from the column; no adjacency list.
#[cfg(grafeo_work_counters)]
#[test]
fn a_scan_counts_what_it_reads_from_the_store() {
    use grafeo_common::work::Work;
    let work = work_of(&people(), "MATCH (n:P) RETURN n.x");
    assert_eq!(work.get(Work::RowGroupLookup), 5, "{}", work.describe());
    assert_eq!(work.get(Work::VisibilityTest), 1 + 5, "{}", work.describe());
    assert_eq!(work.get(Work::ColumnRead), 5, "{}", work.describe());
    assert_eq!(work.get(Work::AdjacencyRead), 0, "{}", work.describe());
    assert!(
        work.get(Work::StampedRead) <= work.get(Work::VisibilityTest),
        "a stamped read is a visibility test: {}",
        work.describe()
    );
}

/// Every visibility test counts, whichever path makes it, and a scan tests a word of rows at a
/// time. Five `:P` nodes and three `:Q` nodes, written by one statement, so their word reads alike:
/// - a label scan returning `id(n)` tests the word once and reads no column;
/// - a range scan with a label tests the label's word once and reads the label's five values,
///   not the `:Q` nodes' (`id(n)` reads nothing more);
/// - a range scan without one walks the store's vector of 2,048 rows as its 32 words, each tested
///   once, and reads all eight values.
#[cfg(grafeo_work_counters)]
#[test]
fn every_visibility_test_counts_whichever_path_makes_it() {
    use grafeo_common::work::Work;
    let db = GrafeoDB::new_in_memory();
    db.execute(concat!(
        "INSERT (:P {x: 1}), (:P {x: 2}), (:P {x: 3}), (:P {x: 4}), (:P {x: 5}), ",
        "(:Q {x: 1}), (:Q {x: 2}), (:Q {x: 3})"
    ))
    .unwrap();
    let label = work_of(&db, "MATCH (n:P) RETURN id(n)");
    assert_eq!(label.get(Work::VisibilityTest), 1, "{}", label.describe());
    assert_eq!(label.get(Work::ColumnRead), 0, "{}", label.describe());
    let range = work_of(&db, "MATCH (n:P) WHERE n.x > 2 RETURN id(n)");
    assert_eq!(range.get(Work::VisibilityTest), 1, "{}", range.describe());
    assert_eq!(range.get(Work::ColumnRead), 5, "{}", range.describe());
    let unlabelled = work_of(&db, "MATCH (n) WHERE n.x > 2 RETURN id(n)");
    assert_eq!(
        unlabelled.get(Work::VisibilityTest),
        32,
        "{}",
        unlabelled.describe()
    );
    assert_eq!(
        unlabelled.get(Work::ColumnRead),
        8,
        "{}",
        unlabelled.describe()
    );
}

/// An aggregate over a label's range folds the range in one store call: the five `:P` rows'
/// values of `x` read once each (the average takes the value the range read), the label's word
/// tested once, and no node read again by id. With the fold off, the range scan's three nodes in
/// range are each looked up again and read again by the aggregate.
#[cfg(grafeo_work_counters)]
#[test]
fn a_range_fold_reads_no_node_by_id() {
    use grafeo_common::work::Work;
    use grafeo_engine::query::optimizer::{PlannerRule, PlannerRules};
    let db = GrafeoDB::new_in_memory();
    db.execute(concat!(
        "INSERT (:P {x: 1}), (:P {x: 2}), (:P {x: 3}), (:P {x: 4}), (:P {x: 5}), ",
        "(:Q {x: 1}), (:Q {x: 2}), (:Q {x: 3})"
    ))
    .unwrap();
    let query = "MATCH (n:P) WHERE n.x > 2 RETURN count(*), avg(n.x)";
    let folded = work_of(&db, query);
    assert_eq!(folded.get(Work::RowGroupLookup), 0, "{}", folded.describe());
    assert_eq!(folded.get(Work::ColumnRead), 5, "{}", folded.describe());
    assert_eq!(folded.get(Work::VisibilityTest), 1, "{}", folded.describe());
    let mut session = db.session();
    session.set_planner_rules(PlannerRules::all().with(PlannerRule::RangeFold, false));
    let _ = grafeo_common::work::take();
    session.execute(query).unwrap();
    let scanned = grafeo_common::work::take();
    assert_eq!(
        scanned.get(Work::RowGroupLookup),
        3,
        "{}",
        scanned.describe()
    );
    assert_eq!(
        scanned.get(Work::ColumnRead),
        5 + 3,
        "{}",
        scanned.describe()
    );
}

/// A hop reads one adjacency list per source node.
#[cfg(grafeo_work_counters)]
#[test]
fn a_hop_reads_one_adjacency_list_per_source() {
    use grafeo_common::work::Work;
    let work = work_of(&people(), "MATCH (a:P)-[:K]->(b) RETURN b.x");
    assert_eq!(work.get(Work::AdjacencyRead), 5, "{}", work.describe());
}

/// Values: DISTINCT makes each of its five rows a hash key, ORDER BY compares (four times for
/// five rows already in order), and a string column clones each string it returns.
#[cfg(grafeo_work_counters)]
#[test]
fn values_count_their_hashes_comparisons_and_string_clones() {
    use grafeo_common::work::Work;
    let db = people();
    let distinct = work_of(&db, "MATCH (n:P) RETURN DISTINCT n.g");
    assert_eq!(distinct.get(Work::ValueHash), 5, "{}", distinct.describe());
    let sorted = work_of(&db, "MATCH (n:P) RETURN n.x ORDER BY n.x");
    assert_eq!(sorted.get(Work::ValueCompare), 4, "{}", sorted.describe());
    assert_eq!(sorted.get(Work::ValueHash), 0, "{}", sorted.describe());
    let names = work_of(&db, "MATCH (n:P) RETURN n.name");
    assert_eq!(names.get(Work::StringClone), 5, "{}", names.describe());
    let numbers = work_of(&db, "MATCH (n:P) RETURN n.x");
    assert_eq!(numbers.get(Work::StringClone), 0, "{}", numbers.describe());
}

/// Rows: the sort builds one chunk of its five rows and copies one column of each; the grouping
/// builds one chunk of its two groups and copies a key and an aggregate of each.
#[cfg(grafeo_work_counters)]
#[test]
fn rows_count_the_chunks_built_and_the_columns_copied() {
    use grafeo_common::work::Work;
    let db = people();
    let sorted = work_of(&db, "MATCH (n:P) RETURN n.x ORDER BY n.x");
    assert_eq!(sorted.get(Work::ChunkBuilt), 1, "{}", sorted.describe());
    assert_eq!(sorted.get(Work::RowBuilt), 5, "{}", sorted.describe());
    assert_eq!(sorted.get(Work::ColumnCopied), 5, "{}", sorted.describe());
    let grouped = work_of(&db, "MATCH (n:P) RETURN n.g, count(*)");
    assert_eq!(grouped.get(Work::RowBuilt), 2, "{}", grouped.describe());
    assert_eq!(grouped.get(Work::ColumnCopied), 4, "{}", grouped.describe());
    assert!(
        grouped.get(Work::DeadlineCheck) >= 1,
        "{}",
        grouped.describe()
    );
}

/// Per query: the first run parses, misses the plan cache and builds a plan; the second finds
/// the plan in the cache, parses nothing, and still builds the physical plan. A correlated
/// subquery runs again for each of its five outer rows.
#[cfg(grafeo_work_counters)]
#[test]
fn a_query_counts_its_parse_its_plan_and_its_subplans() {
    use grafeo_common::work::Work;
    let db = people();
    let first = work_of(&db, "MATCH (n:P) RETURN n.x");
    assert_eq!(first.get(Work::Parse), 1, "{}", first.describe());
    assert_eq!(first.get(Work::PlanCacheMiss), 1, "{}", first.describe());
    assert_eq!(first.get(Work::PlanBuilt), 1, "{}", first.describe());
    let second = work_of(&db, "MATCH (n:P) RETURN n.x");
    assert_eq!(second.get(Work::Parse), 0, "{}", second.describe());
    assert_eq!(second.get(Work::PlanCacheHit), 1, "{}", second.describe());
    assert_eq!(second.get(Work::PlanBuilt), 1, "{}", second.describe());
    let correlated = work_of(
        &db,
        "MATCH (n:P) CALL { WITH n MATCH (n)-[:K]->(m) RETURN count(m) AS c } RETURN n.x, c",
    );
    assert_eq!(
        correlated.get(Work::SubplanReplanned),
        5,
        "{}",
        correlated.describe()
    );
}

/// MVCC: with an older snapshot open while a node is deleted, the reads go through stamped
/// vectors, and the deleted node is not read.
#[cfg(grafeo_work_counters)]
#[test]
fn an_open_older_snapshot_reads_through_stamped_vectors() {
    use grafeo_common::work::Work;
    let db = people();
    let mut session = db.session();
    session.begin_transaction().unwrap();
    session.execute("MATCH (n:P) RETURN n.x").unwrap();
    db.execute("MATCH (n:P {x: 5}) DETACH DELETE n").unwrap();
    let work = work_of(&db, "MATCH (n:P) RETURN n.x");
    assert!(work.get(Work::StampedRead) > 0, "{}", work.describe());
    assert_eq!(work.get(Work::ColumnRead), 4, "{}", work.describe());
    session.rollback().unwrap();
}

/// A statement called with an empty parameter map, as graph-bench calls every statement without
/// parameters, is parsed and planned once: a repeat finds its plan by one cache lookup and parses
/// nothing, in GQL, Cypher and SQL/PGQ alike.
#[cfg(all(grafeo_work_counters, feature = "cypher", feature = "sql-pgq"))]
#[test]
fn a_repeat_with_an_empty_parameter_map_parses_nothing() {
    use grafeo_common::work::{self, Work};
    use std::collections::HashMap;

    let db = people();
    type Call = fn(&GrafeoDB) -> usize;
    let calls: [(&str, Call); 3] = [
        ("GQL", |db| {
            db.execute_with_params("MATCH (n:P) RETURN n.x", HashMap::new())
                .unwrap()
                .rows()
                .len()
        }),
        ("Cypher", |db| {
            db.execute_cypher_with_params("MATCH (n:P) RETURN n.x", HashMap::new())
                .unwrap()
                .rows()
                .len()
        }),
        ("SQL/PGQ", |db| {
            db.execute_sql_with_params(
                "SELECT x FROM GRAPH_TABLE (MATCH (n:P) COLUMNS (n.x AS x))",
                HashMap::new(),
            )
            .unwrap()
            .rows()
            .len()
        }),
    ];
    for (language, call) in calls {
        assert_eq!(call(&db), 5, "{language}");
        let _ = work::take();
        assert_eq!(call(&db), 5, "{language}");
        let repeat = work::take();
        let cache = [
            Work::Parse,
            Work::PlanCacheHit,
            Work::PlanCacheMiss,
            Work::PlanBuilt,
        ]
        .map(|work| repeat.get(work));
        assert_eq!(cache, [0, 1, 0, 1], "{language}: {}", repeat.describe());
    }
}

/// The memory classes wait for the operators' memory parameter (their own patch): until then they
/// read 0.
#[cfg(grafeo_work_counters)]
#[test]
fn the_memory_classes_read_nothing_yet() {
    use grafeo_common::work::Work;
    let work = work_of(&people(), "MATCH (n:P) RETURN n.x ORDER BY n.x");
    assert_eq!(work.get(Work::BudgetCheck), 0);
    assert_eq!(work.get(Work::GrantGrowth), 0);
}
