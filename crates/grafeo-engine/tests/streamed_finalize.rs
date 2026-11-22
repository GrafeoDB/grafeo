//! A spilled GQL sort hands its merged rows to the operators after it as it
//! makes them: a limit, a filter and an aggregate after it give the rows of
//! the same query on a database with room for the whole sort (the oracle),
//! and leave no spill file, on success and on an error. How much it holds is
//! measured in `streamed_finalize_peak.rs` (a counting allocator); that the
//! merge stops at a limit, and the pipeline's own rules, in `grafeo-core`.
//!
//! A limit directly after the sort runs as one top-k: within the query memory
//! limit it needs no spill, and past it the top-k goes on as the spilling sort
//! and a limit. The other queries here take no top-k (a filter or an aggregate
//! follows the sort) and spill: without a spill path they refuse at the query
//! memory limit.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main streamed_finalize::
//! ```

#![cfg(all(feature = "gql", feature = "spill"))]

use std::path::{Path, PathBuf};

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

const MIB: usize = 1024 * 1024;
const PEOPLE: i64 = 30_000;

/// 30,000 people, their ages a permutation of 0..30,000.
fn people(config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    let cities = ["Amsterdam", "Berlin", "Paris", "Prague", "Utrecht"];
    for i in 0..PEOPLE {
        let city = cities[usize::try_from(i % 5).unwrap()];
        db.create_node_with_props(
            &["Person"],
            [
                ("name", Value::from(format!("Person {i:05}"))),
                ("age", Value::Int64(i * 7919 % PEOPLE)),
                ("city", Value::from(city)),
            ],
        )
        .unwrap();
    }
    db
}

/// A database that spills at 1 MiB, one that cannot spill at 1 MiB, and the
/// oracle, with room for the whole sort.
struct Databases {
    spill: tempfile::TempDir,
    spilling: GrafeoDB,
    without_a_spill_path: GrafeoDB,
    oracle: GrafeoDB,
}

fn databases() -> Databases {
    let spill = tempfile::tempdir().unwrap();
    let spilling = people(
        Config::in_memory()
            .with_query_memory_limit(MIB)
            .with_spill_path(spill.path()),
    );
    Databases {
        spill,
        spilling,
        without_a_spill_path: people(Config::in_memory().with_query_memory_limit(MIB)),
        oracle: people(Config::in_memory().with_query_memory_limit(512 * MIB)),
    }
}

/// The files under `dir`, but the owner lock of a spill directory.
fn files_left(dir: &Path) -> Vec<PathBuf> {
    let mut left = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap().flatten() {
        let path = entry.path();
        if path.is_dir() {
            left.extend(files_left(&path));
        } else if entry.file_name() != "owner.lock" {
            left.push(path);
        }
    }
    left
}

fn rows(db: &GrafeoDB, query: &str) -> Vec<Vec<Value>> {
    db.execute(query)
        .unwrap_or_else(|error| panic!("{query}: {error}"))
        .rows()
        .to_vec()
}

/// `query` spills on `dbs.spilling`, and gives the oracle's rows in the
/// oracle's order there, leaving no spill file.
fn assert_oracle_rows(dbs: &Databases, query: &str) {
    let refused = dbs.without_a_spill_path.execute(query);
    assert!(refused.is_err(), "{query}: spills ({refused:?})");
    let want = rows(&dbs.oracle, query);
    assert!(!want.is_empty(), "{query}");
    assert_eq!(rows(&dbs.spilling, query), want, "{query}");
    assert_eq!(
        files_left(dbs.spill.path()),
        Vec::<PathBuf>::new(),
        "{query}"
    );
}

/// A limit directly after the sort, behind an augmenting projection (the sort key is not
/// returned), runs as one top-k: a limit within the first merged chunk and one past it (3,000
/// rows, over two chunks of 2,048) keep their rows within the query memory limit, without a
/// spill path, and give the oracle's rows.
#[test]
fn a_top_k_within_the_limit_needs_no_spill() {
    let dbs = databases();
    for query in [
        "MATCH (n:Person) RETURN n.name ORDER BY n.age LIMIT 10",
        "MATCH (n:Person) RETURN n.city ORDER BY n.age DESC LIMIT 3000",
    ] {
        assert_eq!(
            rows(&dbs.without_a_spill_path, query),
            rows(&dbs.oracle, query),
            "{query}"
        );
    }
}

/// A top-k whose rows outgrow the statement's memory goes on as the spilling sort and a limit:
/// with a spill path it gives the oracle's rows in the oracle's order (ties in the order the
/// rows came, as the sort keeps them) and leaves no spill file; without one it fails with
/// `GRAFEO-S001`, as the sort does.
#[test]
fn a_top_k_past_the_limit_spills_as_the_sort_does() {
    let dbs = databases();
    for query in [
        "MATCH (n:Person) RETURN n.name ORDER BY n.age LIMIT 30000",
        "MATCH (n:Person) RETURN n.name ORDER BY n.age DESC LIMIT 20000",
        "MATCH (n:Person) RETURN n.name ORDER BY n.city LIMIT 25000",
    ] {
        let refused = dbs
            .without_a_spill_path
            .execute(query)
            .expect_err("past the query memory limit without a spill path");
        assert!(
            refused.to_string().contains("GRAFEO-S001"),
            "{query}: {refused}"
        );
        assert_oracle_rows(&dbs, query);
    }
}

/// A filter and an aggregate after a spilled sort: the filter's rows in the
/// sort's order, the groups in the order the sort first gives them.
#[test]
fn a_spilled_sort_then_a_filter_or_an_aggregate_gives_the_oracle_rows() {
    let dbs = databases();
    for query in [
        "MATCH (n:Person) WITH n.name AS name, n.age AS age ORDER BY age WHERE age % 1000 = 0 RETURN name, age",
        "MATCH (n:Person) WITH n.city AS c, n.age AS a ORDER BY a RETURN c, count(*) AS k",
    ] {
        assert_oracle_rows(&dbs, query);
    }
}

/// A filter after a spilled sort that fails part way through the merged
/// rows (a division by zero at age 20,000) fails the query and leaves no
/// spill file.
#[test]
fn a_spilled_sort_then_a_failing_filter_leaves_no_file() {
    let dbs = databases();
    let query = "MATCH (n:Person) WITH n.name AS name, n.age AS age ORDER BY age \
                 WHERE 1 / (age - 20000) <= 1 RETURN name";
    assert!(dbs.oracle.execute(query).is_err(), "the oracle fails too");
    let error = dbs.spilling.execute(query).expect_err("division by zero");
    assert_eq!(
        files_left(dbs.spill.path()),
        Vec::<PathBuf>::new(),
        "{error}"
    );
}

/// A stop does not lose the rows that came with it: a limit that hands on its
/// last rows and stops in the same push, followed by a `DISTINCT`, still gives
/// those rows. The oracle is the same limit without the `DISTINCT`, made
/// distinct here.
#[test]
fn a_limit_before_a_distinct_keeps_the_rows_it_stopped_with() {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();
    for i in 0..50 {
        session
            .execute(&format!(
                "INSERT (:Person {{name: 'Mia {i}', age: {}}})",
                i % 7
            ))
            .unwrap();
    }
    let limited = rows(&db, "MATCH (n:Person) WITH n.age AS a LIMIT 10 RETURN a");
    assert_eq!(limited.len(), 10, "the limit alone");
    let mut want: Vec<String> = limited.iter().map(|row| format!("{row:?}")).collect();
    want.sort();
    want.dedup();
    let mut got: Vec<String> = rows(
        &db,
        "MATCH (n:Person) WITH n.age AS a LIMIT 10 RETURN DISTINCT a",
    )
    .iter()
    .map(|row| format!("{row:?}"))
    .collect();
    got.sort();
    assert_eq!(got, want, "the distinct values of the limited rows");
}

/// The same in Cypher: `WITH ... LIMIT` followed by `RETURN DISTINCT` keeps the
/// rows the limit stopped with.
#[cfg(feature = "cypher")]
#[test]
fn a_cypher_limit_before_a_distinct_keeps_the_rows_it_stopped_with() {
    let db = GrafeoDB::new_in_memory();
    for i in 0..50 {
        db.execute_cypher(&format!(
            "CREATE (:Person {{name: 'Mia {i}', age: {}}})",
            i % 7
        ))
        .unwrap();
    }
    let cypher_rows = |query: &str| -> Vec<String> {
        db.execute_cypher(query)
            .unwrap_or_else(|error| panic!("{query}: {error}"))
            .rows()
            .iter()
            .map(|row| format!("{row:?}"))
            .collect()
    };
    let mut want = cypher_rows("MATCH (n:Person) WITH n.age AS a LIMIT 10 RETURN a");
    assert_eq!(want.len(), 10, "the limit alone");
    want.sort();
    want.dedup();
    let mut got = cypher_rows("MATCH (n:Person) WITH n.age AS a LIMIT 10 RETURN DISTINCT a");
    got.sort();
    assert_eq!(got, want, "the distinct values of the limited rows");
}
