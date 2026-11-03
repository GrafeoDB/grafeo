//! Groups without ORDER BY come out in the order their first row was seen
//! (0.6.1), so the same query on the same data gives the same rows in the same
//! order on every run. The push aggregate kept its groups in a hash map with a
//! seed per table, so the order changed from one database to the next. Each
//! case runs on 20 fresh databases and must give the first-seen order each
//! time: one key, two keys, a key from an expand, and in GQL, Cypher and
//! SQL/PGQ.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test main group_order_is_first_seen::
//! ```

#![cfg(all(feature = "gql", feature = "cypher", feature = "sql-pgq"))]

use grafeo_engine::GrafeoDB;

/// Twelve cities in an order that is neither sorted nor random-looking.
const CITIES: [&str; 12] = [
    "Prague",
    "Amsterdam",
    "Paris",
    "Berlin",
    "Barcelona",
    "Lisbon",
    "Vienna",
    "Rome",
    "Oslo",
    "Madrid",
    "Dublin",
    "Athens",
];

/// 60 people, the i-th in CITIES[i % 12], inserted in order; each knows the
/// next one.
fn database() -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let cities: Vec<String> = CITIES.iter().map(|c| format!("'{c}'")).collect();
    db.execute(&format!(
        "UNWIND range(0, 59) AS i \
         INSERT (:Person {{i: i, city: [{}][i % 12], even: i % 2 = 0}})",
        cities.join(", ")
    ))
    .unwrap();
    db.execute("MATCH (a:Person), (b:Person) WHERE b.i = a.i + 1 INSERT (a)-[:KNOWS]->(b)")
        .unwrap();
    db
}

fn rows(db: &GrafeoDB, language: &str, query: &str) -> Vec<String> {
    db.execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"))
        .rows()
        .iter()
        .map(|row| format!("{row:?}"))
        .collect()
}

/// `query` gives `first_seen` on each of 20 fresh databases.
fn same_order_every_run(language: &str, query: &str, first_seen: &[String]) {
    for run in 0..20 {
        let got = rows(&database(), language, query);
        assert_eq!(got, first_seen, "{language} run {run}: `{query}`");
    }
}

fn city_rows(count: i64) -> Vec<String> {
    CITIES
        .iter()
        .map(|city| format!("[String({city:?}), Int64({count})]"))
        .collect()
}

#[test]
fn one_key_in_gql_cypher_and_sql_pgq() {
    let want = city_rows(5);
    same_order_every_run(
        "gql",
        "MATCH (p:Person) RETURN p.city AS city, count(*) AS n",
        &want,
    );
    same_order_every_run(
        "cypher",
        "MATCH (p:Person) RETURN p.city AS city, count(*) AS n",
        &want,
    );
    same_order_every_run(
        "sql-pgq",
        "SELECT city, COUNT(*) AS n FROM GRAPH_TABLE (MATCH (p:Person) COLUMNS (p.city AS city)) \
         GROUP BY city",
        &want,
    );
}

#[test]
fn two_keys() {
    // Person i has city i % 12 and evenness i % 2: the first 12 people give
    // the 12 groups, each city with one evenness (an even index for an even
    // position in CITIES).
    let want: Vec<String> = CITIES
        .iter()
        .enumerate()
        .map(|(i, city)| format!("[String({city:?}), Bool({}), Int64(5)]", i % 2 == 0))
        .collect();
    same_order_every_run(
        "gql",
        "MATCH (p:Person) RETURN p.city AS city, p.even AS even, count(*) AS n",
        &want,
    );
}

#[test]
fn a_key_reached_through_an_expand() {
    // a.i from 0 to 58 knows a.i + 1: the city of b, first seen for b = 1,
    // is CITIES[1]; Prague (b.i = 12, 24, 36, 48) comes last, four times.
    let mut want: Vec<String> = CITIES[1..]
        .iter()
        .map(|city| format!("[String({city:?}), Int64(5)]"))
        .collect();
    want.push(format!("[String({:?}), Int64(4)]", CITIES[0]));
    same_order_every_run(
        "gql",
        "MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN b.city AS city, count(*) AS n",
        &want,
    );
}

/// The groups of `UNWIND range(0, rows - 1) AS i RETURN (i * 7919) % 3000`,
/// in the order their first rows come: 7919 is prime to 3000, so the first
/// 3000 rows give each of the 3000 groups once, in an order that is not
/// sorted.
fn three_thousand_groups(rows: i64) -> (String, Vec<String>) {
    let query = format!(
        "UNWIND range(0, {}) AS i RETURN (i * 7919) % 3000 AS g, count(*) AS n",
        rows - 1
    );
    let want = (0..3000_i64)
        .map(|i| format!("[Int64({}), Int64({})]", (i * 7919) % 3000, rows / 3000))
        .collect();
    (query, want)
}

/// A database in a file has a spill path, so its aggregates keep their
/// groups in partitions that can go to disk; they come out in the order
/// their first rows came too, the same order as in memory, on every run.
#[cfg(feature = "spill")]
#[test]
fn a_file_database_gives_the_first_row_order_of_an_in_memory_one() {
    use grafeo_engine::Config;

    let (query, want) = three_thousand_groups(9000);
    let dir = tempfile::tempdir().unwrap();
    let file = GrafeoDB::with_config(Config::persistent(dir.path().join("lisbon.grafeo"))).unwrap();
    assert!(file.buffer_manager().config().spill_path.is_some());
    let memory = GrafeoDB::new_in_memory();
    for run in 0..5 {
        for language in ["gql", "cypher"] {
            assert_eq!(
                rows(&memory, language, &query),
                want,
                "{language} in memory, run {run}"
            );
            assert_eq!(
                rows(&file, language, &query),
                want,
                "{language} in a file, run {run}"
            );
        }
    }
}

/// Groups that go to disk and come back keep the order their first rows
/// came: the buffer manager is under pressure, so the aggregate spills its
/// largest partition after each chunk.
#[cfg(feature = "spill")]
#[test]
fn spilled_groups_keep_the_first_row_order() {
    use grafeo_common::memory::buffer::{MemoryRegion, PressureLevel};
    use grafeo_engine::Config;

    let (query, want) = three_thousand_groups(30_000);
    let dir = tempfile::tempdir().unwrap();
    let db = GrafeoDB::with_config(
        Config::persistent(dir.path().join("vienna.grafeo"))
            .with_memory_limit(64 * 1024 * 1024)
            .without_query_timeout(),
    )
    .unwrap();
    let manager = db.buffer_manager();
    let pressure = manager
        .try_allocate(
            manager.config().budget * 9 / 10,
            MemoryRegion::ExecutionBuffers,
        )
        .unwrap();
    assert_eq!(manager.pressure_level(), PressureLevel::High);
    for run in 0..3 {
        for language in ["gql", "cypher"] {
            assert_eq!(rows(&db, language, &query), want, "{language} run {run}");
        }
    }
    drop(pressure);
}
