//! Groups without ORDER BY come out in the order their first row was seen
//! (0.6.1), so the same query on the same data gives the same rows in the same
//! order on every run. The push aggregate kept its groups in a hash map with a
//! seed per table, so the order changed from one database to the next. Each
//! case runs on 20 fresh databases and must give the first-seen order each
//! time: one key, two keys, a key from an expand, and in GQL, Cypher and
//! SQL/PGQ.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test group_order_is_first_seen
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
