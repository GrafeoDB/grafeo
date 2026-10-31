//! A two-hop expand whose product passes the most rows a chunk can hold
//! (65,535, the u16 selection) must not panic, and must count every path.
//! 0.6.0 flattened a whole factorized chain, or a whole input chunk's product,
//! into one chunk, so a filter above `(a)-[:E]->(b)-[:E]->(c)` panicked at
//! selection.rs (`index <= MAX_CAPACITY`) once the product passed 65,536 rows
//! (found by grafeo-31, 2026-10-31). Each case checks the exact count or rows,
//! and the same query with factorized execution off.
//!
//! The graphs: sources into a hub, the hub out to `n` leaves (`k` x `n`
//! paths), and a chain through two hubs whose products multiply.
//!
//! ```bash
//! cargo test -p grafeo-engine --all-features --test factorized_large_products
//! ```

#![cfg(all(feature = "gql", feature = "cypher"))]

use std::collections::HashSet;

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};

/// `k` sources (`id` -1 to -k) with an E edge into one hub (`id` -1,000,000),
/// and `n` E edges out of the hub to leaves (`id` 0 to n - 1): `k` x `n`
/// two-hop paths.
fn hub(config: Config, k: i64, n: i64) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    let hub = db
        .create_node_with_props(&["N"], [("id", Value::Int64(-1_000_000))])
        .unwrap();
    for source in 1..=k {
        let id = db
            .create_node_with_props(&["N"], [("id", Value::Int64(-source))])
            .unwrap();
        db.create_edge(id, hub, "E").unwrap();
    }
    for leaf in 0..n {
        let id = db
            .create_node_with_props(&["N"], [("id", Value::Int64(leaf))])
            .unwrap();
        db.create_edge(hub, id, "E").unwrap();
    }
    db
}

fn count(db: &GrafeoDB, language: &str, query: &str) -> i64 {
    let result = db
        .execute_language(query, language, None)
        .unwrap_or_else(|error| panic!("{language} `{query}`: {error}"));
    match result.rows()[0][0] {
        Value::Int64(n) => n,
        ref other => panic!("{query}: {other:?}"),
    }
}

/// The count of `query` on the database with factorized execution on and on
/// the one with it off, which must agree; returns it.
fn agreed_count(k: i64, n: i64, language: &str, query: &str) -> i64 {
    let on = count(&hub(Config::in_memory(), k, n), language, query);
    let off = count(
        &hub(Config::in_memory().without_factorized_execution(), k, n),
        language,
        query,
    );
    assert_eq!(on, off, "factorized on and off: `{query}`");
    on
}

const FILTERED: &str = "MATCH (a)-[:E]->(b)-[:E]->(c) WHERE c.id >= 0 RETURN count(*)";

#[test]
fn a_filter_over_a_two_hop_product_of_70000_counts_every_path() {
    assert_eq!(agreed_count(1, 70_000, "gql", FILTERED), 70_000);
}

#[test]
fn an_edge_inequality_over_a_two_hop_product_of_70000_counts_every_path() {
    assert_eq!(
        agreed_count(
            1,
            70_000,
            "gql",
            "MATCH (a)-[e1:E]->(b)-[e2:E]->(c) WHERE e1 <> e2 RETURN count(*)"
        ),
        70_000
    );
}

#[test]
fn cypher_returns_every_end_of_a_two_hop_product_of_70000() {
    let db = hub(Config::in_memory(), 1, 70_000);
    let query = "MATCH (a)-[:E]->(b)-[:E]->(c) WHERE c.id >= 0 RETURN c.id AS id";
    let ids: Vec<i64> = db
        .execute_cypher(query)
        .unwrap()
        .rows()
        .iter()
        .map(|row| match row[0] {
            Value::Int64(id) => id,
            ref other => panic!("{other:?}"),
        })
        .collect();
    assert_eq!(ids.len(), 70_000);
    let distinct: HashSet<i64> = ids.iter().copied().collect();
    assert_eq!(distinct.len(), 70_000, "each leaf once");
    assert_eq!(distinct, (0..70_000).collect::<HashSet<_>>());
    assert_eq!(
        count(
            &db,
            "cypher",
            "MATCH (a)-[:E]->(b)-[:E]->(c) WHERE c.id >= 0 RETURN count(*)"
        ),
        70_000
    );
    // From a bound source.
    assert_eq!(
        count(
            &db,
            "cypher",
            "MATCH (a {id: -1})-[:E]->(b)-[:E]->(c) WHERE c.id >= 0 RETURN count(*)"
        ),
        70_000
    );
    assert_eq!(
        count(
            &db,
            "gql",
            "MATCH (a {id: -1})-[:E]->(b)-[:E]->(c) WHERE c.id >= 0 RETURN count(*)"
        ),
        70_000
    );
}

/// The count of the filtered two-hop query over `k` x `n` paths, factorized
/// on and off.
fn boundary(k: i64, n: i64) {
    assert_eq!(
        agreed_count(k, n, "gql", FILTERED),
        k * n,
        "{k} sources x {n} leaves"
    );
}

#[test]
fn a_product_of_65535_counts_every_path() {
    boundary(1, 65_535);
}

#[test]
fn a_product_of_65536_counts_every_path() {
    boundary(1, 65_536);
}

#[test]
fn a_product_of_65537_counts_every_path() {
    boundary(1, 65_537);
}

#[test]
fn three_sources_into_a_hub_of_21845_count_every_path() {
    // 65,535 paths from three sources: just under the limit.
    boundary(3, 21_845);
}

#[test]
fn two_sources_into_a_hub_of_65536_count_every_path() {
    boundary(2, 65_536);
}

#[test]
fn without_factorized_execution_a_product_of_70000_counts_every_path() {
    // The regular path alone, so a failure here is not the factorized one.
    let db = hub(
        Config::in_memory().without_factorized_execution(),
        1,
        70_000,
    );
    assert_eq!(count(&db, "gql", FILTERED), 70_000);
    assert_eq!(
        count(
            &db,
            "gql",
            "MATCH (a)-[e1:E]->(b)-[e2:E]->(c) WHERE e1 <> e2 RETURN count(*)"
        ),
        70_000
    );
}

/// A chain through two hubs: s -> h1 -> m (`m` nodes) -> h2 -> c (`c`
/// leaves): `m` x `c` paths of four hops, with only 2m + c + 1 edges, so the
/// product grows at the third level.
fn two_hubs(config: Config, m: i64, c: i64) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).unwrap();
    let node = |id: i64| {
        db.create_node_with_props(&["N"], [("id", Value::Int64(id))])
            .unwrap()
    };
    let source = node(-1);
    let first = node(-2);
    let second = node(-3);
    db.create_edge(source, first, "E").unwrap();
    for i in 0..m {
        let middle = node(1_000_000 + i);
        db.create_edge(first, middle, "E").unwrap();
        db.create_edge(middle, second, "E").unwrap();
    }
    for i in 0..c {
        let leaf = node(i);
        db.create_edge(second, leaf, "E").unwrap();
    }
    db
}

#[test]
fn a_chain_through_two_hubs_counts_a_product_past_a_chunk_at_each_level() {
    let query = "MATCH (a {id: -1})-[:E]->(b)-[:E]->(m)-[:E]->(h)-[:E]->(c) \
                 WHERE c.id >= 0 RETURN count(*)";
    for (m, c) in [(300, 300), (400, 400), (70_000, 2)] {
        let on = count(&two_hubs(Config::in_memory(), m, c), "gql", query);
        let off = count(
            &two_hubs(Config::in_memory().without_factorized_execution(), m, c),
            "gql",
            query,
        );
        assert_eq!(on, m * c, "{m} x {c}");
        assert_eq!(on, off, "{m} x {c}: factorized on and off");
    }
}

#[test]
fn a_limit_over_the_hub_expansion_returns_its_rows() {
    // 50 sources into a hub of 70,000: 3,500,000 paths; LIMIT 10 needs ten.
    let db = hub(Config::in_memory(), 50, 70_000);
    let rows = db
        .execute("MATCH (a)-[:E]->(b)-[:E]->(c) WHERE c.id >= 0 RETURN c.id AS id LIMIT 10")
        .unwrap()
        .rows()
        .to_vec();
    assert_eq!(rows.len(), 10);
    for row in &rows {
        assert!(
            matches!(row[0], Value::Int64(id) if (0..70_000).contains(&id)),
            "{row:?}"
        );
    }
}
