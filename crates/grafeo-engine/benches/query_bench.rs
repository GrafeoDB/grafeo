//! End-to-end query benchmarks for the Grafeo engine.
//!
//! Measures full query execution time from query string to result,
//! covering parsing, planning, optimization, and execution.
//!
//! Run with: cargo bench -p grafeo-engine
//! RDF benchmarks require: cargo bench -p grafeo-engine --all-features
// reason: criterion_group! expansion from codspeed-criterion-compat does not
// carry doc comments on the generated wrapper functions.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};

use grafeo_engine::GrafeoDB;
#[cfg(all(feature = "sparql", feature = "triple-store"))]
use {
    grafeo_engine::{Config, GraphModel},
    std::fmt::Write,
};

// ============================================================================
// LPG Benchmarks (GQL)
// ============================================================================

/// Sets up a small social graph for benchmarking.
/// Returns the database instance ready for queries.
fn setup_social_graph(node_count: usize, edge_multiplier: usize) -> GrafeoDB {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Create Person nodes with properties
    for i in 0..node_count {
        let query = format!(
            "INSERT (:Person {{id: {}, name: 'User{}', age: {}}})",
            i,
            i,
            20 + (i % 50)
        );
        session.execute(&query).unwrap();
    }

    // Create KNOWS edges (CREATE and INSERT are equivalent after MATCH in GQL)
    let edge_count = node_count * edge_multiplier;
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (i * 7 + 13) % node_count;
        if src != dst {
            let query = format!(
                "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) CREATE (a)-[:KNOWS]->(b)",
                src, dst
            );
            let _ = session.execute(&query);
        }
    }

    db
}

fn bench_node_lookup(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_node_lookup_by_property", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (n:Person {id: 42}) RETURN n.name")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_pattern_match(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_1hop_pattern", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.name")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_two_hop_pattern(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_2hop_pattern", |b| {
        b.iter(|| {
            let result = session
                .execute(
                    "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN DISTINCT c.id",
                )
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_aggregation_count(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_count_all", |b| {
        b.iter(|| {
            let result = session.execute("MATCH (n:Person) RETURN COUNT(n)").unwrap();
            black_box(result)
        });
    });
}

fn bench_filter_range(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("query_filter_range", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (n:Person) WHERE n.age > 50 RETURN n.id")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_fan_out_expand_1k(c: &mut Criterion) {
    let db = setup_social_graph(1_000, 5);
    let session = db.session();

    // Expands from ALL Person nodes, testing scatter performance.
    c.bench_function("query_fan_out_expand_1k", |b| {
        b.iter(|| {
            let result = session
                .execute("MATCH (a:Person)-[:KNOWS]->(b) RETURN COUNT(b)")
                .unwrap();
            black_box(result)
        });
    });
}

fn bench_insert_single_node(c: &mut Criterion) {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let mut counter = 0u64;
    c.bench_function("query_insert_single_node", |b| {
        b.iter(|| {
            let query = format!("INSERT (:Bench {{id: {}}})", counter);
            counter += 1;
            let result = session.execute(&query).unwrap();
            black_box(result)
        });
    });
}

criterion_group!(
    lpg_benches,
    bench_node_lookup,
    bench_pattern_match,
    bench_two_hop_pattern,
    bench_fan_out_expand_1k,
    bench_aggregation_count,
    bench_filter_range,
    bench_insert_single_node,
);

// ============================================================================
// RDF Benchmarks (SPARQL)
// ============================================================================

#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn setup_rdf_social_graph(person_count: usize, edge_multiplier: usize) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();

    // Batch insert persons with name and age
    let mut triples = String::from("INSERT DATA {\n");
    for i in 0..person_count {
        let age = 20 + (i % 50);
        let _ = write!(
            triples,
            "  <http://ex.org/p{i}> <http://ex.org/name> \"User{i}\" .\n\
             <http://ex.org/p{i}> <http://ex.org/age> \"{age}\" .\n\
             <http://ex.org/p{i}> <http://ex.org/type> \"Person\" .\n"
        );
    }

    // Add knows edges
    let edge_count = person_count * edge_multiplier;
    for i in 0..edge_count {
        let src = i % person_count;
        let dst = (i * 7 + 13) % person_count;
        if src != dst {
            let _ = writeln!(
                triples,
                "  <http://ex.org/p{src}> <http://ex.org/knows> <http://ex.org/p{dst}> ."
            );
        }
    }
    triples.push('}');

    session.execute_sparql(&triples).unwrap();
    db
}

/// Single triple pattern lookup with subject bound.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_single_pattern(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("rdf_single_pattern_lookup", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name WHERE {
                        <http://ex.org/p42> <http://ex.org/name> ?name
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// Two-pattern star join (shared subject variable).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_star_join(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("rdf_star_join_2pattern", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?age WHERE {
                        ?s <http://ex.org/name> ?name .
                        ?s <http://ex.org/age> ?age
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// Three-pattern star join (shared subject, three predicates).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_star_join_3(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("rdf_star_join_3pattern", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?age ?type WHERE {
                        ?s <http://ex.org/name> ?name .
                        ?s <http://ex.org/age> ?age .
                        ?s <http://ex.org/type> ?type
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// Chain join: ?a knows ?b, ?b has name.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_chain_join(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("rdf_chain_join", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name WHERE {
                        <http://ex.org/p0> <http://ex.org/knows> ?friend .
                        ?friend <http://ex.org/name> ?name
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// OPTIONAL pattern: name required, email optional.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_optional(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    // Add email for ~half the persons
    let mut triples = String::from("INSERT DATA {\n");
    for i in (0..1_000).step_by(2) {
        let _ = writeln!(
            triples,
            "  <http://ex.org/p{i}> <http://ex.org/email> \"user{i}@example.org\" ."
        );
    }
    triples.push('}');
    session.execute_sparql(&triples).unwrap();

    c.bench_function("rdf_optional_pattern", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?email WHERE {
                        ?s <http://ex.org/name> ?name .
                        OPTIONAL { ?s <http://ex.org/email> ?email }
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// COUNT aggregation over all triples.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_count(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("rdf_count_all", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT (COUNT(?s) AS ?count) WHERE {
                        ?s <http://ex.org/name> ?name
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// FILTER with string comparison.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_filter(c: &mut Criterion) {
    let db = setup_rdf_social_graph(1_000, 5);
    let session = db.session();

    c.bench_function("rdf_filter_string", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name WHERE {
                        ?s <http://ex.org/name> ?name
                        FILTER(CONTAINS(?name, "5"))
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// Insert single triple.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_insert_single(c: &mut Criterion) {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();

    let mut counter = 0u64;
    c.bench_function("rdf_insert_single_triple", |b| {
        b.iter(|| {
            let query = format!(
                "INSERT DATA {{ <http://ex.org/bench{counter}> <http://ex.org/val> \"v\" }}"
            );
            counter += 1;
            let result = session.execute_sparql(&query).unwrap();
            black_box(result)
        });
    });
}

// ============================================================================
// RDF Join-Specific Benchmarks (larger dataset for join performance)
// ============================================================================

/// Sets up a 10K triple dataset with diverse predicates for join benchmarks.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn setup_rdf_join_dataset(person_count: usize) -> GrafeoDB {
    let db = GrafeoDB::with_config(Config::in_memory().with_graph_model(GraphModel::Rdf)).unwrap();
    let session = db.session();

    let mut triples = String::from("INSERT DATA {\n");
    for i in 0..person_count {
        let age = 20 + (i % 50);
        let city_idx = i % 5;
        let city = ["Amsterdam", "Berlin", "Paris", "Prague", "Barcelona"][city_idx];
        let _ = write!(
            triples,
            "  <http://ex.org/p{i}> <http://ex.org/name> \"User{i}\" .\n\
             <http://ex.org/p{i}> <http://ex.org/age> \"{age}\" .\n\
             <http://ex.org/p{i}> <http://ex.org/city> \"{city}\" .\n\
             <http://ex.org/p{i}> <http://ex.org/type> \"Person\" .\n"
        );
        // Half have email
        if i % 2 == 0 {
            let _ = writeln!(
                triples,
                "  <http://ex.org/p{i}> <http://ex.org/email> \"user{i}@example.org\" ."
            );
        }
    }
    // Add knows edges (5x multiplier)
    let edge_count = person_count * 5;
    for i in 0..edge_count {
        let src = i % person_count;
        let dst = (i * 7 + 13) % person_count;
        if src != dst {
            let _ = writeln!(
                triples,
                "  <http://ex.org/p{src}> <http://ex.org/knows> <http://ex.org/p{dst}> ."
            );
        }
    }
    triples.push('}');
    session.execute_sparql(&triples).unwrap();
    db
}

/// 2-pattern star join on 10K persons.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_join_star_2_10k(c: &mut Criterion) {
    let db = setup_rdf_join_dataset(10_000);
    let session = db.session();

    c.bench_function("rdf_join_star_2_10k", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?age WHERE {
                        ?s <http://ex.org/name> ?name .
                        ?s <http://ex.org/age> ?age
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// 3-pattern star join on 10K persons.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_join_star_3_10k(c: &mut Criterion) {
    let db = setup_rdf_join_dataset(10_000);
    let session = db.session();

    c.bench_function("rdf_join_star_3_10k", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?age ?city WHERE {
                        ?s <http://ex.org/name> ?name .
                        ?s <http://ex.org/age> ?age .
                        ?s <http://ex.org/city> ?city
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// 4-pattern star join (one pattern has ~50% selectivity via OPTIONAL-like density).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_join_star_4_10k(c: &mut Criterion) {
    let db = setup_rdf_join_dataset(10_000);
    let session = db.session();

    c.bench_function("rdf_join_star_4_10k", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?age ?city ?email WHERE {
                        ?s <http://ex.org/name> ?name .
                        ?s <http://ex.org/age> ?age .
                        ?s <http://ex.org/city> ?city .
                        ?s <http://ex.org/email> ?email
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// Chain join: ?a knows ?b, ?b has name (traversal pattern).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_join_chain_10k(c: &mut Criterion) {
    let db = setup_rdf_join_dataset(10_000);
    let session = db.session();

    c.bench_function("rdf_join_chain_10k", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name WHERE {
                        <http://ex.org/p0> <http://ex.org/knows> ?friend .
                        ?friend <http://ex.org/name> ?name
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// 2-hop chain join: ?a knows ?b, ?b knows ?c, ?c has name.
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_join_chain_2hop_10k(c: &mut Criterion) {
    let db = setup_rdf_join_dataset(10_000);
    let session = db.session();

    c.bench_function("rdf_join_chain_2hop_10k", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name WHERE {
                        <http://ex.org/p0> <http://ex.org/knows> ?f1 .
                        ?f1 <http://ex.org/knows> ?f2 .
                        ?f2 <http://ex.org/name> ?name
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

/// OPTIONAL join on 10K persons (left join performance).
#[cfg(all(feature = "sparql", feature = "triple-store"))]
fn bench_rdf_join_optional_10k(c: &mut Criterion) {
    let db = setup_rdf_join_dataset(10_000);
    let session = db.session();

    c.bench_function("rdf_join_optional_10k", |b| {
        b.iter(|| {
            let result = session
                .execute_sparql(
                    r#"SELECT ?name ?email WHERE {
                        ?s <http://ex.org/name> ?name .
                        OPTIONAL { ?s <http://ex.org/email> ?email }
                    }"#,
                )
                .unwrap();
            black_box(result)
        });
    });
}

#[cfg(all(feature = "sparql", feature = "triple-store"))]
criterion_group!(
    rdf_benches,
    bench_rdf_single_pattern,
    bench_rdf_star_join,
    bench_rdf_star_join_3,
    bench_rdf_chain_join,
    bench_rdf_optional,
    bench_rdf_count,
    bench_rdf_filter,
    bench_rdf_insert_single,
);

#[cfg(all(feature = "sparql", feature = "triple-store"))]
criterion_group!(
    rdf_join_benches,
    bench_rdf_join_star_2_10k,
    bench_rdf_join_star_3_10k,
    bench_rdf_join_star_4_10k,
    bench_rdf_join_chain_10k,
    bench_rdf_join_chain_2hop_10k,
    bench_rdf_join_optional_10k,
);

#[cfg(all(feature = "sparql", feature = "triple-store"))]
criterion_main!(lpg_benches, rdf_benches, rdf_join_benches);

#[cfg(not(all(feature = "sparql", feature = "triple-store")))]
criterion_main!(lpg_benches);
