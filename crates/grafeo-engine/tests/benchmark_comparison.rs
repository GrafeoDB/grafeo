//! Benchmark comparison: Factorized vs Non-factorized execution

use grafeo_common::types::Value;
use grafeo_engine::{Config, GrafeoDB};
use std::time::Instant;

fn setup_graph_with_config(node_count: usize, avg_degree: usize, config: Config) -> GrafeoDB {
    let db = GrafeoDB::with_config(config).expect("Failed to create database");
    let session = db.session();

    // Create nodes
    let mut nodes = Vec::new();
    for i in 0..node_count {
        let id = session.create_node_with_props(&["Person"], [("id", Value::Int64(i as i64))]);
        nodes.push(id);
    }

    // Create edges with controlled fan-out
    let edge_count = node_count * avg_degree;
    for i in 0..edge_count {
        let src_idx = i % node_count;
        let dst_idx = (src_idx + 1 + (i / node_count)) % node_count;
        if src_idx != dst_idx {
            session.create_edge(nodes[src_idx], nodes[dst_idx], "KNOWS");
        }
    }

    db
}

fn measure<F: FnMut() -> R, R>(mut f: F, iterations: usize) -> (f64, f64) {
    let mut times = Vec::new();
    for _ in 0..iterations {
        let start = Instant::now();
        let _ = f();
        times.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mean = times.iter().sum::<f64>() / times.len() as f64;
    let variance = times.iter().map(|t| (t - mean).powi(2)).sum::<f64>() / times.len() as f64;
    (mean, variance.sqrt())
}

#[test]
fn benchmark_factorized_comparison() {
    println!("\n============================================================");
    println!("FACTORIZED vs NON-FACTORIZED EXECUTION COMPARISON");
    println!("============================================================\n");

    let node_count = 500;
    let avg_degree = 15;

    println!("Graph: {} nodes, avg degree {}", node_count, avg_degree);
    println!("Expected paths from node 0:");
    println!("  1-hop: ~{}", avg_degree);
    println!("  2-hop: ~{}", avg_degree * avg_degree);
    println!("  3-hop: ~{}", avg_degree * avg_degree * avg_degree);
    println!();

    // Setup graphs - one with factorized, one without
    println!("Setting up test graphs...");
    let db_factorized = setup_graph_with_config(
        node_count,
        avg_degree,
        Config::in_memory(), // factorized enabled by default
    );
    let db_flat = setup_graph_with_config(
        node_count,
        avg_degree,
        Config::in_memory().without_factorized_execution(),
    );

    let session_fact = db_factorized.session();
    let session_flat = db_flat.session();

    // Warmup
    for _ in 0..3 {
        let _ = session_fact.execute("MATCH (n:Person) RETURN n LIMIT 10");
        let _ = session_flat.execute("MATCH (n:Person) RETURN n LIMIT 10");
    }

    // Test queries
    let queries = [
        (
            "1-HOP",
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b) RETURN b.id",
            20,
        ),
        (
            "2-HOP",
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c) RETURN c.id",
            10,
        ),
        (
            "3-HOP",
            "MATCH (a:Person {id: 0})-[:KNOWS]->(b)-[:KNOWS]->(c)-[:KNOWS]->(d) RETURN d.id LIMIT 5000",
            5,
        ),
    ];

    println!(
        "\n{:12} | {:>15} | {:>15} | {:>10} | Rows (flat vs fact)",
        "Query", "Non-Factorized", "Factorized", "Speedup"
    );
    println!("{}", "-".repeat(80));

    for (name, query, iterations) in queries {
        // Measure non-factorized (flat)
        let (mean_flat, _) = measure(|| session_flat.execute(query).unwrap(), iterations);
        let result_flat = session_flat.execute(query).unwrap();

        // Measure factorized
        let (mean_fact, _) = measure(|| session_fact.execute(query).unwrap(), iterations);
        let result_fact = session_fact.execute(query).unwrap();

        // Check for row count mismatch (may indicate filter not being applied)
        let rows_match = result_flat.row_count() == result_fact.row_count();
        let speedup = mean_flat / mean_fact;

        let status = if rows_match { "OK" } else { "MISMATCH" };
        println!(
            "{:12} | {:>12.2} ms | {:>12.2} ms | {:>9.2}x | {} vs {} [{}]",
            name,
            mean_flat,
            mean_fact,
            speedup,
            result_flat.row_count(),
            result_fact.row_count(),
            status
        );
    }

    println!("\n============================================================");
    println!("SUMMARY");
    println!("============================================================");
    println!("- Speedup > 1.0x means factorized is faster");
    println!("- Factorized avoids materializing Cartesian products");
    println!("- Benefit increases with number of hops and fan-out");
    println!("============================================================\n");
}
