//! Comprehensive Graph Database Benchmarks
//!
//! Performance benchmarks similar to Memgraph vs Neo4j comparisons.
//! Tests insertion throughput, traversal performance, and query patterns.
//!
//! Run with: cargo test -p grafeo-engine --release -- graph_benchmarks --nocapture
//!
//! Some heavy benchmarks are marked `#[ignore]` to keep default `cargo test` fast.
//! To run all benchmarks including heavy ones:
//!   cargo test -p grafeo-engine --release -- graph_benchmarks --nocapture --include-ignored

use std::time::{Duration, Instant};

use grafeo_engine::GrafeoDB;

// ============================================================================
// Benchmark Configuration
// ============================================================================

// In debug mode, use small datasets for fast iteration
#[cfg(debug_assertions)]
mod scale {
    pub const SMALL_SCALE: usize = 100;
    pub const MEDIUM_SCALE: usize = 500;
    pub const LARGE_SCALE: usize = 1_000;
    pub const NODE_COUNT: usize = 100;
    pub const TRAVERSAL_NODES: usize = 500;
    pub const FILTER_NODES: usize = 1_000;
    pub const LOOKUP_NODES: usize = 1_000;
    pub const CONCURRENT_NODES: usize = 1_000;
    pub const PATTERN_NODES: usize = 200;
    pub const MIXED_NODES: usize = 500;
    pub const AGGREGATION_NODES: usize = 1_000;
    pub const EDGE_NODES: usize = 100;
}

// In release mode, use larger datasets for meaningful benchmarks
// Keep values reasonable for CI (~10-30s per test max)
#[cfg(not(debug_assertions))]
mod scale {
    pub const SMALL_SCALE: usize = 500;
    pub const MEDIUM_SCALE: usize = 2_000;
    pub const LARGE_SCALE: usize = 5_000;
    pub const NODE_COUNT: usize = 5_000;
    pub const TRAVERSAL_NODES: usize = 1_000;
    pub const FILTER_NODES: usize = 2_000;
    pub const LOOKUP_NODES: usize = 1_000;
    pub const CONCURRENT_NODES: usize = 2_000;
    pub const PATTERN_NODES: usize = 500;
    pub const MIXED_NODES: usize = 1_000;
    pub const AGGREGATION_NODES: usize = 2_000;
    pub const EDGE_NODES: usize = 500;
}

use scale::*;

const EDGE_MULTIPLIER: usize = 10; // Average edges per node

// ============================================================================
// Utility Functions
// ============================================================================

fn format_rate(count: usize, duration: Duration) -> String {
    let rate = count as f64 / duration.as_secs_f64();
    if rate >= 1_000_000.0 {
        format!("{:.2}M/sec", rate / 1_000_000.0)
    } else if rate >= 1_000.0 {
        format!("{:.2}K/sec", rate / 1_000.0)
    } else {
        format!("{:.2}/sec", rate)
    }
}

fn format_duration(duration: Duration) -> String {
    if duration.as_secs() > 0 {
        format!("{:.2}s", duration.as_secs_f64())
    } else if duration.as_millis() > 0 {
        format!("{:.2}ms", duration.as_secs_f64() * 1000.0)
    } else {
        format!("{:.2}us", duration.as_secs_f64() * 1_000_000.0)
    }
}

fn print_result(name: &str, count: usize, duration: Duration) {
    println!(
        "  {:<40} {:>10} in {:>10} ({:>12})",
        name,
        count,
        format_duration(duration),
        format_rate(count, duration)
    );
}

fn print_header(section: &str) {
    println!("\n{}", "=".repeat(80));
    println!("  {}", section);
    println!("{}", "=".repeat(80));
}

// ============================================================================
// Benchmark: Bulk Node Insertion
// ============================================================================

#[test]
#[ignore = "heavy benchmark: 100K+ nodes in debug mode"]
fn bench_bulk_node_insertion() {
    print_header("BULK NODE INSERTION BENCHMARKS");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Warmup
    for i in 0..1000 {
        let query = format!("INSERT (:Warmup {{id: {}}})", i);
        session.execute(&query).unwrap();
    }

    // Benchmark: Simple nodes (no properties)
    let start = Instant::now();
    for i in 0..NODE_COUNT {
        let query = format!("INSERT (:Person {{id: {}}})", i);
        session.execute(&query).unwrap();
    }
    let duration = start.elapsed();
    print_result("Simple nodes (1 property)", NODE_COUNT, duration);

    // Benchmark: Nodes with multiple properties
    let db2 = GrafeoDB::new_in_memory();
    let session2 = db2.session();

    let start = Instant::now();
    for i in 0..NODE_COUNT {
        let query = format!(
            "INSERT (:Person {{id: {}, name: 'User{}', age: {}, active: true, score: {}}})",
            i,
            i,
            20 + (i % 60),
            (i % 1000) as f64 / 10.0
        );
        session2.execute(&query).unwrap();
    }
    let duration = start.elapsed();
    print_result("Nodes with 5 properties", NODE_COUNT, duration);

    // Benchmark: Multiple labels
    let db3 = GrafeoDB::new_in_memory();
    let session3 = db3.session();

    let start = Instant::now();
    for i in 0..NODE_COUNT {
        let labels = match i % 4 {
            0 => "Person",
            1 => "Employee",
            2 => "Customer",
            _ => "Admin",
        };
        let query = format!("INSERT (:{} {{id: {}}})", labels, i);
        session3.execute(&query).unwrap();
    }
    let duration = start.elapsed();
    print_result("Nodes with varying labels", NODE_COUNT, duration);
}

// ============================================================================
// Benchmark: Bulk Edge Insertion
// ============================================================================

#[test]
#[ignore = "heavy benchmark: bulk edge creation on 10K nodes"]
fn bench_bulk_edge_insertion() {
    print_header("BULK EDGE INSERTION BENCHMARKS");

    // First, create nodes
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node_count = EDGE_NODES; // Use fewer nodes for edge tests
    let edge_count = node_count * EDGE_MULTIPLIER;

    println!("  Setting up {} nodes...", node_count);
    for i in 0..node_count {
        let query = format!("INSERT (:Person {{id: {}}})", i);
        session.execute(&query).unwrap();
    }

    // Benchmark: Simple edges
    let start = Instant::now();
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (i * 7 + 13) % node_count; // Pseudo-random but deterministic
        if src != dst {
            let query = format!(
                "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) INSERT (a)-[:KNOWS]->(b)",
                src, dst
            );
            let _ = session.execute(&query); // Some may fail due to duplicates
        }
    }
    let duration = start.elapsed();
    print_result("Simple edges (KNOWS)", edge_count, duration);

    // Benchmark: Edges with properties
    let db2 = GrafeoDB::new_in_memory();
    let session2 = db2.session();

    for i in 0..node_count {
        let query = format!("INSERT (:Person {{id: {}}})", i);
        session2.execute(&query).unwrap();
    }

    let start = Instant::now();
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (i * 11 + 17) % node_count;
        if src != dst {
            let query = format!(
                "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) INSERT (a)-[:FRIENDS {{since: {}, weight: {}}}]->(b)",
                src,
                dst,
                2000 + (i % 24),
                (i % 100) as f64 / 100.0
            );
            let _ = session2.execute(&query);
        }
    }
    let duration = start.elapsed();
    print_result("Edges with 2 properties", edge_count, duration);
}

// ============================================================================
// Benchmark: Graph Traversals
// ============================================================================

#[test]
fn bench_graph_traversals() {
    print_header("GRAPH TRAVERSAL BENCHMARKS");

    // Create a scale-free-like graph
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node_count = TRAVERSAL_NODES;
    let edge_count = node_count * 5;

    println!(
        "  Setting up graph: {} nodes, ~{} edges...",
        node_count, edge_count
    );

    // Create nodes
    for i in 0..node_count {
        let query = format!(
            "INSERT (:Person {{id: {}, name: 'User{}', age: {}}})",
            i,
            i,
            20 + (i % 50)
        );
        session.execute(&query).unwrap();
    }

    // Create edges with preferential attachment pattern (more edges to lower IDs)
    for i in 0..edge_count {
        let src = i % node_count;
        // Preferential attachment: more likely to connect to lower-numbered nodes
        let dst = ((i * 37) % (src.max(1))) % node_count;
        if src != dst {
            let query = format!(
                "MATCH (a:Person {{id: {}}}), (b:Person {{id: {}}}) INSERT (a)-[:KNOWS]->(b)",
                src, dst
            );
            let _ = session.execute(&query);
        }
    }

    println!("  Graph setup complete. Running traversal benchmarks...\n");

    // Benchmark: 1-hop traversal (find neighbors)
    let iterations = 1000;
    let start = Instant::now();
    for i in 0..iterations {
        let node_id = i % node_count;
        let query = format!(
            "MATCH (n:Person {{id: {}}})-[:KNOWS]->(friend) RETURN friend.name",
            node_id
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("1-hop traversal (neighbors)", iterations, duration);

    // Benchmark: 2-hop traversal (friends of friends)
    let iterations = 500;
    let start = Instant::now();
    for i in 0..iterations {
        let node_id = i % node_count;
        let query = format!(
            "MATCH (n:Person {{id: {}}})-[:KNOWS]->()-[:KNOWS]->(fof) RETURN DISTINCT fof.id",
            node_id
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("2-hop traversal (friends of friends)", iterations, duration);

    // Benchmark: 3-hop traversal
    let iterations = 100;
    let start = Instant::now();
    for i in 0..iterations {
        let node_id = i % node_count;
        let query = format!(
            "MATCH (n:Person {{id: {}}})-[:KNOWS]->()-[:KNOWS]->()-[:KNOWS]->(target) RETURN DISTINCT target.id",
            node_id
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("3-hop traversal", iterations, duration);

    // Benchmark: Bidirectional 1-hop
    let iterations = 1000;
    let start = Instant::now();
    for i in 0..iterations {
        let node_id = i % node_count;
        let query = format!(
            "MATCH (n:Person {{id: {}}})-[:KNOWS]-(neighbor) RETURN neighbor.id",
            node_id
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("1-hop bidirectional", iterations, duration);
}

// ============================================================================
// Benchmark: Filtering and Predicates
// ============================================================================

#[test]
fn bench_filtering() {
    print_header("FILTERING AND PREDICATE BENCHMARKS");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node_count = FILTER_NODES;

    println!(
        "  Setting up {} nodes with varied properties...",
        node_count
    );

    // Create nodes with varied properties
    for i in 0..node_count {
        let age = 18 + (i % 62);
        let score = (i % 1000) as f64 / 10.0;
        let active = i % 3 != 0;
        let category = match i % 5 {
            0 => "A",
            1 => "B",
            2 => "C",
            3 => "D",
            _ => "E",
        };
        let query = format!(
            "INSERT (:User {{id: {}, age: {}, score: {}, active: {}, category: '{}'}})",
            i, age, score, active, category
        );
        session.execute(&query).unwrap();
    }

    println!("  Setup complete. Running filter benchmarks...\n");

    // Benchmark: Equality filter (high selectivity - ~20% match)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (u:User) WHERE u.category = 'A' RETURN u.id";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("Equality filter (20% selectivity)", iterations, duration);

    // Benchmark: Range filter (low selectivity - ~2% match)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (u:User) WHERE u.age > 75 RETURN u.id";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("Range filter (2% selectivity)", iterations, duration);

    // Benchmark: Range filter (high selectivity - ~50% match)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (u:User) WHERE u.age > 48 RETURN u.id";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("Range filter (50% selectivity)", iterations, duration);

    // Benchmark: Boolean filter
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (u:User) WHERE u.active = true RETURN u.id";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("Boolean filter (67% selectivity)", iterations, duration);

    // Benchmark: Compound filter (AND)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (u:User) WHERE u.age > 30 AND u.active = true RETURN u.id";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("Compound filter (AND)", iterations, duration);

    // Benchmark: Compound filter (OR)
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (u:User) WHERE u.age < 25 OR u.age > 70 RETURN u.id";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("Compound filter (OR)", iterations, duration);
}

// ============================================================================
// Benchmark: Aggregations
// ============================================================================

#[test]
#[ignore = "heavy benchmark: aggregations over 50K nodes"]
fn bench_aggregations() {
    print_header("AGGREGATION BENCHMARKS");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node_count = AGGREGATION_NODES;

    println!("  Setting up {} nodes...", node_count);

    for i in 0..node_count {
        let department = match i % 10 {
            0 => "Engineering",
            1 => "Sales",
            2 => "Marketing",
            3 => "Support",
            4 => "HR",
            5 => "Finance",
            6 => "Legal",
            7 => "Operations",
            8 => "Research",
            _ => "Other",
        };
        let salary = 50000 + (i % 100) * 1000;
        let query = format!(
            "INSERT (:Employee {{id: {}, department: '{}', salary: {}, years: {}}})",
            i,
            department,
            salary,
            i % 30
        );
        session.execute(&query).unwrap();
    }

    println!("  Setup complete. Running aggregation benchmarks...\n");

    // Benchmark: COUNT
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (e:Employee) RETURN COUNT(e)";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("COUNT(*)", iterations, duration);

    // Benchmark: COUNT with filter
    let iterations = 100;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (e:Employee) WHERE e.salary > 100000 RETURN COUNT(e)";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("COUNT with filter", iterations, duration);

    // Benchmark: GROUP BY
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (e:Employee) RETURN e.department, COUNT(e)";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("GROUP BY department", iterations, duration);

    // Benchmark: SUM
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (e:Employee) RETURN SUM(e.salary)";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("SUM(salary)", iterations, duration);

    // Benchmark: AVG
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (e:Employee) RETURN AVG(e.salary)";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("AVG(salary)", iterations, duration);

    // Benchmark: MIN/MAX
    let iterations = 50;
    let start = Instant::now();
    for _ in 0..iterations {
        let query = "MATCH (e:Employee) RETURN MIN(e.salary), MAX(e.salary)";
        let _ = session.execute(query);
    }
    let duration = start.elapsed();
    print_result("MIN/MAX", iterations, duration);
}

// ============================================================================
// Benchmark: Point Lookups
// ============================================================================

#[test]
fn bench_point_lookups() {
    print_header("POINT LOOKUP BENCHMARKS");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node_count = LOOKUP_NODES;

    println!("  Setting up {} nodes...", node_count);

    for i in 0..node_count {
        let query = format!(
            "INSERT (:Record {{id: {}, uuid: 'uuid-{}', data: 'data-{}'}})",
            i, i, i
        );
        session.execute(&query).unwrap();
    }

    println!("  Setup complete. Running lookup benchmarks...\n");

    // Benchmark: Lookup by ID (sequential)
    let iterations = 10_000;
    let start = Instant::now();
    for i in 0..iterations {
        let query = format!("MATCH (r:Record {{id: {}}}) RETURN r", i);
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("ID lookup (sequential)", iterations, duration);

    // Benchmark: Lookup by ID (random)
    let iterations = 10_000;
    let start = Instant::now();
    for i in 0..iterations {
        let id = (i * 7919) % node_count; // Prime-based pseudo-random
        let query = format!("MATCH (r:Record {{id: {}}}) RETURN r", id);
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("ID lookup (random access)", iterations, duration);

    // Benchmark: Lookup by string property
    let iterations = 1_000;
    let start = Instant::now();
    for i in 0..iterations {
        let query = format!("MATCH (r:Record {{uuid: 'uuid-{}'}}) RETURN r", i);
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("String property lookup", iterations, duration);

    // Benchmark: Existence check
    let iterations = 10_000;
    let start = Instant::now();
    for i in 0..iterations {
        let id = (i * 7919) % node_count;
        let query = format!("MATCH (r:Record {{id: {}}}) RETURN COUNT(r)", id);
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("Existence check", iterations, duration);
}

// ============================================================================
// Benchmark: Pattern Matching
// ============================================================================

#[test]
#[ignore = "heavy benchmark: dense graph pattern matching"]
fn bench_pattern_matching() {
    print_header("PATTERN MATCHING BENCHMARKS");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let node_count = PATTERN_NODES;
    let edge_count = node_count * 8;

    println!(
        "  Setting up graph: {} nodes, ~{} edges...",
        node_count, edge_count
    );

    // Create a dense graph suitable for pattern matching
    for i in 0..node_count {
        let query = format!("INSERT (:TestNode {{id: {}}})", i);
        session.execute(&query).unwrap();
    }

    // Create triangles and other patterns
    for i in 0..edge_count {
        let src = i % node_count;
        let dst = (src + 1 + (i % 10)) % node_count;
        if src != dst {
            let query = format!(
                "MATCH (a:TestNode {{id: {}}}), (b:TestNode {{id: {}}}) INSERT (a)-[:LINK]->(b)",
                src, dst
            );
            let _ = session.execute(&query);
        }
    }

    println!("  Setup complete. Running pattern matching benchmarks...\n");

    // Benchmark: Find triangles (a->b->c->a)
    let iterations = 10;
    let start = Instant::now();
    for i in 0..iterations {
        let query = format!(
            "MATCH (a:TestNode {{id: {}}})-[:LINK]->(b)-[:LINK]->(c)-[:LINK]->(a) RETURN a.id, b.id, c.id LIMIT 100",
            i
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("Triangle detection", iterations, duration);

    // Benchmark: Path pattern (a->b->c->d)
    let iterations = 50;
    let start = Instant::now();
    for i in 0..iterations {
        let query = format!(
            "MATCH (a:TestNode {{id: {}}})-[:LINK]->(b)-[:LINK]->(c)-[:LINK]->(d) RETURN d.id LIMIT 100",
            i % node_count
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("4-node path pattern", iterations, duration);

    // Benchmark: Star pattern (hub with multiple neighbors)
    let iterations = 100;
    let start = Instant::now();
    for i in 0..iterations {
        let query = format!(
            "MATCH (hub:TestNode {{id: {}}})-[:LINK]->(n1), (hub)-[:LINK]->(n2) WHERE n1.id <> n2.id RETURN n1.id, n2.id LIMIT 50",
            i % node_count
        );
        let _ = session.execute(&query);
    }
    let duration = start.elapsed();
    print_result("Star pattern (2 branches)", iterations, duration);
}

// ============================================================================
// Benchmark: Mixed Workload (OLTP-like)
// ============================================================================

#[test]
#[ignore = "heavy benchmark: 10K mixed OLTP operations"]
fn bench_mixed_workload() {
    print_header("MIXED WORKLOAD BENCHMARK (OLTP-like)");

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    let initial_nodes = MIXED_NODES;

    println!("  Setting up {} initial nodes...", initial_nodes);

    for i in 0..initial_nodes {
        let query = format!(
            "INSERT (:Account {{id: {}, balance: {}}})",
            i,
            1000 + (i % 10000)
        );
        session.execute(&query).unwrap();
    }

    // Add some relationships
    for i in 0..initial_nodes * 2 {
        let src = i % initial_nodes;
        let dst = (i * 7 + 3) % initial_nodes;
        if src != dst {
            let query = format!(
                "MATCH (a:Account {{id: {}}}), (b:Account {{id: {}}}) INSERT (a)-[:TRANSFER {{amount: {}}}]->(b)",
                src,
                dst,
                (i % 500) + 10
            );
            let _ = session.execute(&query);
        }
    }

    println!("  Setup complete. Running mixed workload...\n");

    let total_ops = 10_000;
    let mut reads = 0usize;
    let mut writes = 0usize;
    let mut node_counter = initial_nodes;

    let start = Instant::now();

    for i in 0..total_ops {
        match i % 10 {
            // 30% writes (inserts)
            0..=2 => {
                let query = format!(
                    "INSERT (:Account {{id: {}, balance: {}}})",
                    node_counter,
                    1000 + (i % 5000)
                );
                let _ = session.execute(&query);
                node_counter += 1;
                writes += 1;
            }
            // 20% updates (property changes)
            3..=4 => {
                let id = i % initial_nodes;
                let query = format!(
                    "MATCH (a:Account {{id: {}}}) SET a.balance = {}",
                    id,
                    2000 + (i % 8000)
                );
                let _ = session.execute(&query);
                writes += 1;
            }
            // 50% reads (queries)
            _ => {
                match i % 5 {
                    0 => {
                        // Point lookup
                        let id = (i * 7919) % initial_nodes;
                        let query = format!("MATCH (a:Account {{id: {}}}) RETURN a.balance", id);
                        let _ = session.execute(&query);
                    }
                    1 => {
                        // Range query
                        let query = "MATCH (a:Account) WHERE a.balance > 5000 RETURN COUNT(a)";
                        let _ = session.execute(query);
                    }
                    2 => {
                        // Traversal
                        let id = i % initial_nodes;
                        let query = format!(
                            "MATCH (a:Account {{id: {}}})-[:TRANSFER]->(b) RETURN b.id",
                            id
                        );
                        let _ = session.execute(&query);
                    }
                    3 => {
                        // Aggregation
                        let query = "MATCH (a:Account) RETURN AVG(a.balance)";
                        let _ = session.execute(query);
                    }
                    _ => {
                        // Multi-hop
                        let id = i % initial_nodes;
                        let query = format!(
                            "MATCH (a:Account {{id: {}}})-[:TRANSFER]->()-[:TRANSFER]->(c) RETURN DISTINCT c.id LIMIT 10",
                            id
                        );
                        let _ = session.execute(&query);
                    }
                }
                reads += 1;
            }
        }
    }

    let duration = start.elapsed();

    println!("  Total operations: {}", total_ops);
    println!("  - Reads:  {} ({}%)", reads, reads * 100 / total_ops);
    println!("  - Writes: {} ({}%)", writes, writes * 100 / total_ops);
    print_result("Mixed workload throughput", total_ops, duration);
}

// ============================================================================
// Benchmark: Concurrent Reads
// ============================================================================

#[test]
fn bench_concurrent_reads() {
    use std::sync::Arc;
    use std::thread;

    print_header("CONCURRENT READ BENCHMARKS");

    let db = Arc::new(GrafeoDB::new_in_memory());
    let node_count = CONCURRENT_NODES;

    println!("  Setting up {} nodes...", node_count);

    {
        let session = db.session();
        for i in 0..node_count {
            let query = format!("INSERT (:Item {{id: {}, value: {}}})", i, i * 10);
            session.execute(&query).unwrap();
        }
    }

    println!("  Setup complete. Running concurrent read benchmarks...\n");

    // Benchmark with varying thread counts
    for num_threads in [1, 2, 4, 8] {
        let ops_per_thread = 5_000;
        let total_ops = ops_per_thread * num_threads;

        let start = Instant::now();

        let handles: Vec<_> = (0..num_threads)
            .map(|t| {
                let db = Arc::clone(&db);
                thread::spawn(move || {
                    let session = db.session();
                    for i in 0..ops_per_thread {
                        let id = (t * 1000 + i) % node_count;
                        let query = format!("MATCH (item:Item {{id: {}}}) RETURN item.value", id);
                        let _ = session.execute(&query);
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let duration = start.elapsed();
        print_result(
            &format!("Concurrent reads ({} threads)", num_threads),
            total_ops,
            duration,
        );
    }
}

// ============================================================================
// Summary Report
// ============================================================================

#[test]
fn bench_summary() {
    print_header("GRAPHOS BENCHMARK SUMMARY");

    println!("\n  Configuration:");
    println!("  - Small scale:  {} nodes", SMALL_SCALE);
    println!("  - Medium scale: {} nodes", MEDIUM_SCALE);
    println!("  - Large scale:  {} nodes", LARGE_SCALE);
    println!("  - Edge multiplier: {}x", EDGE_MULTIPLIER);
    println!("\n  Run individual benchmarks with:");
    println!("    cargo test -p grafeo-engine --release -- bench_ --nocapture");
    println!("\n  Run fast benchmarks only (default):");
    println!("    cargo test -p grafeo-engine --release -- graph_benchmarks --nocapture");
    println!("\n  Run ALL benchmarks including heavy ones:");
    println!(
        "    cargo test -p grafeo-engine --release -- graph_benchmarks --nocapture --include-ignored"
    );
}
