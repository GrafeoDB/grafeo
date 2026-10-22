//! Graph algorithms via CALL procedures.
//!
//! Run with: `cargo run -p grafeo-examples --bin algorithms`

use grafeo::{GrafeoDB, NodeId};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // ── Build a collaboration network ─────────────────────────────
    // A small graph of people connected by WORKS_WITH edges.
    // Weights represent collaboration intensity.
    let people = ["Alix", "Gus", "Vincent", "Jules", "Mia", "Butch"];
    for name in &people {
        session.execute(&format!("INSERT (:Person {{name: '{name}'}})"))?;
    }

    let edges = [
        ("Alix", "Gus", 3),
        ("Alix", "Vincent", 2),
        ("Gus", "Vincent", 5),
        ("Gus", "Jules", 1),
        ("Vincent", "Jules", 4),
        ("Jules", "Mia", 2),
        ("Mia", "Butch", 3),
    ];

    for (from, to, weight) in &edges {
        session.execute(&format!(
            "MATCH (a:Person {{name: '{from}'}}), (b:Person {{name: '{to}'}})
             INSERT (a)-[:WORKS_WITH {{weight: {weight}}}]->(b)"
        ))?;
    }

    println!(
        "Created collaboration network: {} people, {} connections\n",
        people.len(),
        edges.len()
    );

    // ── PageRank ──────────────────────────────────────────────────
    // Identifies the most "influential" nodes in the graph.
    // Parameters are passed as a map literal: {key: value}.
    let result = session.execute("CALL grafeo.pagerank({damping: 0.85, max_iterations: 20})")?;

    // Collect and sort by score descending
    let mut scores: Vec<_> = result
        .iter()
        .map(|row| {
            let node_id = row[0].as_int64().unwrap_or(0);
            let score = row[1].as_float64().unwrap_or(0.0);
            (node_id, score)
        })
        .collect();
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("PageRank (most influential people):");
    for (node_id, score) in &scores {
        let name = get_person_name(&db, *node_id);
        println!("  {:<10} {:.4}", name, score);
    }

    // ── Connected Components ──────────────────────────────────────
    // Finds groups of nodes that are all reachable from each other.
    let result = session.execute("CALL grafeo.connected_components()")?;

    println!("\nConnected Components:");
    for row in result.iter() {
        let node_id = row[0].as_int64().unwrap_or(0);
        let component = row[1].as_int64().unwrap_or(0);
        let name = get_person_name(&db, node_id);
        println!("  {:<10} component {}", name, component);
    }

    // ── Louvain Community Detection ───────────────────────────────
    // Detects communities by optimizing modularity.
    let result = session.execute("CALL grafeo.louvain()")?;

    println!("\nLouvain Communities:");
    for row in result.iter() {
        let node_id = row[0].as_int64().unwrap_or(0);
        let community = row[1].as_int64().unwrap_or(0);
        let name = get_person_name(&db, node_id);
        println!("  {:<10} community {}", name, community);
    }

    // ── Degree Centrality ─────────────────────────────────────────
    // Measures connectivity: in-degree, out-degree, and total.
    // Columns: node_id, in_degree, out_degree, total_degree
    let result = session.execute("CALL grafeo.degree_centrality()")?;

    // Collect and sort by total_degree descending
    let mut degrees: Vec<_> = result
        .iter()
        .map(|row| {
            let node_id = row[0].as_int64().unwrap_or(0);
            let in_deg = row[1].as_int64().unwrap_or(0);
            let out_deg = row[2].as_int64().unwrap_or(0);
            let total = row[3].as_int64().unwrap_or(0);
            (node_id, in_deg, out_deg, total)
        })
        .collect();
    degrees.sort_by_key(|row| std::cmp::Reverse(row.3));

    println!("\nDegree Centrality:");
    println!("  {:<10} {:<5} {:<6} Total", "Name", "In", "Out");
    println!("  {}", "-".repeat(30));
    for (node_id, in_deg, out_deg, total) in &degrees {
        let name = get_person_name(&db, *node_id);
        println!("  {:<10} {:<5} {:<6} {}", name, in_deg, out_deg, total);
    }

    println!("\nDone!");
    Ok(())
}

/// Look up a person's name by their raw node ID from CALL procedure results.
fn get_person_name(db: &GrafeoDB, raw_id: i64) -> String {
    let node_id = NodeId::from(raw_id as u64);
    db.get_node(node_id)
        .and_then(|n| {
            n.get_property("name")
                .and_then(|v| v.as_str().map(String::from))
        })
        .unwrap_or_else(|| "?".to_string())
}
