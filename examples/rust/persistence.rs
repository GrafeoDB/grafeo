//! Persistence: WAL-backed storage and snapshot export/import.
//!
//! Run with: `cargo run -p grafeo-examples --bin persistence --features storage`

use grafeo::{GrafeoDB, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Part 1: WAL-backed persistence ────────────────────────────
    // Create a persistent database that writes to disk via WAL
    // (write-ahead log). Data survives process restarts.
    let temp_dir = std::env::temp_dir().join("grafeo_persistence_example");

    // Clean up from any previous run
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir)?;
    }

    println!("Creating persistent database at: {}", temp_dir.display());

    // Open creates the directory and WAL files automatically
    let db = GrafeoDB::open(&temp_dir)?;
    let session = db.session();

    // Insert some data
    session.execute("INSERT (:Person {name: 'Alix', city: 'Utrecht'})")?;
    session.execute("INSERT (:Person {name: 'Gus', city: 'Leiden'})")?;
    session.execute(
        "MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
         INSERT (a)-[:KNOWS]->(b)",
    )?;

    let count: i64 = session
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("Inserted {count} people");

    // Close the database (flushes WAL)
    db.close()?;
    println!("Database closed\n");

    // Reopen the same path: data should still be there
    let db2 = GrafeoDB::open(&temp_dir)?;
    let session2 = db2.session();

    let count: i64 = session2
        .execute("MATCH (p:Person) RETURN COUNT(p)")?
        .scalar()?;
    println!("Reopened database: {count} people found");

    let result = session2.execute(
        "MATCH (p:Person)
         RETURN p.name, p.city
         ORDER BY p.name",
    )?;
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let city = row[1].as_str().unwrap_or("?");
        println!("  {} ({})", name, city);
    }

    db2.close()?;

    // ── Part 2: Snapshot export/import ─────────────────────────────
    // Snapshots serialize the entire graph to a byte array.
    // Useful for backups, replication, or transferring data.
    println!("\n--- Snapshot round-trip ---\n");

    let db = GrafeoDB::new_in_memory();
    let alix = db.create_node(&["Person"]).unwrap();
    db.set_node_property(alix, "name", Value::from("Alix"))
        .unwrap();
    db.set_node_property(alix, "age", Value::from(30_i64))
        .unwrap();

    let gus = db.create_node(&["Person"]).unwrap();
    db.set_node_property(gus, "name", Value::from("Gus"))
        .unwrap();
    db.set_node_property(gus, "age", Value::from(28_i64))
        .unwrap();

    db.create_edge(alix, gus, "KNOWS").unwrap();

    println!(
        "Original: {} nodes, {} edges",
        db.node_count(),
        db.edge_count()
    );

    // Export to bytes
    let snapshot = db.export_snapshot()?;
    println!("Snapshot size: {} bytes", snapshot.len());

    // Import into a fresh database
    let restored = GrafeoDB::import_snapshot(&snapshot)?;
    println!(
        "Restored: {} nodes, {} edges",
        restored.node_count(),
        restored.edge_count()
    );

    // Verify data integrity
    let session = restored.session();
    let result = session.execute(
        "MATCH (p:Person)
         RETURN p.name, p.age
         ORDER BY p.name",
    )?;
    println!("\nRestored data:");
    for row in result.iter() {
        let name = row[0].as_str().unwrap_or("?");
        let age = row[1].as_int64().unwrap_or(0);
        println!("  {} (age {})", name, age);
    }

    // Clean up the temp directory
    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir)?;
    }
    println!("\nCleaned up temp directory");

    println!("\nDone!");
    Ok(())
}
