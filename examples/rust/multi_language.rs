//! Multi-language queries: same data, different query languages.
//!
//! Run with: `cargo run -p grafeo-examples --bin multi_language --features full`

use grafeo::GrafeoDB;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // ── Build a movie database ────────────────────────────────────
    // Insert actors and movies using GQL (the default language).
    session.execute("INSERT (:Person {name: 'Vincent', born: 1954})")?;
    session.execute("INSERT (:Person {name: 'Jules', born: 1948})")?;
    session.execute("INSERT (:Person {name: 'Mia', born: 1963})")?;

    session.execute("INSERT (:Movie {title: 'Pulp Fiction', year: 1994})")?;
    session.execute("INSERT (:Movie {title: 'Kill Bill', year: 2003})")?;

    session.execute(
        "MATCH (a:Person {name: 'Vincent'}), (m:Movie {title: 'Pulp Fiction'})
         INSERT (a)-[:ACTED_IN {role: 'Vincent Vega'}]->(m)",
    )?;
    session.execute(
        "MATCH (a:Person {name: 'Jules'}), (m:Movie {title: 'Pulp Fiction'})
         INSERT (a)-[:ACTED_IN {role: 'Jules Winnfield'}]->(m)",
    )?;
    session.execute(
        "MATCH (a:Person {name: 'Mia'}), (m:Movie {title: 'Pulp Fiction'})
         INSERT (a)-[:ACTED_IN {role: 'Mia Wallace'}]->(m)",
    )?;
    session.execute(
        "MATCH (a:Person {name: 'Mia'}), (m:Movie {title: 'Kill Bill'})
         INSERT (a)-[:ACTED_IN {role: 'The Bride'}]->(m)",
    )?;

    println!(
        "Movie database: {} nodes, {} edges\n",
        db.node_count(),
        db.edge_count()
    );

    // ── GQL (ISO standard) ────────────────────────────────────────
    // GQL is the default query language, used with session.execute().
    let result = session.execute("MATCH (p:Person) RETURN p.name, p.born ORDER BY p.name")?;

    println!("=== GQL ===");
    println!("  MATCH (p:Person) RETURN p.name, p.born ORDER BY p.name");
    for row in result.iter() {
        println!(
            "    {} (born {})",
            row[0].as_str().unwrap_or("?"),
            row[1].as_int64().unwrap_or(0)
        );
    }

    // ── Cypher ────────────────────────────────────────────────────
    // Cypher uses the same MATCH/RETURN syntax as GQL for basic queries.
    let result =
        session.execute_cypher("MATCH (p:Person) RETURN p.name, p.born ORDER BY p.name")?;

    println!("\n=== Cypher ===");
    println!("  MATCH (p:Person) RETURN p.name, p.born ORDER BY p.name");
    for row in result.iter() {
        println!(
            "    {} (born {})",
            row[0].as_str().unwrap_or("?"),
            row[1].as_int64().unwrap_or(0)
        );
    }

    // ── SQL/PGQ (SQL:2023 GRAPH_TABLE) ────────────────────────────
    // SQL/PGQ wraps graph patterns in GRAPH_TABLE() with COLUMNS for projection.
    // WHERE goes outside GRAPH_TABLE, filtering on column aliases.
    let result = session.execute_sql(
        "SELECT * FROM GRAPH_TABLE (
             MATCH (p:Person)
             COLUMNS (p.name AS name, p.born AS born)
         ) AS g
         ORDER BY g.name",
    )?;

    println!("\n=== SQL/PGQ ===");
    println!("  SELECT * FROM GRAPH_TABLE (");
    println!("      MATCH (p:Person)");
    println!("      COLUMNS (p.name AS name, p.born AS born)");
    println!("  ) AS g ORDER BY g.name");
    for row in result.iter() {
        println!(
            "    {} (born {})",
            row[0].as_str().unwrap_or("?"),
            row[1].as_int64().unwrap_or(0)
        );
    }

    // ── Dynamic dispatch with execute_language() ──────────────────
    // When the language is determined at runtime (e.g., from user input
    // or configuration), use execute_language() for dynamic dispatch.
    println!("\n=== Dynamic dispatch ===");
    println!("  Query: MATCH (m:Movie) RETURN m.title ORDER BY m.title\n");

    let query = "MATCH (m:Movie) RETURN m.title ORDER BY m.title";
    for lang in &["gql", "cypher"] {
        let result = session.execute_language(query, lang, None)?;
        let titles: Vec<_> = result.iter().filter_map(|row| row[0].as_str()).collect();
        println!("  {:<8} -> {}", lang, titles.join(", "));
    }

    println!("\nDone!");
    Ok(())
}
