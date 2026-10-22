//! Vector similarity search with HNSW indexes.
//!
//! Run with: `cargo run -p grafeo-examples --bin vector_search`

use grafeo::{GrafeoDB, NodeId, Value};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = GrafeoDB::new_in_memory();

    // ── Create documents with embeddings ──────────────────────────
    // In a real application, embeddings come from a model (e.g., OpenAI,
    // sentence-transformers). Here we use 4-dimensional vectors for clarity.
    let documents: &[(&str, [f32; 4])] = &[
        ("Graph Databases Explained", [0.9, 0.1, 0.2, 0.0]),
        ("Introduction to Rust", [0.1, 0.8, 0.3, 0.1]),
        ("Network Analysis with Graphs", [0.8, 0.2, 0.1, 0.1]),
        ("Rust for Systems Programming", [0.2, 0.9, 0.2, 0.0]),
        ("Knowledge Graphs in AI", [0.7, 0.3, 0.5, 0.4]),
        ("Memory Safety in Rust", [0.1, 0.7, 0.4, 0.2]),
    ];

    for (title, embedding) in documents {
        // Create a document node using the programmatic API
        let node_id = db.create_node(&["Document"]).unwrap();
        db.set_node_property(node_id, "title", Value::from(*title))
            .unwrap();

        // Store the embedding as a Vector property.
        // Value::Vector wraps an Arc<[f32]>, created from a Vec.
        db.set_node_property(
            node_id,
            "embedding",
            Value::Vector(embedding.to_vec().into()),
        )
        .unwrap();
    }

    println!("Created {} documents with embeddings\n", documents.len());

    // ── Build an HNSW index ───────────────────────────────────────
    // The index enables fast approximate nearest-neighbor search.
    // Parameters: label, property, dimensions, distance metric, m, ef_construction
    //   - dimensions: inferred from data if None
    //   - metric: "cosine", "euclidean", or "dot_product"
    //   - m: max connections per layer (None = default 16)
    //   - ef_construction: build-time search width (None = default 200)
    db.create_vector_index(
        "Document",
        "embedding",
        Some(4),
        Some("cosine"),
        None,
        None,
        None,
    )?;
    println!("Built HNSW index (cosine similarity, 4 dimensions)");

    // ── Search for similar documents ──────────────────────────────
    // Query: find the 3 documents most similar to a "graph" concept
    let query_vector = [0.85_f32, 0.15, 0.2, 0.1];
    let results = db.vector_search(
        "Document",    // label to search
        "embedding",   // property containing vectors
        &query_vector, // query vector
        3,             // k: number of nearest neighbors
        None,          // ef: search-time quality (None = default)
        None,          // filters: optional property filters
    )?;

    println!("\nTop 3 documents similar to 'graph' query:");
    println!("{:<40} Distance", "Title");
    println!("{}", "-".repeat(55));
    for (node_id, distance) in &results {
        let title = get_title(&db, *node_id);
        println!("{:<40} {:.4}", title, distance);
    }

    // ── Search with a different query ─────────────────────────────
    // Query: find documents about "Rust programming"
    let query_vector = [0.15_f32, 0.85, 0.3, 0.1];
    let results = db.vector_search("Document", "embedding", &query_vector, 3, None, None)?;

    println!("\nTop 3 documents similar to 'Rust' query:");
    println!("{:<40} Distance", "Title");
    println!("{}", "-".repeat(55));
    for (node_id, distance) in &results {
        let title = get_title(&db, *node_id);
        println!("{:<40} {:.4}", title, distance);
    }

    println!("\nDone!");
    Ok(())
}

/// Look up a document's title by its node ID.
fn get_title(db: &GrafeoDB, node_id: NodeId) -> String {
    db.get_node(node_id)
        .and_then(|n| {
            n.get_property("title")
                .and_then(|v| v.as_str().map(String::from))
        })
        .unwrap_or_else(|| "?".to_string())
}
