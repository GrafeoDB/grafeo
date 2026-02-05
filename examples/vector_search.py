"""
Vector Search with Grafeo and anywidget-vector

This example demonstrates how to use Grafeo's vector search capabilities
with the anywidget-vector package for 3D visualization of embeddings.

Run with: marimo run vector_search.py
Or convert to notebook: marimo export notebook vector_search.py

Requirements:
    pip install grafeo anywidget-vector marimo numpy
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    mo.md("""
    # Vector Search with Grafeo

    This notebook demonstrates vector similarity search using:
    - **Grafeo**: Graph database with native vector support
    - **anywidget-vector**: 3D vector visualization widget

    We'll create embeddings for documents and find similar items!
    """)
    return (mo,)


@app.cell
def __():
    import numpy as np

    # Simulate embeddings (in real use, these would come from an embedding model)
    np.random.seed(42)

    # Sample documents with simulated 128-dimensional embeddings
    documents = [
        {"title": "Machine Learning Basics", "category": "AI", "year": 2023},
        {"title": "Deep Learning with Python", "category": "AI", "year": 2022},
        {"title": "Natural Language Processing", "category": "AI", "year": 2023},
        {"title": "Computer Vision Fundamentals", "category": "AI", "year": 2021},
        {"title": "Data Structures and Algorithms", "category": "CS", "year": 2020},
        {"title": "Database Systems", "category": "CS", "year": 2019},
        {"title": "Operating Systems Design", "category": "CS", "year": 2021},
        {"title": "Network Programming", "category": "CS", "year": 2022},
        {"title": "Web Development with React", "category": "Web", "year": 2023},
        {"title": "Backend API Design", "category": "Web", "year": 2022},
        {"title": "Cloud Architecture Patterns", "category": "Cloud", "year": 2023},
        {"title": "Kubernetes in Production", "category": "Cloud", "year": 2022},
    ]

    # Generate embeddings - documents in same category have similar embeddings
    category_bases = {
        "AI": np.random.randn(128),
        "CS": np.random.randn(128),
        "Web": np.random.randn(128),
        "Cloud": np.random.randn(128),
    }

    embeddings = []
    for doc in documents:
        base = category_bases[doc["category"]]
        # Add noise to make each embedding unique
        embedding = base + np.random.randn(128) * 0.3
        # Normalize to unit vector
        embedding = embedding / np.linalg.norm(embedding)
        embeddings.append(embedding.tolist())

    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    return category_bases, documents, embeddings, np


@app.cell
def __(documents, embeddings):
    from grafeo import GrafeoDB

    # Create database with vector support
    db = GrafeoDB()

    # Create document nodes with embeddings
    doc_nodes = []
    for doc, embedding in zip(documents, embeddings):
        node = db.create_node(
            ["Document", doc["category"]],
            {
                "title": doc["title"],
                "category": doc["category"],
                "year": doc["year"],
                "embedding": embedding,
            },
        )
        doc_nodes.append(node)

    print(f"Created {len(doc_nodes)} document nodes with vector embeddings")
    return GrafeoDB, db, doc_nodes


@app.cell
def __(db, mo):
    # Query documents by category
    result = db.execute("""
        MATCH (d:Document)
        RETURN d.title as title, d.category as category, d.year as year
        ORDER BY d.category, d.year DESC
    """)

    rows = [f"| {row['title']} | {row['category']} | {row['year']} |" for row in result]

    mo.md(f"""
    ## Document Catalog

    All documents in our collection:

    | Title | Category | Year |
    |-------|----------|------|
    {chr(10).join(rows)}
    """)
    return result, rows


@app.cell
def __(db, documents, embeddings, np):
    # Perform vector similarity search
    # Find documents similar to "Machine Learning Basics"

    query_idx = 0  # Machine Learning Basics
    query_embedding = embeddings[query_idx]
    query_title = documents[query_idx]["title"]

    # Get all documents and their embeddings
    all_docs = db.execute("""
        MATCH (d:Document)
        RETURN id(d) as id, d.title as title, d.embedding as embedding
    """)

    # Calculate cosine similarity
    similarities = []
    for row in all_docs:
        doc_embedding = row["embedding"]
        if doc_embedding and row["title"] != query_title:
            similarity = np.dot(query_embedding, doc_embedding)
            similarities.append((row["title"], similarity, row["id"]))

    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)

    print(f"Query: '{query_title}'")
    print(f"\nTop 5 similar documents:")
    for title, sim, _ in similarities[:5]:
        print(f"  {title}: {sim:.4f}")
    return (
        all_docs,
        query_embedding,
        query_idx,
        query_title,
        similarities,
    )


@app.cell
def __(mo, query_title, similarities):
    rows_sim = [f"| {title} | {sim:.4f} |" for title, sim, _ in similarities[:5]]

    mo.md(f"""
    ## Vector Similarity Search

    Finding documents similar to **"{query_title}"**:

    | Document | Similarity |
    |----------|------------|
    {chr(10).join(rows_sim)}

    Note how AI-related documents cluster together due to similar embeddings!
    """)
    return (rows_sim,)


@app.cell
def __(documents, embeddings, np):
    from anywidget_vector import VectorSpace

    # Reduce dimensions for visualization using simple PCA
    # (In production, use sklearn.decomposition.PCA or UMAP)
    def simple_pca_3d(vectors):
        """Simple PCA to 3D for visualization."""
        X = np.array(vectors)
        X_centered = X - X.mean(axis=0)
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return (U[:, :3] * S[:3]).tolist()

    # Project to 3D
    coords_3d = simple_pca_3d(embeddings)

    # Create points for visualization
    points = []
    for i, (doc, coord) in enumerate(zip(documents, coords_3d)):
        points.append(
            {
                "id": str(i),
                "label": doc["title"],
                "x": coord[0],
                "y": coord[1],
                "z": coord[2],
                "group": doc["category"],
                "metadata": {"year": doc["year"], "category": doc["category"]},
            }
        )

    # Create 3D vector visualization
    vector_widget = VectorSpace(
        points=points,
        height=500,
    )

    vector_widget
    return VectorSpace, coords_3d, points, simple_pca_3d, vector_widget


@app.cell
def __(mo):
    mo.md("""
    ## 3D Embedding Space

    The visualization above shows documents projected into 3D space:
    - **Colors** represent categories (AI, CS, Web, Cloud)
    - **Proximity** indicates semantic similarity
    - Documents in the same category cluster together

    Try rotating and zooming the visualization to explore the embedding space!
    """)
    return ()


@app.cell
def __(db, documents, embeddings, mo, np):
    # Hybrid search: combine vector similarity with property filters

    # Find AI documents from 2022-2023 similar to "Machine Learning Basics"
    query_emb = embeddings[0]

    # First, filter by properties
    filtered_result = db.execute("""
        MATCH (d:Document)
        WHERE d.category = 'AI' AND d.year >= 2022
        RETURN id(d) as id, d.title as title, d.year as year, d.embedding as embedding
    """)

    # Then rank by vector similarity
    hybrid_results = []
    for row in filtered_result:
        if row["embedding"] and row["title"] != documents[0]["title"]:
            similarity = np.dot(query_emb, row["embedding"])
            hybrid_results.append((row["title"], row["year"], similarity))

    hybrid_results.sort(key=lambda x: x[2], reverse=True)

    rows_hybrid = [f"| {title} | {year} | {sim:.4f} |" for title, year, sim in hybrid_results]

    mo.md(f"""
    ## Hybrid Search (Vector + Filters)

    Finding AI documents from 2022+ similar to "Machine Learning Basics":

    | Document | Year | Similarity |
    |----------|------|------------|
    {chr(10).join(rows_hybrid)}

    This combines:
    - **Property filters**: category='AI', year >= 2022
    - **Vector ranking**: cosine similarity to query
    """)
    return filtered_result, hybrid_results, query_emb, rows_hybrid


@app.cell
def __(db, mo):
    # Database statistics
    stats = db.detailed_stats()
    schema = db.schema()

    labels_text = ", ".join([f"{l['name']} ({l['count']})" for l in schema["labels"]])

    mo.md(f"""
    ## Database Statistics

    | Metric | Value |
    |--------|-------|
    | Total Nodes | {stats['node_count']} |
    | Labels | {labels_text} |
    | Properties | {stats['property_key_count']} |
    | Memory | {stats['memory_bytes'] / 1024:.1f} KB |

    Each document stores a 128-dimensional embedding vector alongside its metadata.
    """)
    return labels_text, schema, stats


if __name__ == "__main__":
    app.run()
