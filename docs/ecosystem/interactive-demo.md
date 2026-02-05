---
title: Interactive Demo
description: Live, interactive demos of the anywidget-graph and anywidget-vector widgets running directly in your browser.
tags:
  - demo
  - visualization
  - marimo
  - anywidget
---

# Interactive Demo

These demos run **entirely in your browser** using [marimo](https://marimo.io) and WebAssembly -- no server or Python installation required. Pan, zoom, click, and interact with the widgets below.

!!! info "First load"
    The first time you visit this page, the Python runtime (Pyodide) will be downloaded and
    initialized in the background. This may take a few seconds. Subsequent interactions are instant.

---

## Graph Widget

The [anywidget-graph](anywidget-graph.md) widget renders interactive graph visualizations powered by **Sigma.js**. Nodes are laid out automatically and you can pan, zoom, and click to inspect them.

The example below builds a small social network with people and companies, connected by `KNOWS` and `WORKS_AT` relationships -- the same dataset used in Grafeo's [graph visualization example](https://github.com/GrafeoDB/grafeo/blob/main/examples/graph_visualization.py).

```python {marimo}
import marimo as mo
mo.md(
    """
    ### Social Network Graph

    Drag to pan, scroll to zoom, click a node to select it.
    """
)
```

```python {marimo}
from anywidget_graph import Graph

# Social network: people and companies from Grafeo's example dataset
nodes = [
    {"id": "alice", "label": "Alice", "color": "#6366f1", "size": 8},
    {"id": "bob", "label": "Bob", "color": "#6366f1", "size": 10},
    {"id": "carol", "label": "Carol", "color": "#6366f1", "size": 7},
    {"id": "dave", "label": "Dave", "color": "#6366f1", "size": 9},
    {"id": "eve", "label": "Eve", "color": "#6366f1", "size": 6},
    {"id": "acme", "label": "Acme Corp", "color": "#22c55e", "size": 12},
    {"id": "globex", "label": "Globex Inc", "color": "#22c55e", "size": 11},
]

edges = [
    # KNOWS relationships
    {"source": "alice", "target": "bob", "label": "KNOWS"},
    {"source": "alice", "target": "carol", "label": "KNOWS"},
    {"source": "bob", "target": "carol", "label": "KNOWS"},
    {"source": "bob", "target": "dave", "label": "KNOWS"},
    {"source": "carol", "target": "eve", "label": "KNOWS"},
    {"source": "dave", "target": "eve", "label": "KNOWS"},
    # WORKS_AT relationships
    {"source": "alice", "target": "acme", "label": "WORKS_AT"},
    {"source": "bob", "target": "acme", "label": "WORKS_AT"},
    {"source": "carol", "target": "globex", "label": "WORKS_AT"},
    {"source": "dave", "target": "globex", "label": "WORKS_AT"},
    {"source": "eve", "target": "acme", "label": "WORKS_AT"},
]

graph = Graph(
    nodes=nodes,
    edges=edges,
    height=500,
    dark_mode=True,
    show_toolbar=True,
    show_edge_labels=True,
)
graph
```

??? example "Equivalent Grafeo code"
    With a running Grafeo database, this same graph can be built and queried:

    ```python
    from grafeo import GrafeoDB
    from anywidget_graph import Graph

    db = GrafeoDB()

    db.execute("""
        CREATE (:Person {name: 'Alice'}), (:Person {name: 'Bob'}),
               (:Person {name: 'Carol'}), (:Person {name: 'Dave'}),
               (:Person {name: 'Eve'}),
               (:Company {name: 'Acme Corp'}), (:Company {name: 'Globex Inc'})
    """)

    db.execute("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        CREATE (a)-[:KNOWS]->(b)
        // ... additional relationships
    """)

    result = db.execute("MATCH (n)-[r]->(m) RETURN n, r, m")
    graph = Graph.from_gql(result)
    graph
    ```

---

## Vector Widget

The [anywidget-vector](anywidget-vector.md) widget provides interactive **3D point cloud** visualization powered by **Three.js**. Orbit, pan, and zoom to explore vector spaces -- useful for exploring embeddings, search results, and high-dimensional data.

The example below shows clustered vectors that you might get from a vector similarity search, with three distinct groups colored by category.

```python {marimo}
import marimo as mo
mo.md(
    """
    ### 3D Vector Space

    Click and drag to orbit, scroll to zoom, right-click to pan.
    """
)
```

```python {marimo}
import math
from anywidget_vector import VectorSpace

# Generate three clusters of points to simulate embedding groups
points = []

# Cluster 1: "Documents" -- centered around (0.5, 0.8, 0.3)
for i in range(15):
    angle = i * 0.42
    points.append({
        "id": f"doc_{i}",
        "label": f"Document {i}",
        "x": 0.5 + 0.15 * math.cos(angle),
        "y": 0.8 + 0.15 * math.sin(angle),
        "z": 0.3 + 0.1 * math.sin(angle * 1.5),
        "color": "#6366f1",
        "group": "documents",
    })

# Cluster 2: "Images" -- centered around (-0.6, -0.2, 0.7)
for i in range(12):
    angle = i * 0.52
    points.append({
        "id": f"img_{i}",
        "label": f"Image {i}",
        "x": -0.6 + 0.2 * math.cos(angle),
        "y": -0.2 + 0.12 * math.sin(angle),
        "z": 0.7 + 0.15 * math.cos(angle * 0.8),
        "color": "#f59e0b",
        "group": "images",
    })

# Cluster 3: "Queries" -- centered around (0.0, -0.5, -0.6)
for i in range(10):
    angle = i * 0.63
    points.append({
        "id": f"query_{i}",
        "label": f"Query {i}",
        "x": 0.0 + 0.18 * math.sin(angle),
        "y": -0.5 + 0.18 * math.cos(angle),
        "z": -0.6 + 0.12 * math.sin(angle * 1.2),
        "color": "#22c55e",
        "group": "queries",
    })

vector_widget = VectorSpace(
    points=points,
    height=500,
    background="#1a1a2e",
    show_axes=True,
    show_grid=True,
    axis_labels={"x": "Dim 1", "y": "Dim 2", "z": "Dim 3"},
)
vector_widget
```

??? example "Equivalent Grafeo vector search code"
    With Grafeo's built-in vector search, you can query embeddings and visualize results:

    ```python
    from grafeo import GrafeoDB
    from anywidget_vector import VectorSpace

    db = GrafeoDB()

    # Create nodes with vector embeddings
    db.execute("""
        CREATE (:Document {title: 'Graph databases', embedding: [0.5, 0.8, 0.3]}),
               (:Document {title: 'Vector search', embedding: [0.52, 0.78, 0.35]}),
               (:Image {title: 'Network diagram', embedding: [-0.6, -0.2, 0.7]})
    """)

    # Find nearest neighbors
    results = db.vector_search(
        query_vector=[0.5, 0.8, 0.3],
        index_name="embeddings",
        top_k=10
    )

    # Visualize
    points = [
        {"id": str(r.id), "x": r.embedding[0], "y": r.embedding[1], "z": r.embedding[2]}
        for r in results
    ]
    VectorSpace(points=points)
    ```

---

## How it works

These demos use the [mkdocs-marimo](https://github.com/marimo-team/mkdocs-marimo) plugin.
Each `python {marimo}` code block is executed client-side via [Pyodide](https://pyodide.org) (Python compiled to WebAssembly). The anywidget framework bridges the Python widget state to the JavaScript renderers (Sigma.js / Three.js) that run natively in the browser.

Since everything runs in your browser, there is **no backend server** -- making it ideal for static documentation sites hosted on GitHub Pages.
