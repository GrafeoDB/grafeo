# Grafeo Examples

Interactive examples demonstrating Grafeo's capabilities.

## Quick Start

Install dependencies:

```bash
uv add grafeo anywidget-graph anywidget-vector marimo numpy networkx
# or: pip install grafeo anywidget-graph anywidget-vector marimo numpy networkx
```

Run an example:

```bash
# Using Marimo (recommended)
marimo run graph_visualization.py

# Or convert to Jupyter notebook
marimo export notebook graph_visualization.py
```

## Examples

### Graph Visualization (`graph_visualization.py`)

Interactive graph visualization using anywidget-graph. Demonstrates:
- Creating nodes and edges
- Pattern matching queries
- PageRank algorithm
- Community detection (Louvain)
- Shortest path analysis

### Vector Search (`vector_search.py`)

Vector similarity search with 3D visualization. Demonstrates:
- Storing vector embeddings
- Cosine similarity search
- Hybrid search (vectors + filters)
- PCA projection for visualization

### Fraud Detection (`fraud_detection.py`)

Graph-based fraud detection patterns. Demonstrates:
- Building transaction graphs
- Detecting suspicious patterns
- Ring detection algorithms
- Risk scoring with PageRank

### NetworkX Integration (`networkx_integration.py`)

Seamless integration with NetworkX. Demonstrates:
- Converting Grafeo graphs to NetworkX
- Using NetworkX algorithms
- Visualization with matplotlib
- Bidirectional data flow

## Using with Jupyter

Convert to notebook format:

```bash
marimo export notebook graph_visualization.py > graph_visualization.ipynb
```

Or run directly with Jupyter:

```bash
jupyter lab
# Open the .ipynb files
```

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| grafeo | >=0.3.4 | Graph database |
| anywidget-graph | >=0.2.0 | Graph visualization |
| anywidget-vector | >=0.2.0 | Vector visualization |
| marimo | >=0.19.0 | Interactive notebooks |
| numpy | >=1.24 | Vector operations |
| networkx | >=3.0 | Graph algorithms (optional) |
| matplotlib | >=3.0 | Plotting (optional) |

## Learn More

- [Grafeo Documentation](https://grafeo.dev)
- [Tutorials](https://grafeo.dev/tutorials/)
- [API Reference](https://grafeo.dev/api/)
