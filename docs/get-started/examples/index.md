---
title: Interactive Examples
description: Runnable marimo notebooks showcasing Grafeo features.
---

# Interactive Examples

These examples are [marimo](https://marimo.io/) notebooks you can run locally for an interactive experience. Each page shows the full code with expected output.

## Run Locally

Install the dependencies, then launch any example:

```bash
uv add grafeo anywidget-graph anywidget-vector marimo numpy networkx matplotlib
```

```bash
marimo run examples/graph_visualization.py
```

## Available Examples

<div class="grid cards" markdown>

-   :material-graph:{ .lg .middle } **Graph Visualization**

    ---

    Build a social network, run PageRank and community detection, then visualize it interactively with anywidget-graph.

    [:octicons-arrow-right-24: View Example](graph-visualization.md)

-   :material-vector-point:{ .lg .middle } **Vector Search**

    ---

    Store document embeddings, perform cosine similarity search and hybrid filtering, then explore the embedding space in 3D.

    [:octicons-arrow-right-24: View Example](vector-search.md)

-   :material-shield-alert:{ .lg .middle } **Fraud Detection**

    ---

    Model a transaction network, detect money laundering rings and mule accounts and score risk with PageRank.

    [:octicons-arrow-right-24: View Example](fraud-detection-example.md)

-   :material-swap-horizontal:{ .lg .middle } **NetworkX Integration**

    ---

    Convert Grafeo graphs to NetworkX, run centrality and clustering algorithms and visualize with matplotlib.

    [:octicons-arrow-right-24: View Example](networkx-integration.md)

-   :material-table-arrow-right:{ .lg .middle } **Arrow Export**

    ---

    Bulk export graph data to PyArrow, Polars and pandas. DuckDB integration, lazy evaluation, 10-100x faster at scale.

    [:octicons-arrow-right-24: View Example](arrow-export-example.md)

-   :material-duck:{ .lg .middle } **DuckDB Integration**

    ---

    Combine Grafeo graph storage with DuckDB analytical SQL queries via zero-copy Apache Arrow. JOIN nodes and edges, aggregate, and export to Parquet.

    [:octicons-arrow-right-24: View Example](duckdb-integration.md)

</div>
