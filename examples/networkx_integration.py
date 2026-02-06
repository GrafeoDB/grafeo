"""
NetworkX Integration with Grafeo

This example demonstrates seamless integration between Grafeo and NetworkX
for visualization and additional graph algorithms.

Run with: marimo run networkx_integration.py

Requirements:
    pip install grafeo networkx matplotlib marimo
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    mo.md("""
    # NetworkX Integration

    Grafeo provides seamless integration with NetworkX, the popular Python graph library.
    This enables:

    - **Visualization** with matplotlib
    - **Additional algorithms** not in Grafeo
    - **Interoperability** with the broader Python ecosystem
    - **Two-way data flow** between Grafeo and NetworkX
    """)
    return (mo,)


@app.cell
def __():
    from grafeo import GrafeoDB

    # Create a sample graph
    db = GrafeoDB()

    # Build a small social network
    people = {}
    for name, age, city in [
        ("Alice", 30, "Seattle"),
        ("Bob", 35, "Portland"),
        ("Carol", 28, "Seattle"),
        ("Dave", 40, "San Francisco"),
        ("Eve", 32, "Portland"),
        ("Frank", 45, "Seattle"),
        ("Grace", 29, "Portland"),
        ("Henry", 38, "San Francisco"),
    ]:
        node = db.create_node(["Person"], {"name": name, "age": age, "city": city})
        people[name] = node

    # Create relationships
    connections = [
        ("Alice", "Bob"),
        ("Alice", "Carol"),
        ("Bob", "Carol"),
        ("Bob", "Dave"),
        ("Carol", "Eve"),
        ("Dave", "Eve"),
        ("Dave", "Frank"),
        ("Eve", "Grace"),
        ("Frank", "Grace"),
        ("Frank", "Henry"),
        ("Grace", "Henry"),
    ]

    for p1, p2 in connections:
        db.create_edge(people[p1].id, people[p2].id, "KNOWS")

    print(f"Created graph with {db.node_count} nodes and {db.edge_count} edges")
    return GrafeoDB, connections, db, people


@app.cell
def __(db, mo):
    # Convert to NetworkX
    nx_adapter = db.as_networkx(directed=False)

    mo.md(f"""
    ## Convert Grafeo to NetworkX

    Use `db.as_networkx()` to get a NetworkX-compatible adapter:

    ```python
    nx_adapter = db.as_networkx(directed=False)
    ```

    The adapter provides NetworkX-like methods:
    - `number_of_nodes()`: {nx_adapter.number_of_nodes()}
    - `number_of_edges()`: {nx_adapter.number_of_edges()}
    """)
    return (nx_adapter,)


@app.cell
def __(mo, nx_adapter):
    # Convert to actual NetworkX graph
    G = nx_adapter.to_networkx()

    mo.md(f"""
    ## Full NetworkX Graph

    Convert to a full NetworkX graph object:

    ```python
    G = nx_adapter.to_networkx()
    ```

    This creates a `networkx.Graph` (or `DiGraph` if directed=True) with all
    nodes, edges, and their properties.

    Graph type: `{type(G).__name__}`
    """)
    return (G,)


@app.cell
def __(G, db, mo, people):
    import networkx as nx

    # Run NetworkX algorithms
    # Betweenness centrality
    betweenness = nx.betweenness_centrality(G)

    # Clustering coefficient
    clustering = nx.clustering(G)

    # Get node names
    node_names = {}
    for name, node in people.items():
        node_names[node.id] = name

    # Format results
    betweenness_rows = []
    for node_id in sorted(
        betweenness.keys(), key=lambda x: betweenness[x], reverse=True
    )[:5]:
        name = node_names.get(node_id, f"Node {node_id}")
        betweenness_rows.append(
            f"| {name} | {betweenness[node_id]:.4f} | {clustering[node_id]:.4f} |"
        )

    mo.md(f"""
    ## NetworkX Algorithms

    Running algorithms from NetworkX on our Grafeo graph:

    ### Betweenness Centrality & Clustering

    | Person | Betweenness | Clustering |
    |--------|-------------|------------|
    {chr(10).join(betweenness_rows)}

    **Betweenness centrality** measures how often a node lies on shortest paths.
    **Clustering coefficient** measures how connected a node's neighbors are.
    """)
    return betweenness, betweenness_rows, clustering, node_names, nx


@app.cell
def __(G, nx):
    # More NetworkX algorithms
    # Connected components
    components = list(nx.connected_components(G))

    # Diameter (longest shortest path)
    diameter = nx.diameter(G)

    # Average path length
    avg_path = nx.average_shortest_path_length(G)

    # Density
    density = nx.density(G)

    print(f"Connected components: {len(components)}")
    print(f"Graph diameter: {diameter}")
    print(f"Average path length: {avg_path:.2f}")
    print(f"Graph density: {density:.4f}")
    return avg_path, components, density, diameter


@app.cell
def __(avg_path, components, density, diameter, mo):
    mo.md(f"""
    ## Graph Metrics

    | Metric | Value | Interpretation |
    |--------|-------|----------------|
    | Connected Components | {len(components)} | The graph is {"connected" if len(components) == 1 else "disconnected"} |
    | Diameter | {diameter} | Maximum distance between any two nodes |
    | Average Path Length | {avg_path:.2f} | Average steps to reach any node |
    | Density | {density:.4f} | Ratio of actual to possible edges |
    """)
    return ()


@app.cell
def __(G, mo, node_names, nx):
    # Visualize with matplotlib
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Layout
    pos = nx.spring_layout(G, seed=42)

    # Plot 1: Basic graph
    ax1 = axes[0]
    nx.draw(
        G,
        pos,
        ax=ax1,
        with_labels=True,
        labels={n: node_names.get(n, str(n)) for n in G.nodes()},
        node_color="lightblue",
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    ax1.set_title("Social Network Graph")

    # Plot 2: Colored by betweenness centrality
    ax2 = axes[1]
    bc = nx.betweenness_centrality(G)
    node_colors = [bc[n] for n in G.nodes()]
    nx.draw(
        G,
        pos,
        ax=ax2,
        with_labels=True,
        labels={n: node_names.get(n, str(n)) for n in G.nodes()},
        node_color=node_colors,
        cmap=plt.cm.Reds,
        node_size=700,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
    )
    ax2.set_title("Colored by Betweenness Centrality")

    plt.tight_layout()
    mo.mpl.interactive(fig)
    return ax1, ax2, bc, fig, node_colors, plt, pos


@app.cell
def __(mo):
    mo.md("""
    ## Visualization with Matplotlib

    The plots above show:
    - **Left**: Basic network visualization
    - **Right**: Nodes colored by betweenness centrality (darker = more central)

    This demonstrates how easy it is to use NetworkX's visualization
    capabilities with data from Grafeo.
    """)
    return ()


@app.cell
def __(db, mo, nx):
    # Compare Grafeo vs NetworkX algorithm performance

    # Grafeo's built-in algorithms
    grafeo_pagerank = db.algorithms.pagerank(damping=0.85)

    # NetworkX PageRank on converted graph
    G_new = db.as_networkx().to_networkx()
    nx_pagerank = nx.pagerank(G_new, alpha=0.85)

    mo.md("""
    ## Algorithm Comparison

    Both Grafeo and NetworkX can compute PageRank:

    ```python
    # Grafeo (Rust-native, optimized)
    grafeo_pr = db.algorithms.pagerank(damping=0.85)

    # NetworkX (Python)
    nx_pr = nx.pagerank(G, alpha=0.85)
    ```

    For large graphs, Grafeo's Rust implementation is significantly faster,
    but NetworkX provides many more algorithms for specialized analyses.

    **Best practice:** Use Grafeo for common algorithms (PageRank, BFS, DFS,
    shortest paths) and NetworkX for specialized ones (community detection
    variants, centrality measures, motif finding).
    """)
    return G_new, grafeo_pagerank, nx_pagerank


@app.cell
def __(mo):
    mo.md("""
    ## When to Use Each

    | Use Case | Recommendation |
    |----------|----------------|
    | Large graphs (1M+ nodes) | Grafeo algorithms |
    | Visualization | NetworkX + matplotlib |
    | Specialized algorithms | NetworkX |
    | Graph storage/queries | Grafeo |
    | Interactive exploration | Grafeo + anywidget-graph |
    | Production applications | Grafeo (embedded, no server) |

    The two libraries complement each other well!
    """)
    return ()


if __name__ == "__main__":
    app.run()
