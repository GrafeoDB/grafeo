"""
Graph Visualization with Grafeo and anywidget-graph

This example demonstrates how to visualize graph data using Grafeo's Python API
and the anywidget-graph package for interactive visualization.

Run with: marimo run graph_visualization.py
Or convert to notebook: marimo export notebook graph_visualization.py

Requirements:
    pip install grafeo anywidget-graph marimo
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    mo.md("""
    # Graph Visualization with Grafeo

    This notebook demonstrates interactive graph visualization using:
    - **Grafeo**: High-performance graph database
    - **anywidget-graph**: Interactive graph widget for notebooks

    Let's build a social network and explore it visually!
    """)
    return (mo,)


@app.cell
def __():
    from grafeo import GrafeoDB

    # Create an in-memory database
    db = GrafeoDB()

    # Create some people
    alice = db.create_node(["Person"], {"name": "Alice", "age": 30, "city": "Seattle"})
    bob = db.create_node(["Person"], {"name": "Bob", "age": 35, "city": "Portland"})
    carol = db.create_node(["Person"], {"name": "Carol", "age": 28, "city": "Seattle"})
    dave = db.create_node(["Person"], {"name": "Dave", "age": 40, "city": "San Francisco"})
    eve = db.create_node(["Person"], {"name": "Eve", "age": 32, "city": "Portland"})

    # Create some companies
    acme = db.create_node(["Company"], {"name": "Acme Corp", "industry": "Tech"})
    globex = db.create_node(["Company"], {"name": "Globex Inc", "industry": "Finance"})

    # Create relationships
    db.create_edge(alice.id, bob.id, "KNOWS", {"since": 2020})
    db.create_edge(alice.id, carol.id, "KNOWS", {"since": 2019})
    db.create_edge(bob.id, carol.id, "KNOWS", {"since": 2021})
    db.create_edge(bob.id, dave.id, "KNOWS", {"since": 2018})
    db.create_edge(carol.id, eve.id, "KNOWS", {"since": 2022})
    db.create_edge(dave.id, eve.id, "KNOWS", {"since": 2020})

    # Employment relationships
    db.create_edge(alice.id, acme.id, "WORKS_AT", {"role": "Engineer"})
    db.create_edge(bob.id, acme.id, "WORKS_AT", {"role": "Manager"})
    db.create_edge(carol.id, globex.id, "WORKS_AT", {"role": "Analyst"})
    db.create_edge(dave.id, globex.id, "WORKS_AT", {"role": "Director"})
    db.create_edge(eve.id, acme.id, "WORKS_AT", {"role": "Designer"})

    print(f"Created {db.node_count} nodes and {db.edge_count} edges")
    return acme, alice, bob, carol, dave, db, eve, globex


@app.cell
def __(db, mo):
    # Query to find friends of friends
    result = db.execute("""
        MATCH (p:Person)-[:KNOWS]->(friend)-[:KNOWS]->(fof:Person)
        WHERE p.name = 'Alice' AND p <> fof
        RETURN DISTINCT p.name as person, fof.name as friend_of_friend
    """)

    mo.md(f"""
    ## Friends of Friends Query

    Finding Alice's friends-of-friends:

    | Person | Friend of Friend |
    |--------|-----------------|
    {"".join(f"| {row['person']} | {row['friend_of_friend']} |" + chr(10) for row in result)}
    """)
    return (result,)


@app.cell
def __(db):
    from anywidget_graph import Graph

    # Query all nodes and relationships for visualization
    viz_result = db.execute("""
        MATCH (n)-[r]->(m)
        RETURN n, r, m
    """)

    # Extract nodes and edges
    nodes = viz_result.nodes()
    edges = viz_result.edges()

    # Convert to anywidget-graph format
    graph_nodes = []
    seen_ids = set()
    for node in nodes:
        if node.id not in seen_ids:
            seen_ids.add(node.id)
            labels = node.labels
            props = node.properties
            graph_nodes.append({
                "id": str(node.id),
                "label": props.get("name", f"Node {node.id}"),
                "group": labels[0] if labels else "Unknown",
                "properties": props,
            })

    graph_edges = []
    for edge in edges:
        graph_edges.append({
            "source": str(edge.source_id),
            "target": str(edge.target_id),
            "label": edge.edge_type,
            "properties": edge.properties,
        })

    # Create interactive graph widget
    graph_widget = Graph(
        nodes=graph_nodes,
        edges=graph_edges,
        height=500,
    )

    graph_widget
    return (
        Graph,
        edges,
        graph_edges,
        graph_nodes,
        graph_widget,
        nodes,
        seen_ids,
        viz_result,
    )


@app.cell
def __(db, mo):
    # Run PageRank algorithm
    pagerank_scores = db.algorithms.pagerank(damping=0.85)

    # Get top nodes by PageRank
    sorted_scores = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)

    rows = []
    for node_id, score in sorted_scores[:5]:
        node = db.get_node(node_id)
        name = node.properties.get("name", f"Node {node_id}") if node else f"Node {node_id}"
        rows.append(f"| {name} | {score:.4f} |")

    mo.md(f"""
    ## PageRank Analysis

    Top 5 most influential nodes in the network:

    | Name | PageRank Score |
    |------|----------------|
    {chr(10).join(rows)}

    Higher scores indicate more central/influential nodes in the network.
    """)
    return pagerank_scores, rows, sorted_scores


@app.cell
def __(db, mo):
    # Community detection
    communities = db.algorithms.louvain(resolution=1.0)

    # Group nodes by community
    community_groups = {}
    for node_id, community_id in communities.items():
        if community_id not in community_groups:
            community_groups[community_id] = []
        node = db.get_node(node_id)
        if node:
            name = node.properties.get("name", f"Node {node_id}")
            community_groups[community_id].append(name)

    community_text = []
    for comm_id, members in sorted(community_groups.items()):
        community_text.append(f"- **Community {comm_id}**: {', '.join(members)}")

    mo.md(f"""
    ## Community Detection (Louvain)

    Detected communities in the network:

    {chr(10).join(community_text)}

    The algorithm found **{len(community_groups)}** distinct communities.
    """)
    return communities, community_groups, community_text


@app.cell
def __(db, mo):
    # Shortest paths
    shortest_paths_text = []

    # Find paths between Seattle people
    seattle_result = db.execute("""
        MATCH (p:Person)
        WHERE p.city = 'Seattle'
        RETURN p.name as name, id(p) as id
    """)

    seattle_people = [(row["name"], row["id"]) for row in seattle_result]

    if len(seattle_people) >= 2:
        name1, id1 = seattle_people[0]
        name2, id2 = seattle_people[1]
        distances = db.algorithms.dijkstra(id1)
        if id2 in distances:
            shortest_paths_text.append(
                f"- Distance from **{name1}** to **{name2}**: {distances[id2]:.0f} hops"
            )

    mo.md(f"""
    ## Path Analysis

    Analyzing connections between Seattle residents:

    {chr(10).join(shortest_paths_text) if shortest_paths_text else "No paths found"}
    """)
    return (
        distances,
        id1,
        id2,
        name1,
        name2,
        seattle_people,
        seattle_result,
        shortest_paths_text,
    )


@app.cell
def __(db, mo):
    # Database statistics
    stats = db.detailed_stats()

    mo.md(f"""
    ## Database Statistics

    | Metric | Value |
    |--------|-------|
    | Nodes | {stats['node_count']} |
    | Edges | {stats['edge_count']} |
    | Labels | {stats['label_count']} |
    | Edge Types | {stats['edge_type_count']} |
    | Properties | {stats['property_key_count']} |
    | Memory | {stats['memory_bytes'] / 1024:.1f} KB |
    """)
    return (stats,)


if __name__ == "__main__":
    app.run()
