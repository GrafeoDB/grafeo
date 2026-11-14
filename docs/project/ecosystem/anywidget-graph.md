# anywidget-graph

Interactive graph visualization for Python notebooks.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/anywidget-graph){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/anywidget-graph/){ .md-button }

## Overview

anywidget-graph provides interactive graph visualization powered by **Sigma.js**. Built on the [anywidget](https://anywidget.dev/) framework, it works universally across Jupyter, Marimo, VS Code, Colab and Databricks.

- **Backend-agnostic**: Grafeo, Neo4j, NetworkX, pandas or raw dicts
- **Interactive**: Pan, zoom, click, drag, pin, expand neighbors, box select
- **Customizable**: Colors, sizes, layouts, dark mode
- **Exportable**: HTML, JSON

## Installation

```bash
uv add anywidget-graph
```

Optional extras:

```bash
uv add "anywidget-graph[networkx]"   # NetworkX support
uv add "anywidget-graph[pandas]"     # pandas support
uv add "anywidget-graph[grafeo]"     # Grafeo backend
uv add "anywidget-graph[cosmosdb]"   # CosmosDB / Gremlin support
```

## Quick Start

```python
from anywidget_graph import Graph

graph = Graph.from_dict({
    "nodes": [
        {"id": "alix", "label": "Alix", "group": "person"},
        {"id": "gus", "label": "Gus", "group": "person"},
        {"id": "paper", "label": "Graph Theory", "group": "document"},
    ],
    "edges": [
        {"source": "alix", "target": "gus", "label": "knows"},
        {"source": "alix", "target": "paper", "label": "authored"},
    ]
})

graph
```

## Data Sources

### Dictionary

```python
graph = Graph.from_dict({
    "nodes": [{"id": "a"}, {"id": "b"}],
    "edges": [{"source": "a", "target": "b"}]
})
```

### Direct initialization

```python
graph = Graph(
    nodes=[{"id": "a", "label": "Alix"}, {"id": "b", "label": "Gus"}],
    edges=[{"source": "a", "target": "b", "label": "KNOWS"}],
)
```

### Cypher results (Neo4j)

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

with driver.session() as session:
    result = session.run("MATCH (a)-[r]->(b) RETURN a, r, b LIMIT 100")
    graph = Graph.from_cypher(result)
```

### GQL results

```python
graph = Graph.from_gql(result)
```

### SPARQL results

```python
from rdflib import Graph as RDFGraph

g = RDFGraph()
g.parse("data.ttl")
result = g.query("SELECT ?s ?p ?o WHERE { ?s ?p ?o }")
graph = Graph.from_sparql(result)
```

### Gremlin results (CosmosDB, TinkerPop)

```python
graph = Graph.from_gremlin(result)
```

### GraphQL results

```python
graph = Graph.from_graphql(
    response.json(),
    nodes_path="data.characters.results",
    id_field="id",
    label_field="name",
)
```

### NetworkX

```python
import networkx as nx

G = nx.karate_club_graph()
graph = Graph.from_networkx(G)
```

### pandas DataFrames

```python
import pandas as pd

nodes_df = pd.DataFrame({"id": ["alix", "gus"], "group": ["person", "person"]})
edges_df = pd.DataFrame({"source": ["alix"], "target": ["gus"], "weight": [1.0]})

graph = Graph.from_dataframe(nodes_df, edges_df)
```

## Interactivity

### Events

```python
graph = Graph.from_dict(data)

@graph.on_node_click
def handle_node(node_id, node_data):
    print(f"Clicked: {node_id}")

@graph.on_edge_click
def handle_edge(edge_data):
    print(f"Edge: {edge_data['label']}")

@graph.on_selection
def handle_selection(node_ids):
    print(f"Selected: {node_ids}")
```

### Selection

```python
graph.selected_nodes            # Current selection (list of IDs)
graph.selection_mode = "box"    # Switch to box-select mode
```

### Node expansion

```python
graph.expand_node("alix")      # Fetch and merge neighbors (requires backend)
```

### Node pinning

```python
graph.pin_nodes(["alix", "gus"])   # Pin at current positions
graph.unpin_nodes(["alix"])        # Release back to layout
graph.toggle_pin("gus")            # Toggle pin state
graph.unpin_all()                   # Unpin everything
```

### Clear

```python
graph.clear()                   # Remove all nodes, edges, pins and selection
```

## Styling

### Property-based coloring

```python
graph = Graph.from_dict(
    data,
    color_field="group",               # Color nodes by field
    color_scale="viridis",             # Scale: viridis, plasma, inferno, magma, cividis, turbo
    size_field="score",                # Size nodes by field
    size_range=[5, 30],                # Min/max node size
)
```

### Edge styling

```python
graph.edge_color_field = "type"
graph.edge_color_scale = "plasma"
graph.edge_size_field = "weight"
graph.edge_size_range = [1, 8]
```

### Layouts

```python
Graph.from_dict(data, layout="force")      # ForceAtlas2 (default)
Graph.from_dict(data, layout="circular")
Graph.from_dict(data, layout="random")
```

## Options

```python
graph = Graph(
    nodes=nodes,
    edges=edges,
    width=800,                  # Widget width (px)
    height=600,                 # Widget height (px)
    background="#fafafa",       # Background color
    show_labels=True,           # Node labels
    show_edge_labels=False,     # Edge labels
    show_toolbar=True,          # Toolbar visibility
    show_settings=True,         # Settings panel
    show_query_input=True,      # Query input box
    dark_mode=True,             # Dark theme
    show_tooltip=True,          # Hover tooltips
    tooltip_fields=["label", "id"],
    max_nodes=300,              # Limit for node expansion
)
```

## Database Backends

### Grafeo (default)

```python
import grafeo
db = grafeo.GrafeoDB()
graph = Graph(database_backend="grafeo", grafeo_db=db)
```

### Neo4j (browser-side)

```python
graph = Graph(
    database_backend="neo4j",
    connection_uri="neo4j+s://demo.neo4jlabs.com",
    connection_username="neo4j",
    connection_password="password",
)
```

### Generic backend

```python
graph = Graph(backend=my_backend)  # Any object implementing DatabaseBackend protocol
```

## Export

```python
graph.to_json()                         # JSON string with nodes and edges
graph.to_html()                         # Self-contained HTML string
graph.to_html(title="My Graph")         # Custom title
graph.save_html("graph.html")           # Write HTML to file
```

## Environment Support

| Environment | Supported |
| ----------- | --------- |
| Marimo | Yes |
| JupyterLab | Yes |
| Jupyter Notebook | Yes |
| VS Code | Yes |
| Google Colab | Yes |
| Databricks | Yes |

## Requirements

- Python 3.12+
- Modern browser

## License

Apache-2.0
