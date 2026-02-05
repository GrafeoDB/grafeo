# anywidget-graph

Interactive graph visualization widget for Python notebooks.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/anywidget-graph){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/anywidget-graph/){ .md-button }

## Overview

anywidget-graph provides interactive graph visualization powered by Sigma.js. Built on the anywidget framework, it works universally across Jupyter, Marimo, VS Code, Colab, and Databricks with support for multiple graph database backends.

## Features

### Multi-Backend Support

| Backend | Query Languages |
|---------|-----------------|
| **Grafeo** | GQL, Cypher, SPARQL, Gremlin, GraphQL |
| **Neo4j** | Cypher |
| **ArangoDB** | AQL |
| **LadybugDB** | Cypher |
| **NetworkX** | Python API |

### Query Language Converters
Automatically converts query results from:

- Cypher (Neo4j, Grafeo)
- GQL (ISO standard)
- SPARQL (RDF/semantic web)
- Gremlin (TinkerPop)
- GraphQL (JSON responses)
- AQL (ArangoDB)

### Interactivity
- Pan, zoom, and navigate
- Click nodes/edges to select
- Expand/collapse neighbors
- Multi-select paths
- Export to PNG, SVG, JSON

## Installation

```bash
uv add anywidget-graph
```

With optional backends:

```bash
uv add "anywidget-graph[networkx,neo4j]"
```

## Quick Start

### From Grafeo

```python
from anywidget_graph import GraphWidget
from grafeo import Database

db = Database()
db.query("""
    INSERT (:Person {name: 'Alice'})-[:KNOWS]->(:Person {name: 'Bob'}),
           (:Person {name: 'Bob'})-[:KNOWS]->(:Person {name: 'Carol'})
""")

widget = GraphWidget.from_grafeo(db, "MATCH (n)-[r]->(m) RETURN n, r, m")
widget
```

### From NetworkX

```python
import networkx as nx
from anywidget_graph import GraphWidget

G = nx.karate_club_graph()
widget = GraphWidget.from_networkx(G)
widget
```

### From Edge List

```python
from anywidget_graph import GraphWidget

nodes = [
    {"id": "1", "label": "Alice", "group": "person"},
    {"id": "2", "label": "Bob", "group": "person"},
    {"id": "3", "label": "Acme", "group": "company"},
]

edges = [
    {"source": "1", "target": "2", "label": "knows"},
    {"source": "2", "target": "3", "label": "works_at"},
]

widget = GraphWidget(nodes=nodes, edges=edges)
widget
```

## Styling

### Node Styling

```python
widget = GraphWidget(
    nodes=nodes,
    edges=edges,
    node_color_by="group",
    node_size_by="degree",
    node_colors={
        "person": "#6366f1",
        "company": "#22c55e"
    }
)
```

### Layout Algorithms

```python
widget = GraphWidget(
    nodes=nodes,
    edges=edges,
    layout="force",  # force, hierarchical, circular, grid
)
```

## Event Handling

```python
@widget.on_node_click
def handle_node_click(node_id, node_data):
    print(f"Clicked: {node_data['label']}")

@widget.on_edge_click
def handle_edge_click(edge_id, edge_data):
    print(f"Edge: {edge_data['label']}")
```

## Configuration

```python
widget = GraphWidget(
    nodes=nodes,
    edges=edges,

    # Display
    show_labels=True,
    show_edge_labels=False,
    show_toolbar=True,
    dark_mode=True,

    # Layout
    layout="force",

    # Performance
    virtualize=True,  # For 1000+ nodes
)
```

## Database Connections

### Neo4j

```python
from anywidget_graph import GraphWidget
from anywidget_graph.backends import Neo4jBackend

backend = Neo4jBackend(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password"
)

widget = GraphWidget(
    backend=backend,
    query="MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"
)
```

### ArangoDB

```python
from anywidget_graph.backends import ArangoDBBackend

backend = ArangoDBBackend(
    host="localhost",
    database="mydb",
    username="root",
    password="password"
)

widget = GraphWidget(
    backend=backend,
    query="FOR v, e IN 1..2 OUTBOUND 'nodes/1' edges RETURN {v, e}"
)
```

## Notebook Compatibility

Works in any environment supporting anywidget:

- JupyterLab / Jupyter Notebook
- Marimo
- VS Code Notebooks
- Google Colab
- Databricks

## Requirements

- Python 3.12+
- Modern browser

## License

Apache-2.0
