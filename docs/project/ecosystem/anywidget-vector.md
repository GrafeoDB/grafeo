# anywidget-vector

Interactive 3D vector visualization for Python notebooks.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/anywidget-vector){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/anywidget-vector/){ .md-button }

## Overview

anywidget-vector provides 3D visualization for high-dimensional embeddings and vector data. Built on **Three.js** and the [anywidget](https://anywidget.dev/) framework, it works universally across Jupyter, Marimo, VS Code, Colab and Databricks.

- **6D Visualization**: X, Y, Z position + Color, Shape, Size encoding
- **Backend-agnostic**: NumPy, pandas, Qdrant, Chroma, Pinecone, Weaviate, LanceDB or raw dicts
- **Interactive**: Orbit, pan, zoom, click, hover, box select
- **Customizable**: Color scales, shapes, sizes, themes
- **Performant**: Instanced rendering for large point clouds

## Installation

```bash
uv add anywidget-vector
```

## Quick Start

```python
from anywidget_vector import VectorSpace

widget = VectorSpace(points=[
    {"id": "a", "x": 0.5, "y": 0.3, "z": 0.8, "label": "Point A", "cluster": 0},
    {"id": "b", "x": -0.2, "y": 0.7, "z": 0.1, "label": "Point B", "cluster": 1},
    {"id": "c", "x": 0.1, "y": -0.4, "z": 0.6, "label": "Point C", "cluster": 0},
])

widget
```

## Data Sources

### Dictionary

```python
widget = VectorSpace.from_dict({
    "points": [
        {"id": "a", "x": 0, "y": 0, "z": 0},
        {"id": "b", "x": 1, "y": 1, "z": 1},
    ]
})
```

### NumPy Arrays

```python
import numpy as np

positions = np.random.randn(100, 3)
widget = VectorSpace.from_numpy(positions)
```

### pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    "x": [0.1, 0.5, 0.9],
    "y": [0.2, 0.6, 0.3],
    "z": [0.3, 0.1, 0.7],
    "cluster": ["A", "B", "A"],
    "size": [0.5, 1.0, 0.8],
})

widget = VectorSpace.from_dataframe(
    df,
    color_col="cluster",
    size_col="size",
)
```

### UMAP / t-SNE / PCA

```python
import umap

embedding = umap.UMAP(n_components=3).fit_transform(high_dim_data)
widget = VectorSpace.from_umap(embedding, labels=labels)
```

### Qdrant

```python
from qdrant_client import QdrantClient

client = QdrantClient("localhost", port=6333)
widget = VectorSpace.from_qdrant(client, "my_collection", limit=5000)
```

### ChromaDB

```python
import chromadb

client = chromadb.Client()
collection = client.get_collection("embeddings")
widget = VectorSpace.from_chroma(collection)
```

### Pinecone

```python
from pinecone import Pinecone

pc = Pinecone(api_key="...")
index = pc.Index("my-index")
widget = VectorSpace.from_pinecone(index, limit=5000)
```

### Weaviate

```python
import weaviate

client = weaviate.Client("http://localhost:8080")
widget = VectorSpace.from_weaviate(client, "Article", limit=5000)
```

### LanceDB

```python
import lancedb

db = lancedb.connect("~/.lancedb")
table = db.open_table("vectors")
widget = VectorSpace.from_lancedb(table, limit=5000)
```

## Visual Encoding

### 6 Dimensions

| Dimension | Visual Channel | Example |
| --------- | ------------- | ------- |
| X | Horizontal position | `x` coordinate |
| Y | Vertical position | `y` coordinate |
| Z | Depth position | `z` coordinate |
| Color | Hue/gradient | Cluster, score |
| Shape | Geometry | Category, type |
| Size | Scale | Importance, count |

### Color Scales

```python
widget = VectorSpace(
    points=data,
    color_field="score",           # Field to map
    color_scale="viridis",         # Scale: viridis, plasma, inferno, magma, cividis, turbo
    color_domain=[0, 100],         # Optional: explicit range
)
```

### Shapes

```python
widget = VectorSpace(
    points=data,
    shape_field="category",
    shape_map={
        "type_a": "sphere",        # Available: sphere, cube, cone,
        "type_b": "cube",          #            tetrahedron, octahedron, cylinder
        "type_c": "cone",
    }
)
```

### Size

```python
widget = VectorSpace(
    points=data,
    size_field="importance",
    size_range=[0.02, 0.15],       # Min/max point size
)
```

## Interactivity

### Events

```python
widget = VectorSpace(points=data)

@widget.on_click
def handle_click(point_id, point_data):
    print(f"Clicked: {point_id}")

@widget.on_hover
def handle_hover(point_id, point_data):
    if point_id:
        print(f"Hovering: {point_id}")

@widget.on_selection
def handle_selection(point_ids, points_data):
    print(f"Selected {len(point_ids)} points")
```

### Selection

```python
widget.selected_points              # Current selection
widget.select(["a", "b"])           # Select points
widget.clear_selection()            # Clear
widget.selection_mode = "box"       # Switch to box-select mode
```

### Camera

```python
widget.camera_position              # Get position [x, y, z]
widget.camera_target                # Get target [x, y, z]
widget.reset_camera()               # Reset to default
widget.focus_on(["a", "b"])         # Focus on specific points
```

## Distance Metrics

Compute distances and visualize similarity relationships between points.

### Supported Metrics

| Metric | Description |
| ------ | ----------- |
| `euclidean` | Straight-line distance (L2 norm) |
| `cosine` | Angle-based distance (1 - cosine similarity) |
| `manhattan` | Sum of absolute differences (L1 norm) |
| `dot_product` | Negative dot product (higher = closer) |

### Color by Distance

```python
widget.color_by_distance("point_a")
widget.color_by_distance("point_a", metric="cosine")
```

### Find Neighbors

```python
neighbors = widget.find_neighbors("point_a", k=5)
# Returns: [("point_b", 0.1), ("point_c", 0.2), ...]

neighbors = widget.find_neighbors("point_a", threshold=0.5)
```

### Show Connections

```python
widget.show_neighbors("point_a", k=5)
widget.show_neighbors("point_a", threshold=0.3)

# Manual connection settings
widget = VectorSpace(
    points=data,
    show_connections=True,
    k_neighbors=3,
    distance_metric="cosine",
    connection_color="#00ff00",
    connection_opacity=0.5,
)
```

### Compute Distances

```python
distances = widget.compute_distances("point_a")
# Returns: {"point_b": 0.1, "point_c": 0.5, ...}

# Use high-dimensional vectors (not just x,y,z)
distances = widget.compute_distances(
    "point_a",
    metric="cosine",
    vector_field="embedding"
)
```

## Options

```python
widget = VectorSpace(
    points=data,
    width=1000,
    height=700,
    background="#1a1a2e",         # Dark theme default
    show_axes=True,
    show_grid=True,
    axis_labels={"x": "PC1", "y": "PC2", "z": "PC3"},
    show_tooltip=True,
    tooltip_fields=["label", "x", "y", "z", "cluster"],
    selection_mode="click",       # "click", "multi", or "box"
    use_instancing=True,          # Performance: instanced rendering
)
```

## Backends

Configure a backend for interactive querying:

```python
widget.set_backend("chroma", client=collection)
widget.set_backend("lancedb", client=table)
widget.set_backend("grafeo", client=db)
```

## Export

```python
widget.to_json()                            # Points as JSON string
widget.to_html()                            # Self-contained HTML string
widget.to_html(title="My Vectors")          # Custom title
widget.save_html("vectors.html")            # Write HTML to file
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
- Modern browser with WebGL support

## License

Apache-2.0
