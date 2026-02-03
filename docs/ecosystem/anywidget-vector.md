# anywidget-vector

Interactive 3D vector visualization for Python notebooks.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/anywidget-vector){ .md-button }
[:simple-pypi: PyPI](https://pypi.org/project/anywidget-vector/){ .md-button }

## Overview

anywidget-vector provides 3D visualization for high-dimensional embeddings and vector data. Built on Three.js and the anywidget framework, it works universally across Jupyter, Marimo, VS Code, Colab, and Databricks.

## Features

### 6D Visualization
Encode six dimensions simultaneously:

- **X, Y, Z** — 3D position coordinates
- **Color** — Hue/gradient mapping
- **Shape** — Geometry types (sphere, cube, etc.)
- **Size** — Scale/importance

### Multi-Backend Support

| Side | Backends |
|------|----------|
| **Browser** | Qdrant, Pinecone, Weaviate (REST API) |
| **Python** | ChromaDB, LanceDB, Grafeo |

### Rich Interactivity
- Orbit, pan, zoom controls
- Click/hover/selection events
- Programmable camera positioning
- K-nearest neighbor visualization

## Installation

```bash
uv add anywidget-vector
```

With optional backends:

```bash
uv add "anywidget-vector[chromadb,qdrant]"
```

## Quick Start

### From NumPy Arrays

```python
from anywidget_vector import VectorSpace
import numpy as np

# Generate sample embeddings
embeddings = np.random.randn(1000, 3)
colors = np.random.rand(1000)

widget = VectorSpace(
    points=embeddings,
    colors=colors,
    point_size=0.05
)
widget
```

### From pandas DataFrame

```python
import pandas as pd
from anywidget_vector import VectorSpace

df = pd.DataFrame({
    "x": np.random.randn(500),
    "y": np.random.randn(500),
    "z": np.random.randn(500),
    "category": np.random.choice(["A", "B", "C"], 500)
})

widget = VectorSpace.from_dataframe(
    df,
    x="x", y="y", z="z",
    color_by="category"
)
widget
```

### From Vector Database

```python
from anywidget_vector import VectorSpace
from anywidget_vector.backends import ChromaDBBackend

backend = ChromaDBBackend(
    collection_name="embeddings",
    path="./chroma_db"
)

widget = VectorSpace(backend=backend)
widget
```

## Distance Metrics

Compute and visualize vector relationships:

```python
widget = VectorSpace(
    points=embeddings,
    distance_metric="cosine",  # euclidean, manhattan, dot_product
    show_neighbors=5           # K-nearest neighbors
)
```

## Event Handling

```python
@widget.on_click
def handle_click(point_index, point_data):
    print(f"Clicked point {point_index}: {point_data}")

@widget.on_select
def handle_selection(indices):
    print(f"Selected {len(indices)} points")
```

## Configuration

```python
widget = VectorSpace(
    points=embeddings,

    # Appearance
    point_size=0.05,
    point_shape="sphere",      # sphere, cube, octahedron
    colormap="viridis",
    background_color="#1a1a1a",

    # Controls
    show_axes=True,
    show_grid=True,
    enable_selection=True,

    # Performance
    use_instancing=True,       # For 100K+ points
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
- Modern browser with WebGL support

## License

Apache-2.0
