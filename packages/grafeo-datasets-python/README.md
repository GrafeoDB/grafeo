# grafeo-datasets

The Silicon Valley demo dataset for [Grafeo](https://grafeo.dev): people, companies, products, places and events as nodes and edges, a timeline of dated facts, ready-to-run GQL and Cypher scripts, Turtle, and MiniLM embeddings of every description.

An homage to the HBO series Silicon Valley; not affiliated with or endorsed by HBO. All descriptions are original.

## Install

```bash
pip install grafeo-datasets
```

Python 3.12 or later. The data ships inside the package, so nothing is fetched at run time, and the package itself has no dependencies. To load the dataset into a database, install Grafeo with the `grafeo` extra:

```bash
pip install "grafeo-datasets[grafeo]"
```

## API

Everything lives in `grafeo_datasets.silicon_valley`.

| Function                     | Returns                                            |
| ---------------------------- | -------------------------------------------------- |
| `nodes()`                    | list of node dicts                                 |
| `edges()`                    | list of edge dicts                                 |
| `timeline()`                 | list of dated facts                                |
| `gql()`                      | GQL script as text, one statement per line         |
| `cypher()`                   | Cypher script as text, one statement per line      |
| `turtle()`                   | Turtle text                                        |
| `vectors()`                  | `VectorSet` with `model`, `dimension` and `by_key` |
| `load(db, *, vectors=False)` | number of statements run on `db`                   |

### Data

```python
from grafeo_datasets import silicon_valley

nodes = silicon_valley.nodes()
edges = silicon_valley.edges()
timeline = silicon_valley.timeline()
print(len(nodes), len(edges), len(timeline))
```

A node has `key`, `label`, `name`, `description` and `props`. An edge has `type`, `source` and `target` (node keys) and `props`. A timeline fact has `subject`, `relation` and `object` (node keys), plus `from` and `to`. Both dates are optional: `from` is absent when only the end is known, `to` is absent while the fact still holds. Dates are `YYYY-MM-DD`, or `YYYY-MM` when only the month is known. Each call returns fresh lists.

### GQL, Cypher and Turtle

`gql()` and `cypher()` return the whole script as text, one statement per line. Month-only dates load as the first of the month, with a `<field>_precision` property set to `'month'`.

`turtle()` uses the namespace `https://docs.grafeo.dev/datasets/silicon-valley/` and carries no vectors. Dates are `xsd:date`, so SPARQL range filters work on them; a month-only date is the first of the month, with `sv:<field>Precision "month"` on the same subject (for example `sv:fromPrecision`).

### Vectors

```python
from grafeo_datasets import silicon_valley

vs = silicon_valley.vectors()
print(vs.model, vs.dimension, len(vs.by_key))
embedding = vs.by_key["middle-out"]  # list of 384 floats
```

`by_key` holds one unit-length list of `dimension` floats per node key. The model is `Xenova/all-MiniLM-L6-v2` (quantized), with mean pooling and normalised output. The vectors are read with the standard library only.

## Load into Grafeo

```python
from grafeo import GrafeoDB
from grafeo_datasets import silicon_valley

db = GrafeoDB()
count = silicon_valley.load(db)
print(count, "statements run")
```

`load` runs each statement of the GQL script with `db.execute`. It raises `ValueError` when the database already holds the dataset, fully or in part, so a second load never duplicates the nodes. After a load that failed partway, load into an empty database.

### Vector search

`load(db, vectors=True)` runs the script that stores each description's embedding in an `embedding` property and creates one cosine vector index per label. Then search with `db.vector_search(label, property, query, k)`, which returns `(node_id, distance)` pairs, nearest first:

```python
from grafeo import GrafeoDB
from grafeo_datasets import silicon_valley

db = GrafeoDB()
silicon_valley.load(db, vectors=True)

query = silicon_valley.vectors().by_key["middle-out"]
for node_id, distance in db.vector_search("Technology", "embedding", query, 2):
    print(node_id, distance)
```

A query must be embedded with the same quantized model. A query embedded with a different precision does not line up with the stored vectors.

## Licence

Apache-2.0.
