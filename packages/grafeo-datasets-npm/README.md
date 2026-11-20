# @grafeo-db/datasets

The Silicon Valley demo dataset for [Grafeo](https://grafeo.dev): people, companies, products, places and events as nodes and edges, a timeline of dated facts, ready-to-run GQL and Cypher scripts, Turtle, and MiniLM embeddings of every description.

An homage to the HBO series Silicon Valley; not affiliated with or endorsed by HBO. All descriptions are original.

## Install

```bash
npm install @grafeo-db/datasets
```

Node 20 or later, or any bundler. Every entry point is a plain ES module with the data inlined, so nothing is fetched at run time. The root entry stays small: it does not import the scripts, the Turtle or the vectors, so a web bundle pays only for what it imports.

## Entry points

| Import                                                | Contents                                           |
| ----------------------------------------------------- | -------------------------------------------------- |
| `@grafeo-db/datasets`                                 | `siliconValley`: nodes, edges and timeline as data |
| `@grafeo-db/datasets/silicon-valley/gql`              | GQL script, as text and as statements              |
| `@grafeo-db/datasets/silicon-valley/gql-with-vectors` | GQL script with embeddings and vector indexes      |
| `@grafeo-db/datasets/silicon-valley/cypher`           | Cypher script, as text and as statements           |
| `@grafeo-db/datasets/silicon-valley/turtle`           | Turtle text                                        |
| `@grafeo-db/datasets/silicon-valley/vectors`          | Embeddings as a `Float32Array`, plus `vectorOf`    |

### Data

```js
import { siliconValley } from '@grafeo-db/datasets';

const { nodes, edges, timeline } = siliconValley;
console.log(nodes.length, edges.length, timeline.length);
```

A node has `key`, `label`, `name`, `description` and `props`. An edge has `type`, `source` and `target` (node keys) and `props`. A timeline fact has `subject`, `relation` and `object` (node keys), plus `from` and `to`. Both dates are optional: `from` is absent when only the end is known, `to` is absent while the fact still holds. Dates are `YYYY-MM-DD`, or `YYYY-MM` when only the month is known.

### GQL and Cypher

Each script module has a default export (the whole script as text) and `statements` (one statement per entry, in load order).

```js
import gql, { statements } from '@grafeo-db/datasets/silicon-valley/gql';
import cypher from '@grafeo-db/datasets/silicon-valley/cypher';
```

Month-only dates load as the first of the month, with a `<field>_precision` property set to `'month'`.

### Turtle

```js
import turtle from '@grafeo-db/datasets/silicon-valley/turtle';
```

The namespace is `https://docs.grafeo.dev/datasets/silicon-valley/`. The Turtle carries no vectors. Dates are `xsd:date`, so SPARQL range filters work on them; a month-only date is the first of the month, with `sv:<field>Precision "month"` on the same subject (for example `sv:fromPrecision`).

### Vectors

```js
import { vectors, vectorOf } from '@grafeo-db/datasets/silicon-valley/vectors';

console.log(vectors.model, vectors.dimension, vectors.keys.length);
const embedding = vectorOf('middle-out'); // Float32Array(384), or undefined
```

`vectors.data` holds one unit-length row of `vectors.dimension` floats per entry of `vectors.keys`. The model is `Xenova/all-MiniLM-L6-v2` (quantized), with mean pooling and normalised output.

## Load into Grafeo

### In the browser with `@grafeo-db/web`

```js
import { GrafeoDB } from '@grafeo-db/web';
import { statements } from '@grafeo-db/datasets/silicon-valley/gql';

const db = await GrafeoDB.create();
for (const statement of statements) {
  await db.execute(statement);
}
```

### In Node with `@grafeo-db/js`

```js
import { GrafeoDB } from '@grafeo-db/js';
import { statements } from '@grafeo-db/datasets/silicon-valley/gql';

const db = GrafeoDB.create();
for (const statement of statements) {
  await db.execute(statement);
}
```

Load the script once per database. A second load would duplicate every node.

### Vector search

The `gql-with-vectors` script stores each description's embedding in an `embedding` property and creates one cosine vector index per label, so it needs a Grafeo build with vector indexes. `@grafeo-db/wasm-lite` has none and stops at the first `CREATE VECTOR INDEX`; use the plain `gql` script there. Embed the query with the same model, then search with `@grafeo-db/js`:

```js
import { GrafeoDB } from '@grafeo-db/js';
import { pipeline } from '@huggingface/transformers';
import { statements } from '@grafeo-db/datasets/silicon-valley/gql-with-vectors';

const db = GrafeoDB.create();
for (const statement of statements) {
  await db.execute(statement);
}

const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'q8' });
const output = await embed('a compression algorithm', { pooling: 'mean', normalize: true });
const hits = await db.vectorSearch('Technology', 'embedding', Array.from(output.data), 5);
// hits: [nodeId, distance] pairs, nearest first
```

The stored vectors come from the quantized model. A query embedded with a different precision does not line up with them.

Without a database, `vectorOf(key)` returns the stored embedding of one node, which is enough for a nearest-neighbour search in plain JavaScript.

## Licence

Apache-2.0.
