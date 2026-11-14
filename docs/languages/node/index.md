---
title: Node.js / TypeScript
description: Using Grafeo from Node.js and TypeScript.
tags:
  - node
  - typescript
  - javascript
---

# Node.js / TypeScript

Grafeo provides native Node.js bindings through the `@grafeo-db/js` package, powered by [napi-rs](https://napi.rs). The bindings include full TypeScript definitions.

## Quick Start

```typescript
import { GrafeoDB } from '@grafeo-db/js';

const db = GrafeoDB.create();

db.createNode(['Person'], { name: 'Alix', age: 30 });
db.createNode(['Person'], { name: 'Gus', age: 25 });
db.createEdge(0, 1, 'KNOWS', { since: 2024 });

const result = await db.execute(
  'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a.name, b.name'
);

for (const row of result.toArray()) {
  console.log(row);
}

db.close();
```

## Sections

<div class="grid cards" markdown>

-   **[Database Operations](database.md)**

    ---

    Creating, opening, and configuring databases.

-   **[Queries](queries.md)**

    ---

    Running queries in GQL, Cypher, and other languages.

-   **[Nodes & Edges](nodes-edges.md)**

    ---

    CRUD operations on nodes and edges.

-   **[Transactions](transactions.md)**

    ---

    Atomic operations with commit and rollback.

-   **[Query Results](results.md)**

    ---

    Working with result sets, nodes, and edges.

</div>
