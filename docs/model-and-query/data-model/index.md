---
title: Data Model
description: Understanding the Labeled Property Graph model in Grafeo.
---

# Data Model

Grafeo implements the **Labeled Property Graph (LPG)** model, the most widely used data model for graph databases.

## Overview

A Labeled Property Graph consists of:

- **Nodes** - Entities in the graph (people, products, locations)
- **Edges** - Relationships between nodes (KNOWS, PURCHASED, LOCATED_IN)
- **Labels** - Categories for nodes (Person, Product, Location)
- **Types** - Categories for edges (relationship types)
- **Properties** - Key-value pairs on nodes and edges

```mermaid
graph LR
    A[Alix<br/>:Person<br/>age: 30] -->|KNOWS<br/>since: 2020| B[Gus<br/>:Person<br/>age: 25]
    B -->|WORKS_AT| C[Acme Inc<br/>:Company]
```

## Learn More

<div class="grid cards" markdown>

-   **[Nodes and Labels](nodes.md)**

    ---

    Understanding nodes, their labels and multi-label support.

-   **[Edges and Types](edges.md)**

    ---

    Creating and working with edges between nodes.

-   **[Properties](properties.md)**

    ---

    Property types, values and schema considerations.

</div>
