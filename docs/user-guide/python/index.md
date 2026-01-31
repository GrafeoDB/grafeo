---
title: Python API
description: Using Grafeo from Python.
---

# Python API

Grafeo provides first-class Python support through the `grafeo` package.

## Quick Start

```python
import grafeo

# Create a database
db = grafeo.GrafeoDB()

# Execute queries
db.execute("INSERT (:Person {name: 'Alice'})")

result = db.execute("MATCH (p:Person) RETURN p.name")
for row in result:
    print(row['p.name'])
```

## Sections

<div class="grid cards" markdown>

-   **[Database Operations](database.md)**

    ---

    Creating and configuring databases.

-   **[Working with Nodes](nodes.md)**

    ---

    Creating, reading, updating, and deleting nodes.

-   **[Working with Edges](edges.md)**

    ---

    Managing relationships between nodes.

-   **[Transactions](transactions.md)**

    ---

    Transaction management and isolation.

-   **[Query Results](results.md)**

    ---

    Working with query results.

</div>
