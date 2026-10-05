---
title: Go API
description: API reference for the Grafeo Go bindings.
---

# Go API

Go bindings for Grafeo via CGO. Requires the `grafeo-c` shared library.

```bash
go get github.com/GrafeoDB/grafeo/crates/bindings/go
```

## Requirements

- Go 1.22+
- CGO enabled (`CGO_ENABLED=1`)
- The `grafeo-c` shared library (`libgrafeo_c.so` / `libgrafeo_c.dylib` / `grafeo_c.dll`)

## Quick Start

```go
package main

import (
    "fmt"
    "log"

    grafeo "github.com/GrafeoDB/grafeo/crates/bindings/go"
)

func main() {
    db, err := grafeo.OpenInMemory()
    if err != nil {
        log.Fatal(err)
    }
    defer db.Close()

    db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix", "age": 30})
    db.CreateNode([]string{"Person"}, map[string]any{"name": "Gus", "age": 25})

    result, err := db.Execute("MATCH (p:Person) RETURN p.name, p.age")
    if err != nil {
        log.Fatal(err)
    }
    for _, row := range result.Rows {
        fmt.Printf("Name: %v, Age: %v\n", row["p.name"], row["p.age"])
    }
}
```

## Database

```go
db, err := grafeo.OpenInMemory()                // in-memory
db, err := grafeo.Open("./path")                // persistent (a single file)
db, err := grafeo.OpenSingleFile("./data.grafeo") // same as Open
defer db.Close()

db.NodeCount()   // number of nodes
db.EdgeCount()   // number of edges
```

## Query Languages

```go
result, err := db.Execute(gql)            // GQL (ISO standard)
result, err := db.ExecuteCypher(query)     // Cypher
result, err := db.ExecuteGremlin(query)    // Gremlin
result, err := db.ExecuteGraphQL(query)    // GraphQL
result, err := db.ExecuteSPARQL(query)     // SPARQL
result, err := db.ExecuteSQL(query)        // SQL/PGQ

// Parameterized queries (accepts map[string]any, marshals to JSON internally)
result, err := db.ExecuteParams(
    "MATCH (p:Person) WHERE p.age > $min RETURN p.name",
    map[string]any{"min": 25},
)

// Each language also has a WithParams variant (accepts raw JSON string)
result, err := db.ExecuteCypherWithParams(query, paramsJSON)
result, err := db.ExecuteGremlinWithParams(query, paramsJSON)
result, err := db.ExecuteGraphQLWithParams(query, paramsJSON)
result, err := db.ExecuteSPARQLWithParams(query, paramsJSON)
result, err := db.ExecuteSQLWithParams(query, paramsJSON)

// Generic language dispatch: "gql", "cypher", "gremlin", "graphql", "sparql", "sql"
result, err := db.ExecuteLanguage("cypher", query, paramsJSON)
```

## Node & Edge CRUD

```go
id, err := db.CreateNode([]string{"Person"}, map[string]any{"name": "Alix"})
eid, err := db.CreateEdge(srcID, dstID, "KNOWS", map[string]any{"since": 2024})

node, err := db.GetNode(id)
edge, err := db.GetEdge(eid)

db.SetNodeProperty(id, "age", 31)
db.SetEdgeProperty(eid, "weight", 0.5)

db.DeleteNode(id)
db.DeleteEdge(eid)
```

## Transactions

```go
tx, err := db.BeginTransaction()
result, err := tx.Execute("INSERT (:Person {name: 'Harm'})")
err = tx.Commit()   // or tx.Rollback()
```

## Vector Search

```go
// Create HNSW index
db.CreateVectorIndex("Doc", "emb", 384, "cosine", 16, 200)

// Search
results, err := db.VectorSearch("Doc", "emb", queryVec, 10)

// With options
results, err := db.VectorSearch("Doc", "emb", queryVec, 10,
    grafeo.WithEf(100))

// MMR search
results, err := db.MmrSearch("Doc", "emb", queryVec, 5, -1, 0.5, -1)
```

## Building the Shared Library

```bash
cargo build --release -p grafeo-c --features full
```

## Links

- [pkg.go.dev](https://pkg.go.dev/github.com/GrafeoDB/grafeo/crates/bindings/go)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go)
