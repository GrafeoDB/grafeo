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
db, err := grafeo.OpenSingleFile("./data.grafeo") // deprecated: same as Open, removed in 0.7.0
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

## Batches and Upserts

```go
// One transaction each; IDs in input order
ids, err := db.BatchCreateNodesWithProps([]string{"Person"}, []map[string]any{
    {"name": "Alix"}, {"name": "Gus"},
})
edgeIDs, err := db.BatchCreateEdges([]grafeo.BatchEdge{
    {Src: ids[0], Dst: ids[1], Type: "KNOWS", Properties: map[string]any{"since": 2020}},
})

// Create or update by a key property: a summary of created, updated and skipped rows
summary, err := db.UpsertNodes([]string{"File"}, []map[string]any{{"path": "a.rs"}},
    &grafeo.UpsertNodesOptions{Key: "path"})
summary, err = db.UpsertEdges("IMPORTS", []map[string]any{{"id": "a-b", "src": "a.rs", "dst": "b.rs"}},
    &grafeo.UpsertEdgesOptions{EndpointKey: "path", EndpointLabels: []string{"File"}})
```

A result carries what a write changed in `result.Counters` (`NodesCreated`, `EdgesDeleted`, `PropertiesSet`, ...).

## Named Graphs

`db.Graph(name)` returns a handle that works in one named graph without switching the current graph: `Execute`, `ExecuteParams`, `ExecuteLanguage`, the node and edge calls, the batches and the upserts. Once the graph is dropped, every call fails. `Close` frees it.

```go
trips, err := db.Graph("trips")
defer trips.Close()
prague, err := trips.CreateNode([]string{"City"}, map[string]any{"name": "Prague"})
result, err := trips.Execute("MATCH (c:City) RETURN c.name")
```

## Change History

With change data capture on (`db.SetCDCEnabled(true)`), every committed change is recorded as a `ChangeEvent` (kind, epoch, properties before and after, labels, the endpoints of an edge, and `DeletedWith` on an edge a detach delete removed).

```go
events, err := db.NodeHistory(id)
events, err = db.EdgeHistory(edgeID)
events, err = db.NodeHistorySince(id, epoch)
events, err = db.ChangesBetween(startEpoch, endEpoch)
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

// Several queries at once (ef 0 for the index default), with optional property filters
batches, err := db.BatchVectorSearch("Doc", "emb", [][]float32{q1, q2}, 10, 0,
    map[string]any{"lang": "en"})
```

## Building the Shared Library

```bash
cargo build --release -p grafeo-c --features full
```

## Links

- [pkg.go.dev](https://pkg.go.dev/github.com/GrafeoDB/grafeo/crates/bindings/go)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go)
