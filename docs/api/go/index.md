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
- The `grafeo-c` shared library (`libgrafeo_c.so` / `libgrafeo_c.dylib` / `grafeo_c.dll`): download the `all` variant from a GitHub release (see [prebuilt libraries](../c/index.md#prebuilt-libraries)) or build it from source as below. The Go package calls every query language, so it does not link against the `default` variant

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

nodes, err := db.NodeCount() // number of nodes
edges, err := db.EdgeCount() // number of edges
```

## Decimals

An exact decimal (GQL `1.50`, `CAST(x AS DECIMAL(10, 2))`) comes back as a
`grafeo.Decimal`: up to 38 digits, kept as its
text with its scale (`String()` gives `"1.50"`). Compare decimals with
`Equal`, which ignores the scale; `Rat()` gives the exact `*big.Rat`, and
`grafeo.ParseDecimal("1.50")` makes one to pass as a parameter, exactly.

```go
result, _ := db.Execute("RETURN 19.99 AS price")
price := result.Rows[0]["price"].(grafeo.Decimal)
fmt.Println(price.String()) // 19.99
```

## RDF Terms

A SPARQL result holds each IRI, blank node and literal as a `grafeo.RdfTerm`:
its `Type` (`grafeo.IRI`, `grafeo.BlankNode` or `grafeo.Literal`), its `Value`
(the IRI, the blank node's label without `_:`, or the lexical form), and for a
literal its `Language` tag or `Datatype` (empty for a simple literal;
`LiteralDatatype()` gives `xsd:string` or `rdf:langString` there). An
`RdfTerm` passed as a parameter, or stored as a property, is that exact term.
`grafeo.NewIRI`, `NewBlankNode`, `NewLiteral`, `NewLangLiteral` and
`NewTypedLiteral` build one: language tags compare without case, so a tag is
kept in lowercase, and a literal of datatype `xsd:string` is the simple
literal. `Equal` compares two terms so, also terms built as struct literals.
A `map[string]any` with `type` and `value` keys stays a map: on the wire a
term is the `$term` tag (see the C API).
A result's `Kind` is `grafeo.ResultRows`, `grafeo.ResultBoolean` for a SPARQL
ASK (`Boolean()` gives its answer) or `grafeo.ResultGraph` for CONSTRUCT and
DESCRIBE.

```go
result, _ := db.ExecuteSPARQL(`SELECT ?o WHERE { <http://ex.org/apple> <http://ex.org/label> ?o }`)
label := result.Rows[0]["o"].(grafeo.RdfTerm)
fmt.Println(label.Value, label.Language) // Pomme fr
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

// A node's edges without a query: DirectionOut, DirectionIn or DirectionBoth (a self-loop once),
// optionally of some types; each NodeEdge has NeighborID and EdgeID
edges, err := db.GetNodeEdges(id, grafeo.DirectionOut, "KNOWS")
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

## Time Travel

Reads of the database as it was at an earlier epoch or time. `OpenInMemoryWithHistory(history)` and
`OpenWithHistory(path, history)` set how much history it keeps: `HistoryOff` (the default), `HistoryForever` or
`HistoryFor(time.Duration)`; a database file stores the setting. A point is a `TimePoint`: `AtEpoch(uint64)`,
`AtTime(time.Time)` (the last commit at or before it) or `Ago(time.Duration)`. A point before the history the
database keeps fails ("not retained"). A view keeps the history of its epoch until `Close`; besides `Execute` and
`ExecuteParams` it has `GetNode`, `GetEdge` (an error when the entity did not exist then) and `GetNodeProperty` (nil
when there was none).

```go
db, err := grafeo.OpenInMemoryWithHistory(grafeo.HistoryFor(7 * 24 * time.Hour))
then := db.CurrentEpoch()
view, err := db.AsOf(grafeo.Ago(5 * time.Minute))
defer view.Close()
result, err := view.Execute("MATCH (p:Person) RETURN count(p) AS n")
result, err = db.ExecuteAsOf("MATCH (p:Person) RETURN count(p) AS n", grafeo.AtEpoch(then), nil)
epoch, err := db.EpochAsOf(grafeo.AtTime(time.Now().Add(-time.Hour)))
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
cargo build --release -p grafeo-c --features lpg,rdf,analytics,ai
```

## Links

- [pkg.go.dev](https://pkg.go.dev/github.com/GrafeoDB/grafeo/crates/bindings/go)
- [GitHub](https://github.com/GrafeoDB/grafeo/tree/main/crates/bindings/go)
