---
title: Loading RDF Data
description: Parse and import Turtle, N-Triples and N-Quads files into Grafeo.
tags:
  - rdf
  - turtle
  - ntriples
  - import
---

Grafeo includes zero-dependency parsers for W3C RDF serialization formats.
Turtle and N-Triples support both batch (replace-all) and streaming
(incremental, memory-bounded) loading. N-Quads is supported for export.

## Turtle

Turtle is the most common human-readable RDF format, supporting prefix
declarations, predicate lists (`;`), object lists (`,`), blank nodes, typed
literals and the `a` shorthand for `rdf:type`.

### Turtle batch load (Rust)

Replaces the store contents with the parsed triples:

```rust
use grafeo_core::graph::rdf::RdfStore;

let store = RdfStore::new();
let result = store.load_turtle(r#"
    @prefix ex: <http://example.org/> .
    @prefix foaf: <http://xmlns.com/foaf/0.1/> .
    ex:alix a foaf:Person ; foaf:name "Alix" ; foaf:knows ex:gus .
    ex:gus  foaf:name "Gus" .
"#)?;
println!("{} triples loaded", result.triple_count);
```

### Turtle streaming load (Rust)

Inserts triples incrementally in batches of configurable size, bounding memory
regardless of document size. Does not replace existing data:

```rust
let count = store.load_turtle_streaming(turtle_str, 10_000)?;
```

The `batch_size` parameter controls how many triples are buffered before
flushing to the store. 10,000 is a good default: small enough to cap memory,
large enough to amortize lock overhead.

### Loading via SPARQL

From any language binding you can load Turtle data through SPARQL INSERT DATA:

```sparql
INSERT DATA {
  <http://example.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
  <http://example.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://example.org/gus> .
}
```

## N-Triples

N-Triples is a line-based subset of Turtle with no prefixes, one triple per
line. Its simplicity makes it ideal for streaming large datasets.

### N-Triples batch load (Rust)

```rust
use std::io::BufReader;

let reader = BufReader::new(file);
let result = store.load_ntriples(reader)?;
```

### N-Triples streaming load (Rust)

Parses line-by-line from a buffered reader, never holding the entire file in
memory:

```rust
use std::io::BufReader;

let reader = BufReader::new(file);
let count = store.load_ntriples_streaming(reader, 10_000)?;
```

## Serialization

### Turtle export

```rust
let turtle_string = store.to_turtle()?;
```

Groups triples by subject with predicate/object shorthand for compact output.

### N-Quads export

N-Quads extends N-Triples with an optional fourth field for the graph name,
producing a single stream that includes both the default graph and all named
graphs:

```rust
let nquads_string = store.to_nquads()?;
```

## Custom Sinks

For advanced use cases the `TripleSink` trait decouples parsing from storage.
Three built-in implementations are provided:

| Sink              | Purpose                                                        |
| ----------------- | -------------------------------------------------------------- |
| `BatchInsertSink` | Buffered insert into an `RdfStore` (used by streaming loaders) |
| `CountSink`       | Dry-run counting without storage, useful for validation        |
| `VecSink`         | Collects all triples into a `Vec<Triple>`                      |

```rust
use grafeo_core::graph::rdf::{CountSink, TurtleParser};

let mut sink = CountSink::new();
TurtleParser::new().parse_into(turtle_str, &mut sink)?;
println!("{} triples parsed", sink.count());
```

Implement `TripleSink` on your own types to route triples to custom
destinations (logging, deduplication, remote stores).

## SPARQL LOAD Statement

Grafeo also supports the SPARQL Update `LOAD` statement for importing RDF data from a URI:

```sparql
-- Load triples from a source URI into the default graph
LOAD <file:///path/to/data.ttl>

-- Load into a specific named graph
LOAD <file:///path/to/data.ttl> INTO GRAPH <http://example.org/my-graph>

-- Silent mode: suppress errors if the source is unavailable
LOAD SILENT <file:///path/to/data.ttl>
```

The `LOAD` statement accepts an IRI as its source. The optional `SILENT` keyword suppresses errors when the source cannot be loaded. The optional `INTO GRAPH` clause directs triples into a named graph instead of the default graph.
