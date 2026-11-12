# grafeo-cli

The command-line interface of [Grafeo](https://grafeo.dev), an embeddable graph database: query a
database, inspect it, back it up and maintain it, from the shell or an interactive REPL.

## Installation

```bash
cargo install grafeo-cli
```

The same build ships as pre-built binaries: `uv add grafeo-cli` (or `pip install grafeo-cli`),
`npm install -g @grafeo-db/cli`, and the archives on every
[GitHub release](https://github.com/GrafeoDB/grafeo/releases).

## What the default build has

The default build opens every database the `grafeo` crate's default build writes:

- GQL queries;
- persistent storage: the single `.grafeo` file, its write-ahead log, spill to disk;
- vector, text and hybrid search indexes (`vector-index`, `text-index`, `hybrid-search`);
- change data capture (`cdc`);
- graph algorithms (`algos`);
- JSON Lines import (`jsonl-import`).

`grafeo version` lists the features of the build you run.

## Other features

Add a feature to `cargo install`:

```bash
cargo install grafeo-cli --features triple-store         # RDF databases (grafeo init --mode rdf)
cargo install grafeo-cli --features cypher,sparql        # more query languages
cargo install grafeo-cli --features full                 # every language, RDF and the defaults
```

| Feature | Adds |
| --- | --- |
| `cypher`, `gremlin`, `graphql`, `sql-pgq` | that query language (`--lang`) |
| `sparql` | SPARQL, with `triple-store` |
| `languages` | every query language |
| `triple-store` | RDF triples: databases created with `--mode rdf`, or by a `grafeo` build with the `rdf` profile |
| `arrow-export` | `grafeo data dump --export-format arrow` |
| `full` | every language, `triple-store` and the default features |

A database that holds data this build cannot read (RDF triples without `triple-store`, for
example) is refused before anything on disk changes. The error names the feature and the
`cargo install` line that adds it.

## Usage

```bash
grafeo init ./mydb
grafeo query ./mydb "INSERT (:Person {name: 'Alix', age: 30})"
grafeo query ./mydb "MATCH (n:Person) RETURN n.name, n.age"
grafeo shell ./mydb
grafeo info ./mydb
```

The full guide: [Command-Line Interface](https://grafeo.dev/getting-started/cli/).

## License

Apache-2.0
