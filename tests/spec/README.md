# Spec Tests (`.gtest`)

Declarative test suite for Grafeo's query engine. Each `.gtest` file defines test cases in a YAML-like format that 7 runners execute through their respective bindings (Rust, Python, Node.js, WASM, Go, C#, Dart).

## Quick start

Add a test to an existing file, or create a new `.gtest` file:

```yaml
meta:
  language: gql
  model: lpg
  dataset: empty

tests:
  - name: my_new_test
    setup:
      - "INSERT (:Person {name: 'Alix', age: 30})"
    query: "MATCH (p:Person) RETURN p.name, p.age"
    expect:
      rows:
        - [Alix, 30]
```

Run it:

```bash
cargo test -p grafeo-spec-tests --all-features
```

## Reporting a bug as a `.gtest`

If you find a query that produces wrong results, you can report it directly as a test case. Add it to the appropriate file under `tests/spec/` with a `skip:` reason:

```yaml
  - name: my_bug_label_intersection
    skip: "returns 3 rows instead of 1 when both labels required"
    setup:
      - "INSERT (:A:B {v: 1})"
      - "INSERT (:A {v: 2})"
      - "INSERT (:B {v: 3})"
    query: "MATCH (n:A:B) RETURN n.v"
    expect:
      rows:
        - [1]
```

The `skip:` field keeps CI green while documenting the expected behavior. When the bug is fixed, remove `skip:` and the test enforces the fix across all 7 bindings.

## File format reference

### Meta block

| Field | Default | Description |
|-------|---------|-------------|
| `language` | `gql` | Query language: `gql`, `cypher`, `gremlin`, `graphql`, `sparql`, `sql-pgq` |
| `model` | `lpg` | Data model: `lpg` (labeled property graph) or `rdf` |
| `dataset` | `empty` | Dataset to load: `empty` or a name from `datasets/` (e.g. `social_network`) |
| `requires` | `[]` | Feature gates: `[cypher]`, `[sparql, rdf]`, `[algos]`, etc. |
| `section` | | Spec reference (e.g. `"14.4"` for GQL clause) |
| `title` | | Human-readable file description |
| `tags` | `[]` | Arbitrary tags for filtering |

### Test case fields

| Field | Description |
|-------|-------------|
| `name` | Unique snake_case identifier (required) |
| `query` | Single query to execute. Use `\|` for multi-line block scalar |
| `statements` | List of queries: all execute, last captures result |
| `setup` | List of queries to run before the test (always in the file's `language`) |
| `skip` | Reason string: test is ignored with this message |
| `params` | Key-value map for parameterized queries |
| `variants` | Rosetta map: `{gql: "...", cypher: "..."}` runs the same test in multiple languages |
| `expect` | Assertion block (see below) |

### Expect block

| Field | Description |
|-------|-------------|
| `rows` | Expected rows as `- [col1, col2]`. Default: sorted comparison |
| `ordered` | `true` for exact row order comparison |
| `count` | Expected row count (no value check) |
| `empty` | `true` for zero rows |
| `error` | Substring that must appear in the error message |
| `columns` | Expected column names: `[name, age]` |
| `hash` | MD5 hex digest of sorted pipe-delimited rows |
| `precision` | Float tolerance: cells compared within `10^(-precision)` |

### Value types in rows

| Type | Syntax | Example |
|------|--------|---------|
| String | bare or quoted | `Alix`, `"hello world"` |
| Integer | digits | `42` |
| Float | decimal | `3.14` |
| Boolean | keyword | `true`, `false` |
| Null | keyword | `null` |
| List | JSON array | `[1, 2, 3]` |
| Map | JSON object | `{key: val}` |

### Block scalar queries

For multi-line queries, use `|`:

```yaml
    query: |
      MATCH (a:Person)-[:KNOWS]->(b:Person)
      WHERE a.age > 25
      RETURN a.name, b.name
```

### Rosetta variants (cross-language tests)

Test the same semantics across languages:

```yaml
  - name: count_all_nodes
    variants:
      gql: "MATCH (n) RETURN count(*) AS cnt"
      cypher: "MATCH (n) RETURN count(*) AS cnt"
    expect:
      rows:
        - [4]
```

Setup always runs in the file's declared `language`, not the variant language.

## Directory structure

```
tests/spec/
  datasets/           # Shared .setup files (GQL INSERT statements)
  common/             # Language-agnostic tests (algorithms, search)
  lpg/                # Labeled Property Graph tests
    gql/              # GQL (ISO/IEC 39075:2024)
    cypher/           # openCypher
    gremlin/          # Apache TinkerPop Gremlin
    graphql/          # GraphQL
    sql_pgq/          # SQL/PGQ (SQL:2023 GRAPH_TABLE)
  rdf/                # RDF model tests
    sparql/           # SPARQL 1.1
    graphql/          # GraphQL over RDF
  regression/         # Issue-mapped regression tests
  rosetta/            # Cross-language equivalence tests
  runners/            # Per-language test runners
    python/           # pytest plugin
    node/             # Vitest (shared by WASM)
    wasm/             # Vitest with WASM bindings
    go/               # Go testing.T
    csharp/           # xUnit
    dart/             # package:test
```

## Running the runners

```bash
# Rust (reference, always run first)
cargo test -p grafeo-spec-tests --all-features

# Python
cd tests/spec && pytest runners/python/ -v

# Node.js
npx vitest run tests/spec/runners/node/spec-runner.test.mjs

# WASM (requires wasm-pack build)
npx vitest run tests/spec/runners/wasm/spec-runner.test.mjs

# Go (requires libgrafeo_c)
cd tests/spec/runners/go && go test -count=1 -run TestSpec -v

# C#
cd tests/spec/runners/csharp && dotnet test

# Dart
cd tests/spec/runners/dart && dart test spec_runner_test.dart
```

## Dataset naming conventions

- **Person names**: Alix, Gus, Vincent, Jules, Mia, Butch, Django, Shosanna, Hans, Beatrix, Harm, Maxence, Lucas, Marcus.
- **Cities**: European (Amsterdam, Berlin, Paris, Prague, Barcelona)
- **Never**: Alice, Bob, Charlie or US-centric defaults

## Parser implementation

All 7 runners use line-based parsers (no YAML library dependency). The format is a strict subset of YAML: indentation-based structure, `key: value` pairs, `- item` lists, `|` block scalars, `[inline, lists]`, `#` comments. The parsers split on the first unquoted colon, which allows query syntax containing colons (like `:Label` and `{key: value}`) without quoting.
