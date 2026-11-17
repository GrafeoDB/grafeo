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

The `skip:` field keeps CI green while documenting the expected behavior. When the bug is fixed, remove `skip:` and the test enforces the fix across all 7 bindings. The Rust runner still builds a skipped case, ignored with its reason, so `cargo test -p grafeo-spec-tests --all-features -- --ignored` runs every skipped case and shows which ones pass already.

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
| `values_dialect` | `strict` | How `values:` cells read: `strict`, or `tck` (converted openCypher TCK files), where a quoted cell also matches a temporal whose Cypher `toString()` is that text |

### Test case fields

| Field | Description |
|-------|-------------|
| `name` | Unique snake_case identifier (required) |
| `query` | Single query to execute. Use `\|` for multi-line block scalar |
| `statements` | List of queries: all execute, last captures result |
| `setup` | List of queries to run before the test (always in the file's `language`) |
| `setup_gql` | List of GQL statements run before `setup`, whatever the file's language (the openCypher TCK's test procedures as `CREATE PROCEDURE`) |
| `runner_gap` | A gap of some runners' bindings: `runners:` (any of `rust`, `python`, `node`, `wasm`, `go`, `csharp`, `dart`), `issue:` (`#N` or `fix in X.Y.Z`) and `reason:`. The runners named skip the case with that reason; the others run it. Counts as a skip |
| `skip` | Reason string: test is ignored with this message (the Rust runner runs it with `--ignored`) |
| `xfail` | A strict expected failure: `spec:` (what the expected result is held against) and `issue:` (`#N` or `fix in X.Y.Z`). The case must fail on an assertion or a statement error; a pass fails the run. `passes_in:` lists runners where it passes all the same and must (their binding cannot show what fails it, as JavaScript's one number type) |
| `params` | Key-value map for parameterized queries (see "Parameter values") |
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
| `gql_status` | GQLSTATUS the error must have (`"22012"`), alone or next to `error`; two characters name a class (`"42"` matches `42001`). The case fails when the statement succeeds, and when the binding exposes no status for the error |
| `columns` | Expected column names: `[name, age]` |
| `hash` | MD5 hex digest of sorted pipe-delimited rows |
| `precision` | Float tolerance: cells compared within `10^(-precision)` |
| `terms` | Expected rows of RDF terms, each cell compared as the exact term (see below). Sorted comparison unless `ordered: true` |
| `values` | Typed rows: a table in the openCypher TCK's value notation, its first row the column names (see below). Compared as a bag unless `ordered: true`. Excludes `rows`, `terms`, `count`, `empty`, `error` and `gql_status` |
| `unordered_lists` | `true` compares every list in `values` as a bag |
| `side_effects` | The statement's effect on the graph, as the TCK counts it (see below); `{}` is none |

Any other key under `expect:` fails the test in every runner, so a misspelled key cannot pass without checking anything.

### RDF terms

`rows` compares a SPARQL cell by its plain value, so it cannot tell an IRI from a string with the same text, or `"1"^^xsd:integer` from `"1"`. Use `terms` when the kind of term, the datatype or the language tag is what the test is about. Each cell is the term in N-Triples form, quoted in YAML:

| Term | Written as | Example |
|------|------------|---------|
| IRI | `<iri>` | `'<http://example.org/alix>'` |
| Blank node | `_:label` | `'_:b0'` |
| Simple literal (xsd:string) | `"text"` | `'"Alix"'` |
| Language-tagged literal | `"text"@tag` | `'"Amsterdam"@nl'` |
| Typed literal | `"text"^^<datatype>` or `"text"^^xsd:name` | `'"1.0E3"^^xsd:double'` |
| Unbound | `null` | `null` |

The lexical form is compared exactly, so `"025"^^xsd:integer` is not `"25"^^xsd:integer`. A language tag compares without case, and `"text"^^xsd:string` is the same as `"text"`. The `xsd:` and `rdf:` prefixes expand to their namespaces. A blank node matches any blank node; with `ordered: true` the labels must also correspond one to one between the expected and the actual rows.

```yaml
expect:
  ordered: true
  terms:
    - ['<http://example.org/alix>', '"Alix"@en']
    - ['_:b0', '"30"^^xsd:integer']
```

### Typed rows (`values:`)

`rows` compares each cell as text, so `'1'`, `1` and `1.0` all match `1`. `values` compares type-exact, in the openCypher TCK's value notation, written as the TCK's own table:

```yaml
expect:
  values: |
    | n                       | count |
    | (:Person {name: 'Gus'}) | 1     |
```

| Written | Matches |
|---------|---------|
| `42`, `-7` | an integer of that value, never a float |
| `1.0`, `1e-3`, `NaN`, `Inf`, `-Inf` | a float equal to it, never an integer |
| `'text'` | a string, exactly (`\'` and `\\` escapes) |
| `true`, `false`, `null` | that boolean, null |
| `[a, b]` | a list, element by element (a bag with `unordered_lists: true`) |
| `{k: v}` | a map with exactly these keys |
| `(:A:B {p: 1})` | a node: the same label set and properties |
| `[:T {p: 1}]` | a relationship: the same type and properties |
| `<(:A)-[:T]->(:B)<-[:U]-()>` | a path, node by node and relationship by relationship, each arrow its direction |

A `|` inside a cell is written `\|`. A runner recognises a node, a relationship and a path by the shape results give them (a node is a map with `_id` and `_labels`), so a map a query builds with those keys counts as one. Every runner checks its comparison against `runners/values_cases.json`.

### Side effects (`side_effects:`)

```yaml
expect:
  side_effects:
    +nodes: 1
    +properties: 2
```

The keys are `+nodes`, `-nodes`, `+relationships`, `-relationships`, `+labels`, `-labels`, `+properties` and `-properties`; a key left out is 0. The runner reads the graph right before and right after the statement and counts what changed, as the openCypher TCK defines it: nodes and relationships by id, labels as the set of label names in use, properties as (entity, key, value) triples, so changing a value is one `-properties` and one `+properties`.

### Parameter values

Every runner types a `params` value by the same rule, so a parameter has the same type in every binding:

| Written as | Type | Example |
|------------|------|---------|
| Quoted (single or double) | String, whatever it reads like | `"Alix"`, `"[Amsterdam]"`, `"88"` |
| Bare, starting with `[` or `{` | JSON list or map (must be valid JSON); its numbers are typed as bare ones | `[3, 19, 88]`, `{"city": "Amsterdam"}` |
| Bare decimal integer that fits 64 bits | Integer | `88`, `-3`, `+19` |
| Bare decimal integer past 64 bits, up to 38 digits | Exact decimal (scale 0) | `9223372036854775808` |
| Bare decimal integer past 38 digits | An error naming the parameter, in every binding | `100000000000000000000000000000000000000` |
| Other bare decimal number, representable as a finite f64 | Float | `3.5`, `.5e1`, `19.` |
| Bare `true` or `false` | Boolean | `true` |
| Bare `null` | Null (a quoted `'null'` is the string) | `null` |
| Any other bare value | String | `Alix`, `0x1F`, `1_000`, `inf` |

A number past the f64 range, bare (`1e999`) or inside JSON (`[1e400]`), is an error in every runner: the Rust runner refuses it when it builds, so no test can use one.

A value is written on one line: a block scalar (`|` or `>`) is not supported for params, and the Rust runner refuses one when it builds. Write a multi-line string as a double-quoted string with `\n`, and a list or map as JSON.

A runner passes an integer past 64 bits to its binding with all its digits (Python an `int`, Node.js and WASM a `BigInt`, C# a `BigInteger`, Dart a `BigInt`, Go the digits as written), and the binding types it: the integer rule is the binding's, so the tests check it.

Do not use a JSON object with a key `$date`, `$time`, `$duration`, `$zoned_datetime` or `$timestamp_us` in params: the bindings of Node.js, WASM, Go, C# and Dart decode such an object as a temporal value, while the Rust runner keeps it a map, so the same test would mean two things.

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

For multi-line queries, use `|` (`query`, `variants` and list items; not `params`, see "Parameter values"):

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
