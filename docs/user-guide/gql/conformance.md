# GQL Conformance (ISO/IEC 39075:2024)

This document maps Grafeo's GQL implementation against the ISO/IEC 39075:2024 standard,
declaring which features are supported, partially supported or not yet implemented.

Grafeo targets **minimum conformance** plus a broad set of optional features.

A machine-readable version of this compliance matrix is available as
[`gql-dialect.json`](../../gql-dialect.json) for tool integration (e.g. [GraphGlot](https://graphglot.com/)).
Regenerate it with `python scripts/gql-conformance.py dialect`.

## Minimum Conformance (Section 24.2)

Per the standard, minimum conformance requires:

| Requirement | Status | Notes |
|-------------|--------|-------|
| Open graph type (GG01) | Supported | Default graph type is open |
| STRING type | Supported | |
| BOOLEAN type | Supported | |
| Signed INTEGER type | Supported | 64-bit (INT64) |
| FLOAT type | Supported | 64-bit (FLOAT64) |
| Unicode >= 13.0 | Supported | Uses Rust's native Unicode support |

## Core Language: Statements

### Query Statements (Section 14)

| Feature | Status | Notes |
|---------|--------|-------|
| MATCH clause | Supported | Single and multiple patterns |
| OPTIONAL MATCH | Supported | Left-join semantics |
| WHERE clause | Supported | Standalone and element-level |
| RETURN clause | Supported | With DISTINCT, aliases |
| RETURN * | Supported | Returns all bound variables |
| ORDER BY (ASC/DESC) | Supported | |
| SKIP / OFFSET | Supported | Both keywords accepted |
| LIMIT / FETCH FIRST n ROWS | Supported | Both forms |
| GROUP BY | Supported | Explicit grouping |
| HAVING | Supported | Aggregate result filtering |
| FINISH | Supported | Consume input, return empty |
| EXPLAIN | Supported | Returns query plan |

### Data Modification (Section 13)

| Feature | Status | Notes |
|---------|--------|-------|
| INSERT (graph patterns) | Supported | Nodes and edges |
| SET (property assignment) | Supported | Single and map assignment |
| SET (label operations) | Supported | Add labels |
| REMOVE (properties) | Supported | |
| REMOVE (labels) | Supported | |
| DELETE | Supported | |
| DETACH DELETE | Supported | Auto-removes connected edges |
| MERGE | Supported | ON CREATE / ON MATCH actions |

### Catalog / DDL (Section 12)

| Feature | Status | Notes |
|---------|--------|-------|
| CREATE SCHEMA | Supported | With IF NOT EXISTS |
| DROP SCHEMA | Supported | With IF EXISTS |
| CREATE GRAPH | Supported | With IF NOT EXISTS, optional TYPED |
| DROP GRAPH | Supported | With IF EXISTS |
| CREATE GRAPH TYPE | Supported | Open/closed, node/edge type lists |
| DROP GRAPH TYPE | Supported | With IF EXISTS |
| CREATE NODE TYPE | Supported | With property definitions |
| DROP NODE TYPE | Supported | |
| CREATE EDGE TYPE | Supported | With property definitions |
| DROP EDGE TYPE | Supported | |
| ALTER NODE/EDGE/GRAPH TYPE | Supported | ADD/DROP property, ADD/DROP type |
| CREATE INDEX | Supported | Property, text, vector, B-tree |
| CREATE CONSTRAINT | Supported | UNIQUE, NOT NULL, NODE KEY, EXISTS |
| CREATE PROCEDURE | Supported | Params, returns, GQL body |

### Session Management (Section 7)

| Feature | Status | Notes |
|---------|--------|-------|
| USE GRAPH | Supported | |
| SESSION SET GRAPH | Supported | |
| SESSION SET TIME ZONE | Supported | |
| SESSION SET PARAMETER | Supported | |
| SESSION RESET | Supported | |
| SESSION CLOSE | Supported | |

### Transaction Management (Section 8)

| Feature | Status | Notes |
|---------|--------|-------|
| START TRANSACTION | Supported | |
| READ ONLY / READ WRITE mode | Supported | |
| ISOLATION LEVEL | Supported | READ COMMITTED, SNAPSHOT, SERIALIZABLE |
| COMMIT | Supported | |
| ROLLBACK | Supported | |
| SAVEPOINT | Supported | Named savepoints within transactions |
| ROLLBACK TO SAVEPOINT | Supported | Partial rollback to savepoint |
| RELEASE SAVEPOINT | Supported | Discard savepoint, keep changes |

### Procedure Calling (Section 15)

| Feature | Status | Notes |
|---------|--------|-------|
| CALL named_procedure(args) | Supported | Qualified names |
| YIELD clause | Supported | With aliases |
| Inline CALL { subquery } | Supported | |
| OPTIONAL CALL | Supported | Left-join semantics |

### Composite Queries (Section 14)

| Feature | Status | Notes |
|---------|--------|-------|
| UNION / UNION ALL | Supported | |
| EXCEPT / EXCEPT ALL | Supported | |
| INTERSECT / INTERSECT ALL | Supported | |
| OTHERWISE | Supported | Fallback if left is empty |
| NEXT (linear composition) | Supported | Output feeds into next |

## Graph Patterns (Section 16)

### Node and Edge Patterns

| Feature | Status | Notes |
|---------|--------|-------|
| Node patterns `(n:Label)` | Supported | Variables, labels, properties |
| Edge patterns `-[:TYPE]->` | Supported | All directions |
| Undirected edges `-[]-` | Supported | |
| Element-level WHERE `(n WHERE ...)` | Supported | Nodes and edges |
| Property filters `{key: value}` | Supported | |

### Label Expressions (Section 16.8)

| Feature | Status | Notes |
|---------|--------|-------|
| Single label `:Label` | Supported | |
| IS syntax `IS Label` | Supported | |
| Disjunction `\|` | Supported | `IS Person \| Company` |
| Conjunction `&` | Supported | `IS Person & Employee` |
| Negation `!` | Supported | `IS !Inactive` |
| Wildcard `%` | Supported | Matches any label |

### Path Modes (Section 16.6)

| Feature | Status | Notes |
|---------|--------|-------|
| WALK (default) | Supported | Repeated nodes/edges allowed |
| TRAIL | Supported | No repeated edges |
| SIMPLE | Supported | No repeated nodes (except endpoints) |
| ACYCLIC | Supported | No repeated nodes at all |

### Path Search Prefixes (Section 16.6)

| Feature | Status | Notes |
|---------|--------|-------|
| ALL | Supported | |
| ANY | Supported | |
| ANY k | Supported | |
| ALL SHORTEST | Supported | |
| ANY SHORTEST | Supported | |
| SHORTEST k | Supported | |
| SHORTEST k GROUPS | Supported | |

### Match Modes (Section 16.4)

| Feature | Status | Notes |
|---------|--------|-------|
| DIFFERENT EDGES | Supported | |
| REPEATABLE ELEMENTS | Supported | |
| KEEP clause (per-pattern) | Supported | |

### Quantifiers (Section 16.11)

| Feature | Status | Notes |
|---------|--------|-------|
| `{m,n}` bounded | Supported | |
| `{m,}` lower-bounded | Supported | |
| `{n}` fixed | Supported | Normalized to `{n,n}` |
| `*` (0 or more) | Supported | Normalized to `{0,}` |
| `+` (1 or more) | Supported | Normalized to `{1,}` |
| `?` questioned | Supported | 0 or 1 hop |
| Parenthesized quantified `((a)-[e]->(b)){2,5}` | Supported | |

### Pattern Composition

| Feature | Status | Notes |
|---------|--------|-------|
| Multiple comma-separated patterns | Supported | Implicit join on shared variables |
| Pattern union `\|` | Supported | |
| Path aliases `p = (a)-[*]-(b)` | Supported | |
| shortestPath / allShortestPaths | Supported | Cypher-style functions |

## Expressions and Predicates

### Operators

| Feature | Status | Notes |
|---------|--------|-------|
| Comparison: `=`, `<>`, `<`, `>`, `<=`, `>=` | Supported | |
| Logical: AND, OR, NOT, XOR | Supported | |
| Arithmetic: `+`, `-`, `*`, `/`, `%` | Supported | |
| String concatenation `\|\|` | Supported | |
| LIKE pattern matching | Supported | |
| IN list membership | Supported | |
| STARTS WITH / ENDS WITH / CONTAINS | Supported | |

### Predicates (Section 19)

| Feature | Status | Notes |
|---------|--------|-------|
| IS [NOT] NULL | Supported | |
| IS [NOT] DIRECTED (G110) | Supported | Desugared to function call |
| IS [NOT] LABELED (G111) | Supported | Desugared to function call |
| IS [NOT] SOURCE/DESTINATION OF (G112) | Supported | Desugared to function call |
| IS [NOT] TYPED (GA06) | Supported | Desugared to function call |
| IS [NOT] NORMALIZED | Supported | NFC, NFD, NFKC, NFKD forms supported |
| ALL_DIFFERENT (G113) | Supported | Multi-variable and single-list forms |
| SAME (G114) | Supported | Multi-variable and single-list forms |
| PROPERTY_EXISTS (G115) | Supported | Desugared to function call |

### Value Expressions (Section 20)

| Feature | Status | Notes |
|---------|--------|-------|
| Literals (int, float, string, bool, null) | Supported | |
| Typed temporal literals (DATE, TIME, etc.) | Supported | DATE, TIME, DATETIME, DURATION, ZONED DATETIME/TIME |
| Variable references | Supported | |
| Property access `n.prop` | Supported | |
| Parameter references `$name` | Supported | |
| Function calls | Supported | Built-in and user-defined |
| CASE WHEN / THEN / ELSE | Supported | Simple and searched |
| CAST(expr AS type) (GA05) | Supported | |
| List literals `[1, 2, 3]` | Supported | |
| Map literals `{k: v}` | Supported | |
| Index access `list[0]` | Supported | |
| List comprehension `[x IN list WHERE p \| e]` | Supported | |
| List predicates (all/any/none/single) | Supported | |
| REDUCE accumulator | Supported | |
| LET ... IN ... END (GE03) | Supported | |
| EXISTS { subquery } | Supported | |
| COUNT { subquery } | Supported | |
| VALUE { subquery } (GQ18) | Supported | |
| NULLIF / COALESCE | Supported | Both keyword syntax and function calls |

### Aggregate Functions

| Feature | Status | Notes |
|---------|--------|-------|
| count(*) / count(expr) | Supported | Including DISTINCT |
| sum / avg / min / max | Supported | |
| collect_list | Supported | |
| stdev / percentile | Supported | Extension |
| Vertical aggregation (GROUP BY) | Supported | |
| Horizontal aggregation (GE09) | **Not yet** | Group list variables from var-length patterns |

## Optional Features (Annex D)

### Pattern Features

| ID | Feature | Status |
|----|---------|--------|
| G002 | Different-edges match mode | Supported |
| G003 | Explicit REPEATABLE ELEMENTS | Supported |
| G004 | Path variables | Supported |
| G005 | Path search prefix in path pattern | Supported |
| G006 | KEEP clause: path mode prefix | Supported |
| G007 | KEEP clause: path search prefix | Supported |
| G010 | Explicit WALK keyword | Supported |
| G011 | Advanced path modes: TRAIL | Supported |
| G012 | Advanced path modes: SIMPLE | Supported |
| G013 | Advanced path modes: ACYCLIC | Supported |
| G014 | Explicit PATH/PATHS keywords | Supported |
| G015 | All path search: explicit ALL keyword | Supported |
| G016 | Any path search | Supported |
| G017 | All shortest path search | Supported |
| G018 | Any shortest path search | Supported |
| G019 | Counted shortest path search | Supported |
| G020 | Counted shortest group search | Supported |
| G030 | Path multiset alternation | **Supported** |
| G031 | Path multiset alternation: var-length operands | **Supported** |
| G032 | Path pattern union | Supported |
| G033 | Path pattern union: var-length operands | Supported |
| G035 | Quantified paths | Supported |
| G036 | Quantified edges | Supported |
| G037 | Questioned paths | Supported |
| G038 | Parenthesized path pattern expression | Supported |
| G039 | Simplified path pattern: full defaulting | Supported |
| G041 | Non-local element pattern predicates | **Not yet** |
| G043 | Complete full edge patterns | Supported |
| G044 | Basic abbreviated edge patterns | Supported |
| G045 | Complete abbreviated edge patterns | Supported |
| G046 | Relaxed topological consistency: adjacent vertex | **Not yet** |
| G047 | Relaxed topological consistency: concise edge | **Not yet** |
| G048 | Parenthesized path: subpath variable declaration | Partial (parsed, not yet bound in plan) |
| G049 | Parenthesized path: path mode prefix | Supported |
| G050 | Parenthesized path: WHERE clause | Supported |
| G051 | Parenthesized path: non-local predicates | **Not yet** |
| G060 | Bounded graph pattern quantifiers | Supported |
| G061 | Unbounded graph pattern quantifiers | Supported |
| G074 | Label expression: wildcard label | Supported |
| G080 | Simplified path pattern: basic defaulting | Supported |
| G081 | Simplified path pattern: full overrides | Supported |
| G082 | Simplified path pattern: basic overrides | Supported |

### Predicate & Expression Features

| ID | Feature | Status |
|----|---------|--------|
| G100 | ELEMENT_ID function | Supported |
| G110 | IS DIRECTED predicate | Supported |
| G111 | IS LABELED predicate | Supported |
| G112 | IS SOURCE / IS DESTINATION predicate | Supported |
| G113 | ALL_DIFFERENT predicate | Supported |
| G114 | SAME predicate | Supported |
| G115 | PROPERTY_EXISTS predicate | Supported |
| GA01 | IEEE 754 floating point operations | Supported |
| GA03 | Explicit ordering of nulls | Supported |
| GA04 | Universal comparison | **Not yet** |
| GA05 | Cast specification | Supported |
| GA06 | Value type predicate | Supported |
| GA07 | Ordering by discarded binding variables | **Not yet** |
| GA08 | GQL-status objects with diagnostic records | **Not yet** |
| GA09 | Comparison of paths | **Supported** |
| GE01 | Graph reference value expressions | **Not yet** |
| GE02 | Binding table reference value expressions | **Not yet** |
| GE03 | Let-binding in expressions | Supported |
| GE04 | Graph parameters | **Not yet** |
| GE05 | Binding table parameters | **Not yet** |
| GE06 | Path value construction | **Not yet** |
| GE07 | Boolean XOR | Supported |
| GE08 | Reference parameters | Supported |
| GE09 | Horizontal aggregation | **Not yet** |

### Function Features

| ID | Feature | Status |
|----|---------|--------|
| GF01 | Enhanced numeric functions | **Supported** |
| GF02 | Trigonometric functions | **Supported** |
| GF03 | Logarithmic functions | **Supported** |
| GF04 | Enhanced path functions | **Supported** |
| GF05 | Multi-character TRIM function | **Supported** |
| GF06 | Explicit TRIM function | Supported |
| GF07 | Byte string TRIM function | **Not yet** |
| GF10 | Advanced aggregates: general set functions | **Supported** |
| GF11 | Advanced aggregates: binary set functions | **Supported** |
| GF12 | CARDINALITY function | **Supported** |
| GF13 | SIZE function | Supported |
| GF20 | Aggregate functions in sort keys | Supported |

### Catalog & Graph Features

| ID | Feature | Status |
|----|---------|--------|
| GC01 | Graph schema management | Supported |
| GC02 | Schema management: IF [NOT] EXISTS | Supported |
| GC03 | Graph type: IF [NOT] EXISTS | Supported |
| GC04 | Graph management | Supported |
| GC05 | Graph management: IF [NOT] EXISTS | Supported |
| GD01 | Updatable graphs | Supported |
| GD02 | Graph label set changes | Supported |
| GD03 | DELETE: subquery support | **Supported** |
| GD04 | DELETE: simple expression support | **Supported** |
| GG01 | Graph with open graph type | Supported |
| GG02 | Graph with closed graph type | Supported |
| GG03 | Graph type inline specification | **Supported** |
| GG04 | Graph type like a graph | **Supported** |
| GG05 | Graph from a graph source | **Not yet** |
| GG20 | Explicit element type names | Supported |
| GG21 | Explicit element type key label sets | **Supported** |
| GG22 | Element type key label set inference | **Supported** |
| GG23 | Optional element type key label sets | **Not yet** |
| GG24 | Relaxed structural consistency | Inactive |
| GG25 | Relaxed key label set uniqueness for edge types | Inactive |
| GG26 | Relaxed property value type consistency | Inactive |

### Session & Transaction Features

| ID | Feature | Status |
|----|---------|--------|
| GS01 | SESSION SET: session-local graph parameters | Supported |
| GS02 | SESSION SET: session-local binding table parameters | **Not yet** |
| GS03 | SESSION SET: session-local value parameters | Supported |
| GS04 | SESSION RESET: all characteristics | Supported |
| GS05 | SESSION RESET: session schema | Supported |
| GS06 | SESSION RESET: session graph | Supported |
| GS07 | SESSION RESET: time zone displacement | Supported |
| GS08 | SESSION RESET: all session parameters | Supported |
| GS10 | SESSION SET: binding table parameters based on subqueries | **Not yet** |
| GS11 | SESSION SET: value parameters based on subqueries | **Not yet** |
| GS12 | SESSION SET: graph parameters based on simple expressions | **Not yet** |
| GS13 | SESSION SET: binding table parameters based on simple expressions | **Not yet** |
| GS14 | SESSION SET: value parameters based on simple expressions | **Not yet** |
| GS15 | SESSION SET: time zone displacement | Supported |
| GS16 | SESSION RESET: individual session parameters | **Not yet** |
| GT01 | Explicit transaction commands | Supported |
| GT02 | Specified transaction characteristics | Supported |
| GT03 | Use of multiple graphs in a transaction | **Not yet** |

### Procedure Features

| ID | Feature | Status |
|----|---------|--------|
| GP01 | Inline procedure | Supported |
| GP02 | Inline procedure with implicit nested variable scope | Supported |
| GP03 | Inline procedure with explicit nested variable scope | **Not yet** |
| GP04 | Named procedure calls | Supported |
| GP05 | Procedure-local value variable definitions | **Not yet** |
| GP06 | Procedure-local value variable definitions: based on simple expressions | **Not yet** |
| GP07 | Procedure-local value variable definitions: based on subqueries | **Not yet** |
| GP08 | Procedure-local binding table variable definitions | **Not yet** |
| GP09 | Procedure-local binding table variable definitions: based on simple expressions | **Not yet** |
| GP10 | Procedure-local binding table variable definitions: based on subqueries | **Not yet** |
| GP11 | Procedure-local graph variable definitions | **Not yet** |
| GP12 | Procedure-local graph variable definitions: based on simple expressions | **Not yet** |
| GP13 | Procedure-local graph variable definitions: based on subqueries | **Not yet** |
| GP14 | Binding tables as procedure arguments | **Not yet** |
| GP15 | Graphs as procedure arguments | **Not yet** |
| GP16 | AT schema clause | **Not yet** |
| GP17 | Binding variable definition block | **Not yet** |
| GP18 | Catalog and data statement mixing | **Not yet** |

### Query Composition Features

| ID | Feature | Status |
|----|---------|--------|
| GQ01 | USE graph clause | Supported |
| GQ02 | OTHERWISE | Supported |
| GQ03 | UNION | Supported |
| GQ04 | EXCEPT DISTINCT | Supported |
| GQ05 | EXCEPT ALL | Supported |
| GQ06 | INTERSECT DISTINCT | Supported |
| GQ07 | INTERSECT ALL | Supported |
| GQ08 | FILTER statement | Supported |
| GQ09 | LET statement | Supported |
| GQ10 | FOR statement: list value support | Supported |
| GQ11 | FOR statement: WITH ORDINALITY | Supported |
| GQ12 | ORDER BY: OFFSET clause | Supported |
| GQ13 | ORDER BY: LIMIT clause | Supported |
| GQ14 | Complex expressions in sort keys | Supported |
| GQ15 | GROUP BY clause | Supported |
| GQ16 | Pre-projection aliases in sort keys | Supported |
| GQ17 | Element-wise group variable operations | **Not yet** |
| GQ18 | Scalar subqueries | Supported |
| GQ19 | Graph pattern YIELD clause | **Not yet** |
| GQ20 | Advanced linear composition (NEXT) | Supported |
| GQ21 | OPTIONAL: multiple MATCH statements | Supported |
| GQ22 | EXISTS predicate: multiple MATCH | Supported |
| GQ23 | FOR statement: binding table support | **Not yet** |
| GQ24 | FOR statement: WITH OFFSET | Supported |

### Value Type Features

| ID | Feature | Status |
|----|---------|--------|
| GV01 | 8-bit unsigned integer numbers | **Not yet** |
| GV02 | 8-bit signed integer numbers | **Not yet** |
| GV03 | 16-bit unsigned integer numbers | **Not yet** |
| GV04 | 16-bit signed integer numbers | **Not yet** |
| GV05 | Small unsigned integer numbers | **Not yet** |
| GV06 | 32-bit unsigned integer numbers | **Not yet** |
| GV07 | 32-bit signed integer numbers | **Not yet** |
| GV08 | Regular unsigned integer numbers | **Not yet** |
| GV09 | Specified integer number precision | **Not yet** |
| GV10 | Big unsigned integer numbers | **Not yet** |
| GV11 | 64-bit unsigned integer numbers | **Not yet** |
| GV12 | 64-bit signed integer numbers | Supported |
| GV13 | 128-bit unsigned integer numbers | **Not yet** |
| GV14 | 128-bit signed integer numbers | **Not yet** |
| GV15 | 256-bit unsigned integer numbers | **Not yet** |
| GV16 | 256-bit signed integer numbers | **Not yet** |
| GV17 | Decimal numbers | **Not yet** |
| GV18 | Small signed integer numbers | **Not yet** |
| GV19 | Big signed integer numbers | **Not yet** |
| GV20 | 16-bit floating point numbers | **Not yet** |
| GV21 | 32-bit floating point numbers | **Not yet** |
| GV22 | Specified floating point number precision | **Not yet** |
| GV23 | Floating point type name synonyms | **Not yet** |
| GV24 | 64-bit floating point numbers | Supported |
| GV25 | 128-bit floating point numbers | **Not yet** |
| GV26 | 256-bit floating point numbers | **Not yet** |
| GV30 | Specified character string minimum length | **Not yet** |
| GV31 | Specified character string maximum length | **Not yet** |
| GV32 | Specified character string fixed length | **Not yet** |
| GV35 | Byte string types | **Not yet** |
| GV36 | Specified byte string minimum length | **Not yet** |
| GV37 | Specified byte string maximum length | **Not yet** |
| GV38 | Specified byte string fixed length | **Not yet** |
| GV39 | Temporal: date, local datetime/time | Supported |
| GV40 | Temporal: zoned datetime/time | Supported |
| GV41 | Temporal: duration | Supported |
| GV45 | Record types | **Not yet** |
| GV46 | Closed record types | **Not yet** |
| GV47 | Open record types | **Not yet** |
| GV48 | Nested record types | **Not yet** |
| GV50 | List value types | Supported |
| GV55 | Path value types | **Supported** |
| GV60 | Graph reference value types | **Not yet** |
| GV61 | Binding table reference value types | **Not yet** |
| GV65 | Dynamic union types | **Not yet** |
| GV66 | Open dynamic union types | **Not yet** |
| GV67 | Closed dynamic union types | **Not yet** |
| GV68 | Dynamic property value types | Supported |
| GV70 | Immaterial value types | **Not yet** |
| GV71 | Immaterial value types: null type support | Supported |
| GV72 | Immaterial value types: empty type support | **Not yet** |
| GV90 | Explicit value type nullability | Supported |

### Lexical Features

| ID | Feature | Status |
|----|---------|--------|
| GB01 | Long identifiers | Supported |
| GB02 | Double minus sign comments | Supported |
| GB03 | Double solidus comments | Supported |
| GL01 | Hexadecimal literals | **Supported** |
| GL02 | Octal literals | **Supported** |
| GL03 | Binary literals | **Supported** |
| GL04 | Exact number in common notation without suffix | **Not yet** |
| GL05 | Exact number in common notation or as decimal integer with suffix | **Not yet** |
| GL06 | Exact number in scientific notation with suffix | **Not yet** |
| GL07 | Approximate number in common notation or as decimal integer with suffix | **Not yet** |
| GL08 | Approximate number in scientific notation with suffix | **Not yet** |
| GL09 | Optional float number suffix | **Not yet** |
| GL10 | Optional double number suffix | **Not yet** |
| GL11 | Opt-out character escaping | **Not yet** |
| GL12 | SQL datetime and interval formats | **Not yet** |

### Miscellaneous Features

| ID | Feature | Status |
|----|---------|--------|
| GH01 | External object references | **Not yet** |
| GH02 | Undirected edge patterns | **Not yet** |

## Three-Valued Logic

GQL uses three-valued logic where comparisons involving NULL produce UNKNOWN (not FALSE).

| Behavior | Status |
|----------|--------|
| NULL = NULL returns UNKNOWN | Supported |
| NULL <> NULL returns UNKNOWN | Supported |
| FILTER passes only TRUE (not UNKNOWN) | Supported |
| IS NULL / IS NOT NULL for explicit checks | Supported |
| NULL handling in aggregates (ignored except count(*)) | Supported |

## Known Deviations from Standard

1. **Predicate desugaring**: IS DIRECTED, IS LABELED, IS SOURCE/DESTINATION OF are parsed and
   internally converted to function calls (isDirected(), hasLabel(), etc.) rather than being
   preserved as first-class predicate AST nodes. Semantics are equivalent.

2. **KEEP clause representation**: The standard specifies `KEEP <path pattern prefix>` wrapping
   all patterns. Grafeo implements KEEP as a per-pattern match mode flag, which is simpler but
   semantically equivalent.

3. **Variable scope validation**: The standard defines degree-of-exposure categories (unconditional
   singleton, conditional singleton, etc.) at parse time. Grafeo defers variable scope validation
   to execution time.

4. **Simplified path patterns**: The `-/:Label/->` shorthand is desugared to `-[:Label]->` at parse
   time. Both forms are semantically equivalent.
