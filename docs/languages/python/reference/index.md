---
title: Python API
description: Python API reference.
---

# Python API Reference

Complete reference for the `grafeo` Python package.

## Installation

```bash
uv add grafeo
```

## Quick Start

```python
import grafeo

db = grafeo.GrafeoDB()
db.execute("INSERT (:Person {name: 'Alix'})")
```

## Decimals

An exact decimal (GQL and SQL/PGQ `1.50`, `CAST(x AS DECIMAL(10, 2))`) comes
back as a `decimal.Decimal` with its scale, and a
`decimal.Decimal` parameter or property goes in exactly:

```python
from decimal import Decimal

db.execute("INSERT (:Item {price: 19.99})")
db.execute("MATCH (i:Item) RETURN i.price")[0]["i.price"]  # Decimal('19.99')
db.execute("RETURN $p * 3 AS total", {"p": Decimal("0.10")})  # Decimal('0.30')
```

A float stays a `float`. NaN, the infinities and a `Decimal` of more than 38
digits are refused with a `TypeError`. `to_pandas()` and `to_polars()` keep
the `Decimal` objects.

## RDF Terms

A SPARQL result holds each IRI, blank node and literal as the exact term:
`grafeo.IRI`, `grafeo.BlankNode` (its label, without `_:`) and
`grafeo.Literal`, all subclasses of `str`, so a term formats, indexes and
takes the `str` methods as its text. A `Literal` has `language` (`None`
without a tag, lowercase otherwise) and `datatype` (an `IRI`: `xsd:string` for
a simple literal, `rdf:langString` for a tagged one); `to_python()` gives its
native value (`int`, `Decimal`, `float`, `bool`, `date`, `datetime`, or the
text), and `n3()` the N-Triples form. A term passed as a parameter, or stored
as a property, is that exact term.

Terms compare and hash as RDF terms, as in rdflib: two terms are equal when
they are of the same kind with the same text, language tag and datatype, so
they work as set members and dict keys. A term never equals a plain `str`
(`IRI("http://ex.org/a") != "http://ex.org/a"`): compare `str(term)` for its
text. An `IRI` never equals a `Literal` of the same text, language tags
compare case-insensitively (`Literal("Pomme", language="FR")` is
`Literal("Pomme", language="fr")`), and a literal of the datatype
`xsd:string` is the simple literal of its text (`Literal("Apple")`).

```python
db = grafeo.GrafeoDB()
db.execute_sparql('INSERT DATA { <http://ex.org/apple> <http://ex.org/price> "1.50"^^<http://www.w3.org/2001/XMLSchema#decimal> }')
price = db.execute_sparql("SELECT ?p WHERE { ?s <http://ex.org/price> ?p }")[0]["p"]
price                # Literal('1.50', datatype='http://www.w3.org/2001/XMLSchema#decimal')
price.to_python()    # Decimal('1.50')
price == "1.50"      # False: a term is not a plain str
str(price) == "1.50" # True
db.create_node(["Item"], {"price": price})  # stored as that exact term
```

`to_arrow()`, `to_pandas()` and `to_polars()` give a literal of a known XSD
type as its native value (an `xsd:integer` column is `int64`, `xsd:decimal`
a decimal, `xsd:double` a float, and `xsd:boolean`, `xsd:date` and
`xsd:dateTime` their types), and any other term as text: an IRI, a blank node
as `_:label`, a literal as its lexical form. The language tag and the
datatype of a literal exported as text are dropped.

An ASK result is one row with the column `boolean`, and the result's `kind`
is `"boolean"`; CONSTRUCT and DESCRIBE give rows of `subject`, `predicate` and
`object` with the kind `"graph"`, and every other result `"rows"`.

## Classes

| Class | Description |
|-------|-------------|
| [Database](database.md) | Database connection and management |
| [Node](node.md) | Graph node representation |
| [Edge](edge.md) | Graph edge representation |
| [QueryResult](result.md) | Query result iteration |
| [Transaction](transaction.md) | Transaction management |

## Module Functions

| Function | Description |
|----------|-------------|
| `grafeo.features()` | Features compiled into this build, by Cargo feature name: the query languages (`gql`, `cypher`, `sparql`, `gremlin`, `graphql`, `sql-pgq`) and optional capabilities such as `algos`, `vector-index` or `triple-store` |
| `grafeo.build_info()` | Dict describing the build: `version`, `commit` (git commit, or `None` outside a git checkout), `dirty` (uncommitted changes at build time), `features` and `profile` (`"release"` or `"debug"`) |
| `grafeo.simd_support()` | SIMD instruction set used for vector operations: `"avx2"`, `"sse"`, `"neon"` or `"scalar"` |
| `grafeo.vector(values)` | Builds a vector value from a list of floats |

Check at startup that a build has what your application needs:

```python
import grafeo

missing = {"cypher", "algos"} - set(grafeo.features())
if missing:
    raise RuntimeError(f"grafeo build lacks: {sorted(missing)}")
```
