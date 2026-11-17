# W3C SPARQL Test Suite Integration

This directory contains infrastructure for running the official W3C SPARQL
test suites against Grafeo's SPARQL implementation.

## Setup

The W3C RDF tests are available at https://github.com/w3c/rdf-tests.

To download them:

```bash
# Option 1: Git submodule (preferred)
git submodule add https://github.com/w3c/rdf-tests.git tests/spec/rdf/sparql/w3c/rdf-tests

# Option 2: Clone separately
git clone https://github.com/w3c/rdf-tests.git tests/spec/rdf/sparql/w3c/rdf-tests
```

## Converting W3C Manifests to .gtest Files

The conversion script reads a W3C manifest.ttl and writes one .gtest file of its
query evaluation tests, in the manifest's order:

```bash
# Convert the SPARQL 1.1 aggregates tests
uv run --with rdflib python scripts/convert_w3c_sparql.py \
  --manifest tests/spec/rdf/sparql/w3c/rdf-tests/sparql/sparql11/aggregates/manifest.ttl \
  --output tests/spec/rdf/sparql/w3c/aggregates.gtest

# Convert all SPARQL 1.1 tests
uv run --with rdflib python scripts/convert_w3c_sparql.py --all
```

Each case carries the result the manifest names. The data becomes one `INSERT DATA`
of N-Triples, with literals as the suite writes them. A SELECT result (.srx, .srj,
.tsv) becomes `terms` rows, with the columns in the order the query names its
variables; an ASK result becomes a `true` or `false` row. Rows compare as a sorted
bag, so `ORDER BY` is not checked.

Tests the .gtest format cannot express are left out and listed with their reason in
the file's header: named graphs, updates, syntax-only tests, CSV format tests,
entailment regimes, federated service data and graph results.

## Tracking Known Failures

`known_failures.toml` tracks tests that are expected to fail, with categorized reasons:

- `unimplemented`: Feature not yet supported
- `spec-disagreement`: Intentional deviation from spec
- `known-bug`: Known issue, tracked in GitHub

## Conformance Scorecard

Run `python scripts/sparql_conformance.py` to generate a compliance report
from the known_failures.toml file.
