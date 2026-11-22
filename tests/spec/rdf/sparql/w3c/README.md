# W3C SPARQL Test Suite

The `.gtest` files here are the query evaluation tests of the W3C SPARQL 1.1 test
suite, converted by `scripts/convert_w3c_sparql.py`. Every spec runner runs them
like the other `.gtest` files.

## Source and licence

The cases come from [w3c/rdf-tests](https://github.com/w3c/rdf-tests) at commit
`7b5afb91857a9c282c03d2e45d3f76485797c901` (2026-11-12); the repository has no
release tags, so the commit is the pin. Each file's header names it as well.

The test suite is Copyright W3C (MIT, ERCIM, Keio, Beihang) and is used here under
the [licences for W3C test suites](https://www.w3.org/Consortium/Legal/2008/04-testsuite-copyright.html)
(dual: the W3C Test Suite Licence and the W3C 3-clause BSD Licence). Only the
generated files are kept here; the suite's own files are not.

## Regenerating

Check out the suite at the pinned commit (the folder is ignored by git), then convert:

```bash
git clone https://github.com/w3c/rdf-tests.git tests/spec/rdf/sparql/w3c/rdf-tests
git -C tests/spec/rdf/sparql/w3c/rdf-tests checkout 7b5afb91857a9c282c03d2e45d3f76485797c901

# Convert one section
uv run --with rdflib python scripts/convert_w3c_sparql.py \
  --manifest tests/spec/rdf/sparql/w3c/rdf-tests/sparql/sparql11/aggregates/manifest.ttl \
  --output tests/spec/rdf/sparql/w3c/aggregates.gtest

# Convert every SPARQL 1.1 section
uv run --with rdflib python scripts/convert_w3c_sparql.py --all
```

The output is the same on every run and machine. To move to a newer commit, check
it out, convert, review the differences, and update the commit above.

Each case carries the result the manifest names. The data becomes one `INSERT DATA`
of N-Triples, with literals as the suite writes them. A SELECT result (.srx, .srj,
.tsv) becomes `terms` rows, with the columns in the order the query names its
variables; an ASK result becomes a `true` or `false` row. Rows compare as a sorted
bag, so `ORDER BY` is not checked.

Tests the .gtest format cannot express are left out and listed with their reason in
the file's header: named graphs, updates, syntax-only tests, CSV format tests,
entailment regimes, federated service data and graph results.

## Known failures

`known_failures.toml` lists, by section, the cases Grafeo does not pass:

- `{ issue = "...", note = "..." }`: the case becomes a strict expected failure
  (`xfail:`) held against its W3C test, naming the issue or release that fixes it.
  Remove the entry once fixed and convert again.
- `{ reason = "spec-disagreement", note = "..." }`: the case is left out and named
  in the file's header, because no fix of Grafeo can pass it as written: the suite
  writes some doubles in a non-canonical form, and some computed decimals in the
  XSD 1.0 canonical form (`2.0`) where Grafeo writes the XSD 1.1 one (`2`); it
  compares numbers by value, while `terms` compares lexical forms.

The converter stops when an entry names no converted case, so the list cannot go
stale.
