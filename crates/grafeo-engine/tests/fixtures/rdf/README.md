# Databases with RDF triples

## `triples.grafeo`: a build without `triple-store`

`tests/rdf_file_without_triple_store.rs` opens these: a build without the `triple-store` feature must refuse such a
file (a read-write open, a read-only open and `open_in_memory`) and leave it as it is, never serve or checkpoint the
database without its triples.

- `0.6.0/`: written in the current format (RDF and ring sections of version 4: term ids and a term table).
- `0.6.0-beta.1/`: written by a 0.6 development build of format revision 1 (rewritten on 2026-10-09 when the
  LPG section got stable ids and the column chunks of strings their zone maps), with RDF and ring sections of
  version 3 (N-Triples strings), as 0.6.0-beta.1 writes them. `tests/rdf_term_ids_on_disk.rs` opens it as the
  file a read-write open rewrites in version 4, keeping a copy as `<path>.pre-0.6.0`. Never write it again.

The database went through these steps, in a persistent database (`Config::persistent`) of a build with
`triple-store`, `sparql` and `ring-index`:

1. `INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})`: the LPG data.
2. SPARQL `INSERT DATA { <http://example.org/alix> <http://example.org/knows> <http://example.org/gus> . GRAPH
   <http://example.org/trips> { <http://example.org/gus> <http://example.org/visited> <http://example.org/prague> . } }`:
   a triple in the default graph and one in the named graph `http://example.org/trips`.
3. `rdf_store().rebuild_ring()`: the Ring index over the triples.
4. `close()`.

So the file holds the triples (`RdfStore` section), the Ring index (`RdfRing`) and the people (the LPG section), and
a build that reads all of it finds both triples and Alix and Gus; a build without `triple-store` refuses the file
rather than serving the people without their triples.

Write `0.6.0/triples.grafeo` again (after a change of the file format, which `the_fixture_holds_rdf_triples`
reports) with the test that runs these steps:

```bash
GRAFEO_WRITE_RDF_FIXTURE=$PWD/crates/grafeo-engine/tests/fixtures/rdf/0.6.0/triples.grafeo \
  cargo test -p grafeo-engine --all-features --test rdf_file_without_triple_store write_the_fixture
```

## `0.6.0/terms.grafeo`: the golden fixture of the term ids

`tests/rdf_term_ids_on_disk.rs` pins the format of the RDF and ring sections with it: every kind of term (each
literal the ids inline, a non-canonical integer, a double, a long string, tagged literals, a literal of another
datatype, a blank node) in the default graph and a named one, written by the SPARQL update `TERMS` of that test,
then `rdf_store().rebuild_ring()` and `close()`. `the_golden_fixture_pins_the_format` checks that it reads every
term exactly and that the same steps write the same RDF and ring chunks.

Write it again only with a change of the format, which a release names:

```bash
GRAFEO_WRITE_RDF_TERMS_FIXTURE=$PWD/crates/grafeo-engine/tests/fixtures/rdf/0.6.0/terms.grafeo \
  cargo test -p grafeo-engine --all-features --test rdf_term_ids_on_disk write_the_golden_fixture
```
