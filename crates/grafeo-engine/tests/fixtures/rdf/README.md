# A database with RDF triples

`tests/rdf_file_without_triple_store.rs` opens these: a build without the `triple-store` feature must refuse such a
file (a read-write open, a read-only open and `open_in_memory`) and leave it as it is, never serve or checkpoint the
database without its triples. Each directory holds one database, `triples.grafeo`:

- `0.6.0-dev/`: written by a 0.6 development build (on commit `565a7113` with the 0.6.0 work in progress on top,
  a 0.6 file).

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

Write it again (after a change of the file format, which `the_fixture_holds_rdf_triples` reports) with the test that
runs these steps:

```bash
GRAFEO_WRITE_RDF_FIXTURE=$PWD/crates/grafeo-engine/tests/fixtures/rdf/0.6.0-dev/triples.grafeo \
  cargo test -p grafeo-engine --all-features --test rdf_file_without_triple_store write_the_fixture
```
