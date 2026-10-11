# A snapshot with RDF triples and vector and text indexes

`tests/snapshot_without_its_features.rs` imports and restores these: a build without the `triple-store`,
`vector-index` or `text-index` feature must refuse such a snapshot (`import_snapshot` creates no database,
`restore_snapshot` leaves the database as it was), never import or restore it without the triples or the index
definitions. Each directory holds one snapshot, `documents.snapshot`, the bytes `export_snapshot` returned:

- `0.6.0-dev/`: exported by a 0.6 development build (on commit `565a7113` with the 0.6.0 work in progress on top,
  snapshot format version 4).

The snapshot was exported from an in-memory database (`GrafeoDB::new_in_memory`) of a build with all features,
after these steps:

1. `INSERT (:Document {title: 'Canals', content: 'boats on the canals of Amsterdam', embedding: vector([3.0, 19.0,
   88.0])}), (:Document {title: 'Bridges', content: 'bridges over the river in Prague', embedding: vector([88.0,
   19.0, 3.0])})`.
2. `create_vector_index("Document", "embedding", Some(3), Some("euclidean"), Some(19), Some(88), None)`: a vector
   index with 3 dimensions, the Euclidean distance, `m` 19 and `ef_construction` 88.
3. `create_text_index("Document", "content")`.
4. `batch_insert_rdf` of `<http://example.org/alix> <http://example.org/knows> <http://example.org/gus>`: a triple
   in the default graph.
5. `rdf_store().graph_or_create("http://example.org/trips")` and an insert of `<http://example.org/gus>
   <http://example.org/visited> <http://example.org/prague>` into it: a triple in a named graph.

So the snapshot holds the documents, the definitions of the two indexes and the two triples. A build with the three
features imports it with all of that; a build without one of them used to import it without the triples or the
indexes.

`fixtures/snapshot_v4.bin` (the golden snapshot of `tests/golden_format.rs`) is another snapshot: it checks that the
snapshot format does not change.

Write it again (after a change of the snapshot format, which `the_fixture_holds_triples_and_search_indexes` reports)
with the test that runs these steps:

```bash
GRAFEO_WRITE_SNAPSHOT_FIXTURE=$PWD/crates/grafeo-engine/tests/fixtures/snapshots/0.6.0-dev/documents.snapshot \
  cargo test -p grafeo-engine --all-features --test snapshot_without_its_features write_the_fixture
```
