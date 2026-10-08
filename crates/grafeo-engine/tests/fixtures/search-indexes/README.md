# A database with vector and text indexes

`tests/search_indexes_without_their_features.rs` opens these: a build without the `vector-index` or `text-index`
feature must refuse such a file (a read-write open, a read-only open and `open_in_memory`) and leave it as it is,
never checkpoint it without the definitions of its indexes. Each directory holds one database, `documents.grafeo`:

- `0.6.0-dev/`: written by a 0.6 development build (on commit `565a7113` with the 0.6.0 work in progress on top,
  a 0.6 file).

The database went through these steps, in a persistent database (`Config::persistent`) of a build with
`vector-index` and `text-index`:

1. `INSERT (:Document {title: 'Canals', content: 'boats on the canals of Amsterdam', embedding: vector([3.0, 19.0,
   88.0])}), (:Document {title: 'Bridges', content: 'bridges over the river in Prague', embedding: vector([88.0,
   19.0, 3.0])})`.
2. `create_vector_index("Document", "embedding", Some(3), Some("euclidean"), Some(19), Some(88), None)`: a vector
   index with 3 dimensions, the Euclidean distance, `m` 19 and `ef_construction` 88.
3. `create_text_index("Document", "content")`.
4. `CREATE GRAPH trips`, then in `trips`: `INSERT (:Stop {name: 'Berlin', position: vector([3.0, 19.0])})` and
   `CREATE VECTOR INDEX stop_position ON :Stop(position)`.
5. `close()`.

So the catalog defines the three indexes, and the default graph's two are also in their sections (`VectorStore`,
`TextIndex`); the named graph's vector index only the catalog holds. A build with both features finds the indexes
with their configuration; a build without one used to drop its definitions at its next checkpoint.

Write it again (after a change of the file format, which `the_fixture_holds_vector_and_text_indexes` reports) with
the test that runs these steps:

```bash
GRAFEO_WRITE_SEARCH_INDEXES_FIXTURE=$PWD/crates/grafeo-engine/tests/fixtures/search-indexes/0.6.0-dev/documents.grafeo \
  cargo test -p grafeo-engine --all-features --test search_indexes_without_their_features write_the_fixture
```
