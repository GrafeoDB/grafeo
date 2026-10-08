# A compacted database

`tests/compacted_file_without_compact_store.rs` opens these: a build without the `compact-store` feature must
refuse such a file (a read-write open, a read-only open and `open_in_memory`) and leave it as it is, never serve
or checkpoint the overlay alone. Each directory holds one database, `people.grafeo`:

- `0.6.0-dev/`: written by a 0.6 development build (on commit `1c75897f` with the 0.6.0 work in progress on top,
  a 0.6 file).

The database went through these steps, in a persistent database (`Config::persistent`):

1. `INSERT (:Person {name: 'Alix'})-[:KNOWS]->(:Person {name: 'Gus'})-[:KNOWS]->(:Person {name: 'Vincent'})`.
2. `compact()`: the three people and both edges are in the compacted base.
3. `MATCH (g:Person {name: 'Gus'}) DETACH DELETE g`: Gus and both edges are deleted from the base (the
   deletion log holds them).
4. `MATCH (a:Person {name: 'Alix'}) SET a.city = 'Paris'`: the overlay holds a copy of Alix.
5. `INSERT (:Person {name: 'Mia'})`: the overlay holds Mia.
6. `close()`.

So the file holds the base (`CompactStore` section), the deletion log (`OverlayDeletions`) and the overlay (the LPG
section with Alix and Mia), and a build that reads all of it finds Alix (in Paris), Mia and Vincent, and no edges;
the overlay alone has no Vincent.

Write it again (after a change of the file format, which `the_fixture_is_a_compacted_database` reports) with the
test that runs these steps:

```bash
GRAFEO_WRITE_COMPACTED_FIXTURE=$PWD/crates/grafeo-engine/tests/fixtures/compacted/0.6.0-dev/people.grafeo \
  cargo test -p grafeo-engine --all-features --test compacted_file_without_compact_store write_the_fixture
```
