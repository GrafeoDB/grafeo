# A compacted database

`tests/compacted_file_without_compact_store.rs` opens these: a build without the `compact-store` feature must
refuse such a file (a read-write open, a read-only open and `open_in_memory`) and leave it as it is, never serve
or checkpoint the overlay alone; a build with it folds the base into the store as it opens the file. Each
directory holds one database, `people.grafeo`:

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

No build writes such a file any more (`compact()` no longer builds a compacted base), so the fixture cannot be
written again: it stays as it is until 0.7.0 drops the compacted sections.
