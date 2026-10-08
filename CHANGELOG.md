# Changelog

All notable changes to Grafeo, for future reference (and enjoyment).

## [0.6.0] - Unreleased

File format release: every database is now a single file in a new format, and 0.5.x databases migrate to it on their first read-write open. Encryption at rest now encrypts, writes after `close()` fail instead of being lost, and read-only opens recover the last commits after a crash. Plus graph algorithms on projections and named graphs, algorithm results in a fixed order, corrected k-core and Louvain, and faster GQL statements and multi-label patterns.

> **Breaking: new file format.** The first read-write open migrates a 0.5.x database (a `.grafeo` file or a WAL directory) and keeps the old files, unchanged, under `.pre-0.6` names. 0.5.x cannot open the migrated database, so stop every 0.5.x process that uses it before 0.6 opens it, and see [Upgrading from 0.5](https://grafeo.dev/user-guide/persistence/persistent/#upgrading-from-05). Also breaking: a new database is always a single file, `save()` fails on an existing path, opening a directory that is not a 0.5.x database fails, the minimum Rust version is 1.99.0, property values nest at most 128 levels deep, several Rust calls return `Result`, Python `kcore()` returns a dict and Go `DropVectorIndex` an error (see Changed).

### Added

- **Graph algorithms on a projection or a named graph** ([#566](https://github.com/GrafeoDB/grafeo/issues/566)): `db.algorithms` methods take `projection=`, algorithm procedures a `projection` argument (`CALL grafeo.pagerank({projection: 'people'})`), and `db.graph(name).algorithms` runs on that graph. Rust: `GrafeoDB::selected_graph_store` and `GraphHandle::graph_store`.
- **Undirected PageRank** ([#566](https://github.com/GrafeoDB/grafeo/issues/566)): `directed=False` in `db.algorithms.pagerank()` and `as_solvor().pagerank()`, or `CALL grafeo.pagerank({directed: false})`, counts each pair of connected nodes once in both directions, whatever the number or types of edges between them; self-loops are ignored. The default stays directed.
- **Algorithm results keyed by a node property** ([#566](https://github.com/GrafeoDB/grafeo/issues/566)): the per-node methods of `db.algorithms` take `key=` to key their results by a node property, such as an id from outside the database. PageRank, Louvain and label propagation then also run in key order, so their results do not depend on the load order. A missing, duplicate or unhashable key raises an error before the run (see [Determinism](https://grafeo.dev/algorithms/#determinism)).
- **Error code `GRAFEO-T008` (incomplete commit)** ([#412](https://github.com/GrafeoDB/grafeo/issues/412)): after a commit that panicked midway, commits, checkpoints, saves and copies fail with it until the database is reopened. Reads still work (without the `temporal` feature they can see some of the failed commit's property and label changes). It is not retryable.
- **`ZONED DATETIME`, `LOCAL DATETIME` and typed lists as property types** ([#569](https://github.com/GrafeoDB/grafeo/issues/569)): `CREATE NODE TYPE`, `CREATE EDGE TYPE`, `ALTER ... ADD` and inline graph types accept them (`LIST<ZONED DATETIME>` included, nested at most 128 levels), check values against them, and keep them after a reopen. `LOCAL DATETIME` takes the values `local_datetime()` returns, `ZONED DATETIME` only zoned ones.
- **WASM: `executeRawWithParams` and `executeRawWithLanguageAndParams`** ([#574](https://github.com/GrafeoDB/grafeo/issues/574)): the raw result shape with parameters, lists and maps included.
- **Rust (`grafeo-engine`, feature `encryption`): `GrafeoDB::restore_to_epoch_with`** restores the backups of an encrypted database with its key chain; `restore_to_epoch` refuses an encrypted backup with incremental segments to replay.

### Changed

- **Versioning**: before 1.0, a release that changes the file format or breaks the stable surface (the `grafeo` crate, the bindings, the CLI) bumps the minor version, as Cargo and npm expect for `0.x`, so this release, first planned as 0.5.45, is 0.6.0. See [Versioning and Compatibility](https://grafeo.dev/versioning/).
- **Breaking: new file format, migrated on open** ([#555](https://github.com/GrafeoDB/grafeo/issues/555)): the first read-write open of a 0.5.x `.grafeo` file migrates it and keeps the old file and its WAL as `<file>.pre-0.6` and `<file>.pre-0.6.wal`; read-only opens read it without changing it. 0.7.0 will no longer read 0.5.x databases, so open them read-write with 0.6 first. See [Upgrading from 0.5](https://grafeo.dev/user-guide/persistence/persistent/#upgrading-from-05). Rust: `GrafeoDB::file_manager()` returns the new file manager, without the snapshot and section methods.
- **Breaking: every database is a single file**, whatever its extension, with its WAL in `<path>.wal/` while it is open. The first read-write open of a 0.5.x WAL directory migrates it to a file at the same path and keeps the directory as `<path>.pre-0.6/`. The directory that holds the database must be writable. Builds without the `wal` feature, such as the `grafeo` crate's default profile, refuse a 0.5.x database whose WAL holds changes: open it once with the bindings or the CLI.
- **Breaking: opening a directory that is not a 0.5.x database fails**, an empty one included (0.5.x created a WAL-directory database in it). Use a path where nothing exists yet, such as a file inside that directory.
- **Breaking: `save()` writes a single file and fails if the path exists**; for a path without the `.grafeo` extension, 0.5.x wrote a WAL directory, into an existing database if there was one.
- **Breaking: the minimum Rust version is 1.99.0** (was 1.91.1).
- **Breaking: property values nest at most 128 levels deep** (lists, maps and paths), as deep as the GQL parser reads. Writing a deeper value (through a parameter, the direct API or `restore_snapshot`) fails with `GRAFEO-V001` and writes nothing. The migration of a 0.5.x database that holds one fails, names the node or edge and the property, and changes nothing: change that value with 0.5.x first.
- **Breaking (Rust, `grafeo-engine`): the direct graph, index and RDF calls return `Result`**: `drop_graph`, `create_property_index`, `drop_property_index`, `drop_vector_index`, `drop_text_index`, `batch_insert_rdf` and `create_projection` (and the index calls on `Session`) can fail on a closed database (see Fixed). The bindings report these errors; in C, `grafeo_drop_property_index`, `grafeo_drop_vector_index`, `grafeo_create_projection` and `grafeo_drop_projection` return -1, the two projection calls now as `int32_t`. A failed `rebuild_vector_index` or `rebuild_text_index` keeps the old index.
- **Breaking (Rust, `grafeo-core`): `GraphStoreMut`'s transactional writes and property removals return `Result`** ([#594](https://github.com/GrafeoDB/grafeo/issues/594)): `set_node_property_versioned`, `remove_node_property_versioned`, `remove_edge_property_versioned`, `remove_node_property`, `remove_edge_property`, `delete_node_versioned` and `create_edge_versioned`, so a store can refuse a write whose previous value it cannot read instead of losing that value. A store passed to `GrafeoDB::with_store` wraps its old results in `Ok(...)` and implements the new `GraphStore::try_get_node_property_batch` (`Ok(self.get_node_property_batch(ids, key))` when its reads cannot fail).
- **Breaking (Rust): `GrafeoDB::store()` returns `Arc<LpgStore>`** instead of `&Arc<LpgStore>`: after `compact()` it is the store of the writes since, which a merge replaces, so call it again rather than keeping it. Method calls still work; `&**db.store()` becomes `&*db.store()`.
- **Breaking (Python): `db.algorithms.kcore()` returns `{"core_numbers": {...}, "max_core": n}`**, like `louvain()` ([#563](https://github.com/GrafeoDB/grafeo/issues/563)); `"max_core"` was a key next to the node ids. `kcore(k=...)` still returns the nodes in the k-core.
- **Breaking (Go): `DropVectorIndex` returns `(bool, error)`**, like `DropPropertyIndex`, so a drop the database refuses is an error instead of `true`.
- **Graph data and indexes are stored in chunks** ([#555](https://github.com/GrafeoDB/grafeo/issues/555)): a checkpoint writes the graph and its indexes as it goes, in chunks of at most 65,536 rows and 1 MiB with 64-bit offsets, and an open reads them a chunk at a time, so neither holds a whole section in memory, apart from a few structures still built whole (see [Memory](https://grafeo.dev/architecture/storage/container-format/#memory)). A node can have any number of labels.
- **The catalog is stored as typed records** ([#517](https://github.com/GrafeoDB/grafeo/issues/517)): the schema (node, edge and graph types with their key labels, constraints, index definitions and names, procedures) is written in chunks like the other sections, as records a later version can extend; an edge type's sources and targets are stored as pairs. 0.5.x files keep opening. A schema entry whose stored definition exceeds 2 MiB (for example an edge type with hundreds of source and target types) fails the checkpoint with an error naming it, and the property types of one node or edge type may nest at most 32,768 `LIST` levels in all (DDL refuses more).
- **0.5.x properties stored as null are gone after the migration**: 0.5.x could store a property whose value is null, and wrote `GCounter` and `OnCounter` values to its file as null. In 0.6 a property with a null value does not exist, so after the migration `keys()` and `properties()` no longer list it.
- **Graph algorithm results come in node-id order** ([#592](https://github.com/GrafeoDB/grafeo/issues/592)): `CALL` rows, `db.algorithms` results and the Rust result lists came in hash-map order, which changed from call to call, and so did label propagation's community numbers and the flows of `max_flow` and `min_cost_max_flow`. The same graph, loaded in the same order, now gives bit-identical results in any process, on any platform and with any thread count (see [Determinism](https://grafeo.dev/algorithms/#determinism)).
- **Rust (`grafeo-engine`): the `wal` feature enables `grafeo-file`**, as the WAL is the sidecar of the database file.
- **Rust (`grafeo-engine`): `TransactionState::Committing`**, reported by `TransactionManager::state` while a commit completes.
- **Rust (`grafeo-common`): the deprecated `TieredStore` trait is removed**; `Section` and `MemoryConsumer` cover the same lifecycle. `StorageTier` stays.
- **Rust (`grafeo-core`): `Term::from_ntriples` returns `Result<Term, TermParseError>`**, which names what is wrong and where, instead of an `Option`. `RdfStore::load_ntriples` decodes the `\u`, `\U`, `\b` and `\f` escapes in literals and refuses an unknown escape.

### Fixed

- **Grouped aggregates resume after spilling**: later input rows update reloaded DISTINCT and statistical accumulators, with floating compensation and optional values preserved.

- **Encryption at rest did nothing**: `Config::encryption` was ignored, so databases were written in plaintext. The database file and its WAL are now encrypted, and an open without the right key fails. Encrypted databases do not spill yet, so encryption with a spill path, with `TierOverride::ForceDisk` or without a database path now fails. A 0.5.x database migrated with a key becomes an encrypted file; its `.pre-0.6` copy stays unencrypted.
- **Writes after `close()` were lost**: a write through a handle kept after `close()`, or from another thread while `close()` ran, returned success but was gone after a reopen, and a checkpoint, save, backup or `compact()` from such a handle could overwrite newer data. These now fail with `GRAFEO-T007` (Python: `DatabaseClosedError`); one already in progress completes first. Reads still work, and in-memory databases are not affected.
- **Writes during a commit could land in the middle of it** ([#548](https://github.com/GrafeoDB/grafeo/issues/548)): a direct write, or another transaction writing the same entities, could hide the committed value from point-in-time reads or undo it on rollback, another query could see part of a commit, and crash recovery could replay two commits in the wrong order. A commit now completes before anything that comes after it.
- **A read-only open missed the last commits after a crash**: it read only the file's last checkpoint, not the commits a writer that exited without `close()` left in the WAL. It now replays the WAL into memory and still writes nothing. A build without the `wal` feature refuses such a file, read-only or read-write (a read-write `close()` removed the WAL with those commits).
- **An open with the WAL turned off lost the commits in a WAL**: with `Config::wal_enabled = false` (Rust), a read-write open ignored a WAL left without `close()` and then removed it. The WAL is now always replayed; with the WAL off, the open writes the replayed commits to the file and removes the WAL before it returns.
- **A new database replayed a leftover WAL**: creating a `.grafeo` database next to a non-empty `<file>.wal/` from an earlier database at that path replayed its records into the new one. The open now fails and names the directory.
- **CLI: `grafeo backup restore --force` could lose the target**: it removed the target before reading the backup, so a misspelled or unreadable backup left nothing, and it removed any directory given as the target (and failed on a `.grafeo` one). It now replaces the target only once the restore is complete, refuses a target that is another directory or open in another process, and leaves the backup unchanged.
- **Spilled vector embeddings were not part of the database** ([#594](https://github.com/GrafeoDB/grafeo/issues/594)): spilling a vector index moved its embeddings into `<file>.spill` alone, so copies and the database file lost them, `RETURN n.embedding` read null, a crash after a reload lost them, and removals made while spilled were undone. The database file now always holds them, and vector search reads a spilled index in place, about 15 times faster. Embeddings left spilled by 0.5.x come back at the next open. See [Spilled embeddings](https://grafeo.dev/user-guide/persistence/persistent/#spilled-embeddings).
- **`compact()` turned missing properties into empty values** ([#542](https://github.com/GrafeoDB/grafeo/issues/542)): a node or edge without a property that others of its label or edge type have read `''`, `0`, `0.0` or `false`, so `IS NULL` missed it and filters such as `= 0` matched it. Missing properties now stay missing, also after a reopen; databases compacted by 0.5.44 or older keep the values they stored.
- **A read of a compacted database could miss a value while its writes were merged**: when memory pressure merged the writes made after `compact()` during a read, the read could return nothing, or a wrong count, for a node written after `compact()`.
- **Labels set on a node created in the same transaction were lost** without the `temporal` feature (as in the Python and Node.js packages): `SET n:Label` and `REMOVE n:Label` on such a node did nothing, also within one statement (`INSERT (n:Person) SET n:Admin`), and its change event missed them.
- **A list, map or bytes parameter returned as it is came back as `''`** ([#574](https://github.com/GrafeoDB/grafeo/issues/574)): `RETURN $x` and `WITH $x AS v` returned an empty string, in GQL and Cypher and every binding.
- **`open_in_memory()` opened the database read-write**: it took the exclusive lock, and closing its source checkpointed the file and removed its WAL. It now reads the database like a read-only open and changes nothing on disk.
- **Read-only databases took some writes**: `GrafeoDB::execute_sparql` ran SPARQL updates, and `restore_snapshot`, the imports and `batch_insert_rdf` changed the database. They now fail with the read-only error.
- **WASM: `executeRawWithLanguage` ignored the open transaction** ([#574](https://github.com/GrafeoDB/grafeo/issues/574)); it now runs in it and refuses a closed database, like the other execute methods.
- **A node pattern with several labels scanned every node of the first label** ([#457](https://github.com/GrafeoDB/grafeo/issues/457)): `MATCH (n:Graph:File)` read every `Graph` node, so the label order decided the cost (about 40 ms against 0.05 ms at 40,000 `Graph` and 100 `File` nodes). The label with the fewest nodes is now scanned, in GQL and Cypher, also for `MERGE` and on a compacted database.
- **A later `MATCH` joined to an earlier one by property values scanned its label once per earlier row** ([#455](https://github.com/GrafeoDB/grafeo/issues/455)): in a statement that only reads, `MATCH (f:File) MATCH (t:TypeDefinition) WHERE t.filePath = f.path` now filters each `MATCH` by its own conditions first and joins the values by hash (an indexed key is still looked up per row), with the rows `=` gives for integers, floats, numeric strings and NULL, in the same order. 300 files and 3,000 type definitions went from 303 ms to 3 ms, and the two-`Model` query of the issue on 4,000 nodes from 6.3 s to 11 ms. `EXPLAIN` shows `[hash join: ...]`, and `[label-first]` and `[range: ...]` only where those scans run.
- **A key from an earlier row skipped the property index when the pattern had several labels** ([#455](https://github.com/GrafeoDB/grafeo/issues/455)): `MATCH (f:File) MATCH (t:Graph:TypeDefinition {filePath: f.path})` scanned every `TypeDefinition` node for each row, while `(t:TypeDefinition {filePath: f.path})` looked the key up. Both now use the index, also after `UNWIND` and in subqueries.
- **A `WHERE` condition could be checked where one of its values was missing or meant another variable** ([#455](https://github.com/GrafeoDB/grafeo/issues/455)), so rows disappeared or came back with nulls: a filter on `length(p)` (`MATCH p = (a)-[:R*1..2]->(b) WHERE length(p) >= 1` returned nothing), a condition that read an `UNWIND` variable, a value a `CALL` subquery returns or a variable of an `OPTIONAL MATCH`, and a condition on a variable that a `WITH` dropped or renamed, which also filtered a later `OPTIONAL MATCH` variable of that name. Each condition is now checked where every value it reads is bound.
- **A `CALL` subquery that writes ran only for the rows a later `WHERE` kept** ([#455](https://github.com/GrafeoDB/grafeo/issues/455)): `MATCH (a) CALL { WITH a CREATE (:T) RETURN 1 AS one } WITH * WHERE a.k = 1` created one node instead of one per row.
- **A property lookup after a write in the same statement missed what the statement wrote** ([#455](https://github.com/GrafeoDB/grafeo/issues/455)): with a property index, `UNWIND range(1, 3000) AS i CREATE (:N {k: i}) WITH i MATCH (t:N {k: 3001 - i})` found only some of the nodes. Such a lookup now reads after the writes.
- **Reads at a past epoch matched the property values of now** ([#455](https://github.com/GrafeoDB/grafeo/issues/455)): at a past epoch (`execute_at_epoch`), lookups through a property index (`MATCH (n:X {k: 1})`, `IN` lists, a key from the row) and range filters such as `n.k < 2` (also without an index) found the nodes that have the value now. These reads now scan.
- **`DISTINCT` over a variable-length pattern followed every walk** ([#463](https://github.com/GrafeoDB/grafeo/issues/463)): when the rows of a pattern such as `(n)-[*1..k]-(m)` only reach `DISTINCT`, `count(DISTINCT ...)`, `collect(DISTINCT ...)`, `min` or `max`, each node a source reaches is now found once instead of once per walk, and once over all sources when only the reached nodes are read (`RETURN DISTINCT m.id`). On 100 sources in a 12,000-node graph with hubs, `*1..2` went from 21 ms to 8 ms and `*1..3` from 118 ms to 28 ms, and an unbounded `[*]` on a graph with cycles finishes instead of running out of memory. The rows and their order are the same; EXPLAIN and PROFILE mark such a pattern `[reachability]` or `[reachability: once]`.
- **A later `MATCH` with comma-separated parts ignored a variable bound before it**: in `MATCH (a:Person) MATCH (b:City), (a)-[:VISITED]->(b)` the part that reuses `a` matched any node, so every person came back with every visit; also with a `WHERE` on `a`, with the parts in another order, through a bound edge and in a `CALL` subquery, and a part that read an `UNWIND` value found nothing. Such a part now goes on from the bound node or edge, in GQL and Cypher; so does a part joined to an earlier part of the clause whose property map or inline `WHERE` reads a value of the rows before it (`UNWIND [3, 19] AS w MATCH (b:City), (b)<-[:VISITED {w: w}]-(a)` returned nothing).
- **A vector or text search after a `MATCH` lost that MATCH's rows**: `MATCH (f:File) MATCH (d:Doc) WHERE cosine_similarity(d.emb, $q) > 0.5` returned one row per document instead of one per file and document, and returning `f` failed with "Variable 'f' not found in input"; the same for `text_match`, `text_score`, both together, comma-separated patterns and a preceding `UNWIND`. The condition is now checked for each earlier row; a search without an earlier clause still uses the index.
- **A subquery after a write in the same statement missed what later rows wrote**: in `UNWIND range(1, 3000) AS i CREATE (:N {k: i}) WITH i CALL { WITH i MATCH (t:N {k: 3001 - i}) RETURN t } RETURN count(*)` each row's `CALL` saw only the nodes of the rows before it (1500 instead of 3000), with or without a property index, and so did `OPTIONAL CALL`, `EXISTS`, `COUNT` and `VALUE` subqueries and pattern comprehensions; an `EXISTS` tied to the row by a node of its pattern saw none of the writes. A subquery after a write now runs after the whole input, in GQL and Cypher.
- **An `OPTIONAL MATCH` after a write in the same statement saw none of what the statement wrote**: `UNWIND range(1, 300) AS i CREATE (:P {k: i}) WITH i OPTIONAL MATCH (t:P {k: 301 - i}) RETURN count(t)` returned 0 instead of 300, also after an aggregate over the write, after a `CALL` that writes and from a written node, and a property map or `WHERE` reading a variable a `WITH` passed on after a write never matched. It now reads after the whole input, and a row without a match keeps its nulls, in GQL and Cypher, with and without a property index.
- **A `MATCH` that goes on from a node written earlier in the statement saw only what the rows before it wrote**: in `UNWIND range(1, 4) AS i MERGE (h:Hub) CREATE (h)-[:R]->(:Q {i: i}) WITH h, i MATCH (h)-[:R]->(q) RETURN i, count(q)` the rows counted 1, 2, 3 and 4 edges instead of 4 each; the same for a shortest path from such a node, the labels and properties checked on it, and a part joined to another on a shared variable, which found nothing. Such a `MATCH` now reads after the whole input, in GQL and Cypher.
- **An aggregate without `MATCH` failed with "Internal error: Empty plan"**: `RETURN count(*)`, `RETURN sum(3)`, `WITH count(*) AS c` and `CALL { RETURN count(*) AS c }` now aggregate the one row a query starts from (`count(*)` is 1), in GQL and Cypher.
- **A typed variable-length pattern, a chain of typed expands or a typed shortest path missed the edges written in the same transaction**: in an open transaction, or later in the statement that wrote them, `MATCH (h:Hub)-[:R*1..2]->(q)`, `MATCH (h)-[:R]->(q)-[:S]->(t)` and `shortestPath((h)-[:R|S*]->(t))` (also `allShortestPaths`, `ANY SHORTEST` and `ALL SHORTEST`) found none of them, while untyped patterns and a single typed expand did. They now see them, in GQL and Cypher, and a shortest path no longer walks edges another transaction created and has not committed.
- **`FOREACH` changed the number of rows**: `UNWIND [1] AS i FOREACH (x IN [1, 2] | CREATE (:Z)) RETURN count(*)` returned 2 instead of 1, and an empty or null list dropped the row. Each row now goes on once, as it came in, and the updates see all of what the clauses before wrote; as in openCypher, the `FOREACH` variable and what its updates bind are no longer variables after it. `FOREACH` can also start a query now (it failed with "FOREACH requires preceding input").
- **A `MERGE` after a write in the same statement saw only what the rows before it wrote**: `UNWIND range(1, 300) AS i CREATE (:P {k: i}) WITH i MERGE (t:P {k: 301 - i})` left 450 nodes instead of 300, `ON MATCH SET` and `ON CREATE SET` ran for the wrong rows, and a relationship `MERGE` created duplicate edges. A `MERGE` after a write now reads after the whole input, in GQL and Cypher; it still sees what it created for the rows before.
- **A clause after a write in the same statement read only what the rows up to its own wrote**: in `UNWIND range(1, 4) AS i MERGE (h:Hub) SET h.c = i WITH h, i RETURN i, h.c` the rows returned 1 to 4 instead of 4 each, `WHERE h.c = 4` kept one row instead of four and `sum(h.c)` was 10 instead of 16; the same for `ORDER BY`, a returned node, `labels()` after `REMOVE` and a property of a node a later row deleted. A `RETURN`, `WITH`, `WHERE`, `ORDER BY`, aggregate or `UNWIND` that reads the graph after a write now reads after the whole write, as ISO GQL and openCypher define a statement, in GQL and Cypher; EXPLAIN marks such a clause `[after the write]`, and a statement that only reads is planned as before.
- **A `LIMIT` after a write cut the write short**: `UNWIND range(1, 10) AS i CREATE (:L {i: i}) RETURN i LIMIT 1` created one node instead of ten, and GQL's `FINISH` after a write wrote only the first row. The write now completes first. A `LIMIT 0` or `FINISH` in a `CALL` body wrote nothing at all (`FOR i IN [1, 2] CALL (i) { FOR j IN [1, 2, 3] INSERT (:L) FINISH }` created no node); it now cuts the body's rows, not its write.
- **A write took no value from a node or edge created earlier in the statement**: in `UNWIND [3, 19] AS i CREATE (a:N {k: i}) CREATE (b:P {k: a.k})` the `P` nodes got no `k`, and so did a value read from a created edge, `SET b.k = a.k`, a relationship `MERGE` with `{w: a.k}` and `ON CREATE SET`; a `MERGE (b:P {k: a.k})` from such a node merged every row into one node without `k`. These writes now take the value, in GQL and Cypher.
- **A query that ends with a write returned internal columns**: a Cypher query ending with `CREATE`, `MERGE`, `SET`, `REMOVE`, `DELETE`, `FOREACH` or a unit `CALL` returned a row of internal columns (such as `__list__` and `_anon_0`) for each row it wrote, and a GQL one a row without columns. Such a statement now returns no rows and no columns, as openCypher and ISO GQL define it; so does GQL's `FINISH`.
- **A `CALL` subquery without a final `RETURN` changed the number of rows**: `UNWIND [1] AS i CALL { WITH i UNWIND [1, 2] AS x CREATE (:W) } RETURN count(*)` returned 2 instead of 1, a body that makes no rows dropped the row, and a query that ended with such a `CALL`, or `RETURN *` after one, failed with "duplicate column name". Such a unit subquery now runs for its writes and passes each row on once, in GQL and Cypher; as in openCypher, nothing it binds is a variable after it.
- **`PROFILE` panicked on a query that starts without `MATCH`**: `PROFILE RETURN 1 AS x` and a first `OPTIONAL MATCH`, `WITH`, `CALL { ... }` or `MERGE` failed with "profile entry count must match logical operator count" (in Python a `PanicException`). They now return the profile, in GQL and Cypher.
- **A condition without variables could fail with "Internal error: Empty plan"**: `WITH 1 AS x WHERE $p = 1 RETURN x` (also `WHERE 1 = 1`) failed instead of filtering the row.
- **Rust: a query on a projection found no nodes for a label outside its spec**: on a projection of `Person` read through `GrafeoDB::with_read_store`, `MATCH (n:Admin)` returned nothing for a `Person` node that is also an `Admin`.
- **Every GQL statement on a file database cost about 0.25 ms extra** ([#565](https://github.com/GrafeoDB/grafeo/issues/565)): with the `spill` feature (as in the packages), each statement created and removed a spill directory, about 0.27 ms against 0.01 to 0.03 ms for the same Cypher statement, and left an empty `<file>.spill` directory behind. The directory is now created only when a query spills.
- **`compact()` gave some edge ids to another edge**: when a node's edges were created in another order than the nodes they lead to, an edge id could name another edge of the same node after `compact()`, so reading it returned the other edge's target and properties, and updating or deleting it changed the other edge. Each id now keeps its own edge. A database compacted by an earlier version keeps the ids it stored.
- **Parallel edges reached from their target after `compact()` shared one id**: following two or more edges between the same two nodes backwards (`(b)<-[r]-(a)`) returned the first edge for each of them. Each edge now keeps its own id.
- **A damaged database file could abort the process when it was opened**: counts in a compacted base or an index section were trusted before they were checked, so a damaged file could request terabytes of memory. Such a file is now refused with an error, and a vector, text or RDF ring index section that does not decode is rebuilt from the data (a damaged ring section used to fail the open).
- **Databases larger than 4 GiB** ([#392](https://github.com/GrafeoDB/grafeo/issues/392)): 0.5.42 and older wrote them corrupt (`block 0 CRC mismatch` on open), and 0.5.43 and 0.5.44 refused to checkpoint them, as they did a graph over 65,535 blocks, a node with over 65,535 labels or a property with over 65,535 versions. These limits are gone. A compacted base (`compact()`) still keeps each column and table below 4 GiB and 2^32 rows until [#432](https://github.com/GrafeoDB/grafeo/issues/432); a checkpoint past that fails, names it and keeps the WAL.
- **Ids of deleted nodes and edges were given out again after a reopen**: new nodes and edges took the ids of those deleted above the highest id still in use, so an id kept outside the database could name another node or edge. Ids are now never given out twice.
- **A checkpoint lost sub-millisecond times and counter values**: timestamps and zoned datetimes lost their sub-millisecond part at every checkpoint, and `GCounter` and `OnCounter` values became null. Every value is now stored exactly; what a 0.5.x file already lost stays lost after the migration.
- **Non-ASCII RDF literals were corrupted after a reopen**: a literal such as `"Kraków"` came back as `"KrakÃ³w"` after a checkpoint, a crash recovery or a snapshot import, and no longer matched queries for its value. Blank node ids and language tags that differ only in trailing whitespace are no longer merged, and a term that cannot be read is now an error naming it instead of a triple silently dropped.
- **Each RDF named graph took about 1.3 MB of memory, also when empty**; a named graph now starts empty.
- **Rust (`grafeo-core`): `RdfStore::load_ntriples` panicked on a literal that ends in a backslash**; such a line is now refused with a parse error.
- **A checkpoint while a transaction was open wrote that transaction's changes into the file** ([#412](https://github.com/GrafeoDB/grafeo/issues/412)): a checkpoint, `close()`, `save()` or a backup taken while a transaction was open left out the nodes and edges the transaction was deleting, so a crash afterwards lost them (also after a rollback), and without the `temporal` feature (as in the Python and Node.js packages) it wrote the values and labels the transaction had set or removed but not committed. These now write the committed state, in every named graph and after `compact()`, and writes and rollbacks of open transactions wait while it is written. `to_memory()` and `export_snapshot()` taken during a transaction also hold the committed state now (they copied its uncommitted values and labels, with `temporal` its pending versions, and left out what it was deleting), and text and vector search after a reopen no longer find a value an open transaction had written: a checkpoint taken while a transaction has changed the default graph leaves the text and vector indexes out of the file, and the next open builds them from the data. A transaction still open at `close()` is left out of the file.
- **Deleting data stored by `compact()` ignored the transaction**: another session saw the node or edge gone at once, a rollback (or a rollback to a savepoint) did not bring it back, and a checkpoint during the transaction wrote the delete to the file. It now belongs to its transaction, like any other delete.
- **`SHOW INDEXES` lost rows after a reopen**: of a `CREATE INDEX` on several properties, a reopen kept one row per index name; every property's row now comes back.
- **Some deletes after `compact()` were undone or lost at the next open**: a compacted node or edge that was changed and then deleted came back after a reopen, and after a reopen a node or edge created after `compact()` could not be deleted (the delete did nothing).
- **`compact()` merged the labels of a node with several into one**: a node with the labels `Person` and `Employee` read `labels(n) = ['Employee|Person']` after `compact()`, and neither `MATCH (n:Person)` nor `MATCH (n:Employee)` found it, so writes and deletes through such a match did nothing, label counts and the planner's statistics left it out, and a pattern with several labels could scan the wrong one. Each label now finds the node, also in a database compacted by 0.5.44 or older (where a label that holds `|` reads as two labels).
- **`restore_snapshot()` on a compacted database kept the compacted data**: the snapshot went only into what changed since `compact()`, so the compacted nodes and edges stayed, won where their ids clashed with the snapshot's, and came back after a reopen. The restore now replaces the whole database, which reopens as a plain one.
- **Arrow exports of a compacted database held only what changed since `compact()`**: Python `nodes_to_arrow()`, `edges_to_arrow()` and the Polars variants, and Rust `iter_nodes()` and `iter_edges()`, now return every node and edge, without those deleted since.
- **An edge could end at a deleted node**: creating an edge in the statement that deletes its endpoint (`DETACH DELETE g INSERT (a)-[:KNOWS]->(g)`), or on a compacted database while another transaction deletes the endpoint, committed an edge that patterns skipped but `edge_count` counted. An endpoint the transaction deleted itself is now refused, and a concurrent delete is a write conflict (the second of the two fails); transactions that create edges to one node, or set its properties, do not conflict.
- **Sessions opened before `compact()` lost their writes**: their statements and direct calls went to the store from before `compact()`, and their reads and named graphs stayed there. They now go on with the compacted store, and `compact()` fails while a transaction is open, as `recompact()` does.
- **`compact()` dropped the nodes without labels**: a node without a label (`INSERT ({name: 'Gus'})`) was gone after `compact()` with its properties and edges, so `MATCH (n)` and the counts missed it, and a database of only such nodes compacted to an empty one. They are now kept, also through later writes, merges and a reopen. Nodes an earlier version dropped do not come back.
- **Vector search of a compacted database missed nodes**: after `compact()`, a vector written later (a new node, or a new value of a compacted node) was not linked to the compacted nodes in the vector index, so `vector_search` missed it, also after `recompact()` and a reopen, and the vector scans of a query (also without an index of its metric) and `vector_search_with_threshold` found no compacted node. Searches now find the nearest nodes among the compacted and the later ones; an index written by an earlier version keeps its missing links until `rebuild_vector_index()`.
- **`compact()` dropped the indexes made before it**: property, text and vector indexes created before `compact()` were gone after it: `has_property_index` returned false, lookups ran without the index, `text_search` and `vector_search` failed with "No ... index found", a vector of the wrong size for a vector index was accepted, and the indexes were missing after a reopen while `SHOW INDEXES` still listed them. They now stay with their configuration and find the compacted nodes and those written later, also after a merge, `recompact()`, another `compact()` and a reopen. A database compacted by 0.5.44 or older gets them back by creating them again.
- **Builds without a feature lost the data only it reads** (also in 0.5.x): a build without `compact-store` (the `grafeo` crate, the `grafeo` command line tool) opened a compacted database without its compacted data, one without `triple-store` (the engine's default build, the `grafeo` crate's default and `lpg` profiles, the command line tool) without its RDF triples (in the file, its WAL or a 0.5.x WAL directory), and one without `vector-index` or `text-index` (the `lpg` and `rdf` profiles, the command line tool) without its vector and text index definitions; queries missed that data, and the next checkpoint dropped it for good. Such a build now refuses these databases (read-write, read-only and `open_in_memory()`, 0.5.x databases included) with an error that names the data and the feature, and changes nothing on disk. Open them with the Python, Node.js or C bindings, or a Rust build with the features. The same builds imported or restored a snapshot without its RDF triples and its vector and text index definitions (as the `@grafeo-db/wasm-lite` package does); they now refuse such a snapshot the same way.
- **Search of a compacted database found nodes deleted since `compact()`**: after a compacted node was deleted, `text_search`, `hybrid_search` and `CALL grafeo.search.text` still returned it (as a node without labels or properties) until the next merge, also after a reopen, and so did `vector_search`, `batch_vector_search`, `hybrid_search` and `CALL grafeo.search.vector` with a quantized vector index. They now leave it out and still return `k` results when it would have been among them, and `mmr_search` and `CALL grafeo.search.mmr` no longer return fewer results than asked for.
- **`export_snapshot()` after `compact()` left out the compacted data**: the snapshot held only the nodes and edges added or changed since `compact()`, so `import_snapshot()` lost the rest. It now holds the whole graph, without the nodes and edges deleted since.
- **A merge of a compacted database under memory pressure broke open transactions**: after `compact()`, memory pressure could merge the writes since into the compacted data while a transaction was open. Its uncommitted inserts were lost, its values and deletes became part of the compacted data (a rollback did not undo them, and the commit of a delete did nothing), and its snapshot saw later commits. The merge now waits until no transaction is open, and `recompact()` fails while one is.
- **Writes after such a merge, or after `recompact()`, were lost**: sessions opened before it, and the database's own calls, went on with the replaced store, so their commits were invisible and missing from the file after a reopen. Named graphs (also from the file) and the text, vector and property indexes were gone after the merge, and with `temporal` the epoch went back to 0 after a reopen. With a spill path, spilling or reloading the compacted data after a merge could bring back the data from before it. `compact()` and the merge now also free the store they replace.
- **A transaction could lose sight of data stored by `compact()`**: a transaction that began before another session's commit stopped seeing a compacted node or edge, with its labels and values, once a later write changed it or added an edge to it. It now sees them as they were when it began.
- **Concurrent first writes to data stored by `compact()` could duplicate or hide nodes**: two writes that first changed the same compacted node or edge at the same time could leave two copies (wrong counts), a node or edge created meanwhile could take the id of a compacted one and then read as it, and a checkpoint during such a write could store the node without some of its values, which then replaced the compacted ones at the next open. Each is now copied once and whole, new nodes and edges never take a compacted id, and a compacted node or edge deleted while a write copied it no longer stays alive.

### Result changes

- **k-core decomposition returns core numbers** ([#563](https://github.com/GrafeoDB/grafeo/issues/563)): `kcore` (`db.algorithms.kcore()`, `CALL grafeo.kcore()`) gave each node its degree at the moment it was peeled, below its core number for most nodes, and which node got which value changed between calls. It now returns each node's core number, and self-loops no longer count toward a node's degree. `max_core` was already right.
- **Louvain merges communities level by level** ([#564](https://github.com/GrafeoDB/grafeo/issues/564)): `louvain` only moved single nodes, so it stopped at many small communities (a 1,000-node path gave 500 pairs at modularity 0.50; it now gives 31 communities at 0.94). Results change for every graph with edges. `resolution` now changes the communities, and the reported modularity is that of the returned communities (it came out too high).
- **`as_networkx(directed=False).pagerank()` is undirected** ([#566](https://github.com/GrafeoDB/grafeo/issues/566)); it returned the directed scores.
- **`db.algorithms` follows `set_graph`** ([#566](https://github.com/GrafeoDB/grafeo/issues/566)): it always read the default graph, unlike `CALL grafeo.<algorithm>()` and the direct API.
- **`create_projection` builds over the selected graph** ([#566](https://github.com/GrafeoDB/grafeo/issues/566)): it always used the default graph, unlike GQL `CREATE PROJECTION`; a projection keeps reading the graph it was built over.
- **The average clustering coefficient no longer changes from call to call** ([#592](https://github.com/GrafeoDB/grafeo/issues/592)): `global_clustering_coefficient` and the `clustering_coefficient` average summed in hash-map order, so their last digits changed with each call and the thread count; they now depend only on the graph.

### Deprecated

- **Rust (`grafeo-engine`): `StorageFormat::SingleFile` and `StorageFormat::WalDirectory`**, removed in 0.7.0: every database is a single file, so `SingleFile` does the same as `Auto` (the default), and `WalDirectory` only opens an existing 0.5.x WAL directory, by migrating it, and fails at a new path. `Auto` now only decides what a new path becomes.
- **C, Go and Dart: `grafeo_open_single_file`, `OpenSingleFile` and `GrafeoDB.openSingleFile`**, removed in 0.7.0: they do the same as `grafeo_open`, `Open` and `GrafeoDB.open`, and compilers and analyzers warn on their use.

## [0.5.44] - 2026-10-04

Durability and consistency release. Crash-safe checkpoints and WAL recovery, indexes and constraints that survive a reopen, schema checks on every write path, per-graph conflicts and grants, commit and rollback in O(changes), and fixes for shortest paths, variable-length edges, list comprehensions, subqueries, `OPTIONAL MATCH` and `MERGE`. Plus graph handles, upserts by key and write counters.

> **Heads-up: 0.6.0 (first announced as 0.5.45) changes the on-disk format.** 0.6.0 migrates a database automatically on first open (WAL-directory databases become a single `.grafeo` file); after that, 0.5.44 and older can no longer open it, so keep a backup if you may need to go back.

### Changed

- **Breaking (Rust, `grafeo-engine`): the direct write methods return `Result`** (`create_node`, `create_edge`, `set_node_property`, `delete_node`, the batch calls and the rest, on `GrafeoDB` and `Session`). Each call commits on its own with the checks of a query, so writing to a missing entity or endpoint, or deleting a node that still has edges, now fails. The bindings raise these errors.
- **Breaking (Rust, `grafeo-core`): the mutation operators take a `GraphWriter`** instead of `with_transaction_context`, `with_validator` and `with_write_tracker`; `SetPropertyOperator::with_labels` and `with_edge_type` are gone.
- **Breaking (Rust, `grafeo-core`): operators that only order, cut, deduplicate or compare rows take no output schema** (`LimitOperator`, `SkipOperator`, `LimitSkipOperator`, `SortOperator`, `TopKOperator`, `DistinctOperator`, `ShuffleOperator`, `ExceptOperator`, `IntersectOperator`).
- **Breaking (Rust, `grafeo-core`): a sort key's `NullOrder` no longer flips for a descending key** (`SortKey::descending` sets `NullsFirst`), and `value_utils::compare_values_with_nulls` is replaced by `compare_sort_values`.
- **Breaking (Rust, `grafeo-core`): `FilterExpression::ExistsSubquery` and `CountSubquery` have `end_var` and `edge_var`**, and `ExistsSubquery` reads `min_hops` and `max_hops`.
- **Breaking (Rust, `grafeo-engine`)**: the transaction manager's write and read sets hold `GraphEntity` (graph plus node or edge), the logical `ExpandOp` has a `quantified` field, and `RdfPlanner::with_wal` is no longer public.
- **Breaking (Rust, `grafeo-engine`): `ChangeEvent` has a `graph` field** (`None` for the default graph), and `CdcLog` keys its events by graph and entity, with `history_in` and `history_since_in` for a named graph.
- **Breaking (Rust, `grafeo-adapters`): the GQL `QueryClause::InlineCall` has `scope` and `combined` fields, and the Cypher `Clause::CallSubquery` is a struct variant** with `query`, `scope`, `unions` and `union_all`.
- **CDC label change events**: `labels` is now always the labels after the change, and the new `before_labels` holds the previous ones.
- **Some queries that ran now fail with an error, as in openCypher**: an expression in `WITH` without a name; a `CALL` subquery that returns an outer variable or reads one it does not import (`CALL () { ... }`, a Cypher `CALL { ... }` without an importing `WITH`); an importing `WITH` with a `WHERE`, alias, `ORDER BY`, `SKIP` or `LIMIT`; `UNION` mixed with `UNION ALL`; and a Cypher query that ends with a `CALL` subquery that returns rows (add a `RETURN` after it). Before, they gave wrong or silently null results, or failed with an internal error (see Fixed).

### Fixed

Every query result in the differential test corpus that differs from 0.5.43 is listed, with its reason, in `scripts/difftest/reviewed/0.5.44.txt`.

- **Checkpoints after `compact()` left data out of the `.grafeo` file**: the schema, indexes and RDF data were dropped (also at `close()`), the periodic checkpoint wrote the pre-compaction store, and `async_write_snapshot()` wrote little or nothing. Every checkpoint now writes the whole database.
- **A failed checkpoint could leave a `.grafeo` file that no longer opened** ([#418](https://github.com/GrafeoDB/grafeo/issues/418)). Checkpoints now write to `<file>.checkpoint` first, so the last good state stays readable; this needs free disk space for a second copy of the file.
- **Edges appeared twice after a checkpoint followed by a crash** ([#417](https://github.com/GrafeoDB/grafeo/issues/417)). Replay now skips entities the file already holds and repairs affected databases, and checkpoints trim the sidecar WAL.
- **Direct writes were not durable until `close()`** ([#395](https://github.com/GrafeoDB/grafeo/issues/395)). Each call is now durable when it returns, and batch calls are recovered completely or not at all.
- **Concurrent transactions could corrupt WAL recovery** ([#411](https://github.com/GrafeoDB/grafeo/issues/411)): one session's rollback could erase another's commit, and open or rolled-back writes came back on reopen. Each transaction is now logged as one group at commit.
- **Two processes could open the same WAL-directory database** ([#405](https://github.com/GrafeoDB/grafeo/issues/405)); it is now locked like a `.grafeo` file.
- **`wal_checkpoint()` lost data in WAL-directory databases** ([#419](https://github.com/GrafeoDB/grafeo/issues/419)). It now only syncs the WAL there, and open replays every WAL file.
- **`DurabilityMode::Adaptive` never synced the WAL**, so it behaved like `NoSync`, and backups missed writes made while they ran; those now land in this backup or the next.
- **Schema changes, constraints and `create_graph()` / `drop_graph()` were lost on WAL replay** ([#421](https://github.com/GrafeoDB/grafeo/issues/421), [#422](https://github.com/GrafeoDB/grafeo/issues/422)). An unknown WAL record now fails the open instead of being skipped.
- **`DROP CONSTRAINT` did nothing** ([#420](https://github.com/GrafeoDB/grafeo/issues/420)). Constraints are now stored by name: `CREATE` and `DROP CONSTRAINT` support `IF NOT EXISTS` / `IF EXISTS`, `SHOW CONSTRAINTS` lists them, and unnamed ones get a name such as `City_name_not_null`. Constraints from 0.5.43 files have no name and cannot be dropped by name.
- **Reopening a `.grafeo` database or `to_memory()` lost its indexes** (lookups on a reopened file scanned every node, [#459](https://github.com/GrafeoDB/grafeo/issues/459)), and `to_memory()` also dropped the schema, constraints and property history. WAL-directory databases still lose their indexes until 0.6.0 ([#401](https://github.com/GrafeoDB/grafeo/issues/401)).
- **Rust builds with only the `rdf` profile kept no data across a reopen** ([#544](https://github.com/GrafeoDB/grafeo/issues/544)); the facade's `rdf` profile now includes the LPG store.
- **Embeddings changed while their vector index was spilled reverted on reload** ([#522](https://github.com/GrafeoDB/grafeo/issues/522)).
- **Sessions and direct reads after `compact()` missed parts of the database**: SPARQL writes were lost, CDC missed queries, the selected graph was ignored, and `get_node`, `node_count`, `info()`, `schema()` and similar did not see pre-compaction data (or panicked on a `with_store` database). Queries after `compact()` are still not written to the WAL, so a crash loses them; direct calls are logged.
- **Grants and read-only modes did not stop every write** ([#413](https://github.com/GrafeoDB/grafeo/issues/413)): `use_graph()` skipped the grant check, read-only grants allowed writes, and read-only transactions, roles and databases let non-GQL languages and the direct API write. A session whose graph was dropped also fell back to the default graph; it now fails.
- **Transactions**: a failed statement inside a transaction kept its partial writes (it is now undone, and the transaction goes on); a commit that failed with a write-write conflict left the transaction active ([#409](https://github.com/GrafeoDB/grafeo/issues/409)); a transaction could not delete what it had created; transactions in different named graphs raised false write conflicts; and rolled-back nodes still counted in the planner's statistics.
- **Writes that should fail went through**: `SET`, `REMOVE` and label changes on a node or edge deleted earlier in the statement or transaction (now `... does not exist or has been deleted in this transaction`, as in openCypher), writes from a session reading at an earlier epoch (`set_viewing_epoch`, `execute_at_epoch`), and writes after `compact()` on a database opened read-only. All of these now fail.
- **Writes skipped schema checks**: `SET`, `REMOVE`, `SET n:Label` and `MERGE` did not check property types, `NOT NULL`, `UNIQUE`, `NODE KEY`, edge types, endpoint labels or `DEFAULT` values, `UNIQUE` did not hold within a transaction, and multi-property `UNIQUE` and `NODE KEY` were checked one property at a time. Every write is now checked like `INSERT`, and a failed `ALTER NODE TYPE` or `ALTER EDGE TYPE` applies all or none.
- **Parameterized queries skipped schema and constraint checks** ([#526](https://github.com/GrafeoDB/grafeo/issues/526)), so `execute(query, params)` in Python and Node.js could insert a duplicate `UNIQUE` value. They now run exactly like the same statement with literals.
- **Parameters**: a statement that used a parameter nobody supplied stored the text `$e` (`INSERT (:P {e: $e})`) or failed with an internal error, and now fails with `Missing parameter: $e` before it writes anything; unaliased columns were named after a parameter's value (`RETURN $x` gave a column `5`) and are now named after the query text; and `PROFILE` was ignored when parameters were passed ([#460](https://github.com/GrafeoDB/grafeo/issues/460)).
- **Removed properties stayed visible as null** without the `temporal` feature (as in the Python and Node.js packages); `REMOVE n.p` and `SET n.p = NULL` now remove the property.
- **Commit and rollback got slower as the database grew** ([#410](https://github.com/GrafeoDB/grafeo/issues/410)): a one-node insert took 36 ms at 1M nodes. They now cost only as much as the change.
- **Slow paths on large databases**: setting a property got slower with the node's property count (`NodeRecord::props_count` is now always 0), every hundredth commit scanned the whole store, and concurrent writes could deadlock with `tiered-storage`.
- **Lookups by `id()` or an indexed property scanned every node** ([#454](https://github.com/GrafeoDB/grafeo/issues/454), [#356](https://github.com/GrafeoDB/grafeo/issues/356)), once per row for keys like `UNWIND $rows AS row MATCH (n {id: row.id})`, and with a label (`MATCH (n:File {id: 'x'})`) every node of that label was collected. They now seek directly, also for `IN` lists and edge endpoints.
- **Regular expressions and `LIKE` patterns were compiled for every row** ([#458](https://github.com/GrafeoDB/grafeo/issues/458)); each is now compiled once.
- **Nodes and edges came back as IDs, as `0`, or read another entity's properties** ([#482](https://github.com/GrafeoDB/grafeo/issues/482)): after `ORDER BY`, `SKIP`, `LIMIT`, `DISTINCT`, a later `UNION` branch, a second `MATCH`, `OPTIONAL MATCH`, `EXISTS`, `CALL`, `UNWIND`, grouping or a write, a returned edge could be `0`, a node or edge a raw ID, and a property could come from the entity of the other kind with the same ID (`MATCH (a)-[r]->(b) RETURN r ORDER BY r.w`). `collect()` and group keys returned nodes and edges as IDs, `nodes(p)` and `relationships(p)` returned internal ids, `UNWIND` of them or of a function call after `MATCH` returned no rows, and `EXCEPT` and `INTERSECT` over nodes or edges returned the wrong rows. Nodes, edges and lists of them now keep their kind through every clause, also in Gremlin `union(outE(), out())`.
- **`keys(r)` of an edge returned null**; it now lists the edge's keys, and `keys()` of a node or edge returns them sorted, like `properties()`.
- **A variable used as both a node and an edge matched by a coincidence of IDs** (`MATCH (r) MATCH ()-[r]->()`); it now fails with an error that names the variable.
- **`ORDER BY` over values of different types failed or depended on the input order**: values of different types compared as equal, numbers among them and around NaN came out of order, lists and maps were not ordered, and such a sort (also in `percentileCont` and `percentileDisc`) could fail with `does not correctly implement a total order` (a `PanicException` in Python). `ORDER BY` now uses the openCypher total order: maps, lists, paths, temporal values, strings, booleans, numbers, then null; integers and floats compare exactly and NaN sorts after infinity.
- **Other `ORDER BY` fixes**: GQL `NULLS FIRST` and `NULLS LAST` were reversed with `DESC`, `RETURN * ORDER BY ...` failed with `Variable '*' not found in input`, and sort keys could show up as extra columns (`RETURN r AS e ORDER BY e.w` returned an `e_w` column).
- **Patterns through variables bound earlier matched too much**: a path back to an earlier node (`MATCH (a)-->(b)-->(a)`) returned every two-hop path instead of the cycles, and a pattern through a bound edge (`MATCH ()-[r]->() MATCH (x)-[r]->(y)`) matched every edge of its shape, also after `WITH`, in `CALL` subqueries, in stored procedures and after a shortest path. Both now use what the variable holds.
- **Variable-length edges**: the edge variable held only the last edge, a property map was checked only on the last hop, and a one-hop quantifier (`*1..1`, `*1`, GQL `{1,1}`) bound a single edge, so `size(rs)` returned null. The variable is now the list of the path's edges.
- **Shortest-path searches returned pairs without a path and ignored hop bounds** ([#514](https://github.com/GrafeoDB/grafeo/issues/514)). Unreachable pairs no longer get a row, `->+` finds the shortest cycle, and the path must fit the quantifier.
- **Clauses after a write or a `WITH` in one statement** ([#479](https://github.com/GrafeoDB/grafeo/issues/479), [#480](https://github.com/GrafeoDB/grafeo/issues/480)): a `MATCH` after a write missed the written data (and when it found nothing, the write did not run), and a GQL `MATCH` after `WITH` failed with `Variable '...' not found in input`. Clauses now read the rows the earlier ones pass on.
- **`EXISTS` and `COUNT` subqueries gave wrong answers** ([#543](https://github.com/GrafeoDB/grafeo/issues/543)): they checked only a path's first edge (missing multi-hop matches and minimums like `*2..`), ignored the rest of their pattern (`EXISTS { MATCH (a)-[:KNOWS]->(b), (c:Robot) }` held without any `Robot`), used only their start node from the row (`EXISTS { MATCH (b)-[:KNOWS]->(a) }` held whenever `b` knew anyone), ignored a tie to the row through a value (`{id: s.id}`, an inner `WHERE`), could run in `WHERE` before their variables were bound, and counted edges the query could not see. Each row now gets its own answer from the whole pattern, and a null node makes `EXISTS` false and `COUNT` 0.
- **GQL subqueries**: in `EXISTS`, `COUNT` and `VALUE`, each `MATCH` clause matched on its own; an `EXISTS` that starts with `OPTIONAL MATCH` filtered rows; `VALUE { ... RETURN count(x) }` ignored its argument and `DISTINCT`; and a `VALUE` subquery that is not a count gave every row the same answer, or failed in `WHERE` and `WITH`. They now behave as the same clauses do outside a subquery.
- **Cypher `OPTIONAL MATCH` at the start of a query and in `EXISTS` and `COUNT` failed**; it now runs, with one row of nulls when nothing matches.
- **`CALL` subqueries saw the wrong outer variables**: a GQL `CALL` did not see the outer row, so every row got every match; an importing `WITH` (in GQL, the first `WITH` of the body) ignored its `WHERE` and `WITH a AS b` imported the wrong value; and a Cypher `CALL` without an importing `WITH` read outer variables as null. A GQL `CALL` now sees every outer variable and its `WITH` is an ordinary one, a Cypher `CALL` sees those its importing `WITH` lists, and the scope clause `CALL (a, b) { ... }` (`CALL (*)`, `CALL ()`), which failed to parse, names them in both languages.
- **Nodes and edges a `CALL` subquery returned were copies of their properties**: a later `MATCH` from them failed or matched everything, and `id(b)` and `type(r)` were null. `RETURN *` in a `CALL` also returned nothing, and a two-edge chain in a `CALL` no rows. The subquery now passes on the nodes and edges themselves, and `RETURN *` returns the variables it binds.
- **Cypher `CALL` bodies rejected `ORDER BY`, `SKIP`, `LIMIT`, `UNION`, `DELETE`, `MERGE`, `REMOVE`, `FOREACH` and a nested `CALL`**; they now run, as in openCypher (an ordered `LIMIT` gives the top rows per outer row). A GQL `CALL` body takes `UNION`, `EXCEPT`, `INTERSECT` and `OTHERWISE`.
- **An `OPTIONAL MATCH` condition that reads a variable bound before it dropped rows**: in `MATCH (a)-[:KNOWS]->(b) OPTIONAL MATCH (b)-[:KNOWS]->(c) WHERE c.age > a.age`, a `b` whose matches all failed the condition lost its row instead of keeping it with nulls. The condition now decides which matches count, in GQL and Cypher.
- **`MERGE` bound only the first of several matches**: with two `(:Tag {k: 1})` nodes, `MERGE (t:Tag {k: 1}) SET t.seen = true` set one of them. As in openCypher, `MERGE` now binds every match, one row each, also for relationships and in `upsert_edges`.
- **GQL `NEXT` did not pass rows on**: the statement after `NEXT` matched from every node and returned the first statement's columns too. It now reads the rows the one before returns, and only the last `RETURN` is the result.
- **CDC history mixed graphs**: `history(id)` and `changes_between` returned the events of every graph's entity with that id. Each event now names its graph (`graph`, also in Python and Node.js), `history` reads the database's default graph or the session's current graph, and `changes_between` returns every graph's events.
- **A variable a `WITH` left out failed with an internal error** (`MATCH (a) WITH 1 AS x RETURN a.name`); it now fails with `Undefined variable 'a'`.
- **A Cypher pattern comprehension inside a function call, an aggregate or a map projection failed** (`RETURN size([(a)-->(b) | b])`); it now runs.
- **A transaction on a `.grafeo` file, a WAL directory or a compacted database did not see the type of an edge it had created** until the commit.
- **List comprehensions, list predicates and `reduce()` lost items** ([#538](https://github.com/GrafeoDB/grafeo/issues/538)) when the expression called a function, read a property of a `relationships(p)` item or used another row variable.
- **`EXCEPT`, `INTERSECT` and `OTHERWISE` did not check their branches' columns** ([#481](https://github.com/GrafeoDB/grafeo/issues/481)), and SQL/PGQ checked none of its set operations; they now fail like `UNION`.
- **Cypher `=~` matched any part of the string**; it now has to match the whole string, as in openCypher.
- **Cypher and SQL/PGQ checked only the first label of some node patterns** ([#513](https://github.com/GrafeoDB/grafeo/issues/513)), e.g. `MATCH ()-[r]->(n:A:B)`.
- **GQL `INSERT` with unlabeled or already bound endpoints** failed with `Undefined variable` or attached edges to the wrong node.
- **A `LOAD CSV` row returned whole was null** instead of the row's map.
- **Times with UTC offsets compared inconsistently**; they now compare by instant.
- **`toString()` returned an internal form for temporal values, lists and maps**; it now gives ISO 8601 text and literal form.
- **Very long `^` chains or runs of signs or `NOT`s could overflow the parser stack**.
- **Docs**: fixed the Discord invite and the GQL constraint examples ([#344](https://github.com/GrafeoDB/grafeo/issues/344)).

### Deriva

Changes for [Deriva](https://github.com/StevenBtw/deriva), which generates ArchiMate models from software repositories and keeps its graph in an embedded Grafeo database through the Python binding.

- **Deterministic Louvain**: the same partition on every run, with communities numbered from 0 by their smallest node id.
- **Python: `grafeo.features()` and `grafeo.build_info()`**: the compiled-in languages and capabilities, plus version, git commit and build profile.
- **Dotted access into map properties**: `n.meta.route` (also chained) in GQL and Cypher; a missing key gives null.
- **The direct API follows the selected graph** set with `set_graph()`, like `execute()`.
- **Graph handles**: `db.graph(name)` scopes queries, the direct API and transactions to one graph, usable side by side and across threads.
- **Direct writes are checked against the schema and constraints** like `INSERT`, and advance the epoch, so `changes_between` and `get_node_at_epoch` see them in order.
- **CDC events say what changed**: labels, edge type and endpoints, the last properties on delete, and one create event per entity created in a transaction.
- **Write counters**: `result.counters` reports nodes, edges, properties and labels created, set or removed.
- **Upserts by key**: `upsert_nodes(labels, rows, key="id")` and `upsert_edges(...)` create or update many entities by a key property and report created, updated and skipped rows.
- **`shuffle_unordered` test option**: returns rows without `ORDER BY` in random order, to catch code that relies on an order that is not guaranteed.
- **Batch edges with properties, batch nodes with several labels** ([#462](https://github.com/GrafeoDB/grafeo/issues/462)).
- **Cypher pattern predicates in `WHERE`**: `WHERE (d)-[:CONTAINS]->()` and `WHERE NOT (d)-->()`.

### Internal

- **CI gate and policy checks** ([#511](https://github.com/GrafeoDB/grafeo/issues/511)): a single `CI Gate` check to require before merging, and a `Policy` job for crate boundaries, toolchain pins and public-text rules.
- **CI toolchain pins restored** ([#509](https://github.com/GrafeoDB/grafeo/issues/509)); Dependabot no longer bumps the Rust toolchain.
- **Pull request eligibility check**: a `PR Policy` check applies the contribution rules in CONTRIBUTING.md.
- **Differential test**: `scripts/difftest` runs a GQL and Cypher query corpus on the previous release and on a release build of a commit, and fails on any changed result that was not reviewed for the release; it also compares GQL with Cypher within one run.
- **Browser WASM size limit**: the CI limit for the gzipped browser build is now 760 KB (warning at 740 KB); the build is about 730 KB after this release's fixes.

## [0.5.43] - 2026-09-27

Stabilization release. Fixes for silent wrong results (`ORDER BY` + `LIMIT`, `UNION`, aggregates, duplicate column names, SPARQL named graphs and property paths), queries whose trailing statements were ignored, rollbacks that did not undo changes on persistent databases or SPARQL updates, databases over 4 GiB written corrupt, edges lost after `compact()` and HNSW vector updates, plus dependency and security updates.

### Added

- **Gremlin negated text predicates**: `notRegex()`, `notContaining()`, `notStartingWith()` and `notEndingWith()` in `has()` filters, e.g. `g.V().has('city', notStartingWith('Am'))`. ([#336](https://github.com/GrafeoDB/grafeo/pull/336), [#340](https://github.com/GrafeoDB/grafeo/pull/340), [@jakeboone02](https://github.com/jakeboone02))

### Changed

- **Breaking (Rust API): `QueryResult::new`, `with_types` and `from_rows` return `Result`** and reject repeated column names ([#371](https://github.com/GrafeoDB/grafeo/issues/371)). Add `?` or `.unwrap()` at call sites.
- **Unaliased columns are named after their expression, as written**: `RETURN id(a), n.a + 1, count(b)` yields `id(a)`, `n.a + 1` and `count(b)` instead of `id(...)`, `expr` and `count(...)`. Aliased columns are unchanged. ([#350](https://github.com/GrafeoDB/grafeo/pull/350), [#372](https://github.com/GrafeoDB/grafeo/pull/372))
- **`restore_to_epoch()` refuses to overwrite an existing database** or its `.wal` sidecar; restore to a fresh path instead. ([#363](https://github.com/GrafeoDB/grafeo/pull/363), [@teipsum](https://github.com/teipsum))
- **Node.js: transaction queries run on a worker thread**, like `Database.execute()`, so they no longer block the event loop. `commit()` and `rollback()` now throw while a query from the same transaction is still running.
- **Queries with trailing input are rejected** ([#380](https://github.com/GrafeoDB/grafeo/issues/380)): GQL, Cypher and Gremlin fail with a syntax error on text after the statement, including `;`-separated statements (a trailing `;` is fine) and GQL clause orders not supported yet (e.g. `SET ... DELETE`). These used to run only the first part.
- **Breaking (Rust, `grafeo-core`): `ColumnCodec::write_to`, `write_to_v2`, `write_to_v3` and `CsrAdjacency::write_to` return `Result`**, failing instead of writing a truncated size ([#392](https://github.com/GrafeoDB/grafeo/issues/392)).
- **Rust toolchain pinned to 1.98.1** for local builds and CI. The MSRV stays 1.91.1, and the `grafeo` crate now declares it ([#390](https://github.com/GrafeoDB/grafeo/issues/390)).

### Fixed

- **GQL ran only the first statement and ignored the rest** ([#380](https://github.com/GrafeoDB/grafeo/issues/380)): `INSERT ... INSERT ...` (as in the quickstart) created only the first node and `INSERT ... RETURN` ignored its `RETURN`; both now work. Gremlin and Cypher had similar gaps.
- **GQL `^` (power) was not parsed**: `RETURN 2 ^ 10` returned `2`. It now computes the power, binding tighter than `*` and right associative.
- **`ALTER NODE TYPE / ALTER EDGE TYPE ... ADD PROPERTY name TYPE` added a property called `PROPERTY`** of type `name` and dropped the real type; `DROP PROPERTY name` dropped the wrong property. `PROPERTY` is now an optional keyword.
- **Databases over the storage format's limits were written corrupt** ([#392](https://github.com/GrafeoDB/grafeo/issues/392)): an LPG section over 4 GiB or 65,535 blocks, or over 65,535 labels on one node or versions of one property, wrapped a size field, and the file failed to open with `CRC mismatch`. Such a checkpoint now fails with an error naming the limit and keeps the sidecar WAL, so nothing is lost. Larger sections are planned for 0.6.0.
- **Rolling back a transaction on a persistent database did not undo `SET`, `REMOVE` or label changes**, and single-file databases wrote them to disk on `close()`.
- **`ORDER BY ... LIMIT` over a whole-node `RETURN` returned raw NodeIds** instead of node maps, both for property sort keys ([#335](https://github.com/GrafeoDB/grafeo/issues/335)) and for expression keys such as `text_score(...)`, `CASE` or arithmetic ([#347](https://github.com/GrafeoDB/grafeo/issues/347)). ([#337](https://github.com/GrafeoDB/grafeo/pull/337), [#349](https://github.com/GrafeoDB/grafeo/pull/349), [@temporaryfix](https://github.com/temporaryfix))
- **Duplicate column names silently lost data** ([#371](https://github.com/GrafeoDB/grafeo/issues/371)), e.g. `RETURN id(s), id(t)`. Distinct expressions now get distinct names, and a result that would still repeat a name is an error asking for an alias; in SPARQL, an `AS ?x` that repeats a projected or bound name is a parse error. ([#372](https://github.com/GrafeoDB/grafeo/pull/372), [@teipsum](https://github.com/teipsum); [#350](https://github.com/GrafeoDB/grafeo/pull/350), [@temporaryfix](https://github.com/temporaryfix))
- **`UNION` with differing branches** ([#365](https://github.com/GrafeoDB/grafeo/issues/365)): SPARQL now returns every variable from every branch, and GQL and Cypher reject branches with different columns instead of padding or truncating them. ([#366](https://github.com/GrafeoDB/grafeo/pull/366), [@teipsum](https://github.com/teipsum))
- **Aggregates next to aliased or later items** (GQL and Cypher): `RETURN n.city AS city, count(n)` failed with `Undefined variable '_agg_0'`, columns came back in the wrong order, and unreturned `GROUP BY` keys appeared as extra columns.
- **SPARQL updates ignored open transactions**: `INSERT/DELETE ... WHERE` applied immediately and was not undone by `rollback()`, and `INSERT DATA` / `DELETE DATA` on a named graph was dropped on commit. Updates now apply on commit, and queries in a transaction see its earlier writes.
- **SPARQL `FROM` and `WITH <g>` did not apply inside subqueries**: a nested `SELECT` read the default graph instead of the query's dataset.
- **SPARQL updates on named graphs went to the wrong graph or were lost** ([#367](https://github.com/GrafeoDB/grafeo/issues/367)): `INSERT`/`DELETE ... WHERE` with `GRAPH <g>` or `WITH <g>` now targets the named graph and survives a restart. `USING` / `USING NAMED` is rejected until it is supported. ([#368](https://github.com/GrafeoDB/grafeo/pull/368), [@teipsum](https://github.com/teipsum))
- **SPARQL `path+` returned only direct neighbours** ([#369](https://github.com/GrafeoDB/grafeo/issues/369)); it now returns the full transitive closure. ([#370](https://github.com/GrafeoDB/grafeo/pull/370), [@teipsum](https://github.com/teipsum))
- **Edges disappeared after `compact()`** once an endpoint was modified ([#345](https://github.com/GrafeoDB/grafeo/issues/345)). Deleted edges also no longer reappear or show up as neighbours. ([#346](https://github.com/GrafeoDB/grafeo/pull/346), [@temporaryfix](https://github.com/temporaryfix))
- **Updating an indexed vector could make other vectors unfindable** ([#374](https://github.com/GrafeoDB/grafeo/issues/374)). Removing or replacing the HNSW entry point also no longer leaves the upper index layers unreachable. ([#375](https://github.com/GrafeoDB/grafeo/pull/375), [@jarmen423](https://github.com/jarmen423))
- **Filters on edge properties or map values returned no rows** when nodes had a property with the same name, e.g. `MATCH ()-[r]->() WHERE r.id = 'e1'` or `UNWIND [{id: 'a'}] AS m WHERE m.id = 'a'`, because they were answered from node statistics.
- **Property maps on anonymous edges were ignored**: `()-[:T {since: 2020}]->()` in Cypher and GQL matched every `T` edge.
- **Deleted nodes stayed in property and vector indexes**, so `find_nodes_by_property` and vector search could still return them, and the lookup API returned uncommitted nodes. Rolling back a delete or `SET` now restores the index entries.
- **Computed property values in `CREATE`, `INSERT` and `MERGE`**: `CREATE (:X {id: toString(i)})` failed with an internal error, and `MERGE` silently used `null`, so every row matched or created the same node. They are now evaluated per row.
- **`MERGE` created duplicate relationships** when several rows of one statement merged the same edge (`UNWIND [1, 2] AS i MATCH (a), (b) MERGE (a)-[:T]->(b)` created two). A `null` in a relationship's `MERGE` pattern now also matches an absent property, as it already did for nodes.
- **`=` and `<>` on dates, times, datetimes, durations and vectors were always false / true** in expressions, e.g. `RETURN date('2024-01-01') = date('2024-01-01')`, `x IN [date(...)]` or a comparison after `WITH`. Filters answered directly from a node property were not affected.
- **Python: naive `datetime` values were read as local time** but returned as UTC; they are now UTC both ways. Microseconds are no longer rounded and dates before 1970 work on Windows.
- **Storage docs**: the `.grafeo` sidecar WAL is removed on a clean `close()`, not after every checkpoint. ([#364](https://github.com/GrafeoDB/grafeo/pull/364), [@teipsum](https://github.com/teipsum))

### Security

- Fixed RUSTSEC-2026-0190 (anyhow), RUSTSEC-2026-0186 (memmap2), RUSTSEC-2026-0204 (crossbeam-epoch), RUSTSEC-2026-0258 (h2) and RUSTSEC-2026-0285 (rustls), and replaced two yanked crates (chacha20, der).
- **Python bindings on PyO3 0.29**, fixing RUSTSEC-2026-0176 and RUSTSEC-2026-0177. No Python API changes.

### Dependencies

- Arrow and Parquet 59, aes-gcm 0.11, comfy-table 8, tikv-jemallocator 0.7 and hf-hub 1.0, among others ([#396](https://github.com/GrafeoDB/grafeo/pull/396)). The `embed` feature now downloads models through hf-hub's new client; models already in the Hugging Face cache are reused.

---

Thanks to [@teipsum](https://github.com/teipsum) for six PRs and the detailed reports behind them, to [@temporaryfix](https://github.com/temporaryfix) for [#346](https://github.com/GrafeoDB/grafeo/pull/346), [#349](https://github.com/GrafeoDB/grafeo/pull/349) and [#350](https://github.com/GrafeoDB/grafeo/pull/350) and the root-cause analysis on [#335](https://github.com/GrafeoDB/grafeo/issues/335), to [@jakeboone02](https://github.com/jakeboone02) for the Gremlin predicates, to [@jarmen423](https://github.com/jarmen423) for the HNSW fix, and to [@stiff](https://github.com/stiff), [@halaharvi](https://github.com/halaharvi), [@GanbaruTobi](https://github.com/GanbaruTobi) and [@cuongvo](https://github.com/cuongvo) for their reports. The filter, index and `MERGE` fixes come from reports by the Deriva project.

## [0.5.42] - 2026-05-04

End-to-end tiered storage: section data (LPG, RDF Ring, vector topology) can spill to mmap-backed disk under memory pressure or explicit configuration, with per-section tier overrides, introspection, and reload. Plus per-block columnar zone maps for selective range scans, paged HNSW topology, packed RDF Ring on disk, a WAL overlay for mutating mmap'd compact stores and a streaming top-K operator that fuses `ORDER BY ... LIMIT` into a single bounded-heap pass.

### Added

- **Per-section storage tier configuration**: `Config::with_section_tier(SectionType, TierOverride)` pins individual sections to RAM or disk. `ForceDisk` spills at db open; `ForceRam` is hard-enforced (skipped in every spill loop). `Auto` (default) defers to the buffer manager.
- **`db.storage_tiers()` introspection**: returns a map of section type to current tier (`InMemory` / `OnDisk` / `Uninitialized`). New `MemoryConsumer::current_tier` trait method backs it with authoritative state.
- **`db.reload_eligible(target_fraction)`**: brings spilled sections back into RAM in priority order, stopping when projected usage exceeds the target. Returns the count reloaded.
- **Python bindings for tier control**: `GrafeoDB(path, section_tiers={'VectorStore': 'force_disk'})`, `db.storage_tiers()`, `db.reload_eligible()`. Eight new pytest cases.
- **Paged HNSW topology (vector v2)**: `VectorStoreSection` switches from bincode to a packed `GVST` envelope with per-index `GTOP` paged topology. `MmapTopology` serves neighbor lookups directly from `Bytes` slices; `HnswIndex` dispatches search through `TopologyBackend { Heap, Mmap }`. Heap drops >10x under mmap, recall is bit-identical.
- **Packed RDF Ring (ring v2)**: bincode `RdfRingSection` replaced by a `GRFR` envelope (sorted term dictionary + three packed wavelet trees + two permutations + CRC32). `swap_to_mmap` shares refcounted `Bytes` slices from the spill mmap.
- **Per-block columnar zone maps + iterator bounds**: LPG compact-store columns lay out as 4 KiB blocks with per-block min/max. `find_in_range_iter` skips blocks via zone maps; `RangeScanOperator` streams matches into the push pipeline. Large speedups on selective range scans.
- **WAL overlay for mutable mmap'd LPG**: when the compact base is `OnDisk`, mutations route to a live `LpgStore` overlay; periodic merge folds it back via `LayeredStore::merge_overlay_in_place`. New `merge_guard` makes concurrent reads + writes + merges race-free. Persistent file + spilled base + overlay mutations round-trip across restarts.
- **`SectionType::OverlayDeletions`**: persists `LayeredStore`'s deletion log so deletions of base entities survive close/reopen without an explicit `compact()`. Omitted from the container directory when empty. Marked `required: false`; meaningful once the directory parser learns to skip non-required unknowns (planned for 0.5.43).
- **`tracing` events on tier transitions** (with `tracing` feature): info events under `grafeo::buffer` and `grafeo::tier` for spill, reload, and ForceDisk applications.
- **Storage Tiers documentation page**: `docs/architecture/memory/storage-tiers.md` covering tiers, overrides, introspection, reload, and tracing conventions.
- **`PageFetcher` trait + `MmapPageFetcher` impl**: indirection layer that lets sections receive paged byte access without knowing the source, opening a future swap to vmcache or an explicit pager.
- **Streaming top-K operator** (`TopKOperator`): bounded-heap of size k, O(k) memory regardless of input cardinality, ~12x faster than the unfused `Sort` + `Limit` at N=1M. Planner fuses literal `LIMIT k` over `Sort` into a single physical operator; vector/text top-K still takes precedence and PROFILE-mode plans bypass the fusion for entry-count parity. ([#326](https://github.com/GrafeoDB/grafeo/pull/326), [@temporaryfix](https://github.com/temporaryfix))
- **`var.prop IN [literals]` property-index fast path**: per-value index lookups unioned via `NodeListOperator`, deduped, label/MVCC-filtered. ~56x speedup on `WHERE id IN $ids` against a snapshot-loaded database. ([#326](https://github.com/GrafeoDB/grafeo/pull/326), [@temporaryfix](https://github.com/temporaryfix))
- **`LogicalOperator::map_children`**: child-recursive optimizer passes descend without enumerating every variant.

### Changed

- **Vector store and RDF Ring section formats bumped to v2** (paged `GVST` and packed `GRFR` envelopes). Existing v1 bincode files keep loading via magic-byte detection and upgrade to v2 on the next checkpoint.
- **`TierOverride::ForceDisk` enforcement is now targeted**: previously triggered `spill_all()` for every consumer; now `spill_consumer_by_name(...)` per matching section, so unrelated consumers stay InMemory.
- **Filter pushdown extended through `Filter`, `LeftJoin`, `Apply`, `Union`, `Unwind`**: predicates inside `OPTIONAL MATCH` and correlated subqueries now reach the scan. New pre-pushdown pass propagates filter predicates across `LeftJoin` shared-variable boundaries so the right-side subtree of an OPTIONAL MATCH picks up the same `WHERE` constraints as the main `MATCH` (~8x on `feed.hydrate`-shaped queries). Sibling `Filter` commute is gated by a variable-scope check so predicates over path variables like `LENGTH(p) >= 1` don't push past a label filter into a subtree where the path isn't yet bound. ([#326](https://github.com/GrafeoDB/grafeo/pull/326), [@temporaryfix](https://github.com/temporaryfix))

### Fixed

- **`MERGE ... ON CREATE/ON MATCH SET` could not reference the MERGE variable** (#317): expressions like `ON MATCH SET c.description = coalesce(c.description, 'fallback')` failed at the binder, and any non-trivial action expression that did pass through was silently lowered to `Null`. Binder now scopes the MERGE variable into ON CREATE / ON MATCH (per ISO/IEC 39075:2024 §15.5); planner emits a `PropertySource::Expression` for action expressions; operators evaluate them against an augmented row containing the merged entity. Reported by [@Fraenkstan](https://github.com/Fraenkstan).
- **`TierOverride::ForceRam` was silently a no-op**: prior versions accepted the config but the buffer manager spilled `ForceRam` consumers anyway under pressure. The spill loop now consults a `force_ram_consumers` set and skips matching consumers in `run_eviction_internal`, `spill_all`, and `spill_consumer_by_name`.
- **Persistent reopen lost the compact base after `compact()`**: directory parser had no `SectionType::CompactStore` arm and `load_from_sections` had no handler, so reopens fell back to a legacy path that surfaced as a checksum mismatch. New `extract_compact_base` + `wire_layered_after_load` helpers rebuild the LayeredStore and consumer wiring on open.
- **Concurrent merge could lose writes**: stress test with 4 readers + writer + merger lost 426/500 writes in one run. Fixed via `merge_guard: RwLock<()>` (mutators take read, merge takes write).
- **`GRAFEO-X001: snapshot checksum mismatch` opening v2 section files** ([#323](https://github.com/GrafeoDB/grafeo/issues/323), [#324](https://github.com/GrafeoDB/grafeo/pull/324)): `read_snapshot` was unconditionally a v1 reader and CRC'd zero bytes against the v2 directory CRC when the engine's open path fell through. Reader now early-returns on a v2 header (`snapshot_length == 0` with non-empty header), letting the engine's v2 dispatch take over. Reported by [@teipsum](https://github.com/teipsum) against an embedded production database.
- **Silent corruption masking in `read_section_directory`** (#323 follow-up): the v2 directory parser swallowed `from_bytes` errors with a wildcard, routing truncated directories and torn-page writes into the v1 read logic and surfacing as a misleading snapshot-checksum error. Parser now propagates errors with the failing offset, treats "v2 header but file too short" as an error rather than a v1 fallback, and verifies the directory page CRC. Regression tests cover both paths.
- **`LayeredStore` deletions silently reverted on reopen** (#323 follow-up): base nodes/edges deleted after `compact()` were tracked only in memory, so reopen made them reappear until the next `compact()`. New `OverlayDeletions` section persists the deletion log; open path seeds the layered store from it. Round-trip integration tests lock in the behaviour.
- **Per-query spill directory leak** (#323 follow-up): `<spill_path>/query_<id>/` directories were created per query and never removed; one production day accumulated 358 empty subdirectories. `SpillManager`/`AsyncSpillManager` now expose `with_owned_dir()` so `Drop` removes the directory non-recursively (preserving unexpected siblings).
- **`active_db_header` docstring drift**: the function picks the higher-iteration slot unconditionally; checksum validation lives in the readers. Doc rewritten to match.
- **Equality-scan regression from the Bytes-backed codec refactor**: `BitPackedInts`, `DictionaryEncoding`, and `ColumnCodec::{Float64,RawI64}` switched to a two-variant `Inline(Vec<T>)` / `Mapped(Bytes)` store so in-RAM builds keep native slice iteration and mmap loads stay zero-copy. Closes the 19-29% CodSpeed regression on `compact/find_nodes_by_property/{int_eq,dict_eq}`.
- **Property-index fast path silently disabled after `compact()`**: `LayeredStore::has_property_index` now delegates to the overlay (was using the trait default `false`). Snapshot-loaded databases stop falling back to the ~3.5x-slower label-first scan. ([#326](https://github.com/GrafeoDB/grafeo/pull/326), [@temporaryfix](https://github.com/temporaryfix))
- **`ORDER BY` against a `WITH` alias dropped by `RETURN`**: e.g. `WITH x AS s ... RETURN x ORDER BY s` failed with `Variable 's' not found` because the augmented-projection path skipped Variable sort keys. They now pass through alongside Property keys. ([#326](https://github.com/GrafeoDB/grafeo/pull/326), [@temporaryfix](https://github.com/temporaryfix))
- **`PROFILE` panicked on fused fast-path operators**: property/range/IN-list paths absorbed their child `NodeScan` into one physical op, leaving `build_profile_tree` short an entry. Affected paths now emit a synthetic `NodeScan` entry; the factorized expand-chain fusion is also gated under `PROFILE`. ([#326](https://github.com/GrafeoDB/grafeo/pull/326), [@temporaryfix](https://github.com/temporaryfix))

### Deprecated

- **`TieredStore` trait** (`grafeo_common::memory::buffer::TieredStore`): never implemented anywhere; the `Section` trait (with `swap_to_mmap` + `reload_to_ram`) plus `MemoryConsumer` cover the same lifecycle. Marked `#[deprecated(since = "0.5.42")]` and scheduled for removal in 0.6.0. The `StorageTier` enum it shipped alongside is kept.

---

Thanks to [@teipsum](https://github.com/teipsum) (Michael Lewis Cram) for reporting [#323](https://github.com/GrafeoDB/grafeo/issues/323) with byte-level analysis from a production embedded database, and for [#324](https://github.com/GrafeoDB/grafeo/pull/324)'s `read_snapshot` fix which lands cherry-picked with authorship preserved. The follow-up work in this release (stricter `read_section_directory` validation, durable `LayeredStore` deletions via the new `OverlayDeletions` section, and the per-query spill directory cleanup) addresses the deeper root causes the original report surfaced.

Thanks to [@temporaryfix](https://github.com/temporaryfix) for the top-K operator, IN-list fast path, LeftJoin filter propagation, and the cluster of fixes around `compact()` and PROFILE in [#326](https://github.com/GrafeoDB/grafeo/pull/326), reshaped to fit grafeo's `map_children` recursion pattern before merging.

## [0.5.41] - 2026-04-24

Compact-store correctness (post-`compact()` read path, signed integer round-trip), search procedures, disk-backed compact base, silent-hybrid-on-persistent-DB fix, memory introspection for RDF and CDC, and test-infrastructure hardening (proptest, persistent spec variants, CodSpeed CI).

### Added

- **`CALL grafeo.search.*` procedures**: first-class procedure entry points for text and vector search, with scalar similarity available as an expression in projections. Routes through the existing `GraphStoreSearch` surface so WAL/CDC wrappers and `LayeredStore` all work.
- **WASM transactions + `close()`**: explicit transaction API for the browser bindings, plus `close()` to release handles deterministically instead of waiting for GC.
- **Tamper-evident WASM snapshots**: `exportSnapshotSigned(key)` / `importSnapshotSigned(data, key)` wrap snapshots with a `GSN1` magic header and an HMAC-SHA256 tag over magic + payload, with 128 MiB input cap and constant-time verification. `importSnapshot()` refuses `GSN1`-prefixed payloads so the two entry points can't be confused.
- **Property-based round-trip coverage for `compact()`** (#303): proptest generates arbitrary LPG graphs and asserts equivalence across 28 GQL queries per case on a compacted vs fresh database. 128 cases by default, `PROPTEST_CASES=1024` for local investigation. Surfaced the two compact-store bugs fixed in this release (#301, #302).
- **Persistent spec-test variants** (#309): every `.gtest` case requiring `text-index` or `vector-index` now auto-generates a `_persistent` sibling that opens `GrafeoDB::open(tempdir)`, exercising the WAL-wrapped read path used by every on-disk session. 49 persistent variants land with this release.
- **CodSpeed continuous benchmark regression** (#304): all seven Criterion suites run under Callgrind on every PR via `cargo codspeed`, posting a diff vs `main` as a PR comment. Fork PRs skip cleanly; plain `cargo bench` continues to work unchanged.
- **`ColumnCodec::RawI64`** (#306): native signed 64-bit codec for columns containing at least one negative value, with i64 comparison in `find_eq` / `find_in_range` and signed zone maps. Non-negative columns continue to use the more compact `BitPacked` encoding.
- **wasm32 simd128 distance kernels** (#305): `std::arch::wasm32` implementations of all four HNSW distance metrics (dot product, squared Euclidean, cosine, Manhattan), 4.36x to 5.35x faster on 384-dim f32 vectors. Enabled by default for wasm32 builds via `.cargo/config.toml` (`target-feature=+simd128`); runtime requirement Chrome 91+ / Firefox 89+ / Safari 16.4+. Relaxed-simd FMA deliberately skipped: spec permits runtime-defined rounding, and wasmtime regresses vs plain add+mul.
- **RDF and CDC memory breakdown**: `db.memory_usage()` gains `rdf` (triple count, term dictionary, optional Ring index, named-graph count) and `cdc` (entity count, event count) blocks alongside the existing store/index/MVCC/cache totals. Feature-gated and skipped from JSON when empty.
- **CLI `:memory` meta-command**: prints the hierarchical memory breakdown in the REPL, omitting zero-valued and feature-disabled components.
- **Disk-backed compact base** (`compact-store + mmap` features): `CompactStoreTiered` wraps the columnar base in a two-state `InMemory` / `OnDisk` (mmap) machine. `compact()` registers a `CompactStoreConsumer` with the `BufferManager`; under memory pressure the base serialises to `<spill_path>/compact_base.grafeo` and `ArcSwap` publishes the fresh mmap-backed `Arc` through `LayeredStore`, so queries see no discontinuity and the old heap allocation drops.
- **Contributor docs for CDC, query planner, and MVCC visibility**: module-level `//!` guides covering the CDC event model and epoch relationship, the planner's rewrite and filter-pushdown ordering, and the `EpochId::PENDING` / visibility / `TransactionWriteTracker` flow.

### Changed

- **On-disk codec format extended**: databases written by 0.5.41+ may contain columns under the new `RawI64` discriminant (6). Earlier 0.5.4x binaries reject these with `unknown codec discriminant`. One-way format break, in line with the 0.5.35 precedent.

### Fixed

- **Post-`compact()` writes invisible to `MATCH`** (#307, closes #302): `LayeredStore` get/versioned/epoch/property/type/visibility methods now fall through to the overlay when the base doesn't recognise the id; `edges_from` / `neighbors` always consult the overlay so cross-layer edges surface. Results dedup by `EdgeId` to handle promoted edges.
- **Signed Int64 columns stringified after `compact()`** (#306, closes #301): negative-containing `Int64` columns were routed to `InferredType::Dict`, silently becoming `Value::String` (`WHERE n.num = 100` returned zero rows). Signed columns now use the new `RawI64` codec, preserving type and ordering through compaction.
- **Silent text/vector search on file-backed DBs** (#309, closes #308): `WalGraphStore` and `CdcGraphStore` fell through to the `GraphStoreSearch` trait defaults (all no-ops), so every index lookup on a `GrafeoDB::open()` session silently returned "no index." Both wrappers now delegate every `GraphStoreSearch` method to `self.inner`.
- **Cypher aggregate substitution inside CASE WHEN** (#300): aggregates wrapped in `CASE WHEN ... THEN sum(...) ...` now substitute the reduced value post-aggregation instead of leaving an unresolved reference.
- **VectorScan `k=None` risked HNSW overflow** (#299 follow-up): unbounded k was being passed as `usize::MAX`, degrading HNSW to full traversal and risking overflow in quantized rescore. The planner now bounds k to the label's node count via a new `nodes_by_label_count` trait method.
- **Native SIMD kernels read past `b` on mismatched slice lengths** (#312, closes #311): AVX2, SSE, and NEON kernels drove the main loop from `a.len()` and raw-pointer-loaded `b`, guarded only by a `debug_assert_eq!`. The four public `*_simd` dispatchers now assert length equality in release.
- **`rustls-webpki` 0.103.13**: clears RUSTSEC-2026-0104 (reachable panic in CRL parsing), pulled transitively via `hf-hub -> ureq -> rustls`.
- **GQL schema DDL: `CREATE GRAPH TYPE` bare references no longer corrupt the catalog** (#316): `NODE TYPE Person` and `EDGE TYPE KNOWS` inside a graph-type body are now treated as references to existing catalog entries per ISO/IEC 39075:2024, not as empty inline redeclarations. The previous behavior silently wiped `NOT NULL` and other property constraints on the referenced types. References to undefined types now error cleanly at `CREATE GRAPH TYPE` time.

### Dependencies

- `proptest` 1.x added as workspace dev-dep and enabled on `grafeo-engine` for property-based tests.
- `codspeed-criterion-compat` 3.x added as workspace dev-dep and enabled on `grafeo-common`, `grafeo-core`, `grafeo-storage`, `grafeo-engine`. Drop-in for Criterion; pass-through outside `cargo codspeed run`.
- `tempfile` added as a dev-dep on `grafeo-spec-tests` for the `_persistent` variants.
- `arc-swap` 1.x added to the workspace and enabled on `grafeo-core` under the `compact-store` feature, backing the `LayeredStore` base pointer for lock-free atomic swap.

---

Thanks to [@temporaryfix](https://github.com/temporaryfix) for substantial work this cycle: the two compact-store correctness fixes (#306, #307) and the property-based suite that surfaced them (#303), the hybrid-on-persistent fix (#309), the Cypher CASE aggregate fix (#300), the VectorScan k bounding follow-up to #299, the native SIMD safety assert (#312, closes #311), CodSpeed benchmark CI (#304), and the wasm32 simd128 distance kernels (#305).

## [0.5.40] - 2026-04-20

Unified hybrid queries (graph + vector + text), lazy streaming results, structured Python errors, catalog hierarchy hardening, and compact-store fixes.

### Added

- **Unified hybrid queries**: `text_score()` and `text_match()` usable as filter expressions, with planner pushdown of score predicates to `TextScan` / `VectorScan` operators, compound AND/OR joins, top-K recognition, and score projection. Inspired by [#287](https://github.com/GrafeoDB/grafeo/pull/287) ([@temporaryfix](https://github.com/temporaryfix)); reimplemented via the `GraphStoreSearch` subtrait.
- **BM25 text scan operator**: `TextScanOperator` with top-K and threshold modes. `InvertedIndex` gains `score_document`, `search_with_threshold`, `bm25_term_score`. ([#287](https://github.com/GrafeoDB/grafeo/pull/287), [@temporaryfix](https://github.com/temporaryfix))
- **Native Float64 and Float32Vector codecs**: CompactStore stores them directly instead of falling back to dictionary encoding. Mixed `Int64+Float64` columns coalesce to Float64. ([#286](https://github.com/GrafeoDB/grafeo/pull/286), [@temporaryfix](https://github.com/temporaryfix))
- **Streaming query results** (experimental): `Session::execute_streaming` returns a `ResultStream` that pulls one `DataChunk` at a time, bounded memory regardless of result-set size. Exposed across bindings: Python `execute_lazy()`, Node.js `executeStream()`, C# `ExecuteStream()`, Dart `executeStream()`, and Go and C FFI equivalents. Rejects mutations, EXPLAIN/PROFILE, session commands, and push-only plans.
- **Python `GrafeoError` exception**: subclass of `RuntimeError` carrying `error_code` (`"GRAFEO-Q001"`) and `is_retryable`. Legacy `except RuntimeError:` paths keep working.
- **Error codes reference**: user-guide page documenting every `GRAFEO-*` code, retry semantics, and a Python retry-loop sample.
- **Catalog hierarchy docs** (ISO/IEC 39075): user-guide page covering schemas, named graphs, session state, isolation, and cross-schema transactions.

### Changed

- **`DatabaseStats.memory_bytes` reflects the full heap breakdown**: now equals `memory_usage().total_bytes` (store + indexes + MVCC + caches + string pools + buffer manager) instead of just buffer-manager-tracked bytes.
- **Schema and graph names reject `/`**: `CREATE SCHEMA` / `CREATE GRAPH` now fail on names containing `/`, which Grafeo uses internally as the compound `schema/graph` storage-key separator.

### Fixed

- **Multi-schema transaction atomicity**: `SESSION SET SCHEMA` mid-transaction no longer loses pre-switch writes on COMMIT. Fix centralizes "touched graph" tracking inside the session setters so every active-key change is recorded.
- **Commit failure auto-rollback**: a failed COMMIT now discards pending writes and returns the session to a clean state, instead of leaving the transaction in-flight with uncommitted writes still visible.
- **Parser keyword anti-pattern**: `try_accept_keyword` helpers fix four `CREATE CONSTRAINT ... FOR`, `ON REPLACE`, and similar sites where identifier-fallback tokenization accepted the wrong keyword.
- **Vector strict pushdown boundary leak**: `euclidean_distance(...) < t` and `manhattan_distance(...) < t` now correctly exclude rows at exactly the threshold; strict comparisons attach a residual filter above the vector scan, which was previously dropped.
- **MERGE index lookup**: `MERGE (n:Label {prop: value})` now uses property indexes when available, eliminating O(n) scan on large graphs. ([#288](https://github.com/GrafeoDB/grafeo/issues/288))
- **Index and search after `compact()`**: ~26 vector/text index methods no longer panic with "no built-in LpgStore" or silently return empty results. ([#286](https://github.com/GrafeoDB/grafeo/pull/286), [@temporaryfix](https://github.com/temporaryfix))
- **`LayeredStore` new-node visibility**: `get_node` / `get_node_property` fall back to the overlay for nodes added after `compact()`. ([#286](https://github.com/GrafeoDB/grafeo/pull/286))
- **Named graphs across `compact()` / `recompact()`**: graphs existing before compaction are carried into the new overlay; `list_graphs`, `drop_graph`, `create_graph`, `set_current_graph` see them.
- **Layered scan lock holding**: `nodes_by_label` acquires `dirty_node_ids` once per scan instead of re-locking in the chunk loop. ([#278](https://github.com/GrafeoDB/grafeo/pull/278), [@temporaryfix](https://github.com/temporaryfix))

## [0.5.39] - 2026-04-16

Block-STM conflict partitioning, push-based query execution, AES-256-GCM encryption at rest, runtime metrics with Prometheus export, and a writable layered compact store.

### Added

- **Block-STM conflict partitioning**: groups conflicting transactions into disjoint clusters for parallel re-execution.
- **Encryption at rest** (`encryption` feature): AES-256-GCM for WAL records and `.grafeo` sections. Password-based (Argon2id) or raw-key setup. Zero overhead when disabled.
- **Push-based pipeline execution**: filter, sort, aggregate, limit, and distinct queries execute through a push pipeline, reducing per-row overhead.
- **Runtime metrics** (`metrics` feature): query, transaction, session, cache, and GC counters with Prometheus text export. Python `db.metrics()` / `db.metrics_prometheus()` and Node.js equivalents.
- **C# enterprise APIs**: schema management, backup/restore, compact, projections, CDC toggle, plan cache. `IGrafeoDB` and `ITransaction` interfaces for DI.
- **Resource limits**: default 30-second query timeout (`Config::with_query_timeout()`), 16 MiB property value size limit (`max_property_size`), HNSW `max_elements` bound.
- **Layered store** (`compact-store` feature): `compact()` produces a writable two-layer store (columnar base + overlay) instead of a read-only snapshot. `recompact()` merges the overlay back. Versioned section format with CRC32 integrity and ID-preserving builds.
- **WAL benchmarks**: Criterion benchmarks for write throughput, batch commit, and recovery replay.

### Changed

- **`compact()` is now non-destructive**: creates a writable layered store instead of converting to read-only mode.
- **Cast clippy lints re-enabled**: `cast_possible_truncation`, `cast_sign_loss`, `cast_possible_wrap` promoted to `warn` workspace-wide.
- **Leaner WASM builds**: removed `grafeo-storage`, `crc32fast`, `anyhow` from WASM targets. Binary size: 650 KB gzipped. CI threshold: 660 KB warn, 700 KB fail.
- **Expand locality optimization**: sorts input chunks by source node ID before adjacency lookups on large traversals.
- **CI hardening**: MSRV verification (1.91.1), typos check, supply-chain audit as required status check, benchmark regression gating for core paths, Node.js matrix reduced to 22/24.
- **Release pipeline hardening**: explicit publish errors, pre-publish version consistency gate. Removed stale `deny.toml` skips, updated `rustls-webpki` (RUSTSEC-2026-0098).

### Fixed

- **SSI validation race**: concurrent commits could miss read-write conflicts due to a gap between state update and epoch recording. Both locks now held atomically.
- **Transaction lock ordering**: consistent write-lock ordering in `commit()` and `gc()`, eliminating a potential deadlock from the previous read-then-upgrade pattern.
- **EXPLAIN/PROFILE nesting bypass**: recursive EXPLAIN/PROFILE in GQL and Cypher now counts toward the 128-level nesting limit.
- **Session commit atomicity**: `touched_graphs` clone-then-clear replaced with atomic `mem::take()`.
- **WAL encryption nonces**: fixed reuse on restart, collisions across sections, and u64-to-u32 truncation. Old log file now fsynced before rotation.
- **HKDF key derivation**: added domain-separation salt, preventing cross-protocol key reuse.
- **Parser overflow hardening**: integer overflow in Cypher, SQL/PGQ, Gremlin, GraphQL now returns errors instead of silently producing `0`. Float overflow (`1e999`) in GraphQL rejected.
- **Numeric cast safety**: arena allocator, block serializer, buffer manager, DPccp optimizer, temporal constructors, and `toInteger()` all use checked arithmetic instead of unchecked casts.
- **DPccp join optimizer**: fixed `BitSet` overflow for 64+ relations, added 100K iteration budget to prevent stall on large joins.
- **DISTINCT hash collisions**: content-based hashing for List, Map, Vector, and Path values.
- **Parameters in subqueries**: `$param` inside `EXISTS`/`COUNT`/`VALUE` subqueries now substituted correctly. ([@temporaryfix](https://github.com/temporaryfix/grafeo/pull/2))
- **SHACL SPARQL injection**: IRI validation prevents breakout via crafted `$this` values.
- **CDC history without permission**: now requires RBAC read permission.
- **SIMD and arena checks**: promoted debug-only vector length and alignment validation to release builds.
- **Buffer manager TOCTOU race**: replaced non-atomic check-then-allocate with `compare_exchange` loop.
- **RDF dictionary panic**: graceful error on u32::MAX entry overflow instead of panic.
- **Windows memory detection**: reads actual physical memory instead of falling back to 1 GB.
- **Python `execute_sql` language mismatch**: standardized to `"sql"` across all bindings.
- **Gremlin `range(5, 2)` overflow**: returns error instead of panic when end < start.

## [0.5.38] - 2026-04-13

Hardening, ISO compliance, and vector search improvements driven by persona-based exploratory testing. Parser security limits prevent stack overflow attacks, all six query languages gain EXPLAIN support, Unicode identifiers bring GQL closer to ISO 39075, and quantized vector indexes cut memory usage up to 4x for large embedding workloads.

**Breaking:** `QueryBuilder.param()` now raises `ValueError` on unsupported types instead of silently dropping them: code that relied on silent fallthrough will see exceptions and should fix the type conversion. Parser error messages now include a language prefix (e.g., `[GQL] Unexpected token`): code that pattern-matches on error strings may need updating. SPARQL `SERVICE` clauses now return an explicit error instead of silently executing the inner pattern against the local store: queries that appeared to work but returned incorrect results will now fail with a clear message. SPARQL property path `+`/`*` expansion depth raised from 10 to 50: queries on deep hierarchies will return more complete results, which may increase result set sizes and execution time.

### Added

- **Quantized vector indexes**: `create_vector_index()` accepts `quantization` parameter (`"scalar"`, `"binary"`, `"product"`) for 4x memory reduction on large vector datasets. `VectorIndexKind` enum unifies plain and quantized indexes throughout the engine. All bindings (Python, Node.js, WASM, C) updated.
- **EXPLAIN/PROFILE for all 6 query languages**: Gremlin, GraphQL, and SQL/PGQ now support `EXPLAIN` and `EXPLAIN ANALYZE` prefix, matching existing GQL, Cypher, and SPARQL support. Python `explain()`, `explain_cypher()`, `explain_sql()`, `explain_gremlin()` convenience methods added.
- **Unicode identifiers**: GQL, Cypher, and SQL/PGQ parsers now accept Unicode letters in identifiers (e.g., `CREATE (:人物 {名前: 'Alix'})`), per ISO GQL 39075. Gremlin and GraphQL already supported this.
- **Unicode string escapes**: `\uXXXX` (4-digit BMP) and `\UXXXXXXXX` (8-digit full range) escape sequences in string literals across all query languages.
- **CONSTRUCT output serialization**: Python `QueryResult.to_ntriples()` and `to_turtle()` methods for SPARQL CONSTRUCT results.
- **NetworkX ID round-tripping**: `from_networkx()` preserves original node IDs via `_networkx_id` property, `to_networkx()` restores them. Returns a node mapping dict.
- **GQL `!=` operator**: accepted as alias for `<>` (not-equal comparison).

### Changed

- **Parser error messages now identify the language**: all 6 parsers prefix errors with `[GQL]`, `[Cypher]`, `[SPARQL]`, `[Gremlin]`, `[GraphQL]`, `[SQL/PGQ]`.
- **SPARQL property path depth raised to 50**: `+`/`*` paths now expand to 50 hops (up from 10), covering most real-world taxonomies and org charts.
- **`QueryBuilder.param()` raises `ValueError`**: previously silently dropped unsupported types, now raises with a descriptive message.
- **`execute_async()` documentation**: docstring now explains that it uses `spawn_blocking` (releases GIL, uses thread pool, not truly non-blocking I/O).

### Fixed

- **Parser recursion depth limits**: all 6 parsers now enforce a 128-level nesting limit, preventing stack overflow on deeply nested malicious input (DoS vector).
- **SPARQL SERVICE clause returned wrong results silently**: now returns an explicit error instead of executing the inner pattern locally.
- **GQL integer overflow produced confusing errors**: overflow on integer literals now reports the value and valid i64 range.
- **NetworkX `in_degree()` was O(V*E)**: replaced full-graph scan with direct adjacency index lookup.
- **`AsyncQueryResult` missing `nodes()`/`edges()`**: entity extraction now runs post-`spawn_blocking`, matching sync `QueryResult`.
- **RDF blank node collisions across imports**: Turtle parser now prefixes blank node IDs per-import (`_:imp{N}_b0`), preventing cross-file collisions.
- **Incremental backup always failed after full backup** (#267): the backup cursor stored the active WAL file's sequence without rotating, so post-backup writes stayed invisible to incremental. Both `backup_full` and `backup_incremental` now rotate the WAL after completing, ensuring new writes land in a file the next incremental will pick up.
- **Edge variables in multi-hop queries returned as raw IDs** (#268): `plan_expand_chain` and `plan_factorized_aggregate` did not register edge columns in the planner's tracking set, causing RETURN to emit `NodeResolve` instead of `EdgeResolve`. Edge variables now resolve to full maps with `_id`, `_type`, `_source`, `_target`, and properties.
- **Arrow/DataFrame export dropped user properties named `source`/`target`/`id`/`type`** (#272): structural columns in `edges_to_arrow()`, `edges_df()`, `nodes_to_arrow()`, and `nodes_df()` collided with user property names, silently dropping them. Structural columns are now underscore-prefixed (`_id`, `_type`, `_source`, `_target`, `_labels`) to match the engine's `edge_to_map()`/`node_to_map()` convention. **Breaking:** code referencing `df["source"]` must change to `df["_source"]`.
- **Weighted hybrid search inverted vector ranking**: `hybrid_search()` with `fusion="weighted"` applied min-max normalization to raw vector distances, causing the farthest node to score highest. Vector distances are now negated before fusion so that closer vectors rank higher.

### Documentation

- **Search score conventions**: new table in the Vector Search guide clarifying return value semantics across all search methods (`vector_search` returns distances, lower = better; `hybrid_search` returns fusion scores, higher = better; `mmr_search` returns distances in MMR selection order; `text_search` returns BM25 scores, higher = better).
- **Text Search guide**: new dedicated page covering BM25 index creation, searching, auto-sync behavior, and when to rebuild.
- **Hybrid Search guide**: new dedicated page covering RRF vs weighted fusion, prerequisites, graceful degradation, and the common pitfall of treating fusion scores as distances.
- **MMR Search guide**: new dedicated page covering Maximal Marginal Relevance parameters, lambda tuning, and when to use MMR vs vector search.
- **Index auto-sync clarified**: `rebuild_vector_index()` and `rebuild_text_index()` docs now explain that indexes auto-sync on `set_node_property()` and batch operations; explicit rebuild is rarely needed. Updated across Rust doc comments, Python/Node.js binding docstrings, and API reference pages.

## [0.5.37] - 2026-04-12

RDF Semantic Web overhaul with improved SPARQL support, RDF performance improvements and SHACL validation.

### Added

- **SPARQL compliance pass**: spec gaps closed. `CONSTRUCT`, `BIND`, `OPTIONAL`, `MINUS`, `UNION`, `FILTER`, `EXISTS`/`NOT EXISTS`. Named graph CRUD and SPARQL UPDATE. Composite indexes (SP, PO, OS) for O(1) multi-bound lookups. new W3C tests.
- **Ring Index planner** (`ring-index`): wavelet-tree compact triple index wired into SPARQL planner. Leapfrog WCOJ for multi-way star joins, hash join fallback when LANG/DATATYPE columns needed.
- **Ring Index persistence**: bincode serialization with post-load invariant validation. `RdfRingSection` persists to `.grafeo` container, eliminating rebuild on restart.
- **Dictionary encoding infrastructure**: `TermDictionary` maps terms to u32 IDs. `DictResolveOperator` resolves at result boundaries. Built lazily, invalidated on mutation.
- **COUNT(\*) fast paths**: O(1) for unbound scans via `store.len()`, O(log sigma) for bound patterns via Ring Index.
- **RDF query optimizer**: per-predicate cardinality estimates, cached statistics, cost-based join reordering.
- **SPARQL EXPLAIN / EXPLAIN ANALYZE**: physical plan tree without executing, or profiled execution with per-operator timing. Python `explain_sparql()` binding.
- **SHACL validation** (`shacl`): W3C Shapes Constraint Language with all 28 Core constraint types, SHACL-SPARQL (`sh:sparql`), 7 property path types with cycle detection, `ValidationReport` with `to_triples()` RDF materialization. `session.validate_shacl(shapes_graph)` in Rust, `db.validate_shacl("graph")` in Python. In `rdf` persona and `server` profiles.
- **Arrow bulk export** (#260): `nodes_to_arrow()`/`edges_to_arrow()` (pyarrow Table), `nodes_to_polars()`/`edges_to_polars()` (Polars DataFrame), `nodes_to_pandas()`/`edges_to_pandas()` (pandas DataFrame via Arrow). Builds RecordBatch in Rust, serializes to IPC: ~10-100x faster than per-element `nodes_df()`/`edges_df()` at scale. Existing `nodes_df()`/`edges_df()` auto-use the Arrow fast path when pyarrow is available.

### Changed

- **RDF store indexes upgraded to `foldhash`**: replaced `ahash` with `foldhash::fast::RandomState` for all RDF HashMap indexes.
- **`TxId` renamed to `TransactionId`**: consistent naming across the codebase.

### Fixed

- **Incremental backup could skip WAL records**: backup cursor was not advanced, causing duplicate replay on restore (#258).
- **File manager leaked temp files on checkpoint failure**: temp files now cleaned up in the error path (#258).

## [0.5.36] - 2026-04-11

Authentication at engine level with RBAC, per graph access grants and several query language improvements.

### Added

- **Role-based access control**: `Identity`, `Role` (`Admin`, `ReadWrite`, `ReadOnly`), and `StatementKind` types for scoping sessions to specific permission levels. `db.session_with_identity(identity)` creates a session bound to an identity, `db.session_with_role(role)` is a convenience shorthand. Permission checks run after parsing but before execution across all query languages (GQL, Cypher, Gremlin, GraphQL, SQL/PGQ, SPARQL). No credentials or crypto at this layer: the caller is trusted to assign the correct role.
- **Graph projections**: read-only filtered views of a graph store via `ProjectionSpec` and `GraphProjection`. Filter by node labels and edge types to create virtual subgraphs for algorithms and queries. Manage with `create_projection()`/`drop_projection()`/`list_projections()` in Rust, Python, Node.js, WASM, and C. GQL syntax: `CREATE PROJECTION name LABELS (...) EDGE_TYPES (...)`, `DROP PROJECTION name`, `SHOW PROJECTIONS`.
- **Gremlin `repeat().times()`/`.emit()`**: parse and execute `repeat(out()).times(n)` for fixed-depth traversal and `repeat(out()).emit()` for all-depths traversal. Maps to the existing `VariableLengthExpand` operator. `until()` predicates, `path()`, `simplePath()`, and `loops()` remain pending.
- **CSV/JSON Lines import**: CLI `grafeo import csv`/`grafeo import jsonl` commands, Python `import_csv()`/`import_jsonl()`, Node.js `importCsv()`/`importJsonl()`.
- **Per-graph access grants**: `Grant` type scopes an identity's access to specific named graphs. `Identity::with_grants([Grant::new("social", Role::ReadWrite)])` restricts access to listed graphs only. `USE GRAPH`, `CREATE GRAPH`, `DROP GRAPH` enforce grants when present. Empty grants = unrestricted (backward compatible).

### Changed

- **Unified aggregate accumulator**: push-based aggregate operator now uses the same `AggregateState` as the pull-based operator, gaining support for all 30+ aggregate functions (COLLECT, LAST, STDEV, percentiles, regression, etc.) that previously returned NULL in push mode.
- **`session_read_only()` deprecated**: use `session_with_role(Role::ReadOnly)` instead. The old method remains as an alias.

### Fixed

- **Release workflow missing `grafeo-storage`**: the crate publish sequence now includes `grafeo-storage` before `grafeo-engine`, fixing cascading publish failures.
- **Permission bypass in parameterized queries**: `_with_params` methods used a text heuristic to gate write permissions, which had false negatives for languages like GraphQL. Restricted identities now use plan-based mutation detection.
- **Projection `neighbors()` ignored edge-type filter**: neighbors connected via excluded edge types were incorrectly returned.
- **Projection `edge_type()` leaked hidden edges**: edges whose endpoints were excluded by label filtering could still have their type queried.
- **Spill serialization dropped DISTINCT semantics**: DISTINCT aggregate variants are now serialized via finalized-value fallback to avoid corrupting results after reload.
- **Gremlin `times()` accepted negative values**: negative loop counts silently wrapped to huge values, now returns a parse error.
- **Gremlin nested repeat modifiers**: `.times()`/`.until()`/`.emit()` now work inside `union()`, `coalesce()`, and other nested traversals.
- **Projections retained stale store after `compact()`**: `compact()` now clears all projections to prevent stale data and memory leaks.
- **`rand` RUSTSEC-2026-0097**: updated to 0.10.1.

## [0.5.35] - 2026-04-11

Breaking: `QueryResult.rows` is now private (use `rows()`/`into_rows()`), all public enums are `#[non_exhaustive]` (add `_ =>` arms), old feature profiles (`embedded`, `browser`, `server`, `full`) are deprecated in favor of persona-based profiles (`lpg`, `rdf`, `analytics`, `ai`, `edge`, `enterprise`) and the on-disk storage format changed from bincode blobs to block-based sections (databases created with 0.5.34 or earlier must be re-created).

### Added

- **Persona-based feature profiles**: new named profiles `lpg`, `rdf`, `analytics`, `ai`, `edge`, `enterprise` describe use cases instead of deployment targets. Compose them freely: `features = ["lpg", "ai"]` for a graph app with search, `features = ["rdf", "analytics"]` for a knowledge graph with algorithms. Old profiles (`embedded`, `browser`, `server`, `full`) remain as deprecated aliases with unchanged behavior, scheduled for removal in 0.7.0.
- **Python named graph management**: `create_graph()`, `drop_graph()`, `list_graphs()`, `set_graph()`/`reset_graph()`/`current_graph()`, `set_schema()`/`reset_schema()`/`current_schema()` ([#241](https://github.com/GrafeoDB/grafeo/issues/241), [#243](https://github.com/GrafeoDB/grafeo/pull/243) by [@Michaelzag](https://github.com/Michaelzag))
- **Python per-transaction CDC override**: `begin_transaction_with_cdc(True|False)` ([#242](https://github.com/GrafeoDB/grafeo/issues/242), [#244](https://github.com/GrafeoDB/grafeo/pull/244) by [@Michaelzag](https://github.com/Michaelzag))
- **Arrow IPC export** (`arrow-export`): zero-copy export to Arrow IPC for DuckDB, Polars, pandas, DataFusion interop
- **GEXF + GraphML export**: graph interchange for Gephi, Cytoscape, NetworkX, yEd, igraph. CLI `--export-format gexf|graphml`
- **Section-based container format**: `.grafeo` files use a section directory with checksummed, independently addressable sections. Checkpoint writes only dirty sections, recovery loads in parallel.
- **`grafeo-storage` crate**: persistence I/O extracted from `grafeo-adapters`. `grafeo-core` and `grafeo-storage` are now siblings (both depend only on `grafeo-common`).
- **Unified flush model**: checkpoint, `CHECKPOINT`, and memory-pressure eviction share one code path
- **Regression + memory benchmarks**: 11 Criterion benchmarks with per-benchmark thresholds, 5 memory benchmarks with CI bounds checking
- **Section serializers**: Vector Store (HNSW topology) and Text Index (BM25 postings) persist to container, eliminating index rebuild on open
- **Per-section memory config**: `SectionMemoryConfig` with `max_ram` caps and `TierOverride` per section type
- **Mmap for index sections**: zero-copy read via `memmap2`, CRC-verified, cross-platform lifecycle
- **BufferManager section consumers**: sections register as `MemoryConsumer`s for accurate pressure tracking
- **Periodic checkpoint timer**: background flush at `Config::checkpoint_interval`, bounds WAL size
- **Container format spec**: `docs/architecture/storage/container-format.md`
- **Vector spill to disk**: vector columns drain to `MmapStorage` under memory pressure, search reads transparently from mmap
- **BufferManager spill integration**: eviction calls `spill()` on consumers after in-memory eviction exhausted
- **PropertyColumn eviction**: `drain_values()`, `evict_values()`, `restore_values()` with `spilled` flag
- **Block-based LPG section format (v2)**: replaces bincode blob with a structured layout: string table, packed node/edge arrays, columnar property blocks, label assignments, per-block CRC. Enables mmap for data sections.
- **Block-based RDF section format (v2)**: replaces bincode with string-table-deduplicated triple storage, per-block CRC, named graph sub-sections.
- **WAL overlay**: in-memory mutation layer (`WalOverlay`) for tracking inserts, updates, deletes on top of mmap'd base data. Supports drain/clear for checkpoint merge.
- **TieredStore trait**: `StorageTier` enum (InMemory, OnDisk, Uninitialized) and `TieredStore` trait defining `persist()`, `open_mmap()`, `reload_to_ram()` lifecycle in `grafeo-common`.
- **CDC retention and eviction**: `CdcRetentionConfig` with `max_epochs` and `max_events` limits. `CdcLog` implements `MemoryConsumer` for BufferManager-driven eviction. Pruning hooks into MVCC GC cycle. ([#250](https://github.com/GrafeoDB/grafeo/issues/250))
- **EpochAdvance WAL record**: new `WalRecord::EpochAdvance { epoch }` logged after each `TransactionCommit`. `is_metadata()` trait method on `WalEntry`. Enables epoch-bounded WAL replay for point-in-time recovery.
- **Incremental backup**: `backup_full()`, `backup_incremental()`, `restore_to_epoch()` on `GrafeoDB`. Backup manifest tracks the chain, backup cursor in WAL directory prevents premature log truncation. CLI commands: `grafeo backup full`, `grafeo backup incremental`, `grafeo backup status`, `grafeo backup restore-to-epoch`. Exposed in Python, Node.js, and C bindings.

### Changed

- **`VersionChain` uses `Vec` instead of `VecDeque`**: eliminates 4-slot minimum allocation per entity, reducing per-entity memory 14-31% ([#251](https://github.com/GrafeoDB/grafeo/issues/251))
- **Schema DDL types decoupled from GQL parser**: shared types (`SchemaStatement`, `PropertyDefinition`, etc.) moved to `query::schema` module, allowing Cypher to compile without the `gql` feature ([#234](https://github.com/GrafeoDB/grafeo/issues/234))
- **`QueryResult.rows` is now private**: use `rows()` for borrowed access, `into_rows()` for ownership, `push_row()`/`from_rows()` for construction
- **`#[non_exhaustive]` on 95 public enums**: downstream `match` must add `_ =>` wildcard arms
- **Python abi3 wheels**: single wheel per platform supports Python 3.12+
- **`rdf` feature renamed to `triple-store`**: deprecated `rdf` alias kept for one release, `rdf` now names the persona profile
- **`lpg` feature flag**: LPG model is now explicit in grafeo-core/engine/adapters, symmetric to `triple-store`
- **Crate restructure**: storage backends moved to `grafeo-storage`, `grafeo-adapters` is parser-only, `grafeo-core/src/storage/` renamed to `codec/`
- **Removed tokio from grafeo-core**: async spill moved to `grafeo-engine/src/execution/spill/`
- **CI**: benchmark job adds per-benchmark thresholds and baseline persistence on main

### Fixed

- **WAL not replayed on reopen**: data written via mutations was lost across restarts when no explicit checkpoint was called. Session commits now log `TransactionCommit` + `EpochAdvance` to WAL, and the close path only removes the sidecar WAL after verifying the checkpoint wrote data. ([#252](https://github.com/GrafeoDB/grafeo/issues/252))
- **CDC event log unbounded memory**: the CDC log stored every mutation with full property snapshots and never pruned. Added epoch-based and count-based retention (`CdcRetentionConfig`), integrated with BufferManager for memory-pressure eviction. ([#250](https://github.com/GrafeoDB/grafeo/issues/250))
- **`VecDeque::first_mut` compile error**: fixed to `front_mut()` in MVCC `VersionChain` (tiered-storage feature path)
- **Graph/schema context validation**: reject nonexistent targets, `drop_graph()` auto-clears active context ([#245](https://github.com/GrafeoDB/grafeo/issues/245), [#246](https://github.com/GrafeoDB/grafeo/pull/246) by [@Michaelzag](https://github.com/Michaelzag))
- **C binding `grafeo_reset_schema`**: propagates errors instead of silently discarding
- **WASM `setSchema`**: returns proper JS `Error` instead of plain string
- **CI benchmark `--save-baseline`**: scoped to Criterion crates only
- **Aggregate grouping hash collision**: wildcard arm now hashes `std::mem::discriminant` to distinguish future `Value` variants
- **`edges_df()`/`nodes_df()` column overwrite**: properties named after structural columns (`source`, `target`, `type`, `id`, `labels`) no longer silently replace them ([#254](https://github.com/GrafeoDB/grafeo/issues/254))

## [0.5.34] - 2026-04-07

Pre-RC hardening: query engine fixes from external integration testing, format stability, feature matrix CI.

### Added

- **GQL schema hierarchy** (ISO/IEC 39075 Section 4.2.5): `CREATE SCHEMA`/`DROP SCHEMA`, `SESSION SET SCHEMA`, full data isolation between schemas
- **Streaming RDF triple sink**: `TripleSink` trait with `BatchInsertSink` (bounded memory) and `CountSink` (dry-run)
- **Streaming Turtle/N-Triples load**: `load_turtle_streaming()`, `load_ntriples_streaming()` insert incrementally
- **Golden fixture tests**: snapshot v4, `.grafeo` file format, and WAL frame backward-read + byte-equality checks
- **Deterministic snapshot export**: nodes/edges/labels/properties sorted by ID/name for reproducible exports
- **Feature matrix CI**: per-profile build+test jobs (gql-only, gql+vector, gql+rdf, embedded, browser)
- **Serialization benchmarks**: snapshot export/import and `Value` bincode round-trip

### Changed

- **`LabelRegistry` combined lock**: merged `label_to_id` + `id_to_label` into one `RwLock<LabelRegistry>`, reducing write-path lock acquisitions
- **`#[non_exhaustive]` on 13 public enums**: future variants can be added without breaking semver
- **`missing_errors_doc`/`missing_panics_doc` lints enabled**: public functions now document error/panic conditions
- **MVCC types hidden**: `VersionChain`/`VersionInfo` re-exports marked `#[doc(hidden)]`

### Fixed

- **WAL sync counter race**: `fetch_sub` after sync instead of `store(0)`, preserving concurrent increments
- **Multi-aggregate extraction**: `sum(a) + count(b)` now extracts all aggregates
- **Mixed `WITH ... WHERE`/HAVING**: non-aggregate conjuncts stay as WHERE, aggregate parts become HAVING
- **`references_any` completeness**: all `LogicalExpression` variants handled
- **`CREATE SCHEMA` duplicate WAL record**: only logged when graph is actually created
- **`BatchInsertSink` zero batch_size**: defensive `max(1)` clamp
- **`delete_node_edges` self-loop**: deduped via `HashSet`, batch edge lock, batch adjacency lock
- **`cypher` feature needing `gql` dependency** ([#232](https://github.com/GrafeoDB/grafeo/issues/232), [#233](https://github.com/GrafeoDB/grafeo/pull/233) by [@Michaelzag](https://github.com/Michaelzag))
- **Node.js napi cfg-gated methods**: moved methods into per-feature `#[napi] impl` blocks
- **`BitPackedInts::from_bytes`**: `bits_per_value > 64` now returns `Err` instead of panicking
- **WAL `log_files`**: directory read errors now propagated instead of swallowed
- **Per-feature compilation**: missing cfg gates across `vector-index`, `rdf`, `mmap`, `regex`
- **Algorithm unreachable panics**: `expect("node in index")` replaced with `enumerate`/`let-else` in functions
- **Null property pattern matching**: `MATCH (n {key: null})` now matches nodes where the property is absent or explicitly null (MERGE and MATCH)
- **MERGE null key matching**: `MERGE (n:T {a: null, b: 'x'})` correctly finds existing nodes with absent/null `a`
- **SET += {key: null} removes property**: `SET n += {price: null}` now removes the property instead of keeping it as null
- **Negative LIMIT clamped to 0**: `LIMIT -1` returns empty result instead of raising a syntax error (GQL and Cypher)
- **i64 MIN literal parsing**: `-9223372036854775808` now parses correctly by folding `-<integer>` at parse time (GQL and Cypher)
- **NaN/Inf float literals**: `NaN`, `Inf`, `Infinity` recognized as IEEE 754 special float values (GQL and Cypher)
- **`nodes(p)` resolves to property maps**: `nodes(path)` now returns node maps with properties instead of raw Int64 IDs, enabling `[n IN nodes(p) | n.name]`
- **Cyclic VLP pattern matching**: `MATCH p=(s)-[:R*]->(s)` now filters expanded targets to match the source node via `id()` equality
- **VLP default depth raised**: unbounded `[*]` now expands up to `min_hops + 100` (was `+10`)

## [0.5.33] - 2026-04-05

GraphChallenge benchmark suite, RDF-to-LPG bridge, and a large round of query engine correctness fixes.

### Added

- **GraphChallenge algorithms** (DARPA/MIT IEEE HPEC 2026): k-truss decomposition, parallel triangle counting, subgraph isomorphism (VF2), stochastic block partition, partition quality metrics (Rand index, NMI, precision, recall)
- **TSV/MMIO bulk import**: `import_tsv()`, `import_mmio()`, `import_tsv_rdf()` for fast graph loading bypassing per-edge transaction overhead
- **`RdfGraphStoreAdapter`**: bridges `RdfStore` to `GraphStore`, giving RDF graphs access to all graph algorithms
- **grafeo-cli PyPI publish workflow** ([#222](https://github.com/GrafeoDB/grafeo/pull/222))

### Fixed

- **CompactStore multi-table edge types**: same edge type across multiple label pairs now produces separate `RelTable`s. Added `rel_tables_for_type()` ([#221](https://github.com/GrafeoDB/grafeo/issues/221), [#225](https://github.com/GrafeoDB/grafeo/pull/225) by [@Imaclean74](https://github.com/Imaclean74))
- **WAL deadlock on property mutations**: store mutation now applied before WAL logging, matching lock ordering of create/delete methods
- **GQL `CREATE INDEX ... FOR` parsing**: `FOR` accepted whether lexed as keyword or identifier
- **`round()`/`floor()`/`ceil()`**: float inputs return `Float64` instead of truncating to `Int64`
- **`CALL ... YIELD` with aggregation**: aggregates now work over procedure results
- **Cypher keyword-as-label**: `Order`, `By`, `Skip`, `Limit` usable as node labels
- **CompactStore edge type statistics**: counts aggregated across multiple rel tables
- **`CAST(bool AS INT)`**: `true` casts to `1`, `false` to `0`
- **List `+` concatenation**: `[1, 2] + [3, 4]` returns `[1, 2, 3, 4]`
- **Parameter substitution in multi-statement queries**: `$param` variables now forwarded to intermediate statements
- **ORDER BY + LIMIT/SKIP**: SKIP and LIMIT now apply after ORDER BY
- **MIN/MAX aggregate output type**: uses `Any` instead of `Int64`, fixing coercion for Float64 and Date values
- **Cypher ORDER BY after aggregation**: property references resolve correctly after GROUP BY
- **JOIN column deduplication**: multi-pattern MATCH with shared variables no longer produces duplicate columns
- **SET self-reference**: `SET n.value = n.value + 1` pre-computes expressions before the property write
- **`size(collect())` nested aggregate**: no longer panics during aggregate extraction
- **`WITH ... WHERE` on aggregate alias**: WHERE predicate correctly promoted to HAVING
- **`SUM()` on empty result set**: returns `null` per ISO GQL

### Performance

- **Triangle counting**: oriented adjacency built directly from `GraphStore`, improving cache efficiency on CSR-backed stores
- **WAL `sync_all()` outside lock**: reduces lock contention under concurrent writes
- **Kahan compensated summation**: `sum()` uses Kahan algorithm to reduce floating-point rounding errors

## [0.5.32] - 2026-04-03

Correctness hardening, Jepsen readiness, and Hybrid Logical Clock for causal consistency.

### Added

- **`GrafeoDB::compact()`**: converts a live database to a read-only `CompactStore` in one call. Available as `db.compact()` in Python, Node.js, WASM; `grafeo_compact(db)` in C. Included in `embedded` and `browser` profiles by default ([#199](https://github.com/GrafeoDB/grafeo/issues/199))
- **Hybrid Logical Clock (HLC)**: `HlcTimestamp` packs physical ms (48-bit) + logical counter (16-bit) into a u64 with lock-free CAS for monotonic timestamps. Replaces wall-clock `SystemTime::now()` in CDC events
- **CDC for session mutations**: `CdcGraphStore` decorator buffers CDC events during transactions, flushes on commit (discards on rollback). Session-driven mutations via GQL/Cypher now generate CDC events
- **Session CRUD methods**: `set_node_property()`, `set_edge_property()`, `delete_node()`, `delete_edge()`, `create_edge_with_props()` on Session for transaction-aware direct mutations
- **Gremlin `valueMap()` and `elementMap()` with no arguments**: returns all properties (or id + label + all properties) as a map
- **Stress and crash tests**: WAL-disabled crash injection, concurrent MERGE, mixed read/write contention, concurrent schema mutations, and 5 epoch monotonicity stress tests for CDC
- **Expanded gtest suite**: 4 real-world datasets (e-commerce, movies, IT infrastructure, transportation), gap tests for all languages (GQL, Cypher, Gremlin, SQL/PGQ, SPARQL), Rosetta cross-language fidelity, production coverage (data type round-trips, mutation patterns, input validation), parameter substitution, catalog diagnostics, index correctness, temporal queries, and algorithm tests (Dijkstra, PageRank, centrality, BFS, SCC)

### Fixed

- **Sibling CALL block scope collision**: same-named variables in sibling `CALL` blocks no longer clobber each other ([#213](https://github.com/GrafeoDB/grafeo/issues/213))
- **GROUP BY hash collisions**: `hash_value()` now uses discriminant tags for all `Value` variants, preventing cross-type collisions; added `Date`, `Time`, `Timestamp`, `Duration`, `ZonedDatetime`, `Bytes`, `Map` variants to `GroupKeyPart`
- **Cypher ORDER BY zeros with relationship traversal**: planner now resolves to the existing projected column instead of returning zeros ([#218](https://github.com/GrafeoDB/grafeo/issues/218))

### Changed

- **CDC is now opt-in per session**: no longer unconditionally active when compiled in. `Config::with_cdc()` and `GrafeoDB::set_cdc_enabled()` control the default (off). Fixes +251% regression on single-node inserts. Python: `GrafeoDB(cdc=True)`. Node.js: `db.enableCdc()`. C: `grafeo_set_cdc_enabled(db, true)`
- **CompactStore native codec scans**: `find_eq()` and `find_in_range()` push checks into the codec's native domain instead of decoding to `Value` per row. Thanks to [@temporaryfix](https://github.com/temporaryfix) ([#216](https://github.com/GrafeoDB/grafeo/pull/216))

### Internal

- **SPARQL ORDER BY STR() tests tightened**: removed error-accepting fallback; `NullGraphStore` is correct for expression evaluation
- **Vector search `$ne`/`$nin` NULL semantics**: documented and regression-tested (SQL three-valued NULL semantics)

## [0.5.31] - 2026-04-01

CompactStore: a read-optimized columnar graph store for memory-constrained environments. Thanks to [@temporaryfix](https://github.com/temporaryfix) for the design, prototype and implementation ([#199](https://github.com/GrafeoDB/grafeo/issues/199), [#204](https://github.com/GrafeoDB/grafeo/pull/204)). Also, all remaining syntax gaps covered by the gtest suite are now fully implemented!

### Added

- **`compact-store` feature flag**: opt-in columnar read-only store for WASM, edge workers and embedded devices. Per-label `NodeTable`s with typed columns, double-indexed `CsrAdjacency` for O(degree) traversal, zone-map skip optimization, and a fluent `CompactStoreBuilder` API with build-time validation. Integrates via `GrafeoDB::with_read_store(Arc<dyn GraphStore>)`, all query languages work through it
- **Benchmark**: `compact_benches` criterion group with `nodes_by_label`, `get_node_property`, and `edges_from` benchmarks for CompactStore
- **`execute_language(language, query, params)` in Python and Node.js bindings**: generic dispatch for non-standard language keys (e.g. `"graphql-rdf"`) without needing dedicated methods
- **SQL/PGQ UNION, INTERSECT, EXCEPT**: full set operation support between GRAPH_TABLE queries, with optional ALL modifier
- **GraphQL multiple root fields and variable substitution**: `{ person { name } company { name } }` now translates all root fields via Union instead of dropping all but the first; `$variable` references emit `LogicalExpression::Parameter` with default value propagation from query declarations
- **Binding spec runner `params:` support**: Python, Node.js, Go, and C# test runners now pass gtest `params:` fields to parameterized execution methods
- **`DatabaseInfo.features`**: `db.info()` now returns a `features` array listing all compiled feature flags (e.g. `["gql", "cypher", "algos", "vector-index"]`), available in all bindings (Python, Node.js, WASM, C, Go, C#, Dart)
- **WASM `lpg` and `rdf` build profiles**: two new named profiles join `browser` and `full`. `lpg` bundles all LPG query languages plus AI search; `rdf` bundles SPARQL and GraphQL over the RDF model

### Fixed

- **GQL list slice and path search**: `[1..3]`, `[..2]`, `[3..]` slices now work (one-char lexer bug); `MATCH ANY p = ...` and `MATCH p = ANY SHORTEST ...` path search prefixes now use the existing shortest-path BFS operator
- **SPARQL pattern matching**: MINUS with disjoint variables returns left side unchanged per spec; `<p>*`/`<p>?` property paths include zero-length reflexive match; VALUES with UNDEF produces correct partial bindings; anonymous blank node `[]` as subject expanded correctly
- **SPARQL function and type evaluation**: STRLEN, CONCAT, IF, COALESCE, arithmetic work in SELECT/BIND projections; `STRDT()` produces typed values; `DATATYPE()` companion columns track original XSD types through scans; subquery aggregation propagates to outer queries
- **SPARQL graph management**: `GRAPH ?g` scans only named graphs per spec 13.3; `FROM`/`FROM NAMED` restrict visible graphs per spec 13.1-13.2; `CLEAR ALL` clears both default and named graphs; `DESCRIBE` returns Concise Bounded Description
- **SPARQL updates and literals**: `DELETE { ... } WHERE { ... FILTER(...) }` applies the filter correctly; language-tagged literal comparison checks both value and tag
- **Gremlin traversal fixes**: multi-hop dead end no longer causes "Column not found"; `values()` with no keys returns all properties; scalar values in union branches no longer coerced to `NodeId(0)`; `path()` on empty traversal returns empty result set
- **Cypher `CREATE INDEX` / `DROP INDEX` / `SHOW INDEXES`**: indexes now registered in the catalog, persisting across statements
- **GraphQL aggregation**: `personCount`, `personAggregate { sum_age }`, and `_count` field patterns now emit proper aggregate operators
- **RDF GraphQL**: per-test `language: graphql-rdf` dispatch for mutation rejection testing; `first`/`limit`/`skip`/`offset` pagination in the RDF translator

### Performance

- **RDF schema type propagation**: `plan_operator` threads concrete `LogicalType`s through the entire plan tree instead of `LogicalType::Any`, keeping triple scan data in `Vec<ArcStr>` (8 bytes/entry) through joins, sorts, and projections instead of `Vec<Value>` (40 bytes/entry)

### Internal

- **Spec runner feature detection**: all 6 binding spec runners (Python, Node.js, WASM, C#, Dart, Go) now use `db.info().features` to detect available capabilities instead of probing for individual methods, eliminating false skips for non-language features like `algos` and `vector-index`
- **`ValueVector` push safety net**: type-mismatched pushes now fall back to `VectorData::Generic` instead of silently dropping data
- **`derive_rdf_schema` removed**: replaced by concrete type propagation through `plan_operator` return values
- **`eval_function` split**: 1,687-line monolith refactored into a thin dispatcher and 9 focused category methods
- **Dedup macros and utilities**: `impl_algorithm!` for `GraphAlgorithm` boilerplate (17 of 23 implementations), `map_common_keywords!` for shared lexer keyword mapping, `unescape_string` extracted to shared module, `extract_and_map` generic for binding entity extraction

## [0.5.30] - 2026-03-30

Async storage foundation and continued test coverage. Thanks to [@maxwellflitton](https://github.com/maxwellflitton) for the [async storage adapter discussion](https://github.com/orgs/GrafeoDB/discussions/190) that shaped this release.

### Added

- **`async-storage` feature flag**: new opt-in feature for async WAL and storage operations, included in `server` profile
- **`AsyncTypedWal<R>`**: type-safe async WAL wrapper mirroring sync `TypedWal<R>`, with identical on-disk format for cross-recovery compatibility
- **`AsyncLpgWal`**: type alias for `AsyncTypedWal<WalRecord>`, the async equivalent of `LpgWal`
- **`AsyncWalManager::write_frame`**: extracted low-level frame writer enabling generic `WalEntry` types in async context
- **`AsyncWalGraphStore`**: async decorator that logs mutations to `AsyncLpgWal` before applying to `LpgStore`, with named graph context tracking via tokio mutex
- **`GrafeoDB::async_wal_checkpoint()`**: async WAL checkpoint via `spawn_blocking`, avoids blocking the tokio runtime during fsync
- **`GrafeoDB::async_write_snapshot()`**: async snapshot write via `spawn_blocking` for `.grafeo` single-file format
- **`AsyncStorageBackend` trait**: object-safe async trait for pluggable persistence backends (WAL batches, snapshots, sync), enabling community implementations for Postgres, S3, etc.
- **`AsyncLocalBackend`**: built-in local filesystem implementation wrapping `AsyncLpgWal`
- **`SnapshotMetadata`**: metadata type for snapshot listing in async backends
- **Node.js `walCheckpoint()` and `save()`**: new sync methods for checkpoint and persistence in Node.js bindings

### Fixed

- **86 stale spec test skips removed**: path modes (TRAIL, SIMPLE, ACYCLIC, WALK), ALL SHORTEST search prefix, list slice syntax, SPARQL string/datetime/hash functions, RDF term construction, conditional functions, named graphs, property paths, GraphQL directive evaluation, and more
- **SPARQL dateTime extraction functions**: YEAR, MONTH, DAY, HOURS, MINUTES, SECONDS, TIMEZONE, TZ now correctly parse typed `xsd:dateTime` literals with timezone offsets
- **SPARQL LANGMATCHES()**: implemented RFC 4647 basic filtering with case-insensitive prefix matching and wildcard `"*"` support
- **SPARQL LANG() companion columns**: language tags are now tracked through triple scans and available to LANG()/LANGMATCHES() in FILTER
- **SQL/PGQ parameters in WHERE**: `$name`, `$min_age` parameter references now resolved in filter evaluation via gtest runner wiring
- **SQL/PGQ HAVING inline aggregates**: `HAVING COUNT(*) > 0` and other inline aggregates in HAVING clauses now correctly extracted and referenced
- **SQL/PGQ zero-length paths**: `*0..N` variable-length patterns now emit the source node as a 0-hop match
- **Cypher `collect(DISTINCT ...)`**: `size(collect(DISTINCT n.v))` now correctly extracts the wrapped aggregate through non-aggregate function calls

## [0.5.29] - 2026-03-29

Query engine correctness improvements and unified declarative test suite.

### Added

- **Turtle parser and serializer**: zero-dependency W3C Turtle support (`load_turtle()`, `to_turtle()` on `RdfStore`), with prefix detection, subject grouping, numeric/boolean shorthands, `a` shorthand, and line/column error positions
- **N-Quads serializer**: `to_nquads()` on `RdfStore` for exporting default and named graphs in a single stream
- **Declarative `.gtest` spec test framework**: new `grafeo-spec-tests` crate with a YAML-based test format, build.rs code generator, and runtime comparison library. 2500+ tests across all 7 language/model combinations (GQL, Cypher, Gremlin, GraphQL (LPG+RDF), SQL/PGQ, SPARQL and Rosetta cross-language) from a single source of truth, with runners for binding-level verification
- **EXISTS subquery in RETURN**: `RETURN EXISTS { MATCH (n)-[:R]->(:Label) } AS flag` now works for single-hop correlated patterns, including label-filtered endpoints
- **Aggregate detection in GQL WITH**: `WITH count(n) AS cnt, max(n.val) AS mx` now correctly produces an aggregate operator instead of treating aggregates as scalar expressions

### Changed

- **Adjacency list memory**: replaced `SmallVec<8>` with `Vec` (struct 256 to ~144 bytes), added auto-compaction in `add_edge()` to fix unbounded delta buffer growth

### Fixed

- **Integer arithmetic overflow**: `9223372036854775807 + 1` no longer panics; checked arithmetic returns NULL on overflow (SQL semantics) for all operations (+, -, *, /, %, unary negation)
- **Label intersection across MATCH clauses**: `MATCH (n:A) MATCH (n:B)` now correctly filters to nodes with both labels instead of ignoring the second label constraint
- **CASE WHEN with NULL aggregate**: `WITH count(c) AS cc RETURN CASE WHEN cc = 0 THEN 0 ELSE ... END` no longer returns NULL when the WHEN branch is true
- **EXISTS with property filters**: `EXISTS { (n)-[:R]->(m) WHERE m.age > 30 }` silently dropped the WHERE, matching all connected nodes
- **Keywords as property names**: `{order: 3}` and `n.order` rejected `order` and other keywords in property contexts
- **Gremlin `hasLabel` on edges**: `g.E().hasLabel('KNOWS')` returned 0 rows because the translator used node labels instead of edge type
- **Gremlin parser**: added `regex()` predicate, `$param` parameters, mid-traversal `V()` step, bare `label`/`id` keywords in `by()` modifiers
- **Gremlin `coalesce()` semantics**: now uses `OtherwiseOp` for first-non-empty branch selection instead of `Union` which returned all branches
- **Gremlin `group().by()` two-pass**: `group().by(key).by(value)` now correctly sets grouping key and value projection, with `MapCollect` wrapping for single-map output
- **Gremlin `optional()` step**: rewrote translation to produce correct per-row semantics (navigation vs filter cases) instead of returning identity vertex
- **Gremlin `values()` null filtering**: `values('nonexistent')` now returns zero rows instead of a row with null, matching Gremlin semantics
- **Gremlin `addE` with `as()` labels**: `from('a')` / `to('a')` now resolves step labels from the `as()` alias map instead of treating them as literal strings
- **Gremlin `or()` three-valued logic**: `or(hasLabel('X'), has('prop', val))` across different node types now correctly returns matches from both branches (NULL OR true = true)
- **SPARQL functions in SELECT projections**: created `RdfProjectOperator` that delegates to `RdfExpressionPredicate` for full function support (STRLEN, UCASE, LCASE, IF, COALESCE, REPLACE, etc.)
- **SPARQL IN/NOT IN operators**: added `FilterExpression::List` evaluation and `BinaryFilterOp::In` handling in `RdfExpressionPredicate`
- **SPARQL BOUND() with OPTIONAL**: checks vector validity bitmap directly to distinguish unbound variables from null values after LEFT JOIN
- **SQL/PGQ unbounded variable-length paths**: `*1..` no longer silently caps max_hops to 1
- **SQL/PGQ COUNT(column) NULL skipping**: `COUNT(expr)` now uses `CountNonNull` to skip NULL values per SQL standard
- **SQL/PGQ CASE expressions**: CASE WHEN in outer SELECT and WHERE clauses now evaluated by the translator
- **SQL/PGQ outer SELECT projection**: non-aggregate `SELECT col FROM GRAPH_TABLE(... COLUMNS(...))` now projects the correct columns
- **SQL/PGQ ORDER BY on aggregate aliases**: ORDER BY for aggregate queries now placed after the Aggregate operator so output aliases resolve correctly
- **JSON Infinity/NaN lost through C FFI**: `SUM()` overflow returned `null` in bindings because JSON cannot represent infinity; now encoded as string `"Infinity"`
- **C#/Dart temporal values**: dates, times, and durations returned as locale-dependent native types instead of ISO strings
- **Binding spec runners**: replaced YAML library parsers (Go yaml.v3, C# YamlDotNet, Dart package:yaml) with line-based parsers matching Rust/Node.js/Python; fixed SPARQL dispatch, hash assertions, error test logic, WASM feature gating

## [0.5.28] - 2026-03-27

Hotfix: single-file `.grafeo` storage was silently disabled in all bindings.

### Fixed

- **Single-file storage broken in bindings** (#185): `grafeo-file` feature was missing from the `embedded` profile, causing `grafeo_open_single_file` and `.grafeo` auto-detection to silently fall back to WAL directory format. Added `grafeo-file` to engine defaults, `embedded` profile, and all binding crates (C, Python, Node.js, facade)

## [0.5.27] - 2026-03-27

C FFI overhaul, Dart expansion, binding-wide usability audit, grafeo-memory engine support.

### Added

- **C API overhaul** (#185): `grafeo_open_single_file`, `_with_params` for all 5 languages, unified `grafeo_execute_language`, type-safe `GrafeoIsolationLevel` enum
- **Dart bindings expansion**: `openSingleFile`, `openReadOnly`, `executeLanguage`, `execute*WithParams`, schema context, property/vector indexes, `batchCreateNodes`
- **Dart Flutter guide**: native library bundling for Windows, macOS, Linux desktop
- **Go bindings**: `OpenSingleFile`, `ExecuteLanguage`, `Execute*WithParams`, `ExecuteParams(map[string]any)`
- **Rust facade re-exports**: `Error`, `Result`, `QueryResult` now at crate root
- **`batch_create_nodes_with_props`**: engine + Python method accepting list of property dicts with mixed types including vectors
- **Temporal property versioning API** (`temporal` feature): `get_node_property_at_epoch`, `get_node_property_history`, `get_all_node_property_history`
- **Node.js user guide**: 5 pages covering database, queries, CRUD, transactions, results
- **C# P/Invoke completeness**: 11 missing native declarations added, `Transaction.ExecuteLanguage()` with async variant
- **Crash safety testing**: new crash injection point, 6 new recovery/concurrency tests
- **Python API docs**: 45+ undocumented methods added to API reference (DataFrame, batch, search, algorithms, temporal, admin)

### Fixed

- **`labels(n)`/`type(r)` in aggregation** (#187): complex expressions in GROUP BY and ORDER BY failed with "Cannot resolve expression to column". Fixed in all 4 planner locations (LPG aggregate, LPG sort, RDF aggregate, RDF sort)
- **C# isolation level always failed**: P/Invoke passed `string` where `int` expected. Added `IsolationLevel` enum
- **C# `DropVectorIndex` threw on success**: now returns `bool`
- **C# P/Invoke mismatches (3)**: wrong signatures for property indexes, create_vector_index, batch_create_nodes
- **C# double-rollback after commit**: `TransactionHandle` now skips rollback when committed
- **Go stale `grafeo.h`**: 40+ missing declarations prevented compilation
- **Go column order random**: replaced map iteration with ordered JSON key parsing
- **Go thread-local error race**: added `runtime.LockOSThread()` around all C calls (including `GetNodeLabels`, `HasPropertyIndex`)
- **Node.js stale TypeScript definitions**: 6 missing methods, improved `rows()` type
- **Dart iOS loader**: missing `Platform.isIOS` branch
- **Dart Duration decoding**: returned raw ISO string instead of `Duration` object
- **Rust docs (8 errors)**: wrong method names, nonexistent APIs, incorrect fallibility
- **SPARQL docs contradicted themselves**: two pages said "not supported" while it works
- **README missing `pip install grafeo`**: added as primary install command
- **WASM docs**: `createVectorIndex` wrongly listed as unavailable
- **Vector search filter optimization**: operator filters ($gt, $lt, etc.) now scan only the narrowed allowlist instead of all nodes
- **Single-file storage silent failure** (#185): no file created when WAL disabled
- **C API `grafeo_current_schema` memory leak**: returned caller-owned pointer but docs said not to free; now uses thread-local storage
- **C API `out_count` uninitialized on error**: `vector_search`, `mmr_search`, `batch_create_nodes`, and `find_nodes_by_property` now zero all output pointers (`out_count`, `out_ids`, `out_distances`) before the main operation
- **Windows read-only file ops failure**: skipped `sync_all()` on read-only handles in both `close()` and `sync()`
- **Adjacency inline capacity**: raised `SmallVec` from 4 to 8, balancing L1 cache residency with fewer heap allocations for typical node degrees
- **ORDER BY complex expressions leaked columns**: `RETURN n.name ORDER BY labels(n)[0]` included a synthetic `__expr_` column in results. Complex ORDER BY expressions are now computed inside the augmented Return and stripped after sorting
- **GROUP BY on list-valued keys**: `GROUP BY labels(n)` on multi-label nodes produced extra rows because `GroupKeyPart` lacked a `List` variant. Added recursive `List(Vec<GroupKeyPart>)` with proper Hash/Eq, and fixed push-based aggregator `hash_value()` which mapped all lists to `0u8`
- **SPARQL GROUP BY/ORDER BY with expressions**: `GROUP BY (STR(?s))` and `ORDER BY ASC(STR(?s))` failed with "Store required for expression evaluation". RDF planner now passes a `NullGraphStore` to `ProjectOperator` for expression evaluation

## [0.5.26] - 2026-03-25

GQL conformance validation, SQL/PGQ features, and a big batch of bug fixes.

### Added

- **GQL conformance** (ISO/IEC 39075:2024): 234-query corpus cross-validated against GraphGlot. All 24 identified gaps closed: post-edge quantifiers (`->{1,3}`, `->+`, `->*`), path alternation (`|`, `|+|`), FILTER WHERE, SELECT...FROM...MATCH, brace-delimited graph types, and per-pattern path search prefixes
- **SQL/PGQ**: WHERE inside GRAPH_TABLE, SELECT DISTINCT, GROUP BY / HAVING, and graph name references
- **Cross-language correctness tests**: SQL/PGQ queries validated against GQL equivalents, plus CALL block scope isolation tests

### Fixed

- **EXISTS/COUNT subquery bugs**: target-side correlation (#173) now flips traversal direction instead of looking up the anonymous source, end-node labels are verified at runtime (were silently ignored), and complex EXISTS inside OR predicates works via split semi-join + filter
- **WAL directory-format data loss** (#174): `close()` wrote checkpoint metadata that caused recovery to skip older WAL files, silently losing pre-rotation data
- **UNWIND variable in SET clause** (#172): five mutation planner functions assigned `LogicalType::Node` to pass-through columns, silently dropping Map values from UNWIND. All now use `LogicalType::Any`. Present since 0.5.14
- **SET n:Label drops variable binding** (#178, #182): label operators discarded input columns, breaking any subsequent clause referencing the same variable. Now preserves columns per-row
- **Missing expression functions** (#179, #180): `timestamp()` returns epoch milliseconds (was null), `startNode(r)`/`endNode(r)` return node IDs (were unimplemented), zero-argument temporal functions now work in SET clauses
- **CREATE after MATCH creates phantom nodes** (#181): planner now skips node creation when the variable is already bound from a prior MATCH
- **SQL/PGQ GROUP BY** silently dropped non-aggregate columns; **C API typed entity access** (#177) now returns explicit `element_type`/`id`/`labels`/`type` fields in JSON

## [0.5.25] - 2026-03-25

RDF change tracking, CRDT counters, and tracing goes opt-in.

### Added

- **RDF CDC bridge** (`cdc` + `rdf`): SPARQL INSERT/DELETE mutations now emit `ChangeEvent` records to the CDC log, carrying N-Triples-encoded terms. Surfaces RDF changes through `GET /changes` and `POST /sync` for offline-first clients
- **CDC structural metadata**: node Create events now carry `labels`, edge Create events carry `edge_type`/`src_id`/`dst_id`, giving sync clients everything needed to replay creates remotely
- **CRDT counter values**: `Value::GCounter` and `Value::OnCounter` as first-class types with proper merge semantics (per-replica max). All bindings surface them as structured JSON objects

### Changed

- **Tracing is now opt-in** (`tracing` feature): compiles to zero-cost no-ops when disabled. Included in `server` profile, excluded from `embedded`/`browser`. Eliminates ~29% overhead on micro-benchmarks

### Fixed

- **Cypher target node property filter ignored**: `MATCH ()-[r]->(o {name: 'X'})` returned unfiltered results. Translator now applies target and edge property predicates after expand (Discussion #155)
- **Schema isolation for types**: SHOW/CREATE/DROP/ALTER type commands now respect `SESSION SET SCHEMA`. `DROP SCHEMA` rejects non-empty schemas (#167)
- **CREATE GRAPH TYPED regression**: type name resolution now works correctly with session schemas, including cross-schema references like `my_schema.type_name`
- **Schema context in bindings**: all bindings now expose `set_schema`/`reset_schema`/`current_schema` methods that persist across `execute()` calls
- **Temporal feature overhead**: optimized `VersionLog::at()` with O(1) fast path for current-epoch reads, eliminated double HashMap lookups. Reduces overhead from ~16% to ~6%

## [0.5.24] - 2026-03-24

Temporal properties, read-only mode, and snapshot format v4.

### Added

- **Index metadata in snapshots**: property, vector, and text index definitions now persist in v4 snapshots and auto-rebuild on import/restore
- **Read-only open mode**: `GrafeoDB::open_read_only()` uses shared file locks for concurrent reads; mutations rejected at the session level
- **Agent memory migration tests**: Rust and Python integration tests for HNSW at scale, BYOV 384-dim vectors, persistence, concurrent reads, bulk import, and storage size (Discussion #155)
- **Temporal properties** (`temporal` feature): opt-in append-only property versioning with `execute_at_epoch()`, `get_node_at_epoch()`/`get_node_history()` APIs, snapshot roundtrip, and transaction-safe rollback (Discussion #163)

### Breaking

- **Snapshot format v4**: properties stored as version-history lists; not backward-compatible

### Fixed

- **MERGE + UNWIND creates only one node**: planner evaluated MERGE property expressions as constants at plan time, dropping UNWIND variable references. Now uses per-row resolution
- **MERGE with NULL node reference**: `OPTIONAL MATCH (n:NonExistent) MERGE (n)-[:R]->(m)` silently succeeded as a no-op. Now returns a clear type mismatch error

## [0.5.23] - 2026-03-23

Prometheus metrics, tracing spans, and SQL/PGQ optional matching.

### Added

- **Prometheus metrics export** (`metrics`): `MetricsRegistry::to_prometheus()` renders counters, gauges, and histograms in Prometheus text format; `GrafeoDB::metrics_prometheus()` for one-call access; plan cache stats merged into snapshots
- **Tracing spans**: structured spans on query and transaction lifecycle (`session::execute`, `query::parse/optimize/plan/execute`, `tx::begin/commit/rollback`); zero-cost when no subscriber is registered
- **SQL/PGQ LEFT OUTER JOIN**: `LEFT [OUTER] JOIN MATCH` and `OPTIONAL MATCH` inside `GRAPH_TABLE(...)`, producing NULL-padded rows for unmatched patterns

### Changed

- **Read-only expand fast path**: all expand operators skip versioned MVCC lookups for read-only queries, using cheaper epoch-only visibility checks

### Fixed

- **Questioned edge (`->?`) row preservation**: LeftJoin collapsed source rows instead of preserving them with NULLs
- **Negative numeric literals in property maps**: unary negation (e.g. `{lat: -6.248}`) now folds correctly at plan time for both GQL and Cypher (#160)

## [0.5.22] - 2026-03-14

Pretty printing, observability, RDF performance overhaul, and GQL conformance tracking.

### Added

- **Pretty-printed query results**: `QueryResult` now renders as an ASCII table via `Display`, replacing the raw `Vec<Vec<Value>>` output
- **Observability** (`metrics`): lock-free `MetricsRegistry` with atomic counters and fixed-bucket histograms, tracking queries, latency (p50/p99), errors, transactions, sessions, GC sweeps, and plan cache stats across all 6 query languages. Zero overhead when disabled
- **Edge visibility fast path**: `is_edge_visible_at_epoch()` skips full edge construction when only checking MVCC visibility
- **Plan cache bindings**: `clear_plan_cache()` in Python, Node.js, C, and WASM
- **RDF bulk load**: `bulk_load()` builds all indexes in a single pass; `load_ntriples()` parses N-Triples with full term support (IRIs, blank nodes, typed/language-tagged literals)
- **SPARQL EXPLAIN**: returns the optimized logical plan tree without executing
- **GQL conformance tracking**: `// ISO:` test annotations linking to ISO/IEC 39075:2024 feature IDs, with `scripts/gql-conformance.py` for coverage reports and a machine-readable `gql-dialect.json` ([community feedback](https://github.com/orgs/GrafeoDB/discussions/122))
- **GQL binary set functions** (GF11): 12 statistical aggregates (COVAR_SAMP/POP, CORR, REGR_SLOPE/INTERCEPT/R2/COUNT/SXX/SYY/SXY/AVGX/AVGY)

### Changed

- **RDF query performance**: O(N*M) nested loop joins replaced with O(N+M) hash joins for all join types; composite indexes (SP, PO, OS) for O(1) lookup on 2-bound triple patterns; SPARQL optimizer uses RDF-specific statistics
- **Unsafe code enforcement**: `#![forbid(unsafe_code)]` on pure-safe crates, `#![deny(unsafe_code)]` on crates with targeted unsafe
- **GroupKeyPart zero-alloc**: uses `ArcStr` instead of `String`, eliminating allocations during aggregation
- **RDF code consolidation**: scattered `#[cfg]` gates consolidated into dedicated `database/rdf_ops.rs` and `session/rdf.rs` modules

## [0.5.21] - 2026-03-13

First implementation of C# and Dart bindings, single file database completed, snapshot consolidation and test hardening

### Added

- **C# / .NET bindings** (`crates/bindings/csharp`): .NET 8 P/Invoke binding wrapping grafeo-c. Covers GQL + multi-language queries (sync/async), ACID transactions, CRUD, vector search (k-NN + MMR), parameterized queries with temporal types, and SafeHandle resource management. CI on Ubuntu, Windows and macOS
- **Dart bindings** (`crates/bindings/dart`): Dart FFI binding wrapping grafeo-c. Covers parameterized queries with temporal type encoding, ACID transactions, CRUD, vector search (MMR), NativeFinalizer for memory safety, and sealed exception hierarchy. CI on all three platforms. Based on community PR #138 by @CorvusYe
- **Single-file `.grafeo` database format**: stores the entire database in one file with a sidecar WAL during operation (DuckDB-style). Dual-header crash safety with CRC32 checksums, auto format detection by extension, and WAL checkpoint merging. Use `GrafeoDB::open("mydb.grafeo")` or `db.save("mydb.grafeo")`. Realizes feature request #139 by @CorvusYe
- **Exclusive file locking** for `.grafeo` files: prevents multiple processes from opening the same database file simultaneously. Lock is acquired on open and released on close/drop (uses `fs2` for cross-platform advisory locking).
- **DDL schema persistence in snapshots**: CREATE NODE/EDGE/GRAPH TYPE, PROCEDURE and SCHEMA definitions survive close/reopen and export/import. Snapshot format consolidated to v3 with full schema metadata
- **Crash injection testing** (`testing-crash-injection` feature): `maybe_crash()` instrumentation points in `write_snapshot` and `checkpoint_to_file` enable deterministic crash simulation for verifying sidecar WAL recovery
- **Introspection functions**: `RETURN CURRENT_SCHEMA`, `RETURN CURRENT_GRAPH`, `RETURN info()`, `RETURN schema()` for querying session state and database metadata from within GQL

### Breaking

- **Snapshot format v3**: `export_snapshot()`/`import_snapshot()` now produce/consume v3 format (includes schema metadata). Snapshots from previous versions are no longer readable. Re-export from a running database to migrate.

### Testing

- **Spec compliance seam tests**: systematic coverage of ISO/IEC 39075 feature boundaries and negative paths (sessions, transactions, DML, patterns, aggregates, CASE, type coercion, cross-graph isolation). Uncovered 3 spec deviations

### Fixed

- **DDL in READ ONLY transactions** (ISO 39075 §8): CREATE/DROP GRAPH now blocked inside READ ONLY transactions
- **SUM on empty set** (ISO 39075 §20.9): returns NULL instead of 0, matching AVG/MIN/MAX
- **CASE WHEN with NULL conditions** (ISO 39075 §21): NULL conditions now correctly fall through to ELSE
- **SESSION SET SCHEMA / GRAPH separation** (ISO 39075 §7.1-7.2): schema and graph are now independent session fields with independent reset targets, schema-scoped graph keys, and `SHOW SCHEMAS`. `DROP SCHEMA` enforces "must be empty" per §12.3
- **COUNT(\*) parsing** (ISO 39075 §20.9): correctly parsed as a zero-argument aggregate

## [0.5.20] - 2026-03-11

Small release bringing new methods to WASM and adding SESSION SET validation

### Added

- **WASM `memoryUsage()` and `importRows()`**: memory introspection and bulk row import (the DataFrame equivalent) now available in WebAssembly bindings
- **WASM vector search bindings**: `createVectorIndex()`, `dropVectorIndex()`, `rebuildVectorIndex()`, `vectorSearch()` and `mmrSearch()` now exposed in WebAssembly, enabling client-side k-NN and MMR search with HNSW indexes

### Fixed

- **`SESSION SET GRAPH` / `SESSION SET SCHEMA` validation**: now errors when the target graph does not exist, matching the behavior of `USE GRAPH`; previously it silently accepted any name and fell back to the default store

## [0.5.19] - 2026-03-11

GQL translator refactor, new methods, GQL improvements and fixes

### Added

- **Graph type enforcement**: full write-path schema enforcement with node type inheritance, edge endpoint validation, UNIQUE/NOT NULL/CHECK constraints, default value injection, closed graph type guards, MERGE validator support, pattern-form syntax, SHOW commands and Cypher `ALTER CURRENT GRAPH TYPE`
- **LOAD DATA (multi-format import)**: generalized `LOAD DATA FROM 'path' FORMAT CSV|JSONL|PARQUET [WITH HEADERS] AS variable` in GQL, with Cypher-compatible `LOAD CSV` syntax preserved; JSONL behind `jsonl-import` feature, Parquet behind `parquet-import` feature
- **Python `import_df()`**: bulk-import nodes or edges from a pandas or polars DataFrame via `db.import_df(df, 'nodes', label='Person')` or `db.import_df(df, 'edges', edge_type='KNOWS')`
- **Memory introspection**: `db.memory_usage()` returns a hierarchical breakdown of heap usage across store, indexes, MVCC chains, query caches, string pools and buffer manager regions
- **Named graph persistence**: CREATE/DROP GRAPH and all mutations within named graphs are WAL-logged and recovered on restart. Snapshot v2 includes named graph data in all export/import/save paths; v1 snapshots remain backward-compatible
- **SHOW GRAPHS**: `SHOW GRAPHS` lists all named graphs in the database, complementing existing `SHOW NODE TYPES` / `SHOW EDGE TYPES`
- **RDF persistence**: SPARQL INSERT/DELETE/CLEAR/CREATE/DROP operations are now WAL-logged and recovered on restart; snapshot export/import includes RDF triples and RDF named graphs
- **Cross-graph transactions**: `USE GRAPH` and `SESSION SET GRAPH` now work within active transactions; commit/rollback/savepoint operations apply atomically across all touched graphs
- **GrafeoDB graph context**: one-shot `db.execute()` calls now persist `USE GRAPH` context across calls; `current_graph()` and `set_current_graph()` public API for programmatic access
- **WASM batch import**: `importLpg()` and `importRdf()` methods for bulk-loading structured LPG nodes/edges and RDF triples in a single call, with index-relative edge references and typed literal support

### Fixed

- **Named graph data isolation** ([#133](https://github.com/GrafeoDB/grafeo/issues/133)): USE GRAPH / SESSION SET GRAPH now correctly route all queries to the selected graph; query cache keys include graph name; dropping the active graph resets session to default
- **OPTIONAL MATCH WHERE pushdown**: right-side predicates pushed into the join instead of filtering out NULL rows
- **Cypher COUNT(expr) NULL skipping**: `COUNT(expr)` now skips NULLs (using `CountNonNull`), matching `COUNT(*)` behavior
- **Vector validity bitmap**: consecutive NULL pushes no longer silently drop null bits, fixing incorrect results in SPARQL OPTIONAL and RDF left joins

### Improved

- **GQL translator submodules**: split `gql.rs` into `gql/mod.rs`, `expression.rs`, `pattern.rs`, `aggregate.rs` for maintainability
- **Wildcard imports lint**: re-enabled `clippy::wildcard_imports` as warning; replaced `use super::*` in LPG planner submodules with explicit imports
- **Unwrap reduction**: replaced production `.expect()` calls with `Result`/`?` propagation in session initialization, persistence and WAL recovery paths

## [0.5.18] - 2026-03-09

Query language compliance improvements, expanded test coverage and Deriva compatibility fixes

### Added

- **Extensive spec test suites**: 8 Cypher + 12 GQL spec modules covering 1,300+ test cases, plus 67 Cypher exotic integration tests (NOT EXISTS, any()/reduce, list comprehensions, OPTIONAL MATCH, CASE, multi-label, etc.)

### Fixed (Cypher)

- **CALL subquery variable scope**: inner RETURN columns now resolve in the outer query instead of returning NULL
- **RETURN after DELETE**: delete operators pass through input rows for downstream aggregation
- **Inline MERGE with relationship SET**: decomposes inline node patterns into chained MERGE operations
- **WITH \* wildcard**: correctly passes all bound variables through
- **DoubleDash edge patterns**: undirected `--` patterns now parsed alongside `-[]-` syntax

### Fixed (GQL)

- **CALL { subquery }** recognized as query-level clause instead of procedure call
- **WITH + LET bindings**: LET clauses after WITH parsed and attached correctly
- **String concatenation**: `||` (CONCAT) now supported in arithmetic expressions
- **Inline MERGE with relationship SET**: same decomposition fix as Cypher

### Fixed

- **Multiple NOT EXISTS subqueries**: two or more `NOT EXISTS` predicates no longer cause variable-not-found errors
- **Transaction rollback**: SET property, SET/REMOVE label, and MERGE ON MATCH SET changes all correctly undone on ROLLBACK. Savepoint partial rollback preserves earlier changes
- **NPM package missing native binaries** ([#128](https://github.com/GrafeoDB/grafeo/issues/128)): `@grafeo-db/js` now publishes per-platform packages as `optionalDependencies`

## [0.5.17] - 2026-03-09

Cypher query execution bug fixes for Deriva compatibility.

### Fixed

- **Correlated EXISTS subqueries**: `NOT EXISTS { MATCH (a)-[r]->(b) WHERE type(r) = 'X' }` now correctly plans via semi-join instead of failing with "Unsupported EXISTS subquery pattern"
- **CASE WHEN in aggregates**: `sum(CASE WHEN ... THEN 1 ELSE 0 END)` resolves correctly inside aggregate functions
- **any()/all()/none()/single() with IN list**: `any(lbl IN labels(n) WHERE lbl IN ['A', 'B'])` now evaluates the IN operator correctly in list predicate contexts
- **CASE WHEN in reduce()**: `reduce(acc = 0, x IN vals | CASE WHEN x > acc THEN x ELSE acc END)` evaluates CASE expressions with both accumulator and item variable bindings

## [0.5.16] - 2026-03-08

Performance enhancements, bug fixes and Rust examples

### Added

- **LOAD CSV**: `LOAD CSV [WITH HEADERS] FROM 'path' AS row [FIELDTERMINATOR '\t']` in Cypher, with inline CSV parser supporting quoted fields, `file:///` URIs and custom delimiters
- **Cypher schema DDL**: `CREATE/DROP INDEX`, `CREATE/DROP CONSTRAINT`, `SHOW INDEXES`, `SHOW CONSTRAINTS`
- **Relationship WHERE**: inline predicates on relationship patterns (`-[r WHERE r.since > 2020]->`)
- **Temporal map constructors**: `date({year:2024, month:3})`, `time({hour:14})`, `datetime(...)`, `duration({years:1, months:2, days:3})`
- **PROFILE statement**: `PROFILE MATCH ... RETURN ...` executes the query and returns per-operator metrics (rows, self-time, call counts) for GQL and Cypher
- **Rust examples**: 7 runnable examples in `examples/rust/` covering the core API (basic queries, transactions, parameterized queries, vector search, graph algorithms, WAL persistence, multi-language dispatch)
- **Plan cache invalidation**: query plan cache is automatically cleared after DDL operations (CREATE/DROP INDEX, TYPE, CONSTRAINT, etc.), with manual `clear_plan_cache()` API on `GrafeoDB` and `Session`
- **Cache invalidation counter**: `CacheStats.invalidations` tracks how often DDL clears the plan cache

### Improved

- **Cost model calibration**: recursive plan costing, statistics-aware IO estimation, actual child cardinalities for joins, multi-edge-type expand costing
- **Supply chain audit**: replaced `cargo audit` CI job with `cargo-deny` (licenses, advisories, bans, source verification)
- **Benchmark regression detection**: PRs now run all three criterion suites (arena, index, query) and fail on >10% regression via `benchmark-action`
- **Examples CI**: added `cargo build -p grafeo-examples` to CI checks

### Fixed

- **GQL `-->` shorthand**: parser recognizes `-->` as a directed outgoing edge instead of splitting into `--` and `>`
- **EXISTS bare patterns**: `EXISTS { (a)-[r]->(b) }` without explicit MATCH keyword now works in GQL and Cypher
- **CASE WHEN in aggregates**: expressions like `sum(CASE WHEN ... THEN 1 ELSE 0 END)` resolve correctly in the LPG planner
- **SPARQL parameters**: `execute_sparql_with_params()` now substitutes `$param` values instead of ignoring them

## [0.5.15] - 2026-03-07

Full ecosystem feature profile rework and several graph database nice-to-haves

### Added

- **Ecosystem feature profiles**: `embedded`, `browser`, `server` named profiles across all crates. `storage` convenience group (`wal` + `spill` + `mmap`)
- **WASM multi-variant builds**: AI variant (531 KB gzip) and lite variant (513 KB gzip) via `build-wasm-all.sh`, with `regex-lite` for smaller binaries
- **Savepoints and nested transactions**: `SAVEPOINT`/`ROLLBACK TO`/`RELEASE`, inner `START TRANSACTION` auto-creates savepoints
- **Correlated subqueries**: `EXISTS { ... }`, `COUNT { ... }`, `VALUE { ... }` in WHERE/RETURN
- **Subpath variable binding**: `(p = (a)-[e]->(b)){2,5}` with `length(p)`, `nodes(p)`, `edges(p)`
- **Type system extensions**: `LIST<T>` typed lists with coercion, `IS TYPED RECORD/PATH/GRAPH` predicates, `path()` constructor
- **Graph DDL**: `CREATE GRAPH g2 LIKE g1`, `AS COPY OF`, `CREATE GRAPH g ANY/OPEN`
- **GQLSTATUS diagnostics**: ISO sec 23 status codes and diagnostic records on all query results
- **Catalog procedures**: `CALL db.labels()`, `db.relationshipTypes()`, `db.propertyKeys()` with YIELD
- **Python DataFrame bridge**: `result.to_pandas()`, `result.to_polars()`, `db.nodes_df()`, `db.edges_df()` for zero-friction data science integration

### Fixed

- **Temporal functions**: `local_time()`, `local_datetime()`, `zoned_datetime()` constructors, `date_trunc()` truncation
- **Aggregate separators**: `LISTAGG` and `GROUP_CONCAT` with custom separators and per-language defaults

### Changed

- **Default profile**: facade crate default changed from `full` to `embedded`. All binding crates follow
- **WASM**: default changed to `browser` profile, binary size reduced from 1,001 KB to 513 KB gzipped (49%)

## [0.5.14] - 2026-03-06

Moving crates and lots of small improvements and fixes

### Added

- **EXPLAIN statement**: `EXPLAIN <query>` in GQL and Cypher returns the optimized logical plan tree with pushdown hints (`[index: prop]`, `[range: prop]`, `[label-first]`)
- **WASM size optimization**: `wasm-opt -Oz` applied during release builds
- **NetworkX bridge**: `adj` property and `subgraph(nodes)` method
- **SPARQL built-in functions**: date/time (NOW, YEAR, MONTH, ...), hash (MD5, SHA1, SHA256, SHA384, SHA512), RDF term (LANG, DATATYPE, IRI, BNODE, ...) and RAND
- **GROUP_CONCAT / SAMPLE aggregates**: proper implementations replacing the previous Collect stub

### Fixed

- **Auto-commit for mutations**: single-shot `execute()` calls with INSERT/DELETE/SET now auto-commit instead of silently discarding changes
- **WAL persistence for queries**: mutations via GQL/Cypher now persist to WAL (previously only the CRUD API did)
- **WAL property removal**: `remove_node_property` and `remove_edge_property` now log to WAL
- **Cypher count(\*)**: parses correctly when `count` is tokenized as a keyword
- **SPARQL unary plus**: treated as identity instead of `NOT`
- **CLI fixes**: `data dump`/`data load` now work (JSON Lines), `compact` performs real compaction, `index list` shows per-index details, nonexistent databases error instead of being silently created
- **WASM test suite**: fixed compilation and runtime panics on wasm32

### Changed

- **Node.js `nodeCount`/`edgeCount`**: changed from getter properties to methods (`db.nodeCount()`)
- **Arena allocator**: returns `Result<T, AllocError>` instead of panicking on allocation failure
- **Planner refactor**: split into `planner/lpg/` and `planner/rdf/` with shared operator builders
- **Translator refactor**: shared plan-builder functions extracted into `translators/common.rs`, all 7 translators moved into `query/translators/`
- **Dependency cleanup**: removed unused deps, replaced ahash with foldhash, narrowed tokio features

## [0.5.13] - 2026-03-04

Big language compliance push, schema DDL, time-travel and named graphs

### Improved

- **GQL**: full compliance with ISO/IEC 39075:2024, covering all features practical for a graph database
- **Cypher**: improved openCypher v9 compliance, plus pattern comprehensions, CALL subqueries, FOREACH
- **SPARQL**: improved W3C SPARQL 1.1 compliance (no 1.2/SPARQL Star yet)

#### Infrastructure

- **LPG named graphs**: multi-graph support with per-graph storage, labels, indexes and MVCC versioning (`create_graph()`, `drop_graph()`, `list_graphs()`)
- **Apply operator**: correlated subquery execution for CALL, VALUE, NEXT and pattern comprehensions
- **Temporal types**: `Date`, `Time`, `Duration` with ISO 8601 parsing, arithmetic and component extraction. Python round-trips via `datetime.date`/`datetime.time`

#### Schema / DDL

- **Full schema DDL**: CREATE/DROP/ALTER for NODE TYPE, EDGE TYPE, GRAPH TYPE, INDEX, CONSTRAINT and SCHEMA, with `OR REPLACE`, `IF NOT EXISTS`/`IF EXISTS` and WAL persistence
- **Type definitions**: `CREATE NODE TYPE Person (name STRING NOT NULL, age INT64)` with nullability
- **Index DDL**: `CREATE INDEX ... FOR (n:Label) ON (n.property) [USING TEXT|VECTOR|BTREE]`
- **Constraint enforcement**: UNIQUE, NOT NULL, NODE KEY, EXISTS validated on writes

#### Time-Travel

- **Epoch-based time-travel**: `execute_at_epoch(query, epoch)` runs any query against a historical snapshot. Also available via `set_viewing_epoch()` or `SESSION SET PARAMETER viewing_epoch = <n>`
- **Version history**: `get_node_history(id)` / `get_edge_history(id)` return all versions with creation/deletion epochs

#### GQL Spec Compliance (78% to ~97%)

- **New syntax**: LIKE, CAST to temporal, SET map operations (`= {map}`, `+= {map}`), NODETACH DELETE, RETURN \*/WITH \*, list comprehensions, transaction characteristics, zoned temporals, ALTER DDL, CREATE GRAPH TYPED, stored procedures
- **List property storage**: `reduce()` and list operations work correctly after INSERT with list-valued properties

### Fixed

- **Time-travel scans**: now use pure epoch-based visibility instead of transaction-aware checks
- **LIKE parser**: token existed but was never consumed as an infix operator
- **RETURN \* binder**: was incorrectly rejected as an undefined variable
- **List comprehensions**: planner now handles these in RETURN projections
- **Cypher fixes**: standalone DELETE/SET/REMOVE error messages, `^` power operator, anonymous variable name collisions
- **Temporal comparison**: Date/Time/Timestamp comparisons no longer silently return false

### Improved

- **Test coverage**: 80+ GQL parser tests (was 44), 137 Python compliance tests (was 100), new SPARQL and Cypher suites

## [0.5.12] - 2026-03-02

Two-phase commit, snapshot restore, EXISTS subqueries.

### Added

- **PreparedCommit**: two-phase commit via `session.prepare_commit()`, inspect pending mutations and attach metadata before finalizing
- **Atomic snapshot restore**: `db.restore_snapshot(data)` replaces the database in place, with full pre-validation (store unchanged on error)
- **EXISTS subqueries** (GQL, Cypher): complex inner patterns with multi-hop traversals, property filters and label constraints via semi-join rewrite

### Fixed

- **SET on edge variables**: Cypher translator now correctly handles SET when targeting an edge variable

### Improved

- **Variable-length path traversal**: BFS path tracking uses shared-prefix `Rc` segments instead of cloning full vectors, reducing per-edge cost from O(depth) to O(1)

## [0.5.11] - 2026-03-02

Pluggable storage traits, query language compliance, UNION support.

### Added

- **Pluggable storage**: `GraphStore`/`GraphStoreMut` traits decouple all query operators and algorithms from `LpgStore`. Use `GrafeoDB::with_store(Arc<dyn GraphStoreMut>, Config)` to plug in any backend
- **Type-safe WAL**: `WalEntry` trait and `TypedWal<R>` wrapper constrain WAL record types at compile time, preventing cross-model logging
- **Query language compliance tests**: spec-level integration tests for all 6 query languages
- **Cypher UNION / UNION ALL**: combining query results with duplicate elimination or preservation
- **GQL MERGE on relationships**: `MERGE (a)-[r:TYPE]->(b)` with idempotent edge creation
- **Gremlin traversal steps**: `and()`, `or()`, `not()`, `where()`, `filter()`, `choose()`, `optional()`, `union()`, `coalesce()` and more
- **SPARQL improvements**: DISTINCT, HAVING, FILTER NOT EXISTS / EXISTS

## [0.5.10] - 2026-02-29

Robustness: bidirectional shortest path, crash recovery tests, stress tests.

### Added

- **Skip index for adjacency chunks**: compressed cold chunks maintain a zone-map skip index. `contains_edge(src, dst)` provides O(log n) point lookups; `edges_in_range(src, min, max)` supports efficient range queries
- **Bidirectional BFS shortest path**: meet-in-the-middle BFS expanding smaller frontier first, reducing search space from O(b^d) to O(b^(d/2))

### Improved

- **Crash recovery tests**: 7 deterministic crash injection tests verifying WAL recovery at every crash point
- **Concurrent stress tests**: 6 multi-threaded tests covering concurrent writers, mixed read/write, transaction conflicts, epoch pressure and rapid session lifecycle
- **Hardened panic messages**: ~50 bare `unwrap()` calls converted to `expect()` with invariant descriptions; no behavioral change

## [0.5.9] - 2026-02-28

Compact property storage, snapshot validation, crash injection framework.

### Added

- **Snapshot validation**: `import_snapshot()` pre-validates everything before inserting: rejects duplicate IDs and dangling edge references
- **Crash injection framework**: feature-gated `maybe_crash()` / `with_crash_at()` for deterministic recovery testing, with three WAL crash points. Zero overhead when disabled
- **Backward compatibility tests**: pinned v1 snapshot fixture with 8 regression tests for format stability

### Fixed

- **WASM build with `getrandom` 0.4**: added `wasm_js` crate feature for 0.4.x on wasm32 targets
- **WASM binary size regression**: disabled transitive engine features in bindings-common, reducing WASM gzip from 974 KB to 744 KB

### Improved

- **Compact property storage**: property maps switched from `BTreeMap` to `SmallVec<4>`, so nodes with 4 or fewer properties avoid heap allocation
- **Cost model per-type fanout**: the optimizer now tracks per-edge-type average degree instead of a single global estimate

## [0.5.8] - 2026-02-22

Shared bindings crate, unified query dispatch, Node.js/WASM API expansion.

### Added

- **`grafeo-bindings-common` crate**: shared entity extraction, error classification and JSON conversion for all four bindings (Python, Node.js, C, WASM)
- **Unified query dispatch**: `execute_language(query, "gql"|"cypher"|"sparql"|...)` replaces 18 per-language functions
- **Node.js API parity**: property removal, label management, `info()`, `schema()`, `version()` and transaction isolation levels now match the Python binding
- **WASM expansion**: parameterized queries, per-language convenience methods, proper feature gating
- **Batch edge creation**: `batch_create_edges()` with single lock acquisition for bulk imports

### Improved

- **Incremental statistics**: `compute_statistics()` reads atomic delta counters instead of scanning all entities, reducing refresh from O(n+m) to O(|labels|+|edge_types|)
- **Cost model uses real fanout**: optimizer derives average edge fanout from actual graph statistics instead of a hardcoded 10.0

## [0.5.7] - 2026-02-19

UNWIND property access fix, `algos` feature flag.

### Fixed

- **UNWIND mutation property access**: map property access like `e.src`, `e.weight` in CREATE/SET now resolves correctly. Previously only column references and constants worked, so map properties came back as NULL

### Added

- **`algos` feature flag**: graph algorithms gated behind `algos` (included in `full`). Reduces compile time and binary size when algorithms are not needed

## [0.5.6] - 2026-02-18

UNWIND/FOR list expansion, embedding model config, zero unsafe in property storage.

### Added

- **UNWIND clause**: expand lists into rows for batch processing. Works with literals, parameters (`UNWIND $items AS x`) and vectors. Combine with MATCH + INSERT for bulk edge creation
- **FOR statement** (GQL standard): `FOR x IN [1, 2, 3] RETURN x`, with `WITH ORDINALITY` (1-based) and `WITH OFFSET` (0-based) index tracking
- **Text index auto-sync**: text indexes update automatically on property changes, no manual rebuild needed. WASM bindings added too
- **SPARQL COPY/MOVE/ADD**: graph management operators with source-existence validation and SILENT support
- **Embedding model config**: 3 presets (MiniLM-L6-v2, MiniLM-L12-v2, BGE-small-en-v1.5) with HuggingFace auto-download. Exposed in Python and Node.js
- **Native SSSP procedure**: `CALL grafeo.sssp('node_name', 'weight')` for LDBC Graphanalytics compatibility

### Fixed

- **UNWIND scoping**: MATCH clauses after UNWIND now correctly receive UNWIND variables, scalar values no longer resolve as node IDs and `Value::Vector` is handled alongside `Value::List`
- **`RETURN n` returns full entities**: `MATCH (n) RETURN n` now returns `{_id, _labels, ...properties}` instead of a bare integer ID
- **GQL lexer UTF-8 panic**: multi-byte characters no longer cause boundary panics
- **Scalar column tracking**: Gremlin `.values()`, `.count()` and GQL `WITH expr AS alias` no longer return NULL
- **Vector index rebuild after drop**: works without the old index, infers dimensions from data

### Improved

- **Zero unsafe in property storage**: replaced final `transmute_copy` calls with safe `EntityId` conversions
- **Statistics access**: `statistics()` returns `Arc<Statistics>` instead of deep-cloning on every planner invocation
- **Entity resolution**: moved from 6-site post-processing into the ProjectOperator pipeline for single-pass resolution

## [0.5.5] - 2026-02-16

Filter pushdown, query error positions, transaction fixes.

### Added

- **Filter pushdown**: equality predicates on labeled scans are pushed to the store level. Compound predicates like `WHERE n.name = 'Alix' AND n.age > 30` correctly split: equality pushed down, range kept as post-filter
- **Query error positions**: all six parsers now produce errors with line/column positions and source-caret display

### Fixed

- **Transaction edge type visibility**: edges created within a transaction are now visible to subsequent queries in the same transaction
- **SPARQL INSERT/DELETE DATA with GRAPH clause**: triples now route to the named graph instead of the default graph
- **Compound predicate correctness**: filter pushdown no longer drops non-equality parts of compound predicates

## [0.5.4] - 2026-02-15

### Fixed

- **Multi-pattern CREATE**: `CREATE (:A {id: 'x'}), (:B {id: 'y'})` now creates all nodes instead of only the first

## [0.5.3] - 2026-02-13

### Improved

- **Query error quality**: translator errors now produce `QueryError` with semantic error codes (`GRAFEO-Q002`) instead of generic internal errors. More actionable messages
- **GraphQL range filters**: operator suffixes (`_gt`, `_lt`, etc.) now work on direct query arguments, not just `where` clauses

### Fixed

- **SPARQL `FILTER NOT EXISTS`**: parser now recognizes NOT EXISTS/EXISTS, producing correct anti-join/semi-join plans
- **SPARQL `FILTER REGEX`**: REGEX evaluation was missing from the RDF planner (parser/translator already supported it)

## [0.5.2] - 2026-02-13

### Added

- **CALL procedure support**: invoke any of the 22 built-in graph algorithms from query strings: `CALL grafeo.<algorithm>() [YIELD columns]`. Supported in GQL, Cypher and SQL/PGQ
- **Map literal arguments**: `CALL grafeo.pagerank({damping: 0.85, max_iterations: 20})`
- **Procedure listing**: `CALL grafeo.procedures()` returns all available procedures

## [0.5.1] - 2026-02-12

Hybrid search, built-in embeddings, change data capture. The features that make grafeo-memory work.

### Added

- **BM25 text search** (`text-index`): inverted indexes on string properties with BM25 scoring. Built-in tokenizer with Unicode word boundaries, lowercasing and stop word removal
- **Hybrid search** (`hybrid-search`): combine BM25 text + HNSW vector similarity via RRF or weighted fusion. Single `hybrid_search()` call in Python and Node.js
- **Built-in embeddings** (`embed`, opt-in): in-process embedding generation via ONNX Runtime. Load any `.onnx` model, call `embed_text()`. Adds ~17MB, off by default
- **Change data capture** (`cdc`): track all mutations with before/after property snapshots. Query via `history()`, `history_since()`, `changes_between()`. Available in Python and Node.js

## [0.5.0] - 2026-02-11

Error codes, query timeouts, auto-GC, ~50% memory savings for vector workloads.

### Added

- **Standardized error codes**: all errors carry `GRAFEO-XXXX` codes (Q = query, T = transaction, S = storage, V = validation, X = internal) with `error_code()` and `is_retryable()`
- **Query timeout**: `Config::default().with_query_timeout(Duration::from_secs(30))` stops long-running queries cleanly
- **MVCC auto-GC**: version chains garbage-collected every N commits (default 100, configurable). Also `db.gc()` for manual control

### Improved

- **Topology-only HNSW**: vectors no longer duplicated inside the index; reads on-demand via `VectorAccessor` trait. ~50% memory reduction for vector workloads

## [0.4.4] - 2026-02-11

SQL/PGQ queries, MMR search for RAG, auto-syncing vector indexes, CLI overhaul.

### Added

- **SQL/PGQ support**: query with SQL:2023 syntax, `SELECT ... FROM GRAPH_TABLE (MATCH ... COLUMNS ...)`. Includes path functions, DDL and all bindings
- **MMR search**: diverse, relevant results for RAG pipelines via `mmr_search()` with tunable relevance/diversity balance
- **Filtered vector search**: property equality filters on `vector_search()`, `batch_vector_search()` and `mmr_search()` using pre-computed allowlists for efficient HNSW traversal
- **Incremental vector indexing**: indexes stay in sync automatically as nodes change
- **CLI overhaul**: interactive shell with transactions, meta-commands (`:schema`, `:info`, `:stats`), persistent history, CSV output. Install via `cargo install`, `pip install` or `npm install -g`
- **Configurable cardinality estimation**: tune 9 selectivity parameters via `SelectivityConfig`
- **AdminService trait**: unified introspection and maintenance: `info()`, `detailed_stats()`, `schema()`, `validate()`, `wal_status()`
- **GQL `IN` operator**: `WHERE n.name IN ['Alix', 'Gus']`
- **String escape sequences**: `\'`, `\"`, `\\`, `\n`, `\r`, `\t` in GQL, Cypher, SQL/PGQ

### Fixed

- **Node.js ID validation**: rejects negative, NaN, Infinity and values above `MAX_SAFE_INTEGER`

### Changed

- **Python CLI removed**: replaced by the unified `grafeo-cli` Rust binary

## [0.4.3] - 2026-02-08

Per-database graph model selection, snapshot export/import, expanded WASM APIs.

### Added

- **Database creation options**: choose LPG or RDF per database, configure durability mode, toggle schema constraints
- **Snapshot export/import**: serialize to binary snapshots for backups or WASM persistence via IndexedDB
- **WASM expansion**: `executeWithLanguage()`, `exportSnapshot()`/`importSnapshot()`, `schema()`

## [0.4.2] - 2026-02-08

Grafeo now runs in the browser. WebAssembly bindings with TypeScript definitions at 660 KB gzipped.

### Added

- **WebAssembly bindings** (`@grafeo-db/wasm`): `execute()`, `executeRaw()`, `nodeCount()`, `edgeCount()`, full TypeScript definitions. 660 KB gzipped (target was <800 KB)
- **Feature-gated platform subsystems**: `parallel`, `spill`, `mmap`, `wal` are opt-in, making wasm32 compilation straightforward

## [0.4.1] - 2026-02-08

Go and C bindings. Grafeo now embeds in pretty much any language.

### Added

- **Go bindings** (`github.com/GrafeoDB/grafeo`): full CRUD, multi-language queries, ACID transactions, vector search, batch operations, admin APIs
- **C FFI layer** (`grafeo-c`): C-compatible ABI for embedding Grafeo in any language

## [0.4.0] - 2026-02-07

Node.js/TypeScript bindings, Python vector search and transaction isolation.

### Added

- **Node.js/TypeScript bindings** (`@grafeo-db/js`): full CRUD, async queries across all 5 languages, transactions, native type mapping, TypeScript definitions
- **Python vector support**: pass `list[float]` directly, `grafeo.vector()`, distance functions in GQL, HNSW indexes, k-NN search
- **Python transaction isolation**: `"read_committed"`, `"snapshot"` or `"serializable"` per transaction
- **Batch vector APIs**: `batch_create_nodes()` and `batch_vector_search()` for Python and Node.js

### Fixed

- GQL INSERT with list or `vector()` properties no longer silently drops values
- Multi-hop MATCH queries (3+ hops) no longer return duplicate rows
- GQL multi-hop patterns now correctly filter intermediate nodes by label
- Transaction `execute()` rejects queries after commit/rollback

### Improved

- **HNSW recall and speed**: Vamana-style diversity pruning, pre-normalized cosine vectors, pre-allocated structures
- Query optimizer uses actual store statistics instead of hardcoded defaults

## [0.3.4] - 2026-02-06

Query timing, "did you mean?" suggestions, Python pagination.

### Added

- **Query performance metrics**: every result includes `execution_time_ms` and `rows_scanned`
- **"Did you mean?" suggestions**: typo in a variable or label? Grafeo suggests the closest match
- **Python pagination**: `get_nodes_by_label()` supports `offset` for paging

## [0.3.3] - Unreleased

### Added

- **VectorJoin operator**: graph traversal + vector similarity in a single query
- **Vector zone maps**: skips irrelevant data blocks during vector search
- **Product quantization**: 8-32x memory compression with ~90% recall retention
- **Memory-mapped vector storage**: disk-backed with LRU caching for large datasets
- **Python quantization API**: `ScalarQuantizer`, `ProductQuantizer`, `BinaryQuantizer`

## [0.3.2] - Unreleased

### Added

- **Selective property loading**: fetch only the properties you need, much faster for wide nodes
- **Parallel node scan**: 3-8x speedup on large scans (10K+ nodes) across CPU cores

## [0.3.1] - Unreleased

### Added

- **Vector quantization**: f32 to u8 (scalar) or 1-bit (binary) compression with quantized HNSW search + exact rescoring
- **SIMD acceleration**: 4-8x faster distance computations; auto-selects AVX2/FMA, SSE or NEON
- **Vector batch operations**: `batch_insert()` and `batch_search()` for bulk loading
- **VectorScan operators**: vector similarity integrated into the query execution engine
- **Adaptive WAL flusher**: self-tuning background flush based on actual disk speed
- **Fingerprinted hash index**: sharded with 48-bit fingerprints for near-instant miss detection

## [0.3.0] - Unreleased

Vectors are a first-class type. Graph + vector hybrid queries let you do things no pure vector database can.

### Added

- **Vector type**: native storage with dimension-aware schema validation
- **Distance functions**: cosine, euclidean, dot product, manhattan
- **HNSW index**: O(log n) approximate nearest neighbor with tunable presets (`high_recall()`, `fast()`). Also brute-force k-NN with optional predicate filtering
- **GQL vector syntax**: `vector([...])` literals, distance functions, `CREATE VECTOR INDEX`
- **SPARQL vector functions**: `COSINE_SIMILARITY()`, `EUCLIDEAN_DISTANCE()`, `DOT_PRODUCT()`, `MANHATTAN_DISTANCE()`
- **Serializable snapshot isolation**: `ReadCommitted`, `SnapshotIsolation` or `Serializable` per transaction

---

## [0.2.7] - 2026-02-05

Parallel execution primitives, second-chance LRU cache.

### Added

- **Second-chance LRU cache**: lock-free access marking for concurrent workloads
- **Parallel fold-reduce**: `parallel_count`, `parallel_sum`, `parallel_stats`, `parallel_partition` and a composable collector trait

---

## [0.2.6] - 2026-02-04

Zone map filtering, clustering coefficient, faster batch reads.

### Added

- **Local clustering coefficient**: triangle counting with parallel execution
- **Chunk-level zone map filtering**: skip entire data chunks when predicates can't match

### Improved

- Batch property retrieval acquires a single lock instead of one per entity

---

## [0.2.5] - 2026-02-03

Full SPARQL functions, platform allocators, batch property APIs.

### Added

- **Full SPARQL function coverage**: string, type, math functions and REGEX
- **EXISTS/NOT EXISTS**: semi-join and anti-join subqueries
- **Platform allocators**: optional jemalloc (Linux/macOS) or mimalloc (Windows) for 10-20% faster allocations
- **Batch property APIs**, compound predicate pushdown, range queries with zone map pruning

### Improved

- Community detection now O(E) instead of O(V^2 E), roughly 100-500x faster on large graphs

---

## [0.2.4b] - 2026-02-02

Fixed release workflow `--exclude` flag (requires `--workspace`).

## [0.2.4] - 2026-02-02

Benchmark-driven optimizations: lock-free reads, direct lookups, faster filters.

### Improved

- **Lock-free concurrent reads**: hash indexes use DashMap, 4-6x improvement under concurrency
- **Direct lookup APIs**: O(1) point reads bypassing query planning, 10-20x faster than MATCH
- **Filter performance**: 20-50x improvement for equality and range filters

---

## [0.2.3] - Unreleased

### Added

- **Succinct data structures** (`succinct-indexes`): O(1) rank/select bitvectors, Elias-Fano, wavelet trees
- **Block-STM parallel execution** (`block-stm`): optimistic parallel transactions, 3-4x batch speedup
- **Ring index for RDF** (`ring-index`): compact triple storage via wavelet trees (~3x space reduction)
- **Query plan caching**: repeated queries skip parsing and optimization, 5-10x speedup

---

## [0.2.2] - Unreleased

### Added

- **Bidirectional edge indexing**: `edges_to()`, `in_degree()`, `out_degree()`
- **NUMA-aware scheduling**: work-stealing prefers same-node to minimize cross-node memory access
- **Leapfrog TrieJoin**: worst-case optimal joins for cyclic patterns, O(N^1.5) vs O(N^2)

---

## [0.2.1] - Unreleased

### Added

- **Tiered version index**: hot/cold separation for memory-efficient MVCC
- **Compressed epoch store**: zone maps for predicate pushdown on archived data
- **Epoch freeze**: compress and archive old epochs to reclaim memory

---

## [0.2.0] - 2026-02-01

Performance foundation: factorized execution to avoid Cartesian products in multi-hop queries.

### Added

- **Factorized execution**: avoids Cartesian product materialization, inspired by [Kuzu](https://kuzudb.com/)

### Changed

- Switched from Python-based pre-commit to [prek](https://github.com/j178/prek) (Rust-native, faster)

---

## [0.1.4] - 2026-01-31

Label removal, Python label APIs, all languages on by default.

### Added

- **REMOVE clause**: `REMOVE n:Label` and `REMOVE n.property` in GQL
- **Label APIs**: `add_node_label()`, `remove_node_label()`, `get_node_labels()` in Python
- **RDF transactions**: SPARQL now supports proper commit/rollback

### Changed

- All query languages enabled by default, no feature flags needed

## [0.1.3] - 2026-01-30

CLI, Python admin APIs, adaptive execution, property compression.

### Added

- **CLI** (`grafeo-cli`): inspect, backup, export, manage WAL, compact databases
- **Admin APIs**: Python bindings for `info()`, `detailed_stats()`, `schema()`, `validate()`
- **Adaptive execution**: runtime re-optimization when cardinality estimates deviate 3x+ from actuals
- **Property compression**: dictionary, delta, RLE codecs with hot buffer pattern

### Improved

- Query optimizer: projection pushdown, better join reordering, histogram-based cardinality estimation

## [0.1.2] - 2026-01-29

Python test suite, documentation pass.

### Added

- Comprehensive Python test suite covering LPG, RDF, all 5 query languages and plugins
- Docstring pass across all crates

## [0.1.1] - Unreleased

### Added

- **GQL parser**: full ISO/IEC 39075 support
- **Multi-language**: Cypher, Gremlin, GraphQL, SPARQL translators
- **MVCC transactions**: snapshot isolation
- **Indexes**: hash, B-tree, trie, adjacency
- **Storage**: in-memory and write-ahead log
- **Python bindings**: PyO3-based API

### Changed

- Renamed from Graphos to Grafeo, reset version to 0.1.0

## [0.1.0] - Unreleased

### Added

- **Core architecture**: modular crate structure (common, core, adapters, engine, python)
- **Graph models**: LPG and RDF triple store
- **In-memory storage**: fast graph operations without persistence overhead

---

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
