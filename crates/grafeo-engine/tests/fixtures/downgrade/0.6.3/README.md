# Files 0.6.3 writes that 0.6.2 must refuse cleanly

0.6.3 keeps a node or edge type made on its own in force when a graph type lists it (#764), and its catalog writes such
a type as record kind 12 (node type) or 13 (edge type), which 0.6.2 does not know. `scripts/downgrade_check.py` opens
copies of these files with the released 0.6.2 wheel and requires a refusal that leaves every byte of the file and its
WAL as it was; `tests/downgrade_fixtures.rs` checks with the current build that they hold the marked types.

| File | Holds |
| --- | --- |
| `standalone-section.grafeo` | closed: kinds 12 and 13 are in the checkpointed catalog section |
| `standalone-wal.grafeo` with `standalone-wal.grafeo.wal/` | the process ended without `close()`: kinds 12 and 13 are only in the WAL |

Both ran, in a persistent database:

```text
CREATE NODE TYPE A (h FLOAT64)
CREATE EDGE TYPE ROAD (km INT64)
CREATE GRAPH TYPE t (NODE TYPE A)
CREATE GRAPH TYPE u (EDGE TYPE ROAD)
INSERT (:A {h: 1.5e0})
```

Written by a 0.6.3 build of the tree with:

```bash
GRAFEO_WRITE_DOWNGRADE_FIXTURES=1 cargo test -p grafeo-engine --all-features --test main \
    downgrade_fixtures::write_the_fixtures -- --ignored --exact
```

Checked against the 0.6.2 wheel with:

```bash
uv run --no-project --python 3.13 --with grafeo==0.6.2 python scripts/downgrade_check.py 0.6.3 --release 0.6.2
```

0.6.2 answers `GRAFEO-X002` ("kind 12 is required, but this release does not know it (a newer release wrote it)") for
the section file and `GRAFEO-S003` (its replay meets "a catalog record of kind 12 ... a newer release wrote it") for
the WAL file. The tests open copies, never these files.
