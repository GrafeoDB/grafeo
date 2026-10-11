# The sections of a database compacted by 0.5.44

The raw chunks of three sections of `crates/grafeo-engine/tests/fixtures/released/0.5.44/compacted.grafeo` (written
by `scripts/released_fixtures.py --compacted` with the released `grafeo==0.5.44` wheel), extracted once with the
0.5.x container reader (`grafeo_storage::file::legacy::LegacyFile`):

- `compact_store.bin`: the `CompactStore` section, encoding version 3 (the compacted base).
- `overlay_deletions.bin`: the `OverlayDeletions` section, version 1 (the base nodes and edges deleted since).
- `lpg_store.bin`: the `LpgStore` section (the overlay: the writes since `compact()`).

The compact module's reader and fold tests read them: 0.6.0 reads these sections only to migrate a 0.5.x file, and
no build writes them any more, so the tests use bytes a release wrote. They go with the 0.5.x readers in 0.7.0.
