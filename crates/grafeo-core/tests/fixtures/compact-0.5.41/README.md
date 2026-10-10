# The sections of a database compacted by 0.5.41

The raw chunks of two sections of a `compacted.grafeo` written by `scripts/released_fixtures.py --compacted` with the
released `grafeo==0.5.41` wheel, extracted once with the 0.5.x container reader
(`grafeo_storage::file::legacy::LegacyFile`). 0.5.41 cannot write `date`, `zoned_datetime` or `duration` values, so
they were left out of Alix's insert; the sessions are otherwise those of `compact-0.5.44`.

- `compact_store.bin`: the `CompactStore` section, encoding version 1 (columns whole, no blocks), which 0.5.40 and
  0.5.41 wrote.
- `lpg_store.bin`: the `LpgStore` section (the overlay: the writes since `compact()`).

0.5.40 and 0.5.41 wrote no `OverlayDeletions` section: the base node the second session deletes (Vincent) comes back
when the file is opened, as it did with 0.5.42 to 0.5.44. 0.5.41 itself cannot reopen the file ("snapshot checksum
mismatch"). `compact()` came in 0.5.32; up to 0.5.39 it kept the base in memory only, and no release wrote encoding
version 2 (checked with each released wheel).

The compact module's reader and fold tests read them; they go with the 0.5.x readers in 0.7.0.
