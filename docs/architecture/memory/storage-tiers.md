---
title: Storage Tiers
description: How Grafeo manages the RAM ↔ disk lifecycle of section data.
tags:
  - architecture
  - memory
  - storage
---

# Storage Tiers

Grafeo's section data (LPG store, RDF store, vector indexes, ring index, etc.) lives in one of three tiers at any moment. The buffer manager moves data between tiers in response to memory pressure; configuration overrides let you pin specific sections.

## The three tiers

| Tier | Where the data lives | When |
| ---- | -------------------- | ---- |
| `InMemory` | Heap (Rust `Arc`s, `Vec`s, `HashMap`s) | Default for fresh inserts |
| `OnDisk` | Memory-mapped file in the spill directory | After a spill |
| `Uninitialized` | Section registered but holds no data yet | Right after database open with no inserts |

Reads work the same in either tier. The on-disk path serves through the OS page cache; cold pages fault in lazily.

## Default behavior (`Auto`)

By default every section is `TierOverride::Auto`. The buffer manager tracks a unified budget (default 75% of system RAM) and reacts to four pressure levels:

| Level | Allocated | Action |
| ----- | --------- | ------ |
| Normal | < 70% | No action |
| Moderate | 70-85% | Proactive eviction of cold data |
| High | 85-95% | Aggressive eviction, trigger spilling |
| Critical | > 95% | Block new allocations |

When pressure crosses High, the buffer manager picks consumers in priority order (lowest-priority first, e.g., query caches before graph storage) and asks each to spill or evict until the budget recovers.

## Explicit overrides

Use `Config::with_section_tier` to pin a specific section to a tier:

```rust
use grafeo_engine::{Config, GrafeoDB};
use grafeo_common::storage::{SectionType, TierOverride};

// Force the LPG compact base to mmap mode at database open.
let config = Config::persistent("/path/to/db.grafeo")
    .with_section_tier(SectionType::CompactStore, TierOverride::ForceDisk)
    .with_section_tier(SectionType::VectorStore, TierOverride::ForceDisk);

let db = GrafeoDB::with_config(config)?;
```

| Override | Behavior |
| -------- | -------- |
| `Auto` | Default. Buffer manager decides based on pressure. |
| `ForceDisk` | At database open, the matching section is spilled immediately. Subsequent reads serve from mmap. |
| `ForceRam` | This section is pinned in RAM. The buffer manager skips it in every spill path (pressure-driven, explicit `spill_all`, targeted `spill_consumer_by_name`). When pressure exceeds the budget and no other spillable consumers exist, allocations fail rather than spilling a ForceRam consumer. |

`ForceDisk` is targeted: only the matching section is spilled, other sections are unaffected. Configure each section type that should start on disk; the rest follow the `Auto` policy.

You can also pair the tier with a hard `max_ram` cap:

```rust
use grafeo_common::storage::SectionMemoryConfig;

let config = Config::in_memory().with_section_config(
    SectionType::VectorStore,
    SectionMemoryConfig {
        max_ram: Some(500 * 1024 * 1024), // 500 MB cap on vector index heap
        tier: TierOverride::Auto,
    },
);
```

## Reading the current tier

`db.storage_tiers()` returns the tier of every registered section consumer:

```rust
use grafeo_common::storage::SectionType;
use grafeo_common::memory::buffer::StorageTier;

let tiers = db.storage_tiers();
match tiers.get(&SectionType::VectorStore) {
    Some(StorageTier::OnDisk) => println!("vector index is on disk"),
    Some(StorageTier::InMemory) => println!("vector index is in RAM"),
    Some(StorageTier::Uninitialized) | None => println!("no vector index registered"),
}
```

This is observability only: it doesn't move data. Useful for tests, dashboards, and confirming a `ForceDisk` config took effect.

## Bringing data back: `reload_eligible`

After memory pressure drops (a long-running workload finishes, a checkpoint freed mutation overlay state) you can ask the buffer manager to bring spilled sections back into RAM:

```rust
// Reload as long as projected usage stays below 70% of the budget.
// Walks consumers highest-priority-first; stops when budget would exceed target.
let count = db.reload_eligible(0.7);
println!("reloaded {count} sections");
```

`reload_eligible` is best-effort: a per-consumer reload that fails (e.g., spill file missing) is logged-and-skipped. The walk visits highest-priority consumers first (graph storage before index buffers before query caches) so the most-important data comes back first.

The reload is synchronous; for large sections, call from a background thread.

## Spill directory

The spill directory holds the files of spilled sections and of queries that spill. It's set via `Config::with_spill_path`, or derived from the path of a file database.

A spilled vector column keeps its values: the column reads them through a cache file, so queries (`RETURN n.embedding`), vector search, checkpoints and copies (`save`, `to_memory`, backups, snapshots) see them, and the database file always holds them. The cache is never data. A reload deletes its file, and the cache files and their directories are removed when the database is dropped. A closed database still reads its spilled values until then, but spills nothing more. A cache that a crash left in `<file>.spill/cache/` is removed by the next read-write open of the same file; one left in a configured spill path or in the temp directory stays until removed by hand.

| Database | Spill directory | Vector cache |
| --- | --- | --- |
| With `Config::with_spill_path(dir)` | `dir` | `dir/grafeo-<name>-<pid>-<random>/`, one per open, so several databases can share `dir` |
| File database, read-write | `<file>.spill/` | `<file>.spill/cache/` |
| File database, read-only | `grafeo-<name>-<pid>-<random>.spill/` in the system temp directory | the same directory |
| In memory, without a spill path | none | none |

A read-only open writes nothing beside the database. Its directory in the temp directory is created at open, under a new random name, and only if no file or link of that name exists, so another user of a shared temp directory cannot prepare it; on Unix only its user can enter it. When it cannot be created (a missing or read-only temp directory, as on read-only media or in a locked-down container), the open warns and opens without spilling. On Linux the system temp directory (`/tmp`) is often RAM-backed (tmpfs), where spilling frees no memory: for large read-only workloads, set a spill path on disk.

An encrypted database (`Config::encryption`) has no spill directory, as spill files are not encrypted: none is derived, and setting one together with `encryption` is a configuration error. Its sections stay in RAM.

```rust
let config = Config::persistent("/var/lib/grafeo/db.grafeo")
    .with_spill_path("/var/lib/grafeo/spill");
```

## Tracing

When the `tracing` feature is enabled, tier transitions emit events under the `grafeo::buffer` and `grafeo::tier` targets:

| Event target / level | When |
| -------------------- | ---- |
| `grafeo::buffer` info | A consumer spills via `spill_consumer_by_name` |
| `grafeo::buffer` info | A consumer reloads in `reload_eligible` |
| `grafeo::buffer` warn | A consumer's reload returned an error |
| `grafeo::tier` info | A `ForceDisk` override fires at database open |

Wire your tracing subscriber (e.g., `tracing-subscriber`) to these targets to log or export tier transitions.
