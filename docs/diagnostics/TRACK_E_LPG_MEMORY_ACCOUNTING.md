# Track E — LPG memory accounting & compact/lazy residency

**Branch:** `diagnostics/lpg-memory-accounting` (off product pin `9781320f` / `agent/txn-session-batch-20260726`)  
**Worktree:** `/data/worktrees/grafeo-lpg-accounting`  
**Staging policy:** `/data/tmp` only — never `/data/grafeo`  
**Separate from:** AM diagnostics PR #2 / Grafeo `diagnostics/close-forensics`

## Problem

Phase 3/4 forensics (AM docs): live sidecar open ~9–12 GiB RssAnon; jemalloc recovers ~2.4 GiB; residual ~4 GiB consistent with LPG properties/strings/adjacency + allocator blind spots. Prior `memory_usage()` counted property **map slots** (`capacity × sizeof(Value)`) but **omitted decoded Value heap** (especially ArcStr), so Grafeo tracked only ~2.93 GiB (~32% of sidecar Δ) while heaptrack open stacks attributed ~59%.

## Delivered APIs

| API | Where | What |
|-----|-------|------|
| `PropertyStorage::memory_detail()` | `grafeo-core` | Per-column map slots, decoded payloads, string payloads, compressed bytes, capacity waste |
| `PropertyColumn::memory_detail()` / updated `heap_memory_bytes()` | `grafeo-core` | Includes `Value::estimated_size_bytes` + compressed backing |
| `PropertyStorage::shrink_capacities()` | `grafeo-core` | `shrink_to_fit` hot maps |
| Compressed lazy `PropertyColumn::get` | `grafeo-core` | Binary-search `index_to_id` + on-demand decode (strings/bools/ints) |
| `AdjacencyIndex::capacity_memory()` / `shrink_capacities()` | `grafeo-core` | Hot used vs capacity, cold, aux, waste |
| `LpgStore::lpg_residency_detail()` / `shrink_lpg_capacities()` / `force_compress_properties()` | `grafeo-core` | Aggregated residency + experiment knobs |
| `GrafeoDB::lpg_residency_detail()` (+ shrink/compress) | `grafeo-engine` | Operator-facing surface |
| Types | `grafeo-common::memory` | `PropertyColumnMemory`, `PropertyStorageMemory`, `AdjacencyCapacityMemory`, `LpgResidencyMemory`; extended `StoreMemory` / `IndexMemory` fields |

## Measured results (2026-07-29, synthetic N=200k, 256 unique ~60 B strings)

Harness log: `/data/tmp/grafeo-lpg-accounting-track-e/grafeo-lpg-accounting-track-e-run.log`

| Metric | Baseline | A shrink | B force_compress + lazy get |
|--------|---------:|---------:|----------------------------:|
| Estimated total (bytes) | 24,640,531 | 24,640,531 | **4,018,259** (−20.6 MiB / −84%) |
| Map slots | 11,239,424 | same | (hot cleared into dict) |
| String payload (hot) | 13,400,000 | same | **≪** (moved to compressed) |
| Capacity waste | 1,439,424 | 1,439,424 (Δ0) | — |
| Compressed bytes | 0 | 0 | **4,017,152** |
| RssAnon (KiB) | 13,136 | 13,136 (Δ0) | **7,584 (Δ −5,552)** |
| Point-get ns/op | 883.5 | 618.3 | **405.9** |
| Correctness (200k gets) | PASS | PASS | **PASS** |

### Interpretation

- **A:** Post-fill `hashbrown` already sits at its load-factor capacity; `shrink_to_fit` does not reclaim meaningful RSS on this pattern. Still a safe operator knob after over-reserve / partial deletes.
- **B:** Dictionary-compact + lazy decode is the clear win: estimated residency −84%, process RssAnon −5.4 MiB on a 200k toy, **and** point-get correctness holds (fixes the prior compressed-`get` → `None` hole). Production scale (Phase 3 ~1.44 GiB arcstr open stacks) is the extrapolation target — sidecar remeasure when MemAvailable allows.
- Point-get latency **improved** after compress on this workload (dict + binary search beat hot HashMap+ArcStr clone path under the bench); treat as workload-specific, not a universal guarantee.

## How to re-run

```bash
export TMPDIR=/data/tmp
export CARGO_TARGET_DIR=/data/cargo-targets/jfrie-grafeo-lpg-accounting/target
export RUSTC_WRAPPER=sccache SCCACHE_DIR=/data/sccache
cargo test -p grafeo-core --test lpg_residency_experiments -- --nocapture
```

## Operational risk

- **Accounting change:** `heap_memory_bytes` / `StoreMemory.node_properties_bytes` now include decoded payloads → reported totals rise toward reality (not a live RSS change by itself).
- **Lazy get after compress:** integer/bool compressed point-get may decompress a whole column per miss (strings are O(log n) + dict lookup). Prefer scan/block paths for analytics; fine for sparse correctness.
- **force_compress:** still gated by internal ratio thresholds; no-op when compression does not help.
- **Do not** run full sidecar open on this host while `am-server-rs` holds ~18 GiB RSS without reclaiming MemAvailable first.

## Notes for Track F (process isolation) — do not implement here

Numbers F needs (from Phase 3/4, glibc unless noted):

| Quantity | Value | Source |
|----------|------:|--------|
| Sidecar open Δ RssAnon (glibc) | ~9.39 GiB (9,845,496 KiB) | PHASE3 |
| Live open total with account (glibc) | ~11.62 GiB | PHASE4 |
| jemalloc live open total | ~9.14 GiB (−2.36 GiB) | PHASE4D |
| Grafeo tracked pre-Track-E | ~2.93 GiB (~32%) | PHASE3 |
| heaptrack open-stack sum | ~5.58 GiB (~59% of Δ) | PHASE3 |
| Unattributed residual | ~4 GiB (~41% of Δ) | PHASE3 |
| Close wall | ~143–152 s | MITIGATION |
| Account healthy during sidecar close | yes (149/149) | MITIGATION |
| Combined same-file | **worse** (+~0.7 GiB) | PHASE4E |

Isolation estimate (unchanged): moving sidecar RSS into another process protects account latency during close and allows kill-to-reclaim; it does **not** shrink the sidecar working set itself. Track E attribution + compact/lazy wins are the in-process residual levers; F owns process boundary.

## Remaining work

- [ ] Re-measure account/sidecar open with new `lpg_residency_detail()` when MemAvailable ≥ ~14 GiB (copy staging under `/data/tmp`)
- [ ] Optional: ID→index hash for O(1) compressed get (avoid binary search / int full decompress)
- [ ] Optional: unique-ArcStr accounting via pointer set (reduce shared-string overcount)
- [ ] Wire CLI `memory` pretty-print for new StoreMemory / LpgResidency fields
