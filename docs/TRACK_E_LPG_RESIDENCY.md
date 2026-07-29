# Track E — LPG residency accounting + compact/lazy experiments

**Branch:** `diagnostics/lpg-memory-accounting`  
**Base:** `9781320f` (`agent/txn-session-batch-20260726`)  
**Separate from** diagnostics PR #2.

## Attribution API

Expanded `memory_usage` / property-column detail:
- map slot bytes + capacity waste
- decoded payload bytes
- string payload bytes
- compressed backing bytes
- adjacency capacity accounting

## Synthetic experiments (`lpg_residency_experiments`)

Workload: 200k string properties, 256 unique 60-ish-char tokens.

| Experiment | Estimated total | RSS anon | Point-get ns | Correctness |
|------------|----------------:|---------:|-------------:|-------------|
| Baseline | 24,640,531 B (~23.5 MiB) | 13,136 KiB | 208.7 | pass |
| A `shrink_capacities` | same (waste Δ=0 on this fill) | 13,136 (Δ0) | 499.0 | pass |
| B `force_compress_all` + lazy get | 4,018,259 B (−20.6 MiB est.) | 7,584 (Δ −5,552 KiB) | 171.0 | pass |

## Relation to ~4 GiB production residual

Phase3 heaptrack: open-stack attributed ~59% to PropertyColumn/deserialize/arcstr; residual ~4 GiB unattributed. Dictionary-compact strings show **large estimated payload reduction** on high-duplication string columns; production gain depends on unique-string cardinality of account/code graphs. Capacity shrink alone was a no-op on this synthetic fill.

## Track F inputs

- Live open residual still dominated by LPG property/string residency after jemalloc (~2.4 GiB help).
- Compact dictionary representation is a promising in-process lever before process isolation.
