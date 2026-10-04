---
title: Disk Storage Decisions
description: Architectural decisions for the tiered RAM-to-disk storage path.
tags:
  - architecture
  - storage
  - decisions
---

# Disk Storage Decisions

This page records the design decisions that constrain Grafeo's tiered RAM-to-disk storage path. These decisions are baked into the v1 implementation and will not be revisited without a corresponding architectural review.

## D1. Mmap with explicit eviction control

**Decision.** The v1 disk path uses `memmap2::Mmap` for read access, wrapped behind a `PageFetcher` trait so the mechanism can be swapped later without touching consumers.

**Why.** Pure mmap is the cheapest path to ship and works well up to roughly 100 GB on a single node. Above that, well-documented mmap pathologies (TLB shootdown under contention, unrecoverable SIGBUS, no I/O scheduling) become limiting. The `PageFetcher` indirection costs one virtual call per page touch, which is negligible compared to a page fault, and lets us swap in vmcache or an explicit pager later without touching consumers.

**Alternatives considered.**

| Option | Why rejected (for v1) |
|--------|------------------------|
| Pure mmap, no abstraction | Consumers would directly depend on mmap semantics; future swap would be a workspace-wide refactor |
| TUM vmcache (virtual-memory assisted buffer manager) | Research artifact; ~6 months of porting work without a measured workload that justifies it |
| LeanStore-style explicit buffer pool with pointer swizzling | Highest ceiling but requires bespoke index re-engineering; deferred to a later release |

**Sources.**

- [Are You Sure You Want to Use MMAP in Your DBMS? (Crotty/Leis/Pavlo, CIDR 2022)](https://db.cs.cmu.edu/mmap-cidr2022/)
- [Virtual-Memory Assisted Buffer Management (vmcache, TUM)](https://www.cs.cit.tum.de/fileadmin/w00cfj/dis/_my_direct_uploads/vmcache.pdf)
- [LeanStore: A High-Performance Storage Engine for NVMe SSDs (VLDB 2024)](https://dl.acm.org/doi/10.14778/3685800.3685915)

**Implementation.**

- Trait: `crates/grafeo-storage/src/container/page_fetcher.rs`, `PageFetcher` + `AccessHint`
- Initial impl: `MmapPageFetcher` wrapping `Arc<MmapSection>` in the same module

## D2. Packed byte-stable disk format for the RDF Ring

**Decision.** Replace the bincode'd `RdfRingSection` with a packed v2 format: sorted-dictionary string table + bit-packed wavelet level bytes (fixed little-endian) + permutation arrays. Auxiliary rank/select samples are rebuilt on open per the existing `SuccinctBitVector` pattern.

**Why.** Bincode requires a full O(n) deserialize on load, which defeats the purpose of putting the Ring on disk. A packed layout gives O(1) load + paged reads through the OS page cache. Published evidence shows roughly 3x cold-cache query latency improvement at billion-triple scale.

**Layout sketch.** A short header carrying magic, endianness marker, triple count and dictionary size; then the sorted UTF-8 string table with a `u64` offset index, the three wavelet trees (subjects, predicates, objects) as packed `u64` bitvector words in little-endian, the two permutation arrays (`spo→pos`, `spo→osp`) as `u32` arrays, and a CRC32 trailer. Endianness is locked to little-endian and validated at open.

**Sources.**

- [qEndpoint: A Novel Triple Store Architecture (Willerval/Diefenbach/Bonifati, 2024)](https://journals.sagepub.com/doi/10.3233/SW-243616)
- [HDT Technical Specification](https://www.rdfhdt.org/technical-specification/)
- [Compressed and queryable self-indexes for RDF archives (Cerdeira-Pena et al., KAIS 2023)](https://link.springer.com/article/10.1007/s10115-023-01967-7)
- [Faster Wavelet Tree Queries (arXiv 2023)](https://arxiv.org/pdf/2302.09239)

## D3. Paged HNSW (not a DiskANN port) for the vector index

**Decision.** Extend the existing HNSW topology with a paged neighbor format (4 KiB blocks of neighbor lists, indexed by node-offset table). Combine with TurboQuant so quantized codes stay hot in RAM while full embeddings spill to mmap.

The two-phase search keeps the quantized codes plus the rotation matrix and norms in RAM (roughly 245 MB per million 384-dim vectors), and pages full-precision embeddings from disk only during rescoring of the top-K candidates.

**Why.** Paged HNSW reuses the existing search code and quantization integration, costs around six weeks, and serves indexes up to roughly 10M vectors with acceptable latency. DiskANN's single-hop diameter design would deliver O(1) page faults per query but requires a new index module, build pipeline, and optimizer cost model: roughly sixteen weeks of work without a workload that today demands it.

**Future option.** A `DiskAnnIndex` could be added as a parallel `vector-index-disk` feature if real workloads justify it. The optimizer already routes through `VectorAccessor`, so a second index type is pluggable.

**Sources.**

- [TurboQuant: Asymmetric Vector Quantization (Google Research, ICLR 2026)](https://arxiv.org/abs/2504.19874)
- DiskANN family (Microsoft Research) for the deferred design alternative

## D4. Per-block zone maps in LPG columns are the keystone

**Decision.** Bump the LPG section format to v3: block-based columnar layout (4 KiB blocks default, configurable) with per-block min/max + null count + optional bloom filter. Per-label statistics stay as a higher-level summary.

**Why.** Per-label zone maps prune entire labels but cannot terminate a within-label scan early. Per-block maps unlock both block-skip and iterator early-termination. Without them, iterator bounds (D5) are dead code because Grafeo's columns are not value-sorted.

**Tradeoff.** Format-breaking change. The v2 reader stays in the codebase for one release for read-only access; v2 sections are upgraded to v3 on the next checkpoint.

**Sources.**

- [Lance Zone Map Index](https://lance.org/format/table/index/scalar/zonemap/)
- [DuckDB filter pushdown into zone maps (PR #14313)](https://github.com/duckdb/duckdb/pull/14313)
- [LSM Design Space Read Optimizations (Sarkar, ICDE 2023 tutorial)](https://cs-people.bu.edu/mathan/publications/icde23-tutorial.pdf)

## D5. Property column sort order stays insertion-order

**Decision.** Property columns remain stored in insertion order. Iterator bounds rely on per-block zone-map skip, not on a sorted physical layout.

**Why.** A sort index would deliver true binary-search early termination but doubles write cost and adds a second physical layout. Block skip captures roughly 80 percent of the win at roughly 20 percent of the cost. This decision will be revisited only if benchmarks justify the added cost.

**Sources.**

- [LSM Design Space Read Optimizations (Sarkar, ICDE 2023 tutorial)](https://cs-people.bu.edu/mathan/publications/icde23-tutorial.pdf) (informs the cost-of-sorting tradeoff)

## Status

| Decision | Status | Lands in |
|----------|--------|----------|
| D1 PageFetcher trait | Implemented | 0.5.42 |
| D2 Packed Ring format | Planned | 0.6.0 |
| D3 Paged HNSW | Planned | 0.6.0 |
| D4 Per-block zone maps | Planned | not scheduled |
| D5 Insertion-order columns | Active default | (no change) |
