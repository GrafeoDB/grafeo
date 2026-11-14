---
title: Adjacency Lists
description: Chunked adjacency list storage for edges.
tags:
  - architecture
  - storage
---

# Adjacency Lists

Edges are stored in chunked adjacency lists optimized for traversal.

## Why Chunked Lists?

CSR (Compressed Sparse Row) is read-optimized but requires a full rebuild on any insertion. Adjacency matrices waste memory on sparse graphs. Chunked lists allow incremental updates (append to the latest chunk or delta buffer) while still supporting parallel scans across chunks and compression of cold chunks. This fits a database where data changes frequently.

## Structure

Each node maintains separate outgoing and incoming adjacency lists:

```text
Node 1 adjacency:
┌────────────────────────────────────────┐
│ Outgoing: [Node2, Node3, Node5, ...]   │
│ Incoming: [Node4, Node7, ...]          │
└────────────────────────────────────────┘
```

## Three-Tier Storage

Adjacency data moves through three tiers as it ages:

```text
Delta buffer (SmallVec, ≤64 entries):
  Recent insertions, not yet compacted

Hot chunks (uncompressed, 64 edges/chunk):
  Compacted from delta buffer, mutable

Cold chunks (compressed, immutable):
  Sorted + delta-encoded + bit-packed
```

When the delta buffer reaches 64 entries, it is compacted into a hot chunk. When hot chunks exceed 4, the oldest are compressed into cold storage.

## Cold Chunk Compression

Cold chunks sort entries by destination node ID, then apply delta encoding and bit-packing. This exploits locality in the destination ID space for high compression ratios.

A skip index stores (min_destination, max_destination) per cold chunk, enabling O(log n) point lookups without decompressing chunks that can't contain the target.

## Soft Deletion

Deletions use tombstones stored in a hash set, avoiding the need to recompact chunks when edges are removed.
