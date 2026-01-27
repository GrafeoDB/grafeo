---
title: Adjacency Lists
description: Chunked adjacency list storage for edges.
tags:
  - architecture
  - storage
---

# Adjacency Lists

Edges are stored in chunked adjacency lists optimized for traversal.

## Structure

```
Node 1 adjacency:
┌────────────────────────────────────────┐
│ Outgoing: [Node2, Node3, Node5, ...]   │
│ Incoming: [Node4, Node7, ...]          │
└────────────────────────────────────────┘
```

## Chunked Storage

Large adjacency lists are split into chunks:

```
High-degree node (1M edges):
├── Chunk 0: edges 0-1023
├── Chunk 1: edges 1024-2047
├── Chunk 2: edges 2048-3071
└── ... (more chunks)
```

## Benefits

- **Incremental updates** - Add to latest chunk
- **Parallel scans** - Process chunks in parallel
- **Memory efficiency** - Load chunks on demand
- **Cache friendly** - Sequential access within chunks

## Delta Buffer

Recent changes are buffered before merging:

```
Frozen chunks: [Chunk 0, Chunk 1, Chunk 2]
Delta buffer:  [new_edge_1, new_edge_2, ...]

On query: Merge frozen + delta
On checkpoint: Merge delta into new chunk
```
