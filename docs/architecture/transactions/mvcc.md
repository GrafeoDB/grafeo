---
title: MVCC
description: Multi-Version Concurrency Control.
tags:
  - architecture
  - transactions
---

# MVCC

Multi-Version Concurrency Control enables concurrent reads and writes.

## How MVCC Works

Each row has version metadata:

```
Row:
├── created_txn: 100
├── deleted_txn: NULL (or txn that deleted)
└── data: {...}
```

## Visibility Rules

A row is visible to transaction T if:

1. `created_txn < T` (created before T started)
2. `deleted_txn IS NULL OR deleted_txn > T` (not deleted before T)

## Version Chain

Updates create new versions:

```
Row v3 (current) <- Row v2 <- Row v1 (oldest)
```

## Garbage Collection

Old versions are cleaned up when no transaction needs them:

```
Active transactions: [txn 100, txn 105]
Oldest active: 100
Safe to remove: versions with deleted_txn < 100
```
