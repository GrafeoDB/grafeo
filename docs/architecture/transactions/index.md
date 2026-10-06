---
title: Transactions
description: Transaction management and MVCC.
tags:
  - architecture
  - transactions
---

# Transactions

Grafeo provides ACID transactions with MVCC (Multi-Version Concurrency Control).

## Why MVCC with Snapshot Isolation?

Graph traversals can touch many nodes and edges in a single query. Locking all of them would cause severe contention. Snapshot isolation lets readers proceed without blocking writers, and vice versa. Write-write conflicts are detected at commit time. This is the right fit for workloads where reads vastly outnumber writes.

## How MVCC Works

Each node and edge version carries visibility metadata:

```text
VersionInfo:
├── created_epoch: EpochId          # commit epoch (PENDING before commit)
├── created_by: TransactionId       # originating transaction
├── deleted_epoch: Option<EpochId>  # epoch when deleted (if any)
├── deleted_by: Option<TransactionId>
└── data: {...}
```

A version is visible to a reader at epoch E if:

1. `created_epoch <= E` (committed before or at E)
2. Not deleted before E (`deleted_epoch` is None or `> E`)

For in-transaction reads, the system first checks ownership (`created_by == current_tx`), allowing a transaction to see its own uncommitted writes. Uncommitted versions use `EpochId::PENDING` (u64::MAX) so they are invisible to all epoch-based reads from other sessions.

Updates create new versions linked in a chain:

```text
Row v3 (current) <- Row v2 <- Row v1 (oldest)
```

## Garbage Collection

Old versions are cleaned up when no transaction needs them:

```text
Active transactions: [txn 100, txn 105]
Oldest active: 100
Safe to remove: versions with deleted_txn < 100
```

## Snapshot Isolation

- Each transaction sees a consistent snapshot
- Reads never block writes
- Writes never block reads
- Write conflicts detected at commit

### Phenomena Prevented

| Phenomenon | Prevented? |
| ---------- | ---------- |
| Dirty Read | Yes |
| Non-Repeatable Read | Yes |
| Phantom Read | Yes |
| Write Skew | Partially |

## Conflict Detection

Write-write conflicts are detected at commit time via the `WriteTracker` system. Each mutation operator (create, delete, set property, add/remove label) records the affected entity ID in the transaction's write set. When a transaction commits, the write set is compared against all transactions that committed after the current transaction's start epoch. Overlapping writes cause a `WriteConflict` error and the committing transaction is rolled back.

## Session Safety

Sessions implement `Drop` to automatically rollback any active transaction when the session goes out of scope. This prevents uncommitted data from persisting due to forgotten commits or early returns.
