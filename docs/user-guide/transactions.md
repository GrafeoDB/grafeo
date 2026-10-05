# Transactions in Grafeo

Grafeo provides ACID transactions with **Snapshot Isolation** semantics. This guide explains how transactions work, their guarantees and important limitations to be aware of.

## Quick Start

```python
from grafeo import GrafeoDB

db = GrafeoDB()

# Explicit transaction
with db.begin_transaction() as tx:
    tx.execute("CREATE (n:Person {name: 'Alix'})")
    tx.execute("CREATE (n:Person {name: 'Gus'})")
    tx.commit()  # All changes visible atomically

# Auto-commit mode (default)
db.execute("CREATE (n:Person {name: 'Vincent'})")  # Commits immediately
```

## Isolation Levels

Grafeo supports three isolation levels, configurable per transaction:

| Level | Description |
|-------|-------------|
| `read_committed` | See committed data; non-repeatable reads possible |
| `snapshot` | Default: consistent snapshot at transaction start |
| `serializable` | Full SSI with read-write conflict detection |

```python
# Default (snapshot isolation)
tx = db.begin_transaction()

# Explicit isolation level
tx = db.begin_transaction(isolation_level="serializable")
```

```rust
use grafeo::IsolationLevel;

let tx = session.begin_transaction_with_isolation(IsolationLevel::Serializable)?;
```

## Read-Only Transactions

Mark a transaction as read-only to reject mutations at the session level:

```sql
START TRANSACTION READ ONLY
```

DDL operations like `CREATE GRAPH` and `DROP GRAPH` are also blocked in read-only transactions.

## Snapshot Isolation

Grafeo's default isolation level is **Snapshot Isolation (SI)**, which provides strong consistency while maintaining high concurrency.

### Guarantees

| Guarantee | Description |
|-----------|-------------|
| **Repeatable Reads** | Reading the same data twice in a transaction returns the same result |
| **No Dirty Reads** | Uncommitted changes from other transactions are never visible |
| **No Lost Updates** | Write-write conflicts are detected and one transaction is aborted |
| **Consistent Snapshot** | All reads see the database as of transaction start time |

### How It Works

1. When a transaction starts, it receives a **start epoch** representing the current database state
2. All reads within the transaction see data as of that epoch
3. Writes are buffered and only become visible after commit
4. At commit time, the system checks for **write-write conflicts**
5. If another committed transaction wrote to the same entity, the commit fails

## Write-Write Conflict Detection

Grafeo automatically detects when two transactions try to modify the same entity:

```python
# Thread 1
tx1 = db.begin_transaction()
tx1.execute("MATCH (n:Counter {id: 1}) SET n.value = n.value + 10")

# Thread 2 (concurrent)
tx2 = db.begin_transaction()
tx2.execute("MATCH (n:Counter {id: 1}) SET n.value = n.value + 20")

tx1.commit()  # Succeeds
tx2.commit()  # Fails with WriteConflict error
```

When a conflict is detected, the application should:
1. Catch the exception
2. Optionally retry the transaction
3. Or report the conflict

## Important Limitation: Write Skew

Snapshot Isolation does **not** prevent all anomalies. The **write skew** anomaly can occur when transactions read overlapping data but write to different entities.

### Example: The Classic Write Skew

Consider a constraint where `A + B >= 0`:

```python
# Initial: A = 50, B = 50

# Transaction 1
tx1 = db.begin_transaction()
a = tx1.execute("MATCH (n:Account {name: 'A'}) RETURN n.balance").scalar()  # 50
b = tx1.execute("MATCH (n:Account {name: 'B'}) RETURN n.balance").scalar()  # 50
# Check: 50 + 50 - 100 = 0 >= 0, OK
tx1.execute("MATCH (n:Account {name: 'A'}) SET n.balance = -50")

# Transaction 2 (concurrent, sees same snapshot)
tx2 = db.begin_transaction()
a = tx2.execute("MATCH (n:Account {name: 'A'}) RETURN n.balance").scalar()  # 50
b = tx2.execute("MATCH (n:Account {name: 'B'}) RETURN n.balance").scalar()  # 50
# Check: 50 + 50 - 100 = 0 >= 0, OK
tx2.execute("MATCH (n:Account {name: 'B'}) SET n.balance = -50")

tx1.commit()  # Success (wrote to A)
tx2.commit()  # Success (wrote to B, no conflict with A)

# Result: A = -50, B = -50, constraint violated!
```

### Workarounds for Write Skew

**Option 1: Promote Reads to Writes**

Add a dummy write to read entities to force conflict detection:

```python
tx = db.begin_transaction()
# Read both accounts
a = tx.execute("MATCH (n:Account {name: 'A'}) RETURN n").scalar()
b = tx.execute("MATCH (n:Account {name: 'B'}) RETURN n").scalar()

# "Touch" both accounts to register them in write set
tx.execute("MATCH (n:Account {name: 'A'}) SET n._touched = timestamp()")
tx.execute("MATCH (n:Account {name: 'B'}) SET n._touched = timestamp()")

# Now make actual change
tx.execute("MATCH (n:Account {name: 'A'}) SET n.balance = -50")
tx.commit()  # Will conflict if another tx touched A or B
```

**Option 2: Application-Level Validation**

Re-check constraints before commit:

```python
def withdraw(db, account, amount):
    while True:
        tx = db.begin_transaction()
        try:
            # Read current state
            a = tx.execute("MATCH (n:Account {name: 'A'}) RETURN n.balance").scalar()
            b = tx.execute("MATCH (n:Account {name: 'B'}) RETURN n.balance").scalar()

            # Make change
            if account == 'A':
                new_a = a - amount
                if new_a + b < 0:
                    raise ValueError("Would violate constraint")
                tx.execute(f"MATCH (n:Account {{name: 'A'}}) SET n.balance = {new_a}")

            tx.commit()
            return  # Success
        except WriteConflictError:
            continue  # Retry
```

**Option 3: External Locking**

Use database-external locks for critical operations:

```python
import threading

account_lock = threading.Lock()

def withdraw(db, account, amount):
    with account_lock:  # Serializes all withdrawals
        tx = db.begin_transaction()
        # ... perform withdrawal ...
        tx.commit()
```

## What Gets Rolled Back

When a transaction is rolled back (either fully or to a savepoint), all mutations made within the rollback scope are undone:

| Mutation | Rolled Back? |
| -------- | ------------ |
| `SET n.prop = value` (property update) | Yes |
| `SET n.prop = value` (new property) | Yes, property removed |
| `REMOVE n.prop` | Yes, property restored |
| `SET n:Label` (add label) | Yes, label removed |
| `REMOVE n:Label` | Yes, label restored |
| `MERGE ... ON MATCH SET` | Yes, properties restored |
| `INSERT` (new node/edge) | Yes, entity removed |
| `DELETE` (remove node/edge) | Yes, entity restored |

## Savepoints

Savepoints let you create named checkpoints within a transaction. Rolling back to a savepoint undoes only the changes made after it while preserving earlier work.

### Usage (Python)

```python
tx = db.begin_transaction()
tx.execute("MATCH (a:Account {id: 'A001'}) SET a.balance = 1000")

tx.savepoint("before_bonus")
tx.execute("MATCH (a:Account {id: 'A001'}) SET a.bonus = 500")

# Undo only the bonus, keep the balance change
tx.rollback_to_savepoint("before_bonus")

tx.commit()  # balance = 1000, no bonus property
```

### Usage (Rust)

```rust
let mut session = db.session();
session.begin_transaction()?;

session.execute("MATCH (a:Account {id: 'A001'}) SET a.balance = 1000")?;

session.savepoint("before_bonus")?;

session.execute("MATCH (a:Account {id: 'A001'}) SET a.bonus = 500")?;

// Undo only the bonus, keep the balance change
session.rollback_to_savepoint("before_bonus")?;

// Release discards the savepoint but keeps changes
// session.release_savepoint("before_bonus")?;

session.commit()?;  // balance = 1000, no bonus property
```

### Usage (Python via GQL)

```python
tx = db.begin_transaction()

tx.execute("MATCH (a:Account {id: 'A001'}) SET a.balance = 1000")
tx.execute("SAVEPOINT before_bonus")
tx.execute("MATCH (a:Account {id: 'A001'}) SET a.bonus = 500")

# Undo only the bonus, keep the balance change
tx.execute("ROLLBACK TO SAVEPOINT before_bonus")

tx.commit()  # balance = 1000, no bonus property
```

### Savepoint Rules

1. Names must be unique within a transaction
2. Rolling back to a savepoint also releases all savepoints created after it
3. A full `ROLLBACK` undoes everything, including changes before any savepoints

## Transaction Lifecycle

### States

| State | Description |
|-------|-------------|
| `Active` | Transaction is in progress, can read and write |
| `Committed` | Transaction completed successfully, changes visible |
| `Aborted` | Transaction was rolled back, changes discarded |

### Best Practices

1. **Keep transactions short**: Long transactions increase conflict probability
2. **Batch related changes**: Group related writes in a single transaction
3. **Handle conflicts gracefully**: Implement retry logic for write conflicts
4. **Use auto-commit for single operations**: Simpler and equally safe
5. **Don't hold transactions open during user interaction**: Risk of blocking GC

### Session Drop Safety

If a session is dropped (goes out of scope) while a transaction is active, the transaction is automatically rolled back. This prevents data corruption from forgotten commits:

```python
def do_work(db):
    tx = db.begin_transaction()
    tx.execute("CREATE (n:Temp {data: 'test'})")
    # Oops, forgot to commit!
    # When tx goes out of scope, the transaction is rolled back automatically

do_work(db)
# No Temp nodes exist, the uncommitted data was discarded
```

In Rust, the same behavior applies when a `Session` is dropped:

```rust
{
    let mut session = db.session();
    session.begin_transaction()?;
    session.execute("INSERT (:Temp {data: 'test'})")?;
    // session dropped here, transaction auto-rolled back
}
```

## Two-Phase Commit

!!! note "Rust-only API"
    `prepare_commit()` is available on the Rust `Session` type. It is not exposed in the Python, Node.js, or WASM bindings.

For workflows that need to inspect pending mutations before finalizing, use `prepare_commit()`:

```rust
let mut session = db.session();
session.begin_transaction()?;
session.execute("INSERT (:Person {name: 'Alix'})")?;
session.execute("MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'}) INSERT (a)-[:KNOWS]->(b)")?;

// Inspect what would be committed
let prepared = session.prepare_commit()?;
println!("Pending: {} nodes, {} edges", prepared.node_count(), prepared.edge_count());

// Finalize
prepared.commit()?;
```

## GQL Transaction Syntax

Transactions can also be managed via GQL statements:

```sql
-- Start a transaction
START TRANSACTION

-- Start read-only
START TRANSACTION READ ONLY

-- Commit
COMMIT

-- Rollback
ROLLBACK

-- Savepoints
SAVEPOINT my_save
ROLLBACK TO SAVEPOINT my_save
RELEASE SAVEPOINT my_save
```

## API Reference

### Python

```python
# Start explicit transaction
tx = db.begin_transaction()

# With isolation level
tx = db.begin_transaction(isolation_level="serializable")

# Execute within transaction
result = tx.execute("MATCH (n) RETURN n")

# Commit changes
tx.commit()

# Or rollback
tx.rollback()

# Context manager (auto-rollback on exception)
with db.begin_transaction() as tx:
    tx.execute("CREATE (n:Test)")
    tx.commit()

# Savepoints via GQL (no dedicated Python methods)
tx = db.begin_transaction()
tx.execute("SAVEPOINT sp1")
tx.execute("ROLLBACK TO SAVEPOINT sp1")
tx.execute("RELEASE SAVEPOINT sp1")
```

### Rust

```rust
// Start transaction
session.begin_transaction()?;

// With isolation level
session.begin_transaction_with_isolation(IsolationLevel::Serializable)?;

// Execute queries
let result = session.execute("MATCH (n) RETURN n")?;

// Commit
session.commit()?;

// Or rollback
session.rollback()?;

// Savepoints
session.savepoint("sp1")?;
session.rollback_to_savepoint("sp1")?;
session.release_savepoint("sp1")?;
```

## Garbage Collection

Grafeo automatically garbage collects old transaction metadata and version chains:

- Aborted transactions are cleaned up immediately
- Committed transaction metadata is retained until no active transaction can see it
- Version chains are pruned based on the oldest active transaction's start epoch
- GC runs every N commits (default 100, configurable); manual trigger via `db.gc()` in Rust

!!! note
    The `gc()` method is only available on the Rust `GrafeoDB` type. Python and other bindings rely on automatic GC.

## Error Codes

Transaction errors use standardized `GRAFEO-TXXX` codes:

| Code | Description | Retryable? |
|------|-------------|------------|
| `GRAFEO-T001` | Write-write conflict detected | Yes |
| `GRAFEO-T002` | Transaction timeout | Yes |
| `GRAFEO-T003` | Read-only transaction attempted mutation | No |
| `GRAFEO-T004` | Invalid transaction state (e.g., already committed or rolled back) | No |
| `GRAFEO-T005` | Serialization failure (SSI violation) | No |
| `GRAFEO-T006` | Deadlock detected | Yes |
| `GRAFEO-T007` | Database closed: a write after `close()` of a persistent database | No |

In Rust, error codes expose an `is_retryable()` method to indicate whether the operation can be safely retried. This method is not currently exposed in the Python bindings.
