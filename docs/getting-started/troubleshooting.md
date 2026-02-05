---
title: Troubleshooting
description: Common issues and solutions when using Grafeo.
tags:
  - troubleshooting
  - faq
  - errors
---

# Troubleshooting

Solutions to common issues you might encounter when using Grafeo.

## Installation Issues

### Python wheel not found for my platform

**Symptoms:** `uv add grafeo` or `pip install grafeo` fails with "no matching distribution found."

**Solutions:**

1. **Check Python version** - Grafeo requires Python 3.12 or later:
   ```bash
   python --version  # Should be 3.12+
   ```

2. **Build from source** - If no pre-built wheel exists for your platform:
   ```bash
   uv add maturin  # or: pip install maturin
   git clone https://github.com/GrafeoDB/grafeo
   cd grafeo/crates/bindings/python
   maturin develop --release
   ```

3. **Use conda** - Sometimes conda environments resolve better:
   ```bash
   conda create -n grafeo python=3.12
   conda activate grafeo
   uv add grafeo  # or: pip install grafeo
   ```

### Import error: "module 'grafeo' has no attribute..."

**Symptoms:** Import succeeds but classes are missing.

**Solution:** Ensure you have the latest version:
```bash
uv add --upgrade grafeo  # or: pip install --upgrade grafeo
```

---

## Query Errors

### "Property not found" error

**Symptoms:** Query fails with "property 'xyz' not found."

**Cause:** You're querying a property that doesn't exist on the node.

**Solutions:**

1. **Check property name** - Property names are case-sensitive:
   ```python
   # Wrong
   db.execute("MATCH (n:Person) RETURN n.Name")

   # Correct (if property is lowercase)
   db.execute("MATCH (n:Person) RETURN n.name")
   ```

2. **Use OPTIONAL** - If the property might not exist:
   ```python
   db.execute("MATCH (n:Person) RETURN n.name, n.age")  # Fails if age missing
   ```

3. **Check schema** - View existing properties:
   ```python
   schema = db.schema()
   print(schema['property_keys'])
   ```

### "Label not found" error

**Symptoms:** Query returns empty results when you expect matches.

**Solutions:**

1. **Check label spelling** - Labels are case-sensitive:
   ```python
   # These are different labels
   db.execute("MATCH (n:Person) RETURN n")   # "Person"
   db.execute("MATCH (n:PERSON) RETURN n")   # "PERSON"
   ```

2. **View existing labels**:
   ```python
   schema = db.schema()
   for label in schema['labels']:
       print(f"{label['name']}: {label['count']} nodes")
   ```

### Syntax errors in queries

**Symptoms:** "Parse error" or "Unexpected token" messages.

**Solutions:**

1. **Check GQL syntax** - Common mistakes:
   ```python
   # Wrong: Missing colon before label
   db.execute("MATCH (nPerson) RETURN n")

   # Correct
   db.execute("MATCH (n:Person) RETURN n")
   ```

2. **Escape special characters** - Use backticks for unusual names:
   ```python
   db.execute("MATCH (n:`My Label`) RETURN n")
   ```

3. **Use parameterized queries** - Avoid string interpolation:
   ```python
   # Wrong: SQL injection risk
   db.execute(f"MATCH (n:Person {{name: '{user_input}'}}) RETURN n")

   # Correct: Safe parameterized query
   db.execute("MATCH (n:Person {name: $name}) RETURN n", {"name": user_input})
   ```

---

## Performance Issues

### Queries are slow

**Symptoms:** Simple queries take seconds instead of milliseconds.

**Solutions:**

1. **Create indexes** for frequently queried properties:
   ```python
   # Create index on email property
   db.create_property_index("email")

   # Now lookups are O(1) instead of O(n)
   db.find_nodes_by_property("email", "alice@example.com")
   ```

2. **Use batch operations** instead of loops:
   ```python
   # Slow: Individual lookups
   for node_id in node_ids:
       node = db.get_node(node_id)

   # Fast: Batch operation
   nodes = db.get_nodes_by_label("Person", limit=1000)
   props = db.get_property_batch(node_ids, "name")
   ```

3. **Add LIMIT** to exploratory queries:
   ```python
   # Can be slow on large graphs
   db.execute("MATCH (n) RETURN n")

   # Much faster
   db.execute("MATCH (n) RETURN n LIMIT 100")
   ```

4. **Check SIMD support** - Grafeo uses SIMD for vector operations:
   ```python
   import grafeo
   print(grafeo.simd_support())  # Should show "avx2", "sse", "neon", or "scalar"
   ```

### Out of memory errors

**Symptoms:** Process crashes or "memory allocation failed."

**Solutions:**

1. **Use streaming** for large results:
   ```python
   # Instead of loading all at once
   result = db.execute("MATCH (n) RETURN n")

   # Process in batches with SKIP/LIMIT
   offset = 0
   batch_size = 1000
   while True:
       result = db.execute(f"MATCH (n) RETURN n SKIP {offset} LIMIT {batch_size}")
       if len(result) == 0:
           break
       process(result)
       offset += batch_size
   ```

2. **Use persistent storage** to enable spill-to-disk:
   ```python
   # In-memory (limited by RAM)
   db = GrafeoDB()

   # Persistent (can spill to disk)
   db = GrafeoDB("./mydb")
   ```

---

## Transaction Issues

### "Transaction already completed" error

**Symptoms:** Operations fail after commit or rollback.

**Solution:** Don't reuse transactions after completion:
```python
# Wrong
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.commit()
    tx.execute("INSERT (:Person {name: 'Bob'})")  # Error!

# Correct: Start a new transaction
with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Alice'})")
    tx.commit()

with db.begin_transaction() as tx:
    tx.execute("INSERT (:Person {name: 'Bob'})")
    tx.commit()
```

### Write-write conflict error

**Symptoms:** "Write conflict" when committing concurrent transactions.

**Cause:** Two transactions tried to modify the same entity.

**Solution:** Retry with exponential backoff:
```python
import time
import random

def execute_with_retry(db, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            with db.begin_transaction() as tx:
                result = tx.execute(query)
                tx.commit()
                return result
        except Exception as e:
            if "conflict" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(random.uniform(0.1, 0.5) * (2 ** attempt))
                continue
            raise
```

---

## Persistence Issues

### Database file is locked

**Symptoms:** "Database locked" or "File in use" error.

**Cause:** Another process has the database open.

**Solutions:**

1. **Close other connections**:
   ```python
   db.close()  # Always close when done
   ```

2. **Use context manager**:
   ```python
   with GrafeoDB("./mydb") as db:
       # Database is automatically closed
       pass
   ```

3. **Check for zombie processes**:
   ```bash
   # Linux/macOS
   lsof ./mydb

   # Windows
   handle.exe ./mydb
   ```

### WAL file growing too large

**Symptoms:** `.wal` file becomes very large.

**Solution:** Force a checkpoint:
```python
db.wal_checkpoint()
```

Or configure automatic checkpointing (planned for 0.4.x).

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `NodeNotFound` | Node ID doesn't exist | Check ID is valid with `db.get_node(id)` |
| `EdgeNotFound` | Edge ID doesn't exist | Check ID is valid with `db.get_edge(id)` |
| `TypeMismatch` | Property type doesn't match expected | Check property types in schema |
| `ParseError` | Invalid query syntax | Check GQL syntax documentation |
| `TransactionAborted` | Transaction was rolled back | Check for conflicts or errors |
| `IoError` | File system issue | Check permissions and disk space |

---

## Getting Help

If you can't resolve an issue:

1. **Check the documentation** at [grafeo.dev](https://grafeo.dev)
2. **Search existing issues** at [GitHub Issues](https://github.com/GrafeoDB/grafeo/issues)
3. **Open a new issue** with:
   - Grafeo version (`uv pip show grafeo` or `pip show grafeo`)
   - Python version (`python --version`)
   - Operating system
   - Minimal code to reproduce
   - Full error message and stack trace

---

## FAQ

### Can I use Grafeo in production?

Grafeo is currently at version 0.3.x and approaching production readiness. It's suitable for:

- Embedded analytics applications
- Data science workflows
- Microservices with local graph state
- Applications where you control the deployment

For high-availability or distributed scenarios, wait for 0.5.x (beta).

### How do I migrate from Neo4j?

See the [Migration Guide](migration.md) for step-by-step instructions.

### Does Grafeo support clustering?

Not yet. Grafeo is currently single-node only. Distributed deployment is planned for future versions.

### What's the maximum graph size?

Grafeo can handle graphs with billions of edges on a single machine, limited primarily by available RAM. With persistent storage, it can spill to disk for larger-than-memory workloads.

### Is Grafeo thread-safe?

Yes. The Python API uses internal locking to ensure thread safety. Multiple threads can query the same database concurrently.
