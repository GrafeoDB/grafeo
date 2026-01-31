---
title: Fraud Detection
description: Detect fraudulent patterns using graph analysis.
tags:
  - tutorial
  - advanced
---

# Fraud Detection

Use graph patterns to detect potentially fraudulent activity.

## What You'll Learn

- Detecting suspicious connection patterns
- Finding anomalies in transaction graphs
- Using graph metrics for fraud scoring

## Common Fraud Patterns

| Pattern | Description |
|---------|-------------|
| Ring | Circular money flow |
| Burst | Rapid account activity |
| Shared Identity | Multiple accounts, same details |
| First-Party | Self-referential transactions |

## Setup

```python
import grafeo

db = grafeo.GrafeoDB()
```

## Create Transaction Data

```python
# Accounts
db.execute("""
    INSERT (:Account {id: 'A1', name: 'Alice Corp', created: '2023-01-01'})
    INSERT (:Account {id: 'A2', name: 'Bob LLC', created: '2023-01-15'})
    INSERT (:Account {id: 'A3', name: 'Carol Inc', created: '2023-02-01'})
    INSERT (:Account {id: 'A4', name: 'Suspicious Ltd', created: '2024-01-01'})
    INSERT (:Account {id: 'A5', name: 'Shell Corp', created: '2024-01-02'})
    INSERT (:Account {id: 'A6', name: 'Fake Inc', created: '2024-01-03'})
""")

# Normal transactions
db.execute("""
    MATCH (a:Account {id: 'A1'}), (b:Account {id: 'A2'})
    INSERT (a)-[:TRANSFER {amount: 1000, date: '2024-01-10'}]->(b)
""")
db.execute("""
    MATCH (a:Account {id: 'A2'}), (b:Account {id: 'A3'})
    INSERT (a)-[:TRANSFER {amount: 500, date: '2024-01-11'}]->(b)
""")

# Suspicious ring pattern: A4 -> A5 -> A6 -> A4
db.execute("""
    MATCH (a:Account {id: 'A4'}), (b:Account {id: 'A5'})
    INSERT (a)-[:TRANSFER {amount: 10000, date: '2024-01-15'}]->(b)
""")
db.execute("""
    MATCH (a:Account {id: 'A5'}), (b:Account {id: 'A6'})
    INSERT (a)-[:TRANSFER {amount: 9900, date: '2024-01-15'}]->(b)
""")
db.execute("""
    MATCH (a:Account {id: 'A6'}), (b:Account {id: 'A4'})
    INSERT (a)-[:TRANSFER {amount: 9800, date: '2024-01-15'}]->(b)
""")
```

## Detect Fraud Patterns

### Find Circular Money Flows (Rings)

```python
result = db.execute("""
    // Find triangles: A -> B -> C -> A
    MATCH (a:Account)-[:TRANSFER]->(b:Account)-[:TRANSFER]->(c:Account)-[:TRANSFER]->(a)
    WHERE a <> b AND b <> c AND a <> c
    RETURN a.name AS account1, b.name AS account2, c.name AS account3
""")

print("Suspicious circular patterns detected:")
for row in result:
    print(f"  Ring: {row['account1']} -> {row['account2']} -> {row['account3']} -> {row['account1']}")
```

### Find Accounts with High Transaction Velocity

```python
result = db.execute("""
    MATCH (a:Account)-[t:TRANSFER]->()
    WITH a, count(t) AS tx_count, sum(t.amount) AS total
    WHERE tx_count > 2
    RETURN a.name, tx_count, total
    ORDER BY tx_count DESC
""")

print("High-velocity accounts:")
for row in result:
    print(f"  {row['a.name']}: {row['tx_count']} transactions, ${row['total']} total")
```

### Find New Accounts with Large Transactions

```python
result = db.execute("""
    MATCH (a:Account)-[t:TRANSFER]->()
    WHERE a.created > '2024-01-01' AND t.amount > 5000
    RETURN a.name, a.created, t.amount
""")

print("New accounts with large transactions:")
for row in result:
    print(f"  {row['a.name']} (created {row['a.created']}): ${row['t.amount']}")
```

### Calculate Fraud Risk Score

```python
def calculate_fraud_score(db, account_id: str) -> float:
    score = 0.0

    # Check for ring participation
    result = db.execute(f"""
        MATCH (a:Account {{id: '{account_id}'}})-[:TRANSFER*3]->(a)
        RETURN count(*) AS rings
    """)
    rings = next(iter(result))['rings']
    score += rings * 30  # High weight for rings

    # Check transaction velocity
    result = db.execute(f"""
        MATCH (a:Account {{id: '{account_id}'}})-[t:TRANSFER]->()
        RETURN count(t) AS count
    """)
    tx_count = next(iter(result))['count']
    if tx_count > 5:
        score += 20

    # Check account age
    result = db.execute(f"""
        MATCH (a:Account {{id: '{account_id}'}})
        RETURN a.created AS created
    """)
    created = next(iter(result))['created']
    if created > '2024-01-01':
        score += 10  # New account

    return min(score, 100)  # Cap at 100

# Calculate scores
for account_id in ['A1', 'A4', 'A5', 'A6']:
    score = calculate_fraud_score(db, account_id)
    print(f"Account {account_id}: Risk Score = {score}")
```

## Next Steps

- [Path Queries](../user-guide/gql/paths.md) - Advanced graph traversals
- [Architecture](../architecture/index.md) - How Grafeo handles large graphs
