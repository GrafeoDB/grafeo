---
title: Fraud Detection (Interactive)
description: Graph-based fraud detection with pattern analysis and risk scoring.
tags:
  - example
  - fraud-detection
  - algorithms
---

# Fraud Detection

Model a transaction network, inject known fraud patterns, detect them with graph queries and score risk with PageRank.

!!! tip "Run it locally"

    ```bash
    marimo run examples/fraud_detection.py
    ```

    **Requirements:** `grafeo`, `anywidget-graph`, `marimo`

## Build the Transaction Network

Create 50 legitimate accounts with 100 normal transactions:

```python
import random
from grafeo import GrafeoDB

db = GrafeoDB()
random.seed(42)

# Legitimate accounts
accounts = []
for i in range(50):
    account_type = random.choice(["personal", "business"])
    account = db.create_node(
        ["Account", account_type.capitalize()],
        {
            "account_id": f"ACC{i:04d}",
            "type": account_type,
            "created_days_ago": random.randint(30, 1000),
            "verified": random.random() > 0.2,
            "country": random.choice(["US", "UK", "DE", "FR", "CA"]),
        },
    )
    accounts.append(account)

# Normal transactions
for _ in range(100):
    sender = random.choice(accounts)
    receiver = random.choice([a for a in accounts if a.id != sender.id])
    db.create_edge(
        sender.id, receiver.id, "TRANSFERRED",
        {
            "amount": round(random.uniform(10, 500), 2),
            "currency": "USD",
            "days_ago": random.randint(1, 90),
            "flagged": False,
        },
    )
```

## Inject Fraud Patterns

### Pattern 1: Money Laundering Ring

Five recently created accounts passing money in a circle, each transfer just under the reporting threshold:

```python
ring_accounts = []
for i in range(5):
    account = db.create_node(
        ["Account", "Suspicious"],
        {
            "account_id": f"RING{i:02d}",
            "type": "personal",
            "created_days_ago": random.randint(5, 15),
            "verified": False,
            "country": random.choice(["US", "UK"]),
        },
    )
    ring_accounts.append(account)

# Circular: RING00 -> RING01 -> ... -> RING04 -> RING00
for i in range(len(ring_accounts)):
    next_i = (i + 1) % len(ring_accounts)
    db.create_edge(
        ring_accounts[i].id, ring_accounts[next_i].id, "TRANSFERRED",
        {"amount": 9999.00, "currency": "USD", "days_ago": 2, "flagged": False},
    )
```

### Pattern 2: Mule Account

One account receives many small deposits then makes a single large withdrawal:

```python
mule = db.create_node(
    ["Account", "Suspicious"],
    {"account_id": "MULE01", "type": "personal", "created_days_ago": 10,
     "verified": False, "country": "US"},
)

# 15 small deposits from random accounts
for _ in range(15):
    sender = random.choice(accounts)
    db.create_edge(
        sender.id, mule.id, "TRANSFERRED",
        {"amount": random.uniform(100, 500), "currency": "USD",
         "days_ago": random.randint(1, 5), "flagged": False},
    )

# One large withdrawal to an external account
external = db.create_node(
    ["Account", "External"],
    {"account_id": "EXT001", "type": "external", "created_days_ago": 1,
     "verified": False, "country": "XX"},
)
db.create_edge(
    mule.id, external.id, "TRANSFERRED",
    {"amount": 5000.00, "currency": "USD", "days_ago": 1, "flagged": False},
)
```

## Detect Circular Transactions

Find accounts participating in 3-hop cycles:

```python
cycle_result = db.execute("""
    MATCH (a:Account)-[:TRANSFERRED]->(b:Account)
          -[:TRANSFERRED]->(c:Account)-[:TRANSFERRED]->(a)
    RETURN DISTINCT a.account_id AS account1,
                    b.account_id AS account2,
                    c.account_id AS account3
""")

cycle_accounts = set()
for row in cycle_result:
    cycle_accounts.add(row["account1"])
    cycle_accounts.add(row["account2"])
    cycle_accounts.add(row["account3"])

print(f"Accounts in circular patterns: {sorted(cycle_accounts)}")
```

```title="Output"
Accounts in circular patterns: ['RING00', 'RING01', 'RING02', 'RING03', 'RING04']
```

## Detect Mule Accounts

Accounts with many incoming but few outgoing transfers:

```python
mule_result = db.execute("""
    MATCH (a:Account)
    OPTIONAL MATCH (a)<-[incoming:TRANSFERRED]-()
    OPTIONAL MATCH (a)-[outgoing:TRANSFERRED]->()
    WITH a, count(DISTINCT incoming) AS in_count, count(DISTINCT outgoing) AS out_count
    WHERE in_count > 5 AND out_count <= 2
    RETURN a.account_id AS account, in_count, out_count
    ORDER BY in_count DESC
""")

for row in mule_result:
    if row["account"]:
        print(f"{row['account']}: {row['in_count']} in, {row['out_count']} out")
```

```title="Output"
MULE01: 15 in, 1 out
```

## Detect New-Account Burst Activity

Recently created accounts with unusually high transaction counts:

```python
burst_result = db.execute("""
    MATCH (a:Account)
    WHERE a.created_days_ago < 30
    OPTIONAL MATCH (a)-[t:TRANSFERRED]-()
    WITH a, count(t) AS tx_count
    WHERE tx_count > 3
    RETURN a.account_id AS account, a.created_days_ago AS age_days, tx_count
    ORDER BY tx_count DESC
""")

for row in burst_result:
    if row["account"]:
        print(f"{row['account']}: {row['age_days']} days old, {row['tx_count']} transactions")
```

## Risk Scoring with PageRank

Use PageRank to find the most central nodes in the suspicious subgraph:

```python
suspicious_result = db.execute("""
    MATCH (a:Suspicious)
    RETURN id(a) AS id, a.account_id AS account
""")
suspicious_ids = {row["id"]: row["account"] for row in suspicious_result}

pagerank = db.algorithms.pagerank(damping=0.85)

suspicious_scores = [
    (suspicious_ids[nid], score)
    for nid, score in pagerank.items()
    if nid in suspicious_ids
]
suspicious_scores.sort(key=lambda x: x[1], reverse=True)

for account, score in suspicious_scores:
    level = "HIGH" if score > 0.05 else "MEDIUM"
    print(f"{account}: PageRank={score:.4f}  Risk={level}")
```

Higher PageRank means the account is more central to transaction flows, making it a higher-priority target for investigation.

## Visualize the Suspicious Subgraph

```python
from anywidget_graph import Graph

viz_result = db.execute("""
    MATCH (a:Account)-[t:TRANSFERRED]->(b:Account)
    WHERE a:Suspicious OR b:Suspicious
    RETURN a, t, b
""")

nodes = viz_result.nodes()
edges = viz_result.edges()

graph_nodes, seen_ids = [], set()
for node in nodes:
    if node.id not in seen_ids:
        seen_ids.add(node.id)
        is_suspicious = "Suspicious" in node.labels
        graph_nodes.append({
            "id": str(node.id),
            "label": node.properties.get("account_id", f"Node {node.id}"),
            "group": "Suspicious" if is_suspicious else "Normal",
            "properties": node.properties,
        })

graph_edges = [
    {
        "source": str(e.source_id),
        "target": str(e.target_id),
        "label": f"${e.properties.get('amount', 0):.0f}",
        "properties": e.properties,
    }
    for e in edges
]

fraud_graph = Graph(nodes=graph_nodes, edges=graph_edges, height=500)
fraud_graph
```

Suspicious accounts appear in red, normal accounts in blue. Notice the ring pattern (circular cluster) and the mule account's star pattern (many inbound, one outbound).

## Next Steps

- [Fraud Detection tutorial](../fraud-detection.md) for a step-by-step walkthrough
- [Path Queries](../../user-guide/gql/paths.md) for advanced graph traversals
- [Graph Visualization example](graph-visualization.md) for social network analysis
