"""
Fraud Detection with Graph Analysis

This example demonstrates graph-based fraud detection using Grafeo.
We'll build a transaction network and identify suspicious patterns.

Run with: marimo run fraud_detection.py

Requirements:
    pip install grafeo anywidget-graph marimo
"""

import marimo

__generated_with = "0.19.7"
app = marimo.App(width="full")


@app.cell
def __():
    import marimo as mo

    mo.md("""
    # Fraud Detection with Graph Analysis

    This notebook demonstrates how to detect fraudulent patterns using graph analysis:

    1. **Transaction networks** - Model accounts and transactions as a graph
    2. **Pattern detection** - Find suspicious connection patterns
    3. **Risk scoring** - Use centrality metrics to identify high-risk nodes
    4. **Visualization** - Explore the network interactively

    Common fraud patterns we'll detect:
    - **Money laundering rings** - Circular transaction patterns
    - **Mule accounts** - Accounts that quickly move money through
    - **Burst patterns** - Sudden spikes in transaction activity
    """)
    return (mo,)


@app.cell
def __():
    import random
    from datetime import datetime, timedelta

    from grafeo import GrafeoDB

    # Create database
    db = GrafeoDB()

    # Seed for reproducibility
    random.seed(42)

    # Create legitimate accounts
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

    # Create legitimate transactions
    for _ in range(100):
        sender = random.choice(accounts)
        receiver = random.choice([a for a in accounts if a.id != sender.id])
        amount = random.uniform(10, 500)
        db.create_edge(
            sender.id,
            receiver.id,
            "TRANSFERRED",
            {
                "amount": round(amount, 2),
                "currency": "USD",
                "days_ago": random.randint(1, 90),
                "flagged": False,
            },
        )

    print(f"Created {len(accounts)} accounts with legitimate transactions")
    return GrafeoDB, accounts, datetime, db, random, timedelta


@app.cell
def __(accounts, db, random):
    # Add fraudulent patterns

    # Pattern 1: Money laundering ring (circular transactions)
    ring_accounts = []
    for i in range(5):
        account = db.create_node(
            ["Account", "Suspicious"],
            {
                "account_id": f"RING{i:02d}",
                "type": "personal",
                "created_days_ago": random.randint(5, 15),  # Recently created
                "verified": False,
                "country": random.choice(["US", "UK"]),
            },
        )
        ring_accounts.append(account)

    # Create circular pattern: A -> B -> C -> D -> E -> A
    for i in range(len(ring_accounts)):
        next_i = (i + 1) % len(ring_accounts)
        db.create_edge(
            ring_accounts[i].id,
            ring_accounts[next_i].id,
            "TRANSFERRED",
            {"amount": 9999.00, "currency": "USD", "days_ago": 2, "flagged": False},  # Just under reporting threshold
        )

    # Pattern 2: Mule account (many inputs, few outputs)
    mule = db.create_node(
        ["Account", "Suspicious"],
        {
            "account_id": "MULE01",
            "type": "personal",
            "created_days_ago": 10,
            "verified": False,
            "country": "US",
        },
    )

    # Many small deposits from different accounts
    for i in range(15):
        sender = random.choice(accounts)
        db.create_edge(
            sender.id,
            mule.id,
            "TRANSFERRED",
            {"amount": random.uniform(100, 500), "currency": "USD", "days_ago": random.randint(1, 5), "flagged": False},
        )

    # One large withdrawal
    external = db.create_node(
        ["Account", "External"],
        {"account_id": "EXT001", "type": "external", "created_days_ago": 1, "verified": False, "country": "XX"},
    )
    db.create_edge(
        mule.id, external.id, "TRANSFERRED", {"amount": 5000.00, "currency": "USD", "days_ago": 1, "flagged": False}
    )

    print("Added fraudulent patterns: ring (5 accounts) and mule (1 account)")
    return external, mule, ring_accounts


@app.cell
def __(db, mo):
    # Detect circular patterns (potential money laundering)

    # Find accounts that are part of cycles
    cycle_result = db.execute("""
        MATCH (a:Account)-[:TRANSFERRED]->(b:Account)-[:TRANSFERRED]->(c:Account)-[:TRANSFERRED]->(a)
        RETURN DISTINCT a.account_id as account1, b.account_id as account2, c.account_id as account3
    """)

    cycle_accounts = set()
    for row in cycle_result:
        cycle_accounts.add(row["account1"])
        cycle_accounts.add(row["account2"])
        cycle_accounts.add(row["account3"])

    mo.md(f"""
    ## Pattern 1: Circular Transactions (Money Laundering)

    Found **{len(cycle_accounts)}** accounts involved in circular transaction patterns.

    Accounts in cycles: {', '.join(sorted(cycle_accounts)) if cycle_accounts else 'None'}

    **Why this matters:** Circular transactions are a classic money laundering technique
    where funds are moved through multiple accounts to obscure their origin.
    """)
    return cycle_accounts, cycle_result


@app.cell
def __(db, mo):
    # Detect mule accounts (high in-degree, low out-degree)

    mule_result = db.execute("""
        MATCH (a:Account)
        OPTIONAL MATCH (a)<-[incoming:TRANSFERRED]-()
        OPTIONAL MATCH (a)-[outgoing:TRANSFERRED]->()
        WITH a, count(DISTINCT incoming) as in_count, count(DISTINCT outgoing) as out_count
        WHERE in_count > 5 AND out_count <= 2
        RETURN a.account_id as account, in_count, out_count
        ORDER BY in_count DESC
    """)

    mule_rows = [
        f"| {row['account']} | {row['in_count']} | {row['out_count']} |" for row in mule_result if row["account"]
    ]

    mo.md(f"""
    ## Pattern 2: Mule Accounts

    Accounts with many incoming but few outgoing transactions:

    | Account | Incoming | Outgoing |
    |---------|----------|----------|
    {chr(10).join(mule_rows) if mule_rows else "| None found | - | - |"}

    **Why this matters:** Mule accounts collect funds from multiple sources
    and then transfer them out in bulk, often to obscure the money trail.
    """)
    return mule_result, mule_rows


@app.cell
def __(db, mo):
    # Detect recently created accounts with high activity

    burst_result = db.execute("""
        MATCH (a:Account)
        WHERE a.created_days_ago < 30
        OPTIONAL MATCH (a)-[t:TRANSFERRED]-()
        WITH a, count(t) as tx_count
        WHERE tx_count > 3
        RETURN a.account_id as account, a.created_days_ago as age_days, tx_count
        ORDER BY tx_count DESC
    """)

    burst_rows = [
        f"| {row['account']} | {row['age_days']} | {row['tx_count']} |"
        for row in burst_result
        if row["account"]
    ]

    mo.md(f"""
    ## Pattern 3: New Account Burst Activity

    Recently created accounts with unusually high transaction counts:

    | Account | Age (days) | Transactions |
    |---------|------------|--------------|
    {chr(10).join(burst_rows) if burst_rows else "| None found | - | - |"}

    **Why this matters:** Fraudsters often create new accounts and
    quickly use them for illicit transactions before detection.
    """)
    return burst_result, burst_rows


@app.cell
def __(db, mo):
    # Calculate risk scores using PageRank on suspicious subgraph

    # First, find all suspicious accounts
    suspicious_result = db.execute("""
        MATCH (a:Suspicious)
        RETURN id(a) as id, a.account_id as account
    """)

    suspicious_ids = {row["id"]: row["account"] for row in suspicious_result}

    # Run PageRank on full graph
    pagerank = db.algorithms.pagerank(damping=0.85)

    # Get scores for suspicious accounts
    suspicious_scores = [(suspicious_ids[node_id], score) for node_id, score in pagerank.items() if node_id in suspicious_ids]

    suspicious_scores.sort(key=lambda x: x[1], reverse=True)

    risk_rows = [f"| {account} | {score:.4f} | {'HIGH' if score > 0.05 else 'MEDIUM'} |" for account, score in suspicious_scores]

    mo.md(f"""
    ## Risk Scoring with PageRank

    PageRank identifies accounts that are central to transaction flows:

    | Account | PageRank | Risk Level |
    |---------|----------|------------|
    {chr(10).join(risk_rows) if risk_rows else "| None | - | - |"}

    **Interpretation:**
    - Higher PageRank = more central to money flow
    - Suspicious accounts with high centrality are higher risk
    """)
    return pagerank, risk_rows, suspicious_ids, suspicious_result, suspicious_scores


@app.cell
def __(db):
    from anywidget_graph import Graph

    # Visualize the suspicious subgraph
    viz_result = db.execute("""
        MATCH (a:Account)-[t:TRANSFERRED]->(b:Account)
        WHERE a:Suspicious OR b:Suspicious
        RETURN a, t, b
    """)

    nodes = viz_result.nodes()
    edges = viz_result.edges()

    # Convert to graph format
    graph_nodes = []
    seen_ids = set()
    for node in nodes:
        if node.id not in seen_ids:
            seen_ids.add(node.id)
            labels = node.labels
            props = node.properties
            is_suspicious = "Suspicious" in labels
            graph_nodes.append({
                "id": str(node.id),
                "label": props.get("account_id", f"Node {node.id}"),
                "group": "Suspicious" if is_suspicious else "Normal",
                "properties": props,
            })

    graph_edges = []
    for edge in edges:
        graph_edges.append({
            "source": str(edge.source_id),
            "target": str(edge.target_id),
            "label": f"${edge.properties.get('amount', 0):.0f}",
            "properties": edge.properties,
        })

    # Create widget
    fraud_graph = Graph(
        nodes=graph_nodes,
        edges=graph_edges,
        height=500,
    )

    fraud_graph
    return Graph, edges, fraud_graph, graph_edges, graph_nodes, nodes, seen_ids, viz_result


@app.cell
def __(mo):
    mo.md("""
    ## Suspicious Transaction Network

    The visualization shows accounts connected to suspicious activity:
    - **Red nodes**: Flagged as suspicious
    - **Blue nodes**: Normal accounts
    - **Edge labels**: Transaction amounts

    Notice the ring pattern and the mule account's star pattern!
    """)
    return ()


@app.cell
def __(cycle_accounts, db, external, mo, mule, ring_accounts):
    # Flag suspicious accounts for investigation

    # Mark ring accounts
    for account in ring_accounts:
        db.set_node_property(account.id, "risk_level", "HIGH")
        db.set_node_property(account.id, "flag_reason", "circular_transactions")

    # Mark mule account
    db.set_node_property(mule.id, "risk_level", "HIGH")
    db.set_node_property(mule.id, "flag_reason", "mule_pattern")

    # Mark external destination
    db.set_node_property(external.id, "risk_level", "CRITICAL")
    db.set_node_property(external.id, "flag_reason", "suspicious_destination")

    mo.md(f"""
    ## Investigation Summary

    Based on our analysis, the following accounts require investigation:

    **HIGH RISK - Money Laundering Ring:**
    {chr(10).join(f'- {a.properties["account_id"]}' for a in ring_accounts)}

    **HIGH RISK - Mule Account:**
    - {mule.properties["account_id"]}

    **CRITICAL - Suspicious Destination:**
    - {external.properties["account_id"]}

    **Total Suspicious Accounts:** {len(cycle_accounts) + 2}

    All flagged accounts have been marked with `risk_level` and `flag_reason` properties
    for follow-up investigation.
    """)
    return ()


@app.cell
def __(db, mo):
    # Final statistics
    stats = db.detailed_stats()

    total_suspicious = db.execute("MATCH (a:Suspicious) RETURN count(a) as count")
    suspicious_count = list(total_suspicious)[0]["count"]

    total_flagged = db.execute("MATCH (a:Account) WHERE a.risk_level IS NOT NULL RETURN count(a) as count")
    flagged_count = list(total_flagged)[0]["count"]

    mo.md(f"""
    ## Final Statistics

    | Metric | Value |
    |--------|-------|
    | Total Accounts | {stats['node_count']} |
    | Total Transactions | {stats['edge_count']} |
    | Suspicious Accounts | {suspicious_count} |
    | Flagged for Review | {flagged_count} |
    | Detection Rate | {flagged_count / stats['node_count'] * 100:.1f}% |

    This demonstrates how graph analysis can efficiently identify fraud patterns
    that would be difficult to detect with traditional SQL queries.
    """)
    return flagged_count, stats, suspicious_count, total_flagged, total_suspicious


if __name__ == "__main__":
    app.run()
