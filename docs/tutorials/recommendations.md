---
title: Recommendation Engine
description: Build a recommendation system using graph patterns.
tags:
  - tutorial
  - intermediate
---

# Recommendation Engine

Build a product recommendation system using collaborative filtering on a graph.

## What You'll Learn

- Modeling user-product interactions
- Collaborative filtering with graph patterns
- Scoring and ranking recommendations

## The Approach

We'll use the principle: "Users who bought X also bought Y" implemented as graph traversals.

## Setup

```python
import grafeo

db = grafeo.GrafeoDB()
```

## Create the Data Model

```python
# Products
db.execute("""
    INSERT (:Product {id: 'P1', name: 'Laptop', category: 'Electronics', price: 999})
    INSERT (:Product {id: 'P2', name: 'Headphones', category: 'Electronics', price: 149})
    INSERT (:Product {id: 'P3', name: 'Mouse', category: 'Electronics', price: 49})
    INSERT (:Product {id: 'P4', name: 'Keyboard', category: 'Electronics', price: 79})
    INSERT (:Product {id: 'P5', name: 'Monitor', category: 'Electronics', price: 299})
    INSERT (:Product {id: 'P6', name: 'USB Hub', category: 'Electronics', price: 29})
""")

# Users
db.execute("""
    INSERT (:User {id: 'U1', name: 'Alice'})
    INSERT (:User {id: 'U2', name: 'Bob'})
    INSERT (:User {id: 'U3', name: 'Carol'})
    INSERT (:User {id: 'U4', name: 'Dave'})
    INSERT (:User {id: 'U5', name: 'Eve'})
""")
```

## Create Purchase History

```python
# Alice bought Laptop, Headphones, Mouse
db.execute("""
    MATCH (u:User {id: 'U1'}), (p:Product {id: 'P1'})
    INSERT (u)-[:PURCHASED {date: '2024-01-10'}]->(p)
""")
db.execute("""
    MATCH (u:User {id: 'U1'}), (p:Product {id: 'P2'})
    INSERT (u)-[:PURCHASED {date: '2024-01-11'}]->(p)
""")
db.execute("""
    MATCH (u:User {id: 'U1'}), (p:Product {id: 'P3'})
    INSERT (u)-[:PURCHASED {date: '2024-01-12'}]->(p)
""")

# Bob bought Laptop, Keyboard, Mouse
db.execute("""
    MATCH (u:User {id: 'U2'}), (p:Product {id: 'P1'})
    INSERT (u)-[:PURCHASED]->(p)
""")
db.execute("""
    MATCH (u:User {id: 'U2'}), (p:Product {id: 'P4'})
    INSERT (u)-[:PURCHASED]->(p)
""")
db.execute("""
    MATCH (u:User {id: 'U2'}), (p:Product {id: 'P3'})
    INSERT (u)-[:PURCHASED]->(p)
""")

# Carol bought Laptop, Monitor, USB Hub
db.execute("""
    MATCH (u:User {id: 'U3'}), (p:Product {id: 'P1'})
    INSERT (u)-[:PURCHASED]->(p)
""")
db.execute("""
    MATCH (u:User {id: 'U3'}), (p:Product {id: 'P5'})
    INSERT (u)-[:PURCHASED]->(p)
""")
db.execute("""
    MATCH (u:User {id: 'U3'}), (p:Product {id: 'P6'})
    INSERT (u)-[:PURCHASED]->(p)
""")
```

## Generate Recommendations

### Products Frequently Bought Together

```python
result = db.execute("""
    MATCH (p1:Product {id: 'P1'})<-[:PURCHASED]-(u:User)-[:PURCHASED]->(p2:Product)
    WHERE p1 <> p2
    RETURN p2.name AS recommended, count(u) AS buyers
    ORDER BY buyers DESC
    LIMIT 5
""")

print("Frequently bought with 'Laptop':")
for row in result:
    print(f"  {row['recommended']} ({row['buyers']} buyers)")
```

### Personalized Recommendations for a User

```python
def get_recommendations(db, user_id: str, limit: int = 5):
    result = db.execute(f"""
        // Find products similar users bought but this user hasn't
        MATCH (u:User {{id: '{user_id}'}})-[:PURCHASED]->(p:Product)
              <-[:PURCHASED]-(other:User)-[:PURCHASED]->(rec:Product)
        WHERE NOT (u)-[:PURCHASED]->(rec)
        RETURN rec.name AS product,
               rec.price AS price,
               count(DISTINCT other) AS score
        ORDER BY score DESC
        LIMIT {limit}
    """)

    return list(result)

recs = get_recommendations(db, 'U1')
print("Recommendations for Alice:")
for r in recs:
    print(f"  {r['product']} (${r['price']}) - score: {r['score']}")
```

### Category-Based Recommendations

```python
result = db.execute("""
    MATCH (u:User {id: 'U1'})-[:PURCHASED]->(bought:Product)
    WITH u, collect(bought.id) AS purchased
    MATCH (rec:Product)
    WHERE rec.category = 'Electronics'
      AND NOT rec.id IN purchased
    RETURN rec.name, rec.price
""")

print("Other products in categories you've purchased from:")
for row in result:
    print(f"  {row['rec.name']} (${row['rec.price']})")
```

## Next Steps

- [Fraud Detection Tutorial](fraud-detection.md)
- [Aggregations](../user-guide/gql/aggregations.md)
