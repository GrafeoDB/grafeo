---
title: Social Network Graph
description: Build a social network graph with Grafeo.
tags:
  - tutorial
  - beginner
---

# Social Network Graph

In this tutorial, you'll build a social network with users, friendships, posts, and likes.

## What You'll Learn

- Creating nodes with labels and properties
- Creating relationships between nodes
- Querying connected data
- Aggregating graph data

## Prerequisites

- Grafeo installed ([Installation Guide](../getting-started/installation.md))
- Basic understanding of graphs

## Setup

```python
import grafeo

db = grafeo.GrafeoDB()
```

## Step 1: Create Users

```python
db.execute("""
    INSERT (:User {
        id: 1,
        name: 'Alice',
        email: 'alice@example.com',
        joined: '2023-01-15'
    })
    INSERT (:User {
        id: 2,
        name: 'Bob',
        email: 'bob@example.com',
        joined: '2023-02-20'
    })
    INSERT (:User {
        id: 3,
        name: 'Carol',
        email: 'carol@example.com',
        joined: '2023-03-10'
    })
    INSERT (:User {
        id: 4,
        name: 'Dave',
        email: 'dave@example.com',
        joined: '2023-04-05'
    })
""")
```

## Step 2: Create Friendships

```python
# Alice and Bob are friends
db.execute("""
    MATCH (a:User {name: 'Alice'}), (b:User {name: 'Bob'})
    INSERT (a)-[:FRIENDS_WITH {since: '2023-03-01'}]->(b)
    INSERT (b)-[:FRIENDS_WITH {since: '2023-03-01'}]->(a)
""")

# Alice and Carol are friends
db.execute("""
    MATCH (a:User {name: 'Alice'}), (c:User {name: 'Carol'})
    INSERT (a)-[:FRIENDS_WITH {since: '2023-04-15'}]->(c)
    INSERT (c)-[:FRIENDS_WITH {since: '2023-04-15'}]->(a)
""")

# Bob and Dave are friends
db.execute("""
    MATCH (b:User {name: 'Bob'}), (d:User {name: 'Dave'})
    INSERT (b)-[:FRIENDS_WITH {since: '2023-05-20'}]->(d)
    INSERT (d)-[:FRIENDS_WITH {since: '2023-05-20'}]->(b)
""")
```

## Step 3: Create Posts

```python
db.execute("""
    INSERT (:Post {
        id: 1,
        content: 'Hello everyone! Excited to join this network.',
        created: '2023-03-15'
    })
    INSERT (:Post {
        id: 2,
        content: 'Just discovered Grafeo - amazing graph database!',
        created: '2023-04-01'
    })
    INSERT (:Post {
        id: 3,
        content: 'Graph databases make relationships easy to model.',
        created: '2023-04-10'
    })
""")

# Link posts to authors
db.execute("""
    MATCH (u:User {name: 'Alice'}), (p:Post {id: 1})
    INSERT (u)-[:POSTED]->(p)
""")
db.execute("""
    MATCH (u:User {name: 'Bob'}), (p:Post {id: 2})
    INSERT (u)-[:POSTED]->(p)
""")
db.execute("""
    MATCH (u:User {name: 'Carol'}), (p:Post {id: 3})
    INSERT (u)-[:POSTED]->(p)
""")
```

## Step 4: Add Likes

```python
# Bob likes Alice's post
db.execute("""
    MATCH (u:User {name: 'Bob'}), (p:Post {id: 1})
    INSERT (u)-[:LIKES {at: '2023-03-16'}]->(p)
""")

# Carol likes Alice's and Bob's posts
db.execute("""
    MATCH (u:User {name: 'Carol'}), (p:Post {id: 1})
    INSERT (u)-[:LIKES {at: '2023-03-17'}]->(p)
""")
db.execute("""
    MATCH (u:User {name: 'Carol'}), (p:Post {id: 2})
    INSERT (u)-[:LIKES {at: '2023-04-02'}]->(p)
""")

# Dave likes all posts
db.execute("""
    MATCH (u:User {name: 'Dave'}), (p:Post)
    INSERT (u)-[:LIKES]->(p)
""")
```

## Querying the Social Network

### Find Alice's Friends

```python
result = db.execute("""
    MATCH (alice:User {name: 'Alice'})-[:FRIENDS_WITH]->(friend)
    RETURN friend.name, friend.email
""")

print("Alice's friends:")
for row in result:
    print(f"  - {row['friend.name']} ({row['friend.email']})")
```

### Find Friend Recommendations (Friends of Friends)

```python
result = db.execute("""
    MATCH (alice:User {name: 'Alice'})-[:FRIENDS_WITH]->()-[:FRIENDS_WITH]->(fof)
    WHERE fof <> alice
      AND NOT (alice)-[:FRIENDS_WITH]->(fof)
    RETURN DISTINCT fof.name AS recommendation
""")

print("Friend recommendations for Alice:")
for row in result:
    print(f"  - {row['recommendation']}")
```

### Find Most Liked Posts

```python
result = db.execute("""
    MATCH (p:Post)<-[:LIKES]-(u:User)
    MATCH (author)-[:POSTED]->(p)
    RETURN p.content AS post, author.name AS author, count(u) AS likes
    ORDER BY likes DESC
""")

print("Posts by popularity:")
for row in result:
    print(f"  {row['likes']} likes: '{row['post'][:50]}...' by {row['author']}")
```

### Find Users Who Liked Friends' Posts

```python
result = db.execute("""
    MATCH (u:User)-[:FRIENDS_WITH]->(friend)-[:POSTED]->(p)<-[:LIKES]-(u)
    RETURN u.name AS user, friend.name AS friend, p.content AS post
""")

print("Users who liked their friends' posts:")
for row in result:
    print(f"  {row['user']} liked {row['friend']}'s post")
```

## Next Steps

- [Knowledge Graph Tutorial](knowledge-graph.md)
- [GQL Query Language](../user-guide/gql/index.md)
