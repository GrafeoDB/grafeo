---
title: Knowledge Graph
description: Build a knowledge graph with Grafeo.
tags:
  - tutorial
  - intermediate
---

# Knowledge Graph

Build a knowledge graph to organize information about entities and their relationships.

## What You'll Learn

- Modeling diverse entity types
- Creating semantic relationships
- Querying for insights
- Traversing knowledge paths

## The Domain: Movies

We'll build a knowledge graph about movies, actors, directors, and genres.

## Setup

```python
import grafeo

db = grafeo.GrafeoDB()
```

## Create the Knowledge Graph

```python
# Movies
db.execute("""
    INSERT (:Movie {
        title: 'The Matrix',
        year: 1999,
        rating: 8.7
    })
    INSERT (:Movie {
        title: 'Inception',
        year: 2010,
        rating: 8.8
    })
    INSERT (:Movie {
        title: 'The Dark Knight',
        year: 2008,
        rating: 9.0
    })
""")

# People
db.execute("""
    INSERT (:Person {name: 'Keanu Reeves', born: 1964})
    INSERT (:Person {name: 'Leonardo DiCaprio', born: 1974})
    INSERT (:Person {name: 'Christian Bale', born: 1974})
    INSERT (:Person {name: 'Christopher Nolan', born: 1970})
    INSERT (:Person {name: 'Lana Wachowski', born: 1965})
""")

# Genres
db.execute("""
    INSERT (:Genre {name: 'Sci-Fi'})
    INSERT (:Genre {name: 'Action'})
    INSERT (:Genre {name: 'Thriller'})
""")
```

## Create Relationships

```python
# Acting relationships
db.execute("""
    MATCH (p:Person {name: 'Keanu Reeves'}), (m:Movie {title: 'The Matrix'})
    INSERT (p)-[:ACTED_IN {role: 'Neo'}]->(m)
""")
db.execute("""
    MATCH (p:Person {name: 'Leonardo DiCaprio'}), (m:Movie {title: 'Inception'})
    INSERT (p)-[:ACTED_IN {role: 'Cobb'}]->(m)
""")
db.execute("""
    MATCH (p:Person {name: 'Christian Bale'}), (m:Movie {title: 'The Dark Knight'})
    INSERT (p)-[:ACTED_IN {role: 'Batman'}]->(m)
""")

# Directing relationships
db.execute("""
    MATCH (p:Person {name: 'Christopher Nolan'}), (m:Movie {title: 'Inception'})
    INSERT (p)-[:DIRECTED]->(m)
""")
db.execute("""
    MATCH (p:Person {name: 'Christopher Nolan'}), (m:Movie {title: 'The Dark Knight'})
    INSERT (p)-[:DIRECTED]->(m)
""")
db.execute("""
    MATCH (p:Person {name: 'Lana Wachowski'}), (m:Movie {title: 'The Matrix'})
    INSERT (p)-[:DIRECTED]->(m)
""")

# Genre relationships
db.execute("""
    MATCH (m:Movie {title: 'The Matrix'}), (g:Genre {name: 'Sci-Fi'})
    INSERT (m)-[:IN_GENRE]->(g)
""")
db.execute("""
    MATCH (m:Movie {title: 'The Matrix'}), (g:Genre {name: 'Action'})
    INSERT (m)-[:IN_GENRE]->(g)
""")
db.execute("""
    MATCH (m:Movie {title: 'Inception'}), (g:Genre {name: 'Sci-Fi'})
    INSERT (m)-[:IN_GENRE]->(g)
""")
db.execute("""
    MATCH (m:Movie {title: 'Inception'}), (g:Genre {name: 'Thriller'})
    INSERT (m)-[:IN_GENRE]->(g)
""")
db.execute("""
    MATCH (m:Movie {title: 'The Dark Knight'}), (g:Genre {name: 'Action'})
    INSERT (m)-[:IN_GENRE]->(g)
""")
```

## Query the Knowledge Graph

### Find Christopher Nolan's Movies

```python
result = db.execute("""
    MATCH (p:Person {name: 'Christopher Nolan'})-[:DIRECTED]->(m:Movie)
    RETURN m.title, m.year, m.rating
    ORDER BY m.year
""")

print("Christopher Nolan's movies:")
for row in result:
    print(f"  {row['m.title']} ({row['m.year']}) - {row['m.rating']}")
```

### Find Sci-Fi Movies

```python
result = db.execute("""
    MATCH (m:Movie)-[:IN_GENRE]->(g:Genre {name: 'Sci-Fi'})
    RETURN m.title, m.rating
    ORDER BY m.rating DESC
""")

print("Sci-Fi movies:")
for row in result:
    print(f"  {row['m.title']} ({row['m.rating']})")
```

### Find Co-Stars

```python
result = db.execute("""
    MATCH (a1:Person)-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(a2:Person)
    WHERE a1 <> a2
    RETURN a1.name, a2.name, m.title
""")

print("Co-stars:")
for row in result:
    print(f"  {row['a1.name']} and {row['a2.name']} in {row['m.title']}")
```

## Next Steps

- [Recommendation Engine Tutorial](recommendations.md)
- [Path Queries](../user-guide/gql/paths.md)
