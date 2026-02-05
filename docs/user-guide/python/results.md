---
title: Query Results
description: Working with query results in Python.
tags:
  - python
  - results
---

# Query Results

Learn how to work with query results in Python.

## Iterating Results

```python
result = db.execute("""
    MATCH (p:Person)
    RETURN p.name, p.age
""")

# Iterate over rows
for row in result:
    print(f"{row['p.name']}: {row['p.age']}")
```

## Accessing Values

```python
result = db.execute("""
    MATCH (p:Person {name: 'Alice'})
    RETURN p.name, p.age, p.active
""")

row = next(iter(result))

# By column name
name = row['p.name']

# By index
name = row[0]
```

## Converting to List

```python
result = db.execute("MATCH (p:Person) RETURN p.name")

# Convert to list
rows = list(result)

# Or use list comprehension
result = db.execute("MATCH (p:Person) RETURN p.name")
names = [row['p.name'] for row in result]
```

## Single Value

```python
# Get count
result = db.execute("""
    MATCH (p:Person)
    RETURN count(p) AS count
""")

count = next(iter(result))['count']
print(f"Total people: {count}")
```

## Column Names

```python
result = db.execute("""
    MATCH (p:Person)
    RETURN p.name AS name, p.age AS age
""")

# Results have columns
for row in result:
    print(row)  # {'name': 'Alice', 'age': 30}
```

## Empty Results

```python
result = db.execute("""
    MATCH (p:Person {name: 'NonExistent'})
    RETURN p
""")

# Check if empty
rows = list(result)
if not rows:
    print("No results found")
```

## Large Results

```python
result = db.execute("""
    MATCH (p:Person)
    RETURN p.name
""")

# Stream results (memory efficient)
for row in result:
    process(row)
    # Each row is fetched as needed
```
