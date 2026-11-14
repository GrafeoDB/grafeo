---
title: Aggregations
description: Aggregation and grouping operations in Gremlin.
tags:
  - gremlin
  - aggregations
---

# Aggregations

This guide covers aggregation functions, grouping and collection operations in Gremlin.

## count()

Counts the number of elements in the traversal:

```gremlin
// Count all vertices
g.V().count()

// Count people
g.V().hasLabel('Person').count()

// Count Alix's friends
g.V().has('name', 'Alix').out('KNOWS').count()
```

## sum()

Sums numeric values:

```gremlin
// Sum of all ages
g.V().hasLabel('Person').values('age').sum()

// Sum of order totals
g.V().hasLabel('Order').values('total').sum()
```

## mean()

Computes the average of numeric values:

```gremlin
// Average age
g.V().hasLabel('Person').values('age').mean()
```

## min() and max()

Find the minimum or maximum value:

```gremlin
// Youngest person's age
g.V().hasLabel('Person').values('age').min()

// Oldest person's age
g.V().hasLabel('Person').values('age').max()

// Cheapest product
g.V().hasLabel('Product').values('price').min()
```

## fold()

Collects all elements into a single list:

```gremlin
// Collect all names into a list
g.V().hasLabel('Person').values('name').fold()

// Collect friend names
g.V().has('name', 'Alix').out('KNOWS').values('name').fold()
```

## unfold()

Expands a list back into individual elements:

```gremlin
// Unfold a folded list
g.V().hasLabel('Person').values('name').fold().unfold()
```

## groupCount()

Counts elements grouped by value:

```gremlin
// Count people per city
g.V().hasLabel('Person').values('city').groupCount()

// Count edges by label
g.V().outE().label().groupCount()
```

## group()

Groups elements (use with `by()` modulator):

```gremlin
// Group people by city
g.V().hasLabel('Person').group().by('city')
```

## Ordering

Sort results with `order()`:

```gremlin
// Default ascending order
g.V().hasLabel('Person').values('name').order()

// Order by property
g.V().hasLabel('Person').order().by('age')

// Order by T.id or T.label
g.V().order().by(T.id)
g.V().order().by(T.label)
```

## Side Effects

### as()

Label a step for later reference:

```gremlin
// Label vertices for later use
g.V().has('name', 'Alix').as('a').out('KNOWS').as('b').select('a', 'b')
```

### select()

Retrieve labeled elements:

```gremlin
// Select specific labeled elements
g.V().has('name', 'Alix').as('person').out('KNOWS').as('friend').select('person', 'friend')
```

### project()

Create a map projection:

```gremlin
// Project properties into named fields
g.V().hasLabel('Person').project('name', 'age')
```

### aggregate() and store()

Collect elements into a side-effect collection:

```gremlin
// Aggregate into a named collection
g.V().hasLabel('Person').aggregate('people')

// Store (lazy aggregation)
g.V().hasLabel('Person').store('people')
```

### path()

Return the full traversal path:

```gremlin
// Get the path from Alix to her friends' friends
g.V().has('name', 'Alix').out('KNOWS').out('KNOWS').path()
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data
db.execute("INSERT (:Person {name: 'Alix', age: 30, city: 'Utrecht'})")
db.execute("INSERT (:Person {name: 'Gus', age: 25, city: 'Portland'})")
db.execute("INSERT (:Person {name: 'Vincent', age: 35, city: 'Utrecht'})")

# Count
result = db.execute_gremlin("g.V().hasLabel('Person').count()")
for row in result:
    print(row)  # 3

# Average age
result = db.execute_gremlin("g.V().hasLabel('Person').values('age').mean()")
for row in result:
    print(row)  # 30.0

# Min/Max
result = db.execute_gremlin("g.V().hasLabel('Person').values('age').min()")
for row in result:
    print(row)  # 25

# Collect names
result = db.execute_gremlin("g.V().hasLabel('Person').values('name').fold()")
for row in result:
    print(row)  # ['Alix', 'Gus', 'Vincent']
```

## Aggregation Reference

| Step | Description |
|------|-------------|
| `count()` | Count elements |
| `sum()` | Sum numeric values |
| `mean()` | Average of numeric values |
| `min()` | Minimum value |
| `max()` | Maximum value |
| `fold()` | Collect into list |
| `unfold()` | Expand list into elements |
| `group()` | Group elements |
| `groupCount()` | Count per group |
| `order()` | Sort elements |
| `path()` | Full traversal path |
