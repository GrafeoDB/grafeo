---
title: Properties
description: Access vertex and edge properties with values, valueMap and more.
tags:
  - gremlin
  - properties
---

# Properties

This guide covers accessing and working with properties on vertices and edges in Gremlin.

## values()

Returns property values for the given keys:

```gremlin
// Single property
g.V().hasLabel('Person').values('name')

// Multiple properties
g.V().hasLabel('Person').values('name', 'age')
```

## valueMap()

Returns properties as key-value maps:

```gremlin
// All properties
g.V().hasLabel('Person').valueMap()

// Specific properties
g.V().hasLabel('Person').valueMap('name', 'age')
```

## elementMap()

Like `valueMap()` but also includes the element's ID and label:

```gremlin
// Full element details
g.V().hasLabel('Person').elementMap()

// Specific properties with ID and label
g.V().hasLabel('Person').elementMap('name', 'age')
```

## id()

Returns the internal ID of a vertex or edge:

```gremlin
// Get vertex IDs
g.V().hasLabel('Person').id()

// Get edge IDs
g.V().has('name', 'Alix').outE('KNOWS').id()
```

## label()

Returns the label of a vertex or edge:

```gremlin
// Get vertex labels
g.V().label()

// Get edge labels
g.V().has('name', 'Alix').outE().label()
```

## properties()

Returns property objects (key + value) rather than just values:

```gremlin
// All properties
g.V().has('name', 'Alix').properties()

// Specific properties
g.V().has('name', 'Alix').properties('name', 'age')
```

## constant()

Returns a fixed value for each element in the traversal:

```gremlin
// Return a constant value
g.V().hasLabel('Person').constant('found')
```

## Setting Properties

Use `property()` to set properties on vertices:

```gremlin
// Set a single property
g.addV('Person').property('name', 'Alix')

// Set multiple properties
g.addV('Person').property('name', 'Alix').property('age', 30)
```

### Cardinality

Control how property values are stored:

```gremlin
// Single value (replaces existing)
g.V().has('name', 'Alix').property(single, 'email', 'alix@example.com')

// List (appends to existing values)
g.V().has('name', 'Alix').property(list, 'phone', '555-0100')

// Set (adds if not already present)
g.V().has('name', 'Alix').property(set, 'tag', 'developer')
```

## Python Example

```python
import grafeo

db = grafeo.GrafeoDB()

# Create data
db.execute("INSERT (:Person {name: 'Alix', age: 30, city: 'Utrecht'})")
db.execute("INSERT (:Person {name: 'Gus', age: 25, city: 'Portland'})")

# Get names
names = db.execute_gremlin("g.V().hasLabel('Person').values('name')")
for row in names:
    print(row)  # Alix, Gus

# Get IDs
ids = db.execute_gremlin("g.V().hasLabel('Person').id()")
for row in ids:
    print(row)

# Get labels
labels = db.execute_gremlin("g.V().label()")
for row in labels:
    print(row)  # Person, Person
```

## Step Reference

| Step | Description | Returns |
|------|-------------|---------|
| `values(keys...)` | Property values | Values |
| `valueMap(keys...)` | Properties as map | Map |
| `elementMap(keys...)` | Properties + ID + label | Map |
| `id()` | Element ID | ID |
| `label()` | Element label | String |
| `properties(keys...)` | Property objects | Properties |
| `constant(value)` | Fixed value per element | Value |
| `property(key, value)` | Set a property | Element |
