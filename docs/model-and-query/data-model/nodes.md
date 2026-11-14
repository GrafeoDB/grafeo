---
title: Nodes and Labels
description: Working with nodes and labels in Grafeo.
tags:
  - data-model
  - nodes
---

# Nodes and Labels

Nodes are the fundamental entities in a graph. Each node can have one or more labels and any number of properties.

## Creating Nodes

```sql
-- Create a node with a single label
INSERT (:Person {name: 'Alix', age: 30})

-- Create a node with multiple labels
INSERT (:Person:Employee {name: 'Gus', department: 'Engineering'})

-- Create multiple nodes
INSERT (:Person {name: 'Harm'})
INSERT (:Person {name: 'Dave'})
```

## Labels

Labels categorize nodes and enable efficient querying:

```sql
-- Find all Person nodes
MATCH (p:Person)
RETURN p.name

-- Find nodes with multiple labels
MATCH (e:Person:Employee)
RETURN e.name, e.department
```

## Node Properties

Nodes can have any number of properties:

```sql
INSERT (:Person {
    name: 'Alix',
    age: 30,
    email: 'alix@example.com',
    active: true,
    scores: [95, 87, 92]
})
```

## Updating Nodes

```sql
-- Add or update properties
MATCH (p:Person {name: 'Alix'})
SET p.age = 31, p.updated_at = '2024-01-15'

-- Remove a property
MATCH (p:Person {name: 'Alix'})
REMOVE p.email
```

## Deleting Nodes

```sql
-- Delete a node (must have no edges)
MATCH (p:Person {name: 'Alix'})
DELETE p

-- Delete a node and all its edges
MATCH (p:Person {name: 'Alix'})
DETACH DELETE p
```
