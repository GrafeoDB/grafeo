---
title: Schema and DDL
description: GQL schema definition, graph management, type definitions, indexes and constraints.
tags:
  - gql
  - schema
  - ddl
---

# Schema and DDL

GQL provides Data Definition Language (DDL) statements for managing graphs, type definitions, indexes, constraints and stored procedures.

## Graph Management

### Creating Graphs

```sql
-- Create a named graph
CREATE GRAPH my_graph

-- Create with IF NOT EXISTS (no error if it already exists)
CREATE GRAPH IF NOT EXISTS my_graph

-- Create a property graph (equivalent to CREATE GRAPH)
CREATE PROPERTY GRAPH my_graph
```

### Typed Graphs

Bind a graph to a registered graph type to enforce its schema:

```sql
-- First, create a graph type
CREATE GRAPH TYPE social_network (
    NODE TYPE Person (name STRING NOT NULL, age INTEGER),
    EDGE TYPE KNOWS (since INTEGER)
)

-- Then create a graph bound to that type
CREATE GRAPH my_social TYPED social_network
```

### Dropping Graphs

```sql
DROP GRAPH my_graph

-- No error if it doesn't exist
DROP GRAPH IF EXISTS my_graph

DROP PROPERTY GRAPH my_graph
```

### Switching Graphs

```sql
-- Switch the active graph for subsequent queries
USE GRAPH my_graph
```

## Node and Edge Types

Type definitions declare the schema for nodes and edges: which properties they have, their types and constraints.

### Creating Types

```sql
-- Node type with properties
CREATE NODE TYPE Person (
    name STRING NOT NULL,
    age INTEGER,
    email STRING
)

-- Edge type with properties
CREATE EDGE TYPE KNOWS (
    since INTEGER,
    strength FLOAT
)

-- With IF NOT EXISTS
CREATE NODE TYPE IF NOT EXISTS Person (name STRING NOT NULL)

-- With OR REPLACE (drop and recreate if exists)
CREATE OR REPLACE NODE TYPE Person (
    name STRING NOT NULL,
    age INTEGER,
    email STRING NOT NULL
)
```

### Altering Types

Add or remove properties from existing type definitions:

```sql
-- Add a property to a node type
ALTER NODE TYPE Person ADD phone STRING

-- Add a NOT NULL property
ALTER NODE TYPE Person ADD verified BOOLEAN NOT NULL

-- Drop a property
ALTER NODE TYPE Person DROP phone

-- Alter an edge type
ALTER EDGE TYPE KNOWS ADD quality FLOAT
ALTER EDGE TYPE KNOWS DROP quality
```

### Dropping Types

```sql
DROP NODE TYPE Person
DROP EDGE TYPE KNOWS

-- No error if it doesn't exist
DROP NODE TYPE IF EXISTS Person
```

## Graph Types

Graph types define which node and edge types a graph can contain.

### Creating Graph Types

```sql
CREATE GRAPH TYPE social_network (
    NODE TYPE Person (name STRING NOT NULL, age INTEGER),
    NODE TYPE Company (name STRING NOT NULL),
    EDGE TYPE KNOWS (since INTEGER),
    EDGE TYPE WORKS_AT (role STRING)
)

-- With IF NOT EXISTS
CREATE GRAPH TYPE IF NOT EXISTS social_network (
    NODE TYPE Person (name STRING NOT NULL)
)

-- With OR REPLACE
CREATE OR REPLACE GRAPH TYPE social_network (
    NODE TYPE Person (name STRING NOT NULL, age INTEGER),
    EDGE TYPE KNOWS (since INTEGER)
)

-- With key label sets (GG21)
CREATE GRAPH TYPE labeled_type (
    NODE TYPE Person KEY (PersonLabel) (name STRING NOT NULL, age INTEGER),
    EDGE TYPE KNOWS
)
```

### Graph Type from Existing Graph (LIKE)

Clone a graph type from an existing graph's bound type:

```sql
-- Create a type that matches an existing graph's schema
CREATE GRAPH TYPE cloned_type LIKE my_existing_graph
```

### Altering Graph Types

Add or remove allowed node/edge types:

```sql
-- Add a node type to the graph type
ALTER GRAPH TYPE social_network ADD NODE TYPE Company

-- Add an edge type
ALTER GRAPH TYPE social_network ADD EDGE TYPE WORKS_AT

-- Remove types
ALTER GRAPH TYPE social_network DROP NODE TYPE Company
ALTER GRAPH TYPE social_network DROP EDGE TYPE WORKS_AT
```

### Dropping Graph Types

```sql
DROP GRAPH TYPE social_network
DROP GRAPH TYPE IF EXISTS social_network
```

## Schemas

Schemas provide namespace organization:

```sql
-- Create a schema
CREATE SCHEMA analytics

-- Drop a schema
DROP SCHEMA analytics
DROP SCHEMA IF EXISTS analytics
```

## Indexes

Indexes improve query performance for property lookups, text search and vector similarity.

### Property Indexes

```sql
-- Create a property index
CREATE INDEX FOR (p:Person) ON (p.email)

-- B-tree index (explicit)
CREATE INDEX FOR (p:Person) ON (p.age) USING BTREE

-- Drop an index
DROP INDEX index_name
DROP INDEX IF EXISTS index_name
```

### Text Indexes

```sql
-- Full-text search index (BM25)
CREATE INDEX FOR (p:Post) ON (p.content) USING TEXT
```

### Vector Indexes

```sql
-- Vector similarity index (HNSW)
CREATE VECTOR INDEX FOR (d:Document) ON (d.embedding)
    DIMENSION 384
    METRIC 'cosine'
```

## Constraints

Constraints enforce data integrity rules on writes.

### UNIQUE

Ensure a property value is unique across all nodes with a given label:

```sql
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email IS UNIQUE
```

### NODE KEY

Composite uniqueness across multiple properties:

```sql
CREATE CONSTRAINT FOR (p:Person) REQUIRE (p.firstName, p.lastName) IS NODE KEY
```

### NOT NULL

Require a property to always have a value:

```sql
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.name IS NOT NULL
```

### EXISTS

Require a property to exist on every node with the label:

```sql
CREATE CONSTRAINT FOR (p:Person) REQUIRE p.email EXISTS
```

### Naming, Listing and Dropping Constraints

Give a constraint a name after `CREATE CONSTRAINT`; one created without a name is named after its label, properties and kind, such as `Person_email_unique`. A name can be used once: `CREATE CONSTRAINT IF NOT EXISTS name ...` does nothing when the name is taken. `SHOW CONSTRAINTS` lists the constraints with their name, kind, label and properties.

```sql
SHOW CONSTRAINTS
DROP CONSTRAINT constraint_name
DROP CONSTRAINT IF EXISTS constraint_name
```

`DROP CONSTRAINT` fails for a name that does not exist, unless `IF EXISTS` is given.

## Stored Procedures

### Creating Procedures

```sql
-- Create a stored procedure with parameters and return types
CREATE PROCEDURE find_friends(person_name STRING)
    RETURNS (friend_name STRING, mutual_count INTEGER)
AS {
    MATCH (p:Person {name: $person_name})-[:KNOWS]->(friend)
    RETURN friend.name AS friend_name,
        COUNT {
            MATCH (p)-[:KNOWS]->(m)<-[:KNOWS]-(friend)
        } AS mutual_count
}

-- With IF NOT EXISTS
CREATE PROCEDURE IF NOT EXISTS find_friends(person_name STRING)
    RETURNS (friend_name STRING)
AS {
    MATCH (p:Person {name: $person_name})-[:KNOWS]->(f)
    RETURN f.name AS friend_name
}

-- With OR REPLACE
CREATE OR REPLACE PROCEDURE find_friends(person_name STRING)
    RETURNS (friend_name STRING, mutual_count INTEGER)
AS {
    MATCH (p:Person {name: $person_name})-[:KNOWS]->(friend)
    RETURN friend.name AS friend_name,
        COUNT {
            MATCH (p)-[:KNOWS]->(m)<-[:KNOWS]-(friend)
        } AS mutual_count
}
```

### Calling Procedures

```sql
-- Call with YIELD to select output fields
CALL find_friends('Alix') YIELD friend_name, mutual_count
RETURN friend_name, mutual_count
ORDER BY mutual_count DESC

-- Filter yielded results
CALL find_friends('Alix') YIELD friend_name, mutual_count
WHERE mutual_count > 3
RETURN friend_name
```

### Dropping Procedures

```sql
DROP PROCEDURE find_friends
DROP PROCEDURE IF EXISTS find_friends
```

See [Basic Queries](basic-queries.md#calling-procedures) for inline `CALL { ... }` subqueries and `OPTIONAL CALL`.
