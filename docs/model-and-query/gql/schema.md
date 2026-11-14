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

A graph type with a body is closed: a graph bound to it takes only nodes whose labels are its node types, edges of its edge types, and the properties those types declare. Writing anything else to the graph, with a query in it (`USE GRAPH my_social`) or through a graph handle, fails with an error naming the label, edge type or property and the graph type. A graph type declared with `{node_types: [...], edge_types: [...], open: true}` is open and checks only the property types.

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

### Property Types

Each property of a node or edge type has one of these types. A value written to the property (when a node or edge of the type is created, or with `SET`) must have the type, or the write fails; `NULL` is accepted unless the property is `NOT NULL`. Types match strictly, with the conversions of numbers and instants that ISO GQL makes on assignment: an integer written to a `FLOAT64` property is stored as a float (an integer beyond 2^53 that has no exact float fails), a decimal written to a `FLOAT64` property as the nearest float, an integer, float (by its shortest text) or decimal written to a `DECIMAL` property as a decimal, and a zoned datetime written to a `TIMESTAMP` property as its instant in UTC. A float is not an `INT64`, and a zoned datetime is not a `LOCAL DATETIME`.

| Type | Also written | Values |
|------|--------------|--------|
| `STRING` | `VARCHAR`, `TEXT` | Strings |
| `INT64` | `INT`, `INTEGER`, `BIGINT` | Integers |
| `FLOAT64` | `FLOAT`, `DOUBLE`, `REAL` | Floats |
| `DECIMAL` | `DEC` | Exact decimals, each with its own scale |
| `DECIMAL(p, s)` | `DEC(p, s)`, `DECIMAL(p)` | Exact decimals of at most `p` digits, `s` of them after the point (`p` from 1 to 38, `s` up to `p`, 0 when left out): a value is rounded to `s` digits, half away from zero, and one with more than `p - s` digits before the point is refused |
| `BOOLEAN` | `BOOL` | `TRUE` and `FALSE` |
| `DATE` | | Dates, such as `date('2024-03-19')` |
| `TIME` | | Times of day, such as `time('08:30:00')` |
| `LOCAL DATETIME` | | Dates with a time and no time zone, such as `local_datetime('2024-03-19T08:30:00')` |
| `TIMESTAMP` | `DATETIME` | The same values as `LOCAL DATETIME` |
| `ZONED DATETIME` | | Dates with a time and a UTC offset, such as `zoned_datetime('2024-03-19T08:30:00+01:00')` |
| `DURATION` | `INTERVAL` | Durations, such as `duration('P3D')` |
| `BYTES` | `BINARY`, `BLOB` | Byte strings |
| `LIST` | `ARRAY` | Lists of any values |
| `LIST<type>` | | Lists whose elements all have `type`, such as `LIST<STRING>` or `LIST<ZONED DATETIME>`; they nest up to 128 levels, as in `LIST<LIST<INT64>>` |
| `MAP` | `RECORD` | Maps, such as `{mode: 'fast', level: 3}` |
| `NODE` | | Nodes |
| `EDGE` | `RELATIONSHIP` | Edges |
| `ANY` | | Any value |

Any other type name is refused, a typo or a type of ISO GQL that Grafeo does not support (such as `INT32`). A property name may be any name a property map takes, keywords such as `starts`, `ends` and `contains` included.

```sql
CREATE NODE TYPE Event (
    title STRING NOT NULL,
    begins ZONED DATETIME,
    departs LOCAL DATETIME,
    tags LIST<STRING>
)
```

`SHOW NODE TYPES` and `SHOW EDGE TYPES` list each property with its type, in the first spelling of the table.

A property of a node or edge type can have a default value, a literal after `DEFAULT`: a node or edge created without the property gets it. The literal is a string, a number (signed numbers such as `-3` included), `TRUE`, `FALSE` or `NULL`, and must be a value of the property's type, or the statement fails: an integer is a default of a `FLOAT64` property too, `NULL` is no default of a `NOT NULL` property, and the types without such literals (dates, lists, maps and the others) take `NULL` only.

```sql
CREATE NODE TYPE City (name STRING NOT NULL, country STRING DEFAULT 'NL')
CREATE EDGE TYPE ROUTE (km INT64 DEFAULT 88)
CREATE NODE TYPE Reading (celsius INT64 DEFAULT -3, scale FLOAT64 DEFAULT 1)
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
    EDGE TYPE KNOWS ()
)
```

A `NODE TYPE` or `EDGE TYPE` with a property list, even an empty one (`EDGE TYPE KNOWS ()`), or with a `KEY`
clause, defines the type in the graph type. A bare name (`EDGE TYPE KNOWS`) refers to a type that already exists
in the catalog.

A type the graph type defines applies only in the graphs bound to the graph type. A type made on its own with
`CREATE NODE TYPE` or `CREATE EDGE TYPE` applies by its label in every graph, and still does once a graph type
lists it; it then also applies in that graph type's graphs. A graph type names such a type only as it is: a pattern
that gives it other properties or endpoints is refused. `DROP GRAPH TYPE` and `ALTER GRAPH TYPE ... DROP` leave
it, and `DROP NODE TYPE` or `DROP EDGE TYPE` is refused while a graph type lists it.

The element types can also be written as patterns, in braces as ISO GQL writes them or in parentheses. An edge pattern declares its edge type with the node types at its ends as the only sources and targets:

```sql
CREATE GRAPH TYPE routes {
    (:City {name STRING NOT NULL, population INT64})-[:ROUTE {km INT64}]->(:City),
    (:Country {code STRING})
}
```

Several patterns of one edge type are one type, so they must declare the same properties. They may differ on one side only, as in `(:Post)-[:HAS_CREATOR]->(:Person), (:Comment)-[:HAS_CREATOR]->(:Person)`, whose sources are then `Post` and `Comment`. Patterns that differ on both sides are not supported yet, as the type would also accept the pairs no pattern states.

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

Indexes improve query performance for property lookups, text search and vector similarity. In GQL an index
always has a name, the one `SHOW INDEXES` lists and `DROP INDEX` takes (Cypher may leave it out, see the
[Cypher guide](../cypher/mutations.md)).

### Property Indexes

```sql
-- Create a property index
CREATE INDEX person_email FOR (p:Person) ON (p.email)

-- B-tree index (explicit)
CREATE INDEX person_age FOR (p:Person) ON (p.age) USING BTREE

-- Drop an index
DROP INDEX index_name
DROP INDEX IF EXISTS index_name
```

A property or B-tree index holds the nodes of its label only, so writes on other labels pay nothing for it. A query uses it when every result must have that label: `MATCH (p:Person {age: 30})` and `MATCH (p:Person:Author {age: 30})` do. A query whose results have one of several labels, as `MATCH (p:Person|Company {age: 30})`, or that names no label, as `MATCH (p {age: 30})`, uses the indexes there are and reads the property only of the nodes they do not cover. A query for a label with no index on the property, as `MATCH (p:City {age: 30})`, scans that label. Every one returns the same rows.

`create_property_index` in the API makes an index on a property of every label, which also serves a query with no label. An index on the label and property, or one on the property of every label, makes a later `CREATE INDEX` on them build nothing: its name stands for the index that is there. `drop_property_index` drops every index on the property in the current graph.

Index names belong to the database, not to one graph: a name stands for one index definition in all of its graphs.
`CREATE INDEX` with a name that already stands for another label, property or kind of index, in any graph, fails
naming it, and `IF NOT EXISTS` (also on `CREATE VECTOR INDEX`) keeps the existing index.
`DROP INDEX` drops the index the name stands for, whatever its kind, and every property of an index over several.
`SHOW INDEXES` lists each index by name, with its type, label and property (one row per property), also the indexes
made through the API (`create_property_index`, `create_vector_index`, `create_text_index`): those have a null name,
and a property index made that way covers every label, so its label is null too. An index a constraint built without
a name has a null name and its label.

### Text Indexes

```sql
-- Full-text search index (BM25)
CREATE INDEX post_content FOR (p:Post) ON (p.content) USING TEXT

-- With BM25 parameters, a tokenizer or stop words
CREATE INDEX post_content FOR (p:Post) ON (p.content) USING TEXT {k1: 1.2, b: 0.75, stop_words: ['and', 'or']}
```

### Vector Indexes

```sql
-- Vector similarity index (HNSW)
CREATE VECTOR INDEX document_embedding ON :Document(embedding)
    DIMENSION 384
    METRIC 'cosine'

-- The same through CREATE INDEX
CREATE INDEX document_embedding FOR (d:Document) ON (d.embedding) USING VECTOR {dimensions: 384, metric: 'cosine'}
```

## Constraints

Constraints enforce data integrity rules on writes. In GQL a constraint names its label with `FOR` and its properties with `ON`; Cypher writes the same constraints as `REQUIRE p.email IS UNIQUE` (see the [Cypher guide](../cypher/mutations.md)).

### UNIQUE

Ensure a property value is unique across all nodes with a given label:

```sql
CREATE CONSTRAINT FOR (p:Person) ON (p.email) UNIQUE
```

### NODE KEY

A node key names the properties that together identify a node of a label, as a primary key does for a row in SQL:

```sql
CREATE CONSTRAINT person_key FOR (p:Person) ON (p.firstName, p.lastName) NODE KEY
```

- **Required**: every node with the label has a value for each key property. A write that leaves one out or sets it to `NULL` fails: an insert without it, `SET p.lastName = NULL`, `REMOVE p.lastName`, `SET p = {...}` without it, and `SET n:Person` on a node that lacks it.
- **Unique together**: no two nodes with the label hold the same values for all key properties. A key over several properties is unique as a whole, so `('Alix', 'Gus')` and `('Alix', 'Mia')` are two keys. Values compare as `=` does: `1` and `1.0` are one key, `1` and `'1'` are two, and lists compare element by element.
- **Changeable**: a key property may be updated to values no other node holds; an update to a key another node holds fails. Deleting a node, or removing the label from it, frees its key.
- **Checked at each write**: a key is checked as each property is written, not at the end of the statement, so two nodes cannot swap their keys in one `SET`. Move one of them to a free value first.
- **By label**: a node is held to the key when it has the label, whatever other labels it has. A label may have more than one key, and each holds.
- **Per graph**: the constraint belongs to the database and applies in every graph, those created later included, but each graph has its own keys: the same key may be used once in each graph.
- **Concurrent writers**: of two open transactions that write the same key, the second write fails with a write conflict (GQLSTATUS `40000`) right away, without waiting for the first to commit.

A write that breaks a key fails with `GRAFEO-V001`, and nothing of the statement is kept. Like `UNIQUE`, a node key builds an index on its first property, which lookups that name that property use (see below).

A `NODE KEY` on several properties does what `UNIQUE` on those properties together and `NOT NULL` on each of them do, in one constraint with one name. `UNIQUE` alone accepts nodes that lack a key property, as a missing value is never a duplicate.

`NODE KEY` is a Grafeo extension: ISO GQL has no constraint statements, and its `KEY` in a graph type is a set of labels that identifies an element type, not a property key. Cypher's spelling, `REQUIRE (p.firstName, p.lastName) IS NODE KEY`, is accepted too, and a `PRIMARY KEY` column in [SQL/PGQ](../sql-pgq/index.md) `CREATE PROPERTY GRAPH` is the same rule. It is part of the stable query language (see [Stability](../../stability.md)).

### NOT NULL

Require a property to always have a value:

```sql
CREATE CONSTRAINT FOR (p:Person) ON (p.name) NOT NULL
```

### EXISTS

Require a property to exist on every node with the label:

```sql
CREATE CONSTRAINT FOR (p:Person) ON (p.email) EXISTS
```

### Naming, Listing and Dropping Constraints

A constraint over data that already breaks it is refused with `GRAFEO-V001`, naming the constraint and a value that breaks it.

`UNIQUE` and `NODE KEY` build a property index on the constraint's first property, in every graph the constraint reaches and in each graph made later, so that checking an insert does not read the whole label. It holds the constraint's label only, so other labels with that property pay nothing for it, and two constraints on one property and different labels each have their own. An index already on the label and property, or on the property of every label, is used. A `NODE KEY` whose first property has few distinct values stays slower: list a property with many values first.

The index takes the constraint's name, when it was given one that no other index has. `DROP CONSTRAINT` leaves the index and says so, and `DROP INDEX` removes it. A constraint without its index still holds: it reads the label to check each write.

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
