---
title: Filtering
description: Filtering results with WHERE clauses in GQL.
tags:
  - gql
  - filtering
---

# Filtering

The `WHERE` clause filters results based on conditions.

## Comparison Operators

```sql
-- Equality
WHERE p.name = 'Alix'

-- Inequality
WHERE p.age <> 30

-- Greater/less than
WHERE p.age > 25
WHERE p.age < 40
WHERE p.age >= 25
WHERE p.age <= 40
```

## Boolean Logic

```sql
-- AND
WHERE p.age > 25 AND p.active = true

-- OR
WHERE p.city = 'NYC' OR p.city = 'LA'

-- NOT
WHERE NOT p.archived

-- Combined
WHERE (p.age > 25 AND p.active) OR p.role = 'admin'
```

## String Operations

```sql
-- Starts with
WHERE p.name STARTS WITH 'Al'

-- Ends with
WHERE p.email ENDS WITH '@company.com'

-- Contains
WHERE p.bio CONTAINS 'engineer'

-- Regular expression
WHERE p.email =~ '.*@gmail\\.com'
```

## List Operations

```sql
-- IN list
WHERE p.status IN ['active', 'pending']

-- Element in property list
WHERE 'admin' IN p.roles
```

## Null Checks

```sql
-- Is null
WHERE p.email IS NULL

-- Is not null
WHERE p.email IS NOT NULL
```

## Property Existence

```sql
-- Property exists
WHERE p.email IS NOT NULL

-- Combined with value check
WHERE p.age IS NOT NULL AND p.age > 18
```

## Path Filtering

```sql
-- Filter based on relationship properties
MATCH (a)-[r:KNOWS]->(b)
WHERE r.since > 2020
RETURN a.name, b.name
```

## LIKE Pattern Matching

SQL-style pattern matching with `%` (any characters) and `_` (single character):

```sql
-- Names starting with 'Al'
WHERE p.name LIKE 'Al%'

-- Names ending with 'son'
WHERE p.name LIKE '%son'

-- Names with exactly 5 characters
WHERE p.name LIKE '_____'

-- Second character is 'l'
WHERE p.name LIKE '_l%'
```

## XOR (Exclusive Or)

`XOR` is true when exactly one of the two conditions is true:

```sql
-- Active in one system but not both
MATCH (u:User)
WHERE u.active_in_crm XOR u.active_in_erp
RETURN u.name
```

## Type Checking

### IS TYPED / IS NOT TYPED

Check the runtime type of a value:

```sql
-- Find nodes where age is stored as an integer
MATCH (p:Person)
WHERE p.age IS TYPED INTEGER
RETURN p.name, p.age

-- Find mistyped values
MATCH (p:Person)
WHERE p.age IS NOT TYPED INTEGER
RETURN p.name, p.age
```

## Graph Element Predicates

### IS DIRECTED / IS NOT DIRECTED

Check edge directionality:

```sql
MATCH ()-[r]-()
WHERE r IS DIRECTED
RETURN type(r)
```

### IS LABELED / IS NOT LABELED

Check if a node or edge has a specific label:

```sql
MATCH (n)
WHERE n IS LABELED Person
RETURN n.name

MATCH (n)
WHERE n IS NOT LABELED Inactive
RETURN n.name
```

### IS SOURCE OF / IS DESTINATION OF

Check whether a node is the source or destination of an edge:

```sql
MATCH (a)-[r:KNOWS]-(b)
WHERE a IS SOURCE OF r
RETURN a.name AS from, b.name AS to
```

### ALL_DIFFERENT

Check that all elements in the argument list are distinct:

```sql
MATCH (a)-[:KNOWS]->(b)-[:KNOWS]->(c)
WHERE ALL_DIFFERENT(a, b, c)
RETURN a.name, b.name, c.name
```

### SAME

Check that all elements in the argument list are equal:

```sql
MATCH (a)-[:KNOWS]->(b), (a)-[:WORKS_WITH]->(c)
WHERE SAME(b, c)
RETURN a.name, b.name
```

### PROPERTY_EXISTS

Check whether a property key exists on an entity:

```sql
MATCH (p:Person)
WHERE PROPERTY_EXISTS(p, 'email')
RETURN p.name, p.email
```

## Unicode Normalization

### IS [NOT] NORMALIZED

Check whether a string is in a specific Unicode normalization form. The default form is NFC.

```sql
-- Default (NFC) normalization check
MATCH (p:Person)
WHERE p.name IS NORMALIZED
RETURN p.name

-- Specific normalization forms: NFC, NFD, NFKC, NFKD
MATCH (p:Person)
WHERE p.name IS NFC NORMALIZED
RETURN p.name

MATCH (p:Person)
WHERE p.name IS NOT NFKD NORMALIZED
RETURN p.name
```
