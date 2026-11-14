---
title: Composite Queries
description: Combine GQL queries with UNION, EXCEPT, INTERSECT and OTHERWISE.
tags:
  - gql
  - composite
  - set-operations
---

# Composite Queries

Combine multiple query results using set operations.

## UNION

Combine results from two queries. Both sides must return the same columns.

### UNION ALL

Keep all rows, including duplicates:

```sql
-- All people and companies
MATCH (p:Person) RETURN p.name AS name, 'Person' AS type
UNION ALL
MATCH (c:Company) RETURN c.name AS name, 'Company' AS type
```

### UNION DISTINCT

Remove duplicate rows:

```sql
-- Unique names across people and companies
MATCH (p:Person) RETURN p.name AS name
UNION DISTINCT
MATCH (c:Company) RETURN c.name AS name
```

## EXCEPT

Return rows from the first query that are not in the second.

### EXCEPT (Distinct)

```sql
-- People who are not managers
MATCH (p:Person) RETURN p.name
EXCEPT
MATCH (m:Person)-[:MANAGES]->() RETURN m.name
```

### EXCEPT ALL

Multiset difference, preserving duplicate counts:

```sql
-- Remove one occurrence per match
MATCH (p:Person) RETURN p.name
EXCEPT ALL
MATCH (m:Person)-[:MANAGES]->() RETURN m.name
```

## INTERSECT

Return only rows that appear in both queries.

### INTERSECT (Distinct)

```sql
-- People who are both engineers and managers
MATCH (p:Person {role: 'engineer'}) RETURN p.name
INTERSECT
MATCH (m:Person)-[:MANAGES]->() RETURN m.name
```

### INTERSECT ALL

Multiset intersection, preserving duplicate counts:

```sql
MATCH (p:Person {dept: 'Engineering'}) RETURN p.name
INTERSECT ALL
MATCH (p:Person {dept: 'Engineering'}) RETURN p.name
```

## OTHERWISE

Return the left query's results if non-empty; otherwise return the right query's results. Useful for providing defaults or fallback results.

```sql
-- Try to find premium results; fall back to standard
MATCH (p:Product) WHERE p.premium = true
RETURN p.name, p.price
OTHERWISE
MATCH (p:Product) WHERE p.featured = true
RETURN p.name, p.price

-- Default message when no results exist
MATCH (a:Alert) WHERE a.severity = 'critical'
RETURN a.message AS result
OTHERWISE
RETURN 'No critical alerts' AS result
```
