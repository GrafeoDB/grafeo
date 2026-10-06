---
title: String Functions
description: GQL string manipulation functions in Grafeo.
tags:
  - gql
  - functions
  - strings
---

# String Functions

## Summary

| Function | Description |
|----------|-------------|
| `toUpper(str)` / `upper(str)` | Convert to uppercase |
| `toLower(str)` / `lower(str)` | Convert to lowercase |
| `trim(str)` | Remove leading/trailing whitespace |
| `TRIM(BOTH\|LEADING\|TRAILING 'chars' FROM str)` | Trim specific characters |
| `ltrim(str)` | Remove leading whitespace |
| `rtrim(str)` | Remove trailing whitespace |
| `substring(str, start, len)` | Extract substring |
| `char_length(str)` | Character count |
| `octet_length(str)` | Byte length (UTF-8) |
| `replace(str, search, repl)` | Replace occurrences |
| `split(str, delim)` | Split into list |
| `left(str, n)` | First n characters |
| `right(str, n)` | Last n characters |
| `reverse(str)` | Reverse characters |
| `normalize(str)` | Unicode normalization |
| `string_join(list, sep)` | Join list into string |
| `toString(expr)` | Convert to string |

## Case Conversion

```sql
RETURN toUpper('hello')   -- 'HELLO'
RETURN upper('hello')     -- 'HELLO'

RETURN toLower('HELLO')   -- 'hello'
RETURN lower('HELLO')     -- 'hello'
```

## Trimming

```sql
RETURN trim('  hello  ')    -- 'hello'
RETURN ltrim('  hello  ')   -- 'hello  '
RETURN rtrim('  hello  ')   -- '  hello'

-- ISO enhanced TRIM with trim specification (GF05)
RETURN TRIM(BOTH 'xy' FROM 'xxyhelloxyy')     -- 'hello'
RETURN TRIM(LEADING '0' FROM '000123')         -- '123'
RETURN TRIM(TRAILING '.' FROM 'hello...')      -- 'hello'
```

## Substring and Length

```sql
-- Extract substring: substring(string, start, length)
-- Start is 0-based
RETURN substring('hello world', 0, 5)   -- 'hello'
RETURN substring('hello world', 6, 5)   -- 'world'

-- Character count
RETURN char_length('hello')     -- 5
RETURN charlength('hello')      -- 5 (alias)

-- Byte length (differs from char_length for multibyte characters)
RETURN octet_length('hello')    -- 5
RETURN octet_length('café')     -- 5 (UTF-8: é is 2 bytes)
```

## Search and Replace

```sql
-- Replace all occurrences
RETURN replace('hello world', 'world', 'GQL')   -- 'hello GQL'

-- Split into a list
RETURN split('a,b,c', ',')   -- ['a', 'b', 'c']

-- Practical: split and process tags
MATCH (p:Post)
UNWIND split(p.tags, ',') AS tag
RETURN trim(tag), count(*) AS usage
ORDER BY usage DESC
```

## Extraction

```sql
-- First n characters
RETURN left('hello world', 5)    -- 'hello'

-- Last n characters
RETURN right('hello world', 5)   -- 'world'
```

## Transformation

```sql
RETURN reverse('hello')    -- 'olleh'

-- Unicode normalization (identity for UTF-8 strings)
RETURN normalize('café')   -- 'café'
```

## String Joining

Join a list of strings with a separator:

```sql
RETURN string_join(['Alix', 'Gus', 'Harm'], ', ')
-- 'Alix, Gus, Harm'

-- Practical: collect names and join
MATCH (p:Person)-[:WORKS_AT]->(c:Company)
WITH c, collect(p.name) AS employees
RETURN c.name, string_join(employees, ', ') AS team
```

## String Concatenation

Use the `||` operator to concatenate strings:

```sql
RETURN 'Hello' || ' ' || 'World'   -- 'Hello World'

MATCH (p:Person)
RETURN p.firstName || ' ' || p.lastName AS full_name
```

## toString

Convert any value to its string representation:

```sql
RETURN toString(42)       -- '42'
RETURN toString(3.14)     -- '3.14'
RETURN toString(true)     -- 'true'
RETURN toString(null)     -- null
```

## Pattern Matching

GQL provides several string matching operators in `WHERE` clauses. See [Filtering](filtering.md) for details.

```sql
WHERE p.name STARTS WITH 'Al'
WHERE p.name ENDS WITH 'son'
WHERE p.name CONTAINS 'li'
WHERE p.name LIKE 'Al%'           -- SQL-style wildcards
WHERE p.email =~ '.*@gmail\\.com' -- Regular expression
```
