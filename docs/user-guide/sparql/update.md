---
title: SPARQL Update
description: Insert, delete, and manage RDF data with SPARQL Update operations.
tags:
  - sparql
  - update
  - rdf
  - named-graphs
---

# SPARQL Update

Grafeo implements SPARQL 1.1 Update for modifying RDF data. This includes inserting and deleting triples, pattern-based modifications, and named graph management operations.

## INSERT DATA

Add explicit triples to the default graph:

```sparql
INSERT DATA {
    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/age> 30 .
    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/gus> .
}
```

Insert into a named graph:

```sparql
INSERT DATA {
    GRAPH <http://ex.org/friends> {
        <http://ex.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/gus> .
        <http://ex.org/gus> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/alix> .
    }
}
```

Duplicate triples are silently skipped.

## DELETE DATA

Remove explicit triples from the default graph:

```sparql
DELETE DATA {
    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/age> 30 .
}
```

Remove from a named graph:

```sparql
DELETE DATA {
    GRAPH <http://ex.org/friends> {
        <http://ex.org/alix> <http://xmlns.com/foaf/0.1/knows> <http://ex.org/gus> .
    }
}
```

Deleting a triple that does not exist is a no-op (no error is raised).

## DELETE WHERE

Remove all triples matching a pattern. Unlike DELETE DATA, this accepts variables:

```sparql
DELETE WHERE {
    <http://ex.org/alix> <http://xmlns.com/foaf/0.1/knows> ?friend .
}
```

This removes all `foaf:knows` triples where Alix is the subject.

## INSERT ... WHERE

Insert new triples derived from existing data:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

INSERT {
    ?person foaf:status "verified" .
}
WHERE {
    ?person a <http://ex.org/Person> .
    ?person foaf:age ?age
    FILTER(?age >= 18)
}
```

This finds all persons aged 18 or older and adds a `foaf:status "verified"` triple for each.

## DELETE ... WHERE

Delete triples based on a pattern match:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE {
    ?person foaf:status ?status .
}
WHERE {
    ?person foaf:status ?status .
    ?person foaf:age ?age
    FILTER(?age < 18)
}
```

## DELETE/INSERT (Modify)

Combine deletion and insertion in a single operation. The WHERE clause is evaluated first, then deletions are applied, then insertions:

```sparql
PREFIX foaf: <http://xmlns.com/foaf/0.1/>

DELETE {
    ?person foaf:status "pending" .
}
INSERT {
    ?person foaf:status "active" .
}
WHERE {
    ?person foaf:status "pending" .
    ?person foaf:age ?age
    FILTER(?age >= 18)
}
```

This atomically changes the status from "pending" to "active" for qualifying persons.

### WITH Clause

The `WITH` clause sets a default graph for the operation:

```sparql
WITH <http://ex.org/people>
DELETE {
    ?person <http://ex.org/status> "old" .
}
INSERT {
    ?person <http://ex.org/status> "new" .
}
WHERE {
    ?person <http://ex.org/status> "old" .
}
```

## Named Graph Operations

Grafeo supports the full set of SPARQL 1.1 graph management operations.

### CREATE GRAPH

Create an empty named graph:

```sparql
CREATE GRAPH <http://ex.org/newgraph>
```

Use `SILENT` to suppress errors if the graph already exists:

```sparql
CREATE SILENT GRAPH <http://ex.org/newgraph>
```

### DROP GRAPH

Remove a named graph and all its triples:

```sparql
DROP GRAPH <http://ex.org/oldgraph>
```

Drop the default graph:

```sparql
DROP DEFAULT
```

Drop all graphs:

```sparql
DROP ALL
```

Use `SILENT` to suppress errors if the graph does not exist:

```sparql
DROP SILENT GRAPH <http://ex.org/missing>
```

### CLEAR

Remove all triples from a graph without dropping it:

```sparql
CLEAR GRAPH <http://ex.org/mygraph>
```

Clear the default graph:

```sparql
CLEAR DEFAULT
```

Clear all graphs:

```sparql
CLEAR ALL
```

### COPY

Replace the destination graph with the contents of the source graph:

```sparql
COPY <http://ex.org/source> TO <http://ex.org/dest>
```

The destination is completely replaced. The source graph is not modified. Use `SILENT` to suppress errors if the source does not exist:

```sparql
COPY SILENT <http://ex.org/source> TO <http://ex.org/dest>
```

### MOVE

Move all triples from the source graph to the destination (replacing it), then drop the source:

```sparql
MOVE <http://ex.org/staging> TO <http://ex.org/production>
```

Use `SILENT` to suppress errors if the source does not exist:

```sparql
MOVE SILENT <http://ex.org/staging> TO <http://ex.org/production>
```

### ADD

Merge the source graph into the destination. Existing triples in the destination are preserved:

```sparql
ADD <http://ex.org/extra> TO <http://ex.org/main>
```

Use `SILENT` to suppress errors if the source does not exist:

```sparql
ADD SILENT <http://ex.org/extra> TO <http://ex.org/main>
```

## Complete Example

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB()

    # Insert initial data
    db.execute_sparql("""
        INSERT DATA {
            <http://ex.org/alix> a <http://ex.org/Person> ;
                <http://xmlns.com/foaf/0.1/name> "Alix" ;
                <http://xmlns.com/foaf/0.1/age> 30 .
            <http://ex.org/gus> a <http://ex.org/Person> ;
                <http://xmlns.com/foaf/0.1/name> "Gus" ;
                <http://xmlns.com/foaf/0.1/age> 25 .
        }
    """)

    # Pattern-based insert: add a label for everyone
    db.execute_sparql("""
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        INSERT {
            ?person <http://www.w3.org/2000/01/rdf-schema#label> ?name .
        }
        WHERE {
            ?person a <http://ex.org/Person> .
            ?person foaf:name ?name
        }
    """)

    # Delete specific data
    db.execute_sparql("""
        DELETE DATA {
            <http://ex.org/gus> <http://xmlns.com/foaf/0.1/age> 25 .
        }
    """)

    # Copy a named graph
    db.execute_sparql("""
        INSERT DATA {
            GRAPH <http://ex.org/backup> {
                <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
            }
        }
    """)
    db.execute_sparql("COPY <http://ex.org/backup> TO <http://ex.org/archive>")

    # Verify
    result = db.execute_sparql("""
        SELECT ?s ?p ?o WHERE { GRAPH <http://ex.org/archive> { ?s ?p ?o } }
    """)
    for row in result:
        print(row)
    ```

=== "Rust"

    ```rust
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    session.execute_sparql(r#"
        INSERT DATA {
            <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
        }
    "#)?;

    session.execute_sparql(r#"
        DELETE DATA {
            <http://ex.org/alix> <http://xmlns.com/foaf/0.1/name> "Alix" .
        }
    "#)?;
    ```

## Summary

| Operation | Accepts Variables? | Description |
|-----------|-------------------|-------------|
| `INSERT DATA` | No | Add explicit triples |
| `DELETE DATA` | No | Remove explicit triples |
| `DELETE WHERE` | Yes | Remove triples matching a pattern |
| `INSERT ... WHERE` | Yes | Add triples derived from a pattern |
| `DELETE ... INSERT ... WHERE` | Yes | Atomic delete-then-insert |
| `CREATE GRAPH` | No | Create an empty named graph |
| `DROP GRAPH` | No | Remove a named graph |
| `CLEAR` | No | Remove all triples from a graph |
| `COPY` | No | Replace destination with source |
| `MOVE` | No | Move source to destination, drop source |
| `ADD` | No | Merge source into destination |
