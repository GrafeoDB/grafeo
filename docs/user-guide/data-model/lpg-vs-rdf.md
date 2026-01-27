---
title: LPG vs RDF
description: Comparing Labeled Property Graph and RDF data models.
tags:
  - data-model
  - lpg
  - rdf
---

# LPG vs RDF

Understanding the differences between Labeled Property Graph (LPG) and Resource Description Framework (RDF) models.

## Overview

| Aspect | LPG | RDF |
|--------|-----|-----|
| **Primary Use** | Application databases | Knowledge graphs, Linked Data |
| **Query Language** | GQL, Cypher | SPARQL |
| **Schema** | Optional, flexible | Ontologies (OWL, RDFS) |
| **Properties** | On nodes and edges | Predicates only |
| **Identity** | Internal IDs | URIs/IRIs |

## Data Model Comparison

### LPG (Labeled Property Graph)

```
Node:
  - ID: internal identifier
  - Labels: [Person, Employee]
  - Properties: {name: "Alice", age: 30}

Edge:
  - ID: internal identifier
  - Type: KNOWS
  - Source: Node1
  - Target: Node2
  - Properties: {since: 2020}
```

### RDF (Resource Description Framework)

```
Subject-Predicate-Object Triples:

<http://example.org/Alice> <rdf:type> <http://example.org/Person>
<http://example.org/Alice> <foaf:name> "Alice"
<http://example.org/Alice> <foaf:age> "30"^^xsd:integer
<http://example.org/Alice> <foaf:knows> <http://example.org/Bob>
```

## When to Use Each

### Use LPG When:

- Building application databases
- Need rich property support on relationships
- Working with developers familiar with OOP
- Performance is critical
- Schema flexibility is needed

### Use RDF When:

- Building knowledge graphs for data integration
- Need to link to external datasets (Linked Data)
- Require formal ontologies and reasoning
- Publishing data on the Semantic Web
- Need standardized vocabularies (FOAF, Schema.org)

## Feature Comparison

### Properties on Relationships

=== "LPG"

    ```sql
    -- Properties directly on the relationship
    MATCH (a)-[r:KNOWS {since: 2020, strength: 'close'}]->(b)
    RETURN r.since, r.strength
    ```

=== "RDF"

    ```sparql
    # Requires reification or named graphs
    # More complex to model
    SELECT ?since ?strength
    WHERE {
        << :Alice :knows :Bob >> :since ?since ;
                                 :strength ?strength .
    }
    ```

### Multiple Labels

=== "LPG"

    ```sql
    -- Node with multiple labels
    INSERT (:Person:Employee:Manager {name: 'Alice'})

    MATCH (p:Person:Employee)
    RETURN p.name
    ```

=== "RDF"

    ```sparql
    # Multiple types are natural
    INSERT DATA {
        :Alice a :Person, :Employee, :Manager ;
               :name "Alice" .
    }
    ```

### Identity

=== "LPG"

    ```sql
    -- Internal IDs, application-defined keys
    INSERT (:Person {id: 'user-123', name: 'Alice'})
    ```

=== "RDF"

    ```sparql
    # Global URIs enable linking
    INSERT DATA {
        <http://example.org/user-123> a :Person ;
            :name "Alice" .
    }
    ```

## Graphos Approach

Graphos primarily uses the **LPG model** because:

1. **Developer Experience** - More intuitive for application developers
2. **Performance** - Optimized storage and query execution
3. **Flexibility** - Easy schema evolution
4. **Rich Relationships** - First-class properties on edges

!!! note "RDF Support Status"
    Graphos currently focuses on the **LPG model**. RDF support is planned for future releases. This comparison is provided for users evaluating which data model fits their needs.

## Converting Between Models

### LPG to RDF

```python
# Conceptual mapping
# Node (id, labels, properties) ->
#   Subject-type triples + property triples

# Edge (type, source, target, properties) ->
#   Subject-predicate-object + reification for properties
```

### RDF to LPG

```python
# Subject -> Node
# Predicate -> Edge type or property key
# Object -> Target node or property value
# rdf:type -> Labels
```

## Further Reading

- [GQL vs SPARQL](gql-vs-sparql.md)
- [Data Model](index.md)
- [RDF Architecture](../../architecture/index.md)
