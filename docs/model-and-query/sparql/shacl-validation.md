---
title: SHACL Validation
description: Validate RDF data against SHACL shapes in Grafeo.
tags:
  - sparql
  - shacl
  - validation
  - rdf
---

# SHACL Validation

SHACL (Shapes Constraint Language) is a W3C standard for validating RDF graphs against a set of conditions called "shapes." Grafeo implements the SHACL Core specification, letting you define shapes in a named graph and validate your data programmatically.

## Overview

A SHACL workflow has three parts:

1. **Define shapes** in a named graph (using SPARQL INSERT DATA).
2. **Validate** the default graph against those shapes.
3. **Inspect the report** to find violations.

Shapes can be **node shapes** (constraints on focus nodes directly) or **property shapes** (constraints on values reachable via a property path).

## Quick Start

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB()

    # Load data into the default graph
    db.execute_sparql("""
        INSERT DATA {
            <http://ex.org/alix> a <http://ex.org/Person> ;
                <http://ex.org/name> "Alix" ;
                <http://ex.org/age> 30 .
        }
    """)

    # Define shapes in a named graph
    db.execute_sparql("""
        INSERT DATA {
            GRAPH <http://ex.org/shapes> {
                <http://ex.org/PersonShape> a <http://www.w3.org/ns/shacl#NodeShape> ;
                    <http://www.w3.org/ns/shacl#targetClass> <http://ex.org/Person> ;
                    <http://www.w3.org/ns/shacl#property> [
                        <http://www.w3.org/ns/shacl#path> <http://ex.org/name> ;
                        <http://www.w3.org/ns/shacl#minCount> 1 ;
                        <http://www.w3.org/ns/shacl#datatype> <http://www.w3.org/2001/XMLSchema#string>
                    ] .
            }
        }
    """)

    # Validate
    report = db.validate_shacl("http://ex.org/shapes")
    print(report["conforms"])       # True
    print(report["results_text"])   # Validation Report: PASSED
    ```

=== "Rust"

    ```rust
    use grafeo_engine::GrafeoDB;

    let db = GrafeoDB::new_in_memory();
    let session = db.session();

    // Insert data and shapes (omitted for brevity)
    session.execute_sparql(/* ... */)?;

    let report = session.validate_shacl("http://ex.org/shapes")?;
    assert!(report.conforms);
    ```

## Supported Constraint Types

Grafeo supports all SHACL Core constraint types, listed below by category.

### Value Type Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Class | `sh:class` | Value nodes must be instances of the given class |
| Datatype | `sh:datatype` | Value nodes must have the given datatype |
| Node Kind | `sh:nodeKind` | Value nodes must be an IRI, literal, or blank node |

Node kind values: `sh:BlankNode`, `sh:IRI`, `sh:Literal`, `sh:BlankNodeOrIRI`, `sh:BlankNodeOrLiteral`, `sh:IRIOrLiteral`.

### Cardinality Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Min Count | `sh:minCount` | Minimum number of value nodes |
| Max Count | `sh:maxCount` | Maximum number of value nodes |

### Value Range Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Min Exclusive | `sh:minExclusive` | Value must be greater than the bound |
| Max Exclusive | `sh:maxExclusive` | Value must be less than the bound |
| Min Inclusive | `sh:minInclusive` | Value must be greater than or equal to the bound |
| Max Inclusive | `sh:maxInclusive` | Value must be less than or equal to the bound |

### String Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Min Length | `sh:minLength` | Minimum string length |
| Max Length | `sh:maxLength` | Maximum string length |
| Pattern | `sh:pattern` | Regex pattern (with optional `sh:flags`) |
| Language In | `sh:languageIn` | Allowed language tags |
| Unique Lang | `sh:uniqueLang` | Language tags must be unique across value nodes |

### Property Pair Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Equals | `sh:equals` | Values must equal values of another property |
| Disjoint | `sh:disjoint` | Values must not overlap with another property |
| Less Than | `sh:lessThan` | Each value must be less than values of another property |
| Less Than Or Equals | `sh:lessThanOrEquals` | Each value must be <= values of another property |

### Logical Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Not | `sh:not` | Focus node must NOT conform to the given shape |
| And | `sh:and` | Focus node must conform to ALL shapes in the list |
| Or | `sh:or` | Focus node must conform to at least one shape |
| Xone | `sh:xone` | Focus node must conform to exactly one shape |

### Shape-Based Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Node | `sh:node` | Each value node must conform to the given shape |
| Qualified Value Shape | `sh:qualifiedValueShape` | Qualified cardinality on conforming values |

### Other Constraints

| Constraint | SHACL Property | Description |
|-----------|---------------|-------------|
| Closed | `sh:closed` | Only declared properties are allowed |
| Has Value | `sh:hasValue` | Value set must contain the given value |
| In | `sh:in` | Each value node must be in the given list |
| SPARQL | `sh:sparql` | Custom SPARQL-based constraint (see below) |

## Property Paths

SHACL property shapes use `sh:path` to specify how to reach value nodes from the focus node. Grafeo supports all 7 SHACL property path types:

| Path Type | Syntax | Description |
|----------|--------|-------------|
| Predicate | `sh:path ex:name` | Simple predicate (an IRI) |
| Inverse | `sh:inversePath` | Traverse the predicate in reverse |
| Sequence | RDF list of paths | Traverse each path in order |
| Alternative | `sh:alternativePath` | Union of results from each path |
| Zero or More | `sh:zeroOrMorePath` | Zero or more repetitions |
| One or More | `sh:oneOrMorePath` | One or more repetitions |
| Zero or One | `sh:zeroOrOnePath` | Zero or one repetition |

### Example: Inverse Path

```sparql
INSERT DATA {
    GRAPH <http://ex.org/shapes> {
        <http://ex.org/S> a <http://www.w3.org/ns/shacl#NodeShape> ;
            <http://www.w3.org/ns/shacl#targetClass> <http://ex.org/Person> ;
            <http://www.w3.org/ns/shacl#property> [
                <http://www.w3.org/ns/shacl#path> [
                    <http://www.w3.org/ns/shacl#inversePath> <http://ex.org/worksAt>
                ] ;
                <http://www.w3.org/ns/shacl#minCount> 1
            ] .
    }
}
```

This shape requires that every Person must be the object of at least one `ex:worksAt` triple.

## SHACL-SPARQL Constraints

For constraints that go beyond the built-in types, Grafeo supports `sh:sparql`, which lets you write a custom SPARQL SELECT query as a constraint. If the query returns any rows, those rows are reported as violations.

```sparql
INSERT DATA {
    GRAPH <http://ex.org/shapes> {
        <http://ex.org/AgeShape> a <http://www.w3.org/ns/shacl#NodeShape> ;
            <http://www.w3.org/ns/shacl#targetClass> <http://ex.org/Person> ;
            <http://www.w3.org/ns/shacl#sparql> [
                <http://www.w3.org/ns/shacl#select> """
                    SELECT $this WHERE {
                        $this <http://ex.org/age> ?age .
                        FILTER(?age < 0)
                    }
                """ ;
                <http://www.w3.org/ns/shacl#message> "Age must not be negative"
            ] .
    }
}
```

The `$this` variable is bound to each focus node during validation. Prefix declarations can be attached via `sh:prefixes` and `sh:declare`.

## Cycle Detection

Grafeo detects cycles in shape references. If a shape refers back to itself (directly or through a chain of `sh:node`, `sh:not`, `sh:and`, etc.), the validator tracks visited (focus node, shape) pairs and stops recursion without producing spurious violations. Transitive property paths (`sh:zeroOrMorePath`, `sh:oneOrMorePath`) are bounded by a configurable maximum depth to prevent infinite expansion in cyclic RDF data.

## Interpreting the Validation Report

The `validate_shacl()` method returns a report with these fields:

=== "Python"

    ```python
    report = db.validate_shacl("http://ex.org/shapes")

    # Top-level fields
    report["conforms"]       # bool: True if no violations
    report["results_text"]   # str: Human-readable summary

    # Individual results
    for r in report["results"]:
        print(r["focus_node"])    # The node that was validated
        print(r["severity"])      # "Violation", "Warning", or "Info"
        print(r["message"])       # Human-readable description (if set)
        print(r["path"])          # The property path (if applicable)
        print(r["value"])         # The offending value (if applicable)
        print(r["constraint"])    # The constraint component IRI
        print(r["source_shape"])  # The shape that produced this result
    ```

Example output for a failing validation:

```text
Validation Report: FAILED (1 violation(s), 0 warning(s))
  [Violation] <http://ex.org/alix> - Value does not have class <http://ex.org/Person>
```

## Severity Levels

Each shape can specify a severity level via `sh:severity`:

| Level | IRI | Meaning |
|-------|-----|---------|
| Violation | `sh:Violation` | A constraint was violated (default) |
| Warning | `sh:Warning` | Advisory, does not cause `conforms` to be `false` |
| Info | `sh:Info` | Informational, does not cause `conforms` to be `false` |

Only `Violation`-level results cause `conforms` to be `false`.

## Target Declarations

Shapes use targets to specify which nodes to validate:

| Target | SHACL Property | Description |
|--------|---------------|-------------|
| Target Class | `sh:targetClass` | All instances of the given class (via `rdf:type`) |
| Target Node | `sh:targetNode` | A specific node |
| Target Subjects Of | `sh:targetSubjectsOf` | All subjects of triples with the given predicate |
| Target Objects Of | `sh:targetObjectsOf` | All objects of triples with the given predicate |
