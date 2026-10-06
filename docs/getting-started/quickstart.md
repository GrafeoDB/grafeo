---
title: Quick Start
description: Get up and running with Grafeo in minutes.
---

# Quick Start

This guide covers getting up and running with Grafeo in just a few minutes.

## Create a Database

=== "Python"

    ```python
    import grafeo

    # Create an in-memory database
    db = grafeo.GrafeoDB()

    # Or create a persistent database
    db = grafeo.GrafeoDB("my_graph.db")
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    // Create an in-memory database
    let db = GrafeoDB::new_in_memory();

    // Or create a persistent database
    let db = GrafeoDB::open("my_graph.db")?;
    ```

## Add Data

Use GQL to insert nodes and edges:

=== "Python"

    ```python
    # Create nodes
    db.execute("""
        INSERT (:Person {name: 'Alix', age: 30})
        INSERT (:Person {name: 'Gus', age: 25})
        INSERT (:Person {name: 'Harm', age: 35})
    """)

    # Create edges
    db.execute("""
        MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
        INSERT (a)-[:KNOWS {since: 2020}]->(b)
    """)

    db.execute("""
        MATCH (b:Person {name: 'Gus'}), (c:Person {name: 'Harm'})
        INSERT (b)-[:KNOWS {since: 2022}]->(c)
    """)
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Create nodes
    session.execute(r#"
        INSERT (:Person {name: 'Alix', age: 30})
        INSERT (:Person {name: 'Gus', age: 25})
        INSERT (:Person {name: 'Harm', age: 35})
    "#)?;

    // Create edges
    session.execute(r#"
        MATCH (a:Person {name: 'Alix'}), (b:Person {name: 'Gus'})
        INSERT (a)-[:KNOWS {since: 2020}]->(b)
    "#)?;

    session.execute(r#"
        MATCH (b:Person {name: 'Gus'}), (c:Person {name: 'Harm'})
        INSERT (b)-[:KNOWS {since: 2022}]->(c)
    "#)?;
    ```

## Query Data

Retrieve data using pattern matching:

=== "Python"

    ```python
    # Find all people
    result = db.execute("""
        MATCH (p:Person)
        RETURN p.name, p.age
        ORDER BY p.age
    """)

    for row in result:
        print(f"{row['p.name']} is {row['p.age']} years old")

    # Find who Alix knows
    result = db.execute("""
        MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(friend)
        RETURN friend.name
    """)

    for row in result:
        print(f"Alix knows {row['friend.name']}")

    # Find friends of friends
    result = db.execute("""
        MATCH (a:Person {name: 'Alix'})-[:KNOWS]->()-[:KNOWS]->(fof)
        RETURN DISTINCT fof.name
    """)

    for row in result:
        print(f"Friend of friend: {row['fof.name']}")
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Find all people
    let result = session.execute(r#"
        MATCH (p:Person)
        RETURN p.name, p.age
        ORDER BY p.age
    "#)?;

    for row in result.rows() {
        println!("{:?}", row);
    }

    // Find who Alix knows
    let result = session.execute(r#"
        MATCH (a:Person {name: 'Alix'})-[:KNOWS]->(friend)
        RETURN friend.name
    "#)?;

    for row in result.rows() {
        println!("Alix knows {:?}", row);
    }
    ```

## Update Data

Modify existing nodes and edges:

=== "Python"

    ```python
    # Update a property
    db.execute("""
        MATCH (p:Person {name: 'Alix'})
        SET p.age = 31
    """)

    # Add a new property
    db.execute("""
        MATCH (p:Person {name: 'Gus'})
        SET p.email = 'gus@example.com'
    """)
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Update a property
    session.execute(r#"
        MATCH (p:Person {name: 'Alix'})
        SET p.age = 31
    "#)?;
    ```

## Delete Data

Remove nodes and edges:

=== "Python"

    ```python
    # Delete an edge
    db.execute("""
        MATCH (a:Person {name: 'Alix'})-[r:KNOWS]->(b:Person {name: 'Gus'})
        DELETE r
    """)

    # Delete a node (must delete connected edges first)
    db.execute("""
        MATCH (p:Person {name: 'Harm'})
        DETACH DELETE p
    """)
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Delete a node and its edges
    session.execute(r#"
        MATCH (p:Person {name: 'Harm'})
        DETACH DELETE p
    "#)?;
    ```

## Transactions

For atomic operations, use transactions:

=== "Python"

    ```python
    # Begin a transaction
    with db.begin_transaction() as tx:
        tx.execute("INSERT (:Person {name: 'Vincent'})")
        tx.execute("INSERT (:Person {name: 'Jules'})")
        tx.commit()  # Both inserts committed atomically

    # Or rollback on error
    with db.begin_transaction() as tx:
        tx.execute("INSERT (:Person {name: 'Mia'})")
        tx.rollback()  # Changes discarded
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Begin a transaction
    session.begin_transaction()?;

    session.execute("INSERT (:Person {name: 'Vincent'})")?;
    session.execute("INSERT (:Person {name: 'Jules'})")?;

    session.commit()?;  // Both inserts committed atomically
    ```

## Next Steps

- [First Graph](first-graph.md) - Build a complete graph application
- [GQL Query Language](../user-guide/gql/index.md) - Learn more about queries
- [Python API](../user-guide/python/index.md) - Python-specific features
- [Rust API](../user-guide/rust/index.md) - Rust-specific features
