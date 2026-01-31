---
title: Quick Start
description: Get up and running with Grafeo in minutes.
---

# Quick Start

This guide will get you up and running with Grafeo in just a few minutes.

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
    let db = GrafeoDB::new("my_graph.db")?;
    ```

## Add Data

Use GQL to insert nodes and edges:

=== "Python"

    ```python
    # Create nodes
    db.execute("""
        INSERT (:Person {name: 'Alice', age: 30})
        INSERT (:Person {name: 'Bob', age: 25})
        INSERT (:Person {name: 'Carol', age: 35})
    """)

    # Create edges
    db.execute("""
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        INSERT (a)-[:KNOWS {since: 2020}]->(b)
    """)

    db.execute("""
        MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Carol'})
        INSERT (b)-[:KNOWS {since: 2022}]->(c)
    """)
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Create nodes
    session.execute(r#"
        INSERT (:Person {name: 'Alice', age: 30})
        INSERT (:Person {name: 'Bob', age: 25})
        INSERT (:Person {name: 'Carol', age: 35})
    "#)?;

    // Create edges
    session.execute(r#"
        MATCH (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'})
        INSERT (a)-[:KNOWS {since: 2020}]->(b)
    "#)?;

    session.execute(r#"
        MATCH (b:Person {name: 'Bob'}), (c:Person {name: 'Carol'})
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

    # Find who Alice knows
    result = db.execute("""
        MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(friend)
        RETURN friend.name
    """)

    for row in result:
        print(f"Alice knows {row['friend.name']}")

    # Find friends of friends
    result = db.execute("""
        MATCH (a:Person {name: 'Alice'})-[:KNOWS]->()-[:KNOWS]->(fof)
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

    for row in result.rows {
        println!("{:?}", row);
    }

    // Find who Alice knows
    let result = session.execute(r#"
        MATCH (a:Person {name: 'Alice'})-[:KNOWS]->(friend)
        RETURN friend.name
    "#)?;

    for row in result.rows {
        println!("Alice knows {:?}", row);
    }
    ```

## Update Data

Modify existing nodes and edges:

=== "Python"

    ```python
    # Update a property
    db.execute("""
        MATCH (p:Person {name: 'Alice'})
        SET p.age = 31
    """)

    # Add a new property
    db.execute("""
        MATCH (p:Person {name: 'Bob'})
        SET p.email = 'bob@example.com'
    """)
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Update a property
    session.execute(r#"
        MATCH (p:Person {name: 'Alice'})
        SET p.age = 31
    "#)?;
    ```

## Delete Data

Remove nodes and edges:

=== "Python"

    ```python
    # Delete an edge
    db.execute("""
        MATCH (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})
        DELETE r
    """)

    # Delete a node (must delete connected edges first)
    db.execute("""
        MATCH (p:Person {name: 'Carol'})
        DETACH DELETE p
    """)
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Delete a node and its edges
    session.execute(r#"
        MATCH (p:Person {name: 'Carol'})
        DETACH DELETE p
    "#)?;
    ```

## Transactions

For atomic operations, use transactions:

=== "Python"

    ```python
    # Begin a transaction
    with db.begin_transaction() as tx:
        tx.execute("INSERT (:Person {name: 'Dave'})")
        tx.execute("INSERT (:Person {name: 'Eve'})")
        tx.commit()  # Both inserts committed atomically

    # Or rollback on error
    with db.begin_transaction() as tx:
        tx.execute("INSERT (:Person {name: 'Frank'})")
        tx.rollback()  # Changes discarded
    ```

=== "Rust"

    ```rust
    let mut session = db.session();

    // Begin a transaction
    session.begin_tx()?;

    session.execute("INSERT (:Person {name: 'Dave'})")?;
    session.execute("INSERT (:Person {name: 'Eve'})")?;

    session.commit()?;  // Both inserts committed atomically
    ```

## Next Steps

- [Your First Graph](first-graph.md) - Build a complete graph application
- [GQL Query Language](../user-guide/gql/index.md) - Learn more about queries
- [Python API](../user-guide/python/index.md) - Python-specific features
- [Rust API](../user-guide/rust/index.md) - Rust-specific features
