---
title: Sessions
description: Session management in Rust.
tags:
  - rust
  - sessions
---

# Sessions

Sessions provide transactional access to the database.

## Creating Sessions

```rust
let db = GrafeoDB::new_in_memory();

// Create a session
let session = db.session();

// Use the session
session.execute("INSERT (:Person {name: 'Alix'})")?;
```

## Transactions

```rust
let mut session = db.session();

// Begin explicit transaction
session.begin_transaction()?;

session.execute("INSERT (:Person {name: 'Alix'})")?;
session.execute("INSERT (:Person {name: 'Gus'})")?;

// Commit
session.commit()?;
```

## Rollback

```rust
let mut session = db.session();

session.begin_transaction()?;
session.execute("INSERT (:Person {name: 'Alix'})")?;

// Something went wrong, rollback
session.rollback()?;
```

## Multiple Sessions

```rust
let db = GrafeoDB::new_in_memory();

// Each session has isolated transactions
let mut session1 = db.session();
let session2 = db.session();

session1.begin_transaction()?;
session1.execute("INSERT (:Person {name: 'Alix'})")?;
// session2 won't see Alix until session1 commits

session1.commit()?;
// Now session2 can see Alix
```
