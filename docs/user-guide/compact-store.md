---
title: Compact Store
description: What compact() does since 0.6.0, and how databases compacted by 0.5.44 or older open.
tags:
  - storage
  - compact-store
---

# Compact Store

Up to 0.5.44, `compact()` converted the default graph into a separate columnar store: a
read-only base, with an overlay on top that took the writes made after it. Since 0.6.0 a
database keeps one store. `compact()` writes a checkpoint of a persistent database (an
in-memory or read-only one writes none), drops the old versions no open transaction can see
any more, in every graph, and reports what it did: whether it wrote a checkpoint, how many
versions it dropped, and how long it took. Writes after it are logged and recovered like
any other, and transactions, indexes and named graphs work the same before and after it.
`recompact()` (Rust) is a deprecated alias of `compact()`.

=== "Python"

    ```python
    import grafeo

    db = grafeo.GrafeoDB("people.grafeo")
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")
    report = db.compact()
    # {'checkpointed': True, 'versions_collected': 0, 'duration_ms': 4}
    db.execute("INSERT (:Person {name: 'Gus', age: 25})")
    ```

=== "Rust"

    ```rust
    use grafeo::GrafeoDB;

    let mut db = GrafeoDB::open("people.grafeo")?;
    db.execute("INSERT (:Person {name: 'Alix', age: 30})")?;
    let report = db.compact()?;
    assert!(report.checkpointed);
    db.execute("INSERT (:Person {name: 'Gus', age: 25})")?;
    ```

## Databases Compacted by 0.5.44 or Older

A file written after `compact()` by 0.5.44 or older holds the compacted base, the nodes and
edges deleted from it since (from 0.5.42 on), and the writes made since. Opening it folds
the base into the database's store: every node and edge of the base comes back with its
id, its labels and its properties, a change made after `compact()` wins over the base's
version, and what was deleted stays deleted. A read-write open migrates the file, as it
migrates every 0.5.x file (see [Persistent Mode](persistence/persistent.md)), so the
migrated file holds one store; a read-only open folds the base in memory and leaves the
file as it is.

When the process exited without `close()` after `compact()`, the file's WAL holds the
direct calls made after its last checkpoint (`set_node_property()`, `delete_node()` and the
like). The open replays them onto the folded base, also those that changed nodes and edges
of the base, which 0.5.44 lost when it reopened such a file (as long as no 0.5.x release
opened it since). 0.5.x did not log queries after `compact()`
([#558](https://github.com/GrafeoDB/grafeo/issues/558)), so what they changed is not in the
WAL and cannot be recovered.

A node with several labels comes back with each of them
([#595](https://github.com/GrafeoDB/grafeo/issues/595)). Those versions stored its labels
as one name (`"Actor|Person"`), also on the node when a write after `compact()` found it
without a label and changed it. The open splits that name into the labels:
`MATCH (n:Person)` and `MATCH (n:Actor)` both find the node, `labels(n)` lists both, and
`CALL db.labels()` does not list the joined name. A single label that holds a `|` reads as
two labels. A write after `compact()` that matched such a node by one of its labels found
nothing then, so it is not in the file.

What those versions stored differently stays as they stored it:

- **Missing properties**: a property that a node or edge lacked was stored as the
  column's empty value (`''`, `0`, `0.0` or `false`)
  ([#542](https://github.com/GrafeoDB/grafeo/issues/542)).
- **Lists, maps, dates, times and durations** were stored as their text: they read as
  strings, such as `'["amsterdam", "jazz"]'` or `'1994-03-19'`.
- **Vector and text indexes** were dropped by `compact()`: create them again
  (`SHOW INDEXES` may still list the name of a text index).
- **Deletes in 0.5.40 and 0.5.41**: these versions kept no record of the nodes and edges
  of the base deleted after `compact()`, so they come back, as they did when 0.5.42 to
  0.5.44 opened such a file.

## Feature Flag

Since 0.6.0, every build that opens database files reads a compacted base, the `grafeo`
Rust crate and the `grafeo` command line tool included. The `compact-store` feature is
deprecated and enables nothing; it is removed in 0.7.0, so drop it from your build. In
0.5.x, a build without the feature opened a compacted file without its base, and its next
checkpoint lost the base for good.
