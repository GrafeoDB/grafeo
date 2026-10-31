---
title: Upgrading
description: The upgrade guides of the Grafeo releases that change APIs, query results or the database file format.
tags:
  - reference
  - migration
---

# Upgrading

A minor release (`0.5` to `0.6`) may change stable APIs, query results and the database file format; a patch
release (`0.6.1` to `0.6.2`) does not change stable APIs or the file format, but a bug fix in it can change a query
result or turn a wrong answer into an error (see [Versioning](../versioning.md) and [API Stability](../stability.md)).
Each minor release has a guide here that lists what to change, grouped by where it affects you. The
[CHANGELOG](../changelog.md) has every entry, including the fixes that change results.

| From | To | Guide |
| --- | --- | --- |
| 0.5.x | 0.6 | [Upgrading to 0.6](0.6.md) |

Upgrade one minor version at a time: 0.7 will no longer open a 0.5.x database, so open each one for writing with
0.6 first, which migrates it.

To move to Grafeo from another graph database (Neo4j, DGraph and others), see the
[Migration Guide](../getting-started/migration.md).
