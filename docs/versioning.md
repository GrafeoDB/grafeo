---
title: Versioning and Compatibility
description: What each Grafeo release may change, which APIs are stable, and how to pin a version.
tags:
  - reference
---

# Versioning and Compatibility

Grafeo is before 1.0. Versions are `0.MINOR.PATCH` and follow the rule Cargo and npm apply to `0.x` versions: a
new minor version may break you, a new patch version does not. From 1.0 the same rules move up one level, so
breaking changes come only in major versions.

## What a release may change

| Release | May contain |
| --- | --- |
| Patch (`0.6.0` to `0.6.1`) | Bug fixes, additive features, new options whose default keeps the current behavior, performance work, documentation, deprecations |
| Minor (`0.6` to `0.7`) | All of the above, plus a new database file format, breaking changes to the stable surface, removal of deprecated items, a higher minimum Rust version |

A bug fix can change query results, or turn a query that returned a wrong answer into an error. It still ships in
a patch release, because the old result was wrong. The [CHANGELOG](changelog.md) lists such changes under
Fixed, and under Changed when a query that used to run now fails. Breaking changes are marked **Breaking** and
appear only in minor releases.

## The stable surface

The rules above cover:

- the `grafeo` Rust crate and everything it re-exports
- the Python, Node.js, C, WebAssembly, C#, Dart and Go bindings
- the `grafeo` command line tool
- the database file format

`grafeo-common`, `grafeo-core`, `grafeo-storage`, `grafeo-adapters` and `grafeo-engine` are implementation crates.
They are published so that `grafeo` can depend on them, and their APIs may change in any release. Depend on
`grafeo` instead.

## Database files

A patch release never changes the file format. A minor release that does migrates a database the first time it is
opened, and keeps a copy of the old file next to it (for example `data.grafeo.pre-0.7`), so you can return to the
previous version. The release before it announces the change in its notes.

## Deprecation

Before something is removed, it is deprecated, with its replacement named in the CHANGELOG and the API
documentation. Deprecation can happen in any release; removal happens at the earliest in the next minor release.

## Pinning a version

Until 1.0, pin the minor version you test against. For example, to stay on 0.6:

| Package manager | Requirement | Accepts |
| --- | --- | --- |
| Cargo | `grafeo = "0.6"` | `>=0.6.0, <0.7.0` |
| npm | `"@grafeo-db/js": "^0.6.0"` | `>=0.6.0, <0.7.0` |
| pip / uv | `grafeo~=0.6.0` | `>=0.6.0, <0.7` |

For pip and uv, write all three parts: `grafeo~=0.6` means `>=0.6, <1.0`, so it also accepts 0.7 and later.
