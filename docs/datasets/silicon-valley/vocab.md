---
title: Silicon Valley dataset terms
description: The classes and properties of the Silicon Valley demo dataset, the namespace https://docs.grafeo.dev/datasets/silicon-valley/vocab#.
---

# Silicon Valley dataset terms

The namespace `https://docs.grafeo.dev/datasets/silicon-valley/vocab#` (prefix `sv:`) of the [Silicon Valley dataset](index.md). A term's IRI is this namespace plus its name, for example `sv:worksAt`.

## Classes

| Node label | Class |
| --- | --- |
| Person | `schema:Person` |
| Company | `schema:Organization` |
| Product | `schema:Product` |
| Place | `schema:Place` |
| City | `schema:City` |
| Event | `schema:Event` |
| Technology | `sv:Technology` |
| AI | `sv:AI` |

Every node has `schema:name` and `schema:description`. A company's founding date is `schema:foundingDate`.

## Relationships

Each relationship is a triple whose predicate is the relationship type in camel case: `sv:knows`, `sv:rivalOf`, `sv:mentors`, `sv:worksAt`, `sv:founded`, `sv:livesIn`, `sv:basedIn`, `sv:investedIn`, `sv:makes`, `sv:uses`, `sv:competesWith`, `sv:built`, `sv:trainedOn`, `sv:owns`, `sv:attended` and `sv:locatedIn`. `sv:worksAt` also has `schema:worksFor` beside it.

A relationship with properties (a role, a period, a funding round) also gets a resource of its own, which names both ends and carries the properties:

| Relationship | Class | Ends |
| --- | --- | --- |
| `sv:worksAt` | `sv:Employment` | `sv:person`, `sv:company` |
| `sv:basedIn` | `sv:Base` | `sv:company`, `sv:place` |
| `sv:investedIn` | `sv:Investment` | `sv:investor`, `sv:company` |
| `sv:attended` | `sv:Attendance` | `sv:attendee`, `sv:event` |
| `sv:founded` | `sv:Founding` | `sv:founder`, `sv:founded` |
| `sv:livesIn` | `sv:Residence` | `sv:person`, `sv:city` |
| `sv:owns` | `sv:Ownership` | `sv:owner`, `sv:owned` |

Their properties are `sv:role`, `sv:from`, `sv:to`, `sv:date`, `sv:round`, `sv:amountUsd`, `sv:note` and `sv:outcome`.

## Other properties

The other node properties (a few facts and in-jokes per node, such as `sv:weissmanScore` or `sv:catchphrase`) are named after the dataset's property names in camel case.

## Dates

Dates are `xsd:date`. The series rarely states a day, so most dates are known to the month: such a date is the first of its month, with `sv:<field>Precision "month"` on the same subject (`sv:fromPrecision`, `sv:toPrecision`, `sv:datePrecision`, `sv:foundedPrecision`). In GQL and Cypher the same date is a `DATE` with a `<field>_precision` property set to `'month'`.
