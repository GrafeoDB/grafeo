---
title: Silicon Valley dataset
description: The Silicon Valley demo dataset and the namespaces of its RDF form.
---

# Silicon Valley dataset

The Silicon Valley demo dataset follows Pied Piper from its founding in March 2014: people, companies, products, AIs, places, cities and events, with dated employment, moves, funding rounds and events. It ships in the `grafeo-datasets` packages for Python and npm, as GQL, Cypher and Turtle. Its Turtle form names its terms and resources with the IRIs below; each one opens the page that describes it.

An homage to the HBO series Silicon Valley; not affiliated with or endorsed by HBO. All descriptions are original.

## Namespaces

| Prefix | IRI | Holds |
| --- | --- | --- |
| `sv:` | `https://docs.grafeo.dev/datasets/silicon-valley/vocab#` | The dataset's own classes and properties: [vocabulary](vocab.md) |
| `svr:` | `https://docs.grafeo.dev/datasets/silicon-valley/id#` | One resource per node, named by its key (`svr:richard-hendricks`): [resources](id.md) |
| `schema:` | `https://schema.org/` | Names, descriptions and the common classes |

The namespaces moved from `grafeo.dev` to `docs.grafeo.dev` in the dataset release that ships with Grafeo 0.6.4, and resources changed from `.../id/<key>` to `.../id#<key>`, so every IRI opens a page that exists.
