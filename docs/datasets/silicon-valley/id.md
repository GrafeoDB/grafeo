---
title: Silicon Valley dataset resources
description: The resources of the Silicon Valley demo dataset, the namespace https://docs.grafeo.dev/datasets/silicon-valley/id#.
---

# Silicon Valley dataset resources

The namespace `https://docs.grafeo.dev/datasets/silicon-valley/id#` (prefix `svr:`) of the [Silicon Valley dataset](index.md). Each node of the dataset is one resource, named by its key after the `#`: `svr:richard-hendricks` is the node with key `richard-hendricks`, and `svr:pied-piper` the company.

- Its class is the node's label (see [the terms](vocab.md#classes)), its `schema:name` and `schema:description` are the node's name and description, and its other properties and relationships use the `sv:` terms.
- A relationship with properties also gets a resource of its own in this namespace, named after its type and both ends, for example `svr:works-at-richard-hendricks-hooli-1`; the number tells apart two of one type between the same pair.
- The keys are the `key` values of the dataset's nodes (`grafeo_datasets.silicon_valley.nodes()` in Python, `nodes()` in the npm package).
