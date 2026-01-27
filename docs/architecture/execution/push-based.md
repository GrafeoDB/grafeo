---
title: Push-Based Model
description: Push-based query execution.
tags:
  - architecture
  - execution
---

# Push-Based Model

Graphos uses a push-based (producer-driven) execution model.

## Push vs Pull

| Model | Data Flow | Control |
|-------|-----------|---------|
| Pull (Volcano) | Child -> Parent | Parent requests |
| Push | Parent -> Child | Parent pushes |

## Benefits

- **Better pipelining** - No function call overhead per row
- **Parallelism** - Natural morsel-driven parallelism
- **Cache efficiency** - Process data while hot
- **Early termination** - LIMIT can stop pipeline early

## Pipeline Structure

```
Pipeline 1: Scan -> Filter -> Project -> Exchange
Pipeline 2: Exchange -> Aggregate -> Sink
```

## Morsel-Driven Parallelism

Work is divided into morsels (chunks of data):

```
Thread 1: [Morsel 0] -> [Morsel 4] -> [Morsel 8]
Thread 2: [Morsel 1] -> [Morsel 5] -> [Morsel 9]
Thread 3: [Morsel 2] -> [Morsel 6] -> ...
Thread 4: [Morsel 3] -> [Morsel 7] -> ...
```
