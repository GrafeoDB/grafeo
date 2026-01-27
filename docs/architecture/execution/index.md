---
title: Execution Engine
description: Query execution architecture.
---

# Execution Engine

Graphos uses a push-based, vectorized execution engine.

## Overview

```mermaid
graph LR
    SCAN[Scan] --> FILTER[Filter]
    FILTER --> PROJECT[Project]
    PROJECT --> AGG[Aggregate]
    AGG --> SINK[Result Sink]
```

## Sections

- [Push-Based Model](push-based.md)
- [Vectorized Operations](vectorized.md)
- [Adaptive Chunks](chunks.md)
