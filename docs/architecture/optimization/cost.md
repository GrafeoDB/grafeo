---
title: Cost Model
description: Query cost estimation.
tags:
  - architecture
  - optimization
---

# Cost Model

The cost model estimates execution cost for plan selection.

## Cost Components

| Component | Weight | Description |
|-----------|--------|-------------|
| CPU | 1.0 | Computation cost |
| I/O | 10.0 | Disk access cost |
| Memory | 0.5 | Memory allocation |
| Network | 100.0 | Data transfer (future) |

## Cost Formula

```
Total Cost = CPU_cost * cpu_weight
           + IO_cost * io_weight
           + Mem_cost * mem_weight
```

## Operator Costs

| Operator | Cost Formula |
|----------|--------------|
| Scan | rows * column_count |
| Filter | input_rows * selectivity |
| Hash Join | build_rows + probe_rows |
| Sort | rows * log(rows) |
