---
title: Vector Quantization
description: Compress vectors for memory-efficient similarity search.
tags:
  - vector-search
  - quantization
  - compression
---

# Vector Quantization

Quantization compresses vectors to reduce memory usage while maintaining search quality. Grafeo supports three quantization methods with different compression-recall tradeoffs.

## Overview

| Method | Compression | Recall | Best For |
| ------ | ----------- | ------ | -------- |
| **Scalar (SQ)** | 4x | ~97% | General use, high recall |
| **Binary (BQ)** | 32x | ~85% | Fast filtering, massive datasets |
| **Product (PQ)** | 8-192x | ~90-95% | Large datasets, memory-constrained |

## Scalar Quantization

Scalar quantization converts each f32 (4 bytes) to u8 (1 byte), achieving 4x compression with minimal recall loss.

### How It Works

1. **Training**: Learn min/max values per dimension from sample vectors
2. **Quantization**: Map each f32 value to 0-255 range
3. **Search**: Use asymmetric distance (f32 query vs u8 stored)

### Usage

```python
from grafeo import ScalarQuantizer

# Train on sample vectors
vectors = [doc.embedding for doc in documents[:1000]]
quantizer = ScalarQuantizer.train(vectors)

# Quantize vectors
quantized = quantizer.quantize(embedding)  # Returns List[int] (u8 values)

# Compute distance
distance = quantizer.distance_u8(quantized_a, quantized_b)

# Dequantize (approximate reconstruction)
reconstructed = quantizer.dequantize(quantized)
```

### With HNSW Index

```python
from grafeo import QuantizedHnswIndex, HnswConfig, QuantizationType

config = HnswConfig(dimensions=384, metric="cosine")
index = QuantizedHnswIndex(config, QuantizationType.Scalar)

# Insert vectors (automatically quantized after training)
for i, vec in enumerate(vectors):
    index.insert(i, vec)

# Search (rescored with full precision by default)
results = index.search(query_vector, k=10)
```

### Performance

- **Compression**: 4x (384 dims: 1536 → 384 bytes)
- **Recall**: ~97% at k=10
- **Distance computation**: ~424 ns (vs ~38 ns for f32)

## Binary Quantization

Binary quantization converts each f32 to a single bit, achieving 32x compression. Best for fast pre-filtering with rescoring.

### How It Works

1. **Quantization**: `bit = 1 if value > 0 else 0`
2. **Distance**: Hamming distance (popcount of XOR)
3. **Use**: Fast candidate filtering, then rescore top candidates

### Usage

```python
from grafeo import BinaryQuantizer

# Quantize (no training needed)
binary_vec = BinaryQuantizer.quantize(embedding)  # Returns List[int] (packed u64)

# Hamming distance (very fast with SIMD)
distance = BinaryQuantizer.hamming_distance(binary_a, binary_b)
```

### With HNSW Index

```python
from grafeo import QuantizedHnswIndex, HnswConfig, QuantizationType

config = HnswConfig(dimensions=384, metric="cosine")
index = QuantizedHnswIndex(config, QuantizationType.Binary)

# Rescoring is highly recommended for binary quantization
index = index.with_rescore(True).with_rescore_factor(4)  # Rescore top 4k candidates
```

### Performance

- **Compression**: 32x (384 dims: 1536 → 48 bytes)
- **Recall**: ~85% at k=10 (higher with rescoring)
- **Distance computation**: ~50 ns (SIMD popcount)

## Product Quantization

Product quantization (PQ) divides vectors into subvectors and quantizes each using a learned codebook. Achieves high compression with good recall.

### How It Works

1. **Training**: Use k-means to learn K centroids for each of M subvectors
2. **Quantization**: Store M codes (indices into centroid tables)
3. **Distance**: Asymmetric Distance Computation (ADC) via lookup tables

### Configuration

| Parameter | Typical Values | Effect |
| --------- | -------------- | ------ |
| `num_subvectors` (M) | 8, 16, 32, 48 | More = better recall, less compression |
| `num_centroids` (K) | 256 (max) | Usually 256 for u8 codes |
| `iterations` | 10-20 | K-means iterations |

### Compression Ratio

```
compression_ratio = (dimensions * 4) / num_subvectors

# Examples for 384 dimensions:
# M=8:  384*4/8  = 192x compression
# M=16: 384*4/16 = 96x compression
# M=48: 384*4/48 = 32x compression
```

### Usage

```python
from grafeo import ProductQuantizer

# Training vectors (should be representative sample)
training_vectors = [doc.embedding for doc in sample_docs]

# Train quantizer
# - 8 subvectors (48 dims each for 384-dim vectors)
# - 256 centroids per subvector
# - 10 k-means iterations
quantizer = ProductQuantizer.train(
    vectors=training_vectors,
    num_subvectors=8,
    num_centroids=256,
    iterations=10
)

# Quantize to M codes
codes = quantizer.quantize(embedding)  # Returns List[int] of length M

# Fast distance computation using precomputed table
table = quantizer.build_distance_table(query)
distance = quantizer.distance_with_table(table, codes)  # ~4.5 ns!

# Or direct asymmetric distance (builds table internally)
distance = quantizer.asymmetric_distance(query, codes)

# Approximate reconstruction
reconstructed = quantizer.reconstruct(codes)
```

### With HNSW Index

```python
from grafeo import QuantizedHnswIndex, HnswConfig, QuantizationType

config = HnswConfig(dimensions=384, metric="cosine")

# PQ8: 8 subvectors, 192x compression
index = QuantizedHnswIndex(
    config,
    QuantizationType.Product(num_subvectors=8)
)

# Lower training threshold for faster initial training
index = index.with_training_threshold(1000)

# Insert vectors
for i, vec in enumerate(vectors):
    index.insert(i, vec)

# Search
results = index.search(query, k=10)
```

### Performance

- **Compression**: 8-192x depending on M
- **Recall**: ~90-95% at k=10
- **Distance computation**: 4.5 ns with precomputed table (6x faster than raw!)

## Choosing a Quantization Method

### Decision Tree

```
Is memory the primary constraint?
├── No → Use Scalar Quantization (best recall)
└── Yes → How much compression do you need?
    ├── 4x is enough → Scalar Quantization
    ├── 8-50x needed → Product Quantization
    └── 32x+ needed → Binary Quantization (with rescoring)
```

### Comparison for 1M 384-dim Vectors

| Method | Memory | Recall@10 | Search Time |
| ------ | ------ | --------- | ----------- |
| None (f32) | 1.5 GB | 100% | Baseline |
| Scalar | 384 MB | ~97% | ~1.1x |
| PQ8 | 8 MB | ~92% | ~0.8x |
| PQ48 | 48 MB | ~95% | ~0.9x |
| Binary | 48 MB | ~85% | ~0.5x |

## Advanced: Combining Quantization with Zone Maps

For very large datasets, combine quantization with zone maps for block-level pruning:

```python
from grafeo import VectorZoneMap

# Build zone map for a block of vectors
block_vectors = vectors[start:end]
zone_map = VectorZoneMap.build(block_vectors)

# Check if block might contain results
query = get_query()
threshold = 0.5  # Distance threshold

if zone_map.might_contain_within_distance(query, threshold, "euclidean"):
    # Search this block
    results = search_block(block_vectors, query)
else:
    # Skip entire block
    pass
```

## Storage Backends

For datasets too large for RAM, use memory-mapped storage:

```python
from grafeo import MmapStorage, RamStorage

# In-memory storage (fastest)
ram_storage = RamStorage(dimensions=384)

# Memory-mapped storage (disk-backed)
mmap_storage = MmapStorage.create("vectors.bin", dimensions=384)
mmap_storage = mmap_storage.with_cache_limit(10000)  # LRU cache

# Insert vectors
mmap_storage.insert(node_id, embedding)

# Retrieve
vector = mmap_storage.get(node_id)

# Persist
mmap_storage.flush()
```

## Next Steps

- [**Python API**](python-api.md) - Complete Python bindings reference
- [**HNSW Index**](hnsw-index.md) - Index configuration details
