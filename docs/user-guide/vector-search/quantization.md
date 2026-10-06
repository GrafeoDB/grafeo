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
distance = quantizer.distance(quantized_a, quantized_b)

# Dequantize (approximate reconstruction)
reconstructed = quantizer.dequantize(quantized)
```

### Performance

- **Compression**: 4x (384 dims: 1536 --> 384 bytes)
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

### Performance

- **Compression**: 32x (384 dims: 1536 --> 48 bytes)
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

### Performance

- **Compression**: 8-192x depending on M
- **Recall**: ~90-95% at k=10
- **Distance computation**: 4.5 ns with precomputed table (6x faster than raw!)

## Choosing a Quantization Method

### Decision Tree

```
Is memory the primary constraint?
├── No → Use Scalar Quantization (best recall)
└── Yes → How much compression is needed?
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

## Next Steps

- [**Python API**](python-api.md) - Complete Python bindings reference
- [**HNSW Index**](hnsw-index.md) - Index configuration details
