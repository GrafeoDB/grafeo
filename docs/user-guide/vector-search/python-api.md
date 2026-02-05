---
title: Python Vector API
description: Python bindings for vector search and quantization.
tags:
  - vector-search
  - python
  - api
---

# Python Vector API

Complete reference for vector search operations from Python.

## Quantization Types

### QuantizationType

```python
from grafeo import QuantizationType

# No quantization (full f32 precision)
qt_none = QuantizationType.none()

# Scalar quantization (f32 → u8, 4x compression)
qt_scalar = QuantizationType.scalar()

# Binary quantization (f32 → 1 bit, 32x compression)
qt_binary = QuantizationType.binary()

# Product quantization (codebook-based, variable compression)
qt_product = QuantizationType.product(num_subvectors=8)

# Properties
print(qt_scalar.name)              # "scalar"
print(qt_scalar.compression_ratio(384))  # 4
print(qt_scalar.requires_training)  # True
```

## Scalar Quantizer

### Training

```python
from grafeo import ScalarQuantizer

# Train from vectors (learns min/max per dimension)
vectors = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9],
]
quantizer = ScalarQuantizer.train(vectors)

# Or create with explicit ranges
quantizer = ScalarQuantizer.with_ranges(
    min_values=[0.0, 0.0, 0.0],
    max_values=[1.0, 1.0, 1.0]
)
```

### Quantization

```python
# Quantize a vector to u8 values
embedding = [0.5, 0.5, 0.5]
quantized = quantizer.quantize(embedding)
print(quantized)  # [127, 127, 127] (approximately)

# Dequantize (approximate reconstruction)
reconstructed = quantizer.dequantize(quantized)
```

### Distance Computation

```python
vec_a = quantizer.quantize([0.1, 0.2, 0.3])
vec_b = quantizer.quantize([0.4, 0.5, 0.6])

# Quantized distance (fast, approximate)
distance = quantizer.distance_u8(vec_a, vec_b)
```

### Properties

```python
print(quantizer.dimensions)   # 3
print(quantizer.min_values)   # [0.1, 0.2, 0.3]
print(quantizer.max_values)   # [0.7, 0.8, 0.9]
```

## Product Quantizer

### Training

```python
from grafeo import ProductQuantizer

# Training vectors (representative sample of your data)
vectors = load_training_vectors()  # List of List[float]

# Train with:
# - num_subvectors: How many partitions (must divide dimensions)
# - num_centroids: Codebook size per partition (max 256 for u8)
# - iterations: K-means iterations
quantizer = ProductQuantizer.train(
    vectors=vectors,
    num_subvectors=8,      # 384/8 = 48 dims per subvector
    num_centroids=256,     # 256 codes per subvector
    iterations=10          # K-means iterations
)
```

### Quantization

```python
# Quantize to M codes (one per subvector)
embedding = get_embedding()  # 384-dim vector
codes = quantizer.quantize(embedding)
print(len(codes))  # 8 (one code per subvector)
print(codes)       # [42, 128, 7, 255, ...] (u8 values)

# Batch quantization
embeddings = [get_embedding() for _ in range(100)]
all_codes = quantizer.quantize_batch(embeddings)
```

### Distance Computation

```python
query = get_query_embedding()
codes = quantizer.quantize(stored_embedding)

# Option 1: Direct asymmetric distance
distance = quantizer.asymmetric_distance(query, codes)

# Option 2: Using precomputed table (faster for batch queries)
table = quantizer.build_distance_table(query)
distance = quantizer.distance_with_table(table, codes)  # ~4.5 ns!

# Batch distance computation
all_distances = []
for doc_codes in database_codes:
    dist = quantizer.distance_with_table(table, doc_codes)
    all_distances.append(dist)
```

### Reconstruction

```python
# Approximate reconstruction from codes
codes = quantizer.quantize(original)
reconstructed = quantizer.reconstruct(codes)

# Reconstruction error
import numpy as np
error = np.linalg.norm(np.array(original) - np.array(reconstructed))
```

### Properties

```python
print(quantizer.dimensions)       # 384
print(quantizer.num_subvectors)   # 8
print(quantizer.num_centroids)    # 256
print(quantizer.subvector_dim)    # 48
print(quantizer.code_size)        # 8 (bytes per quantized vector)
print(quantizer.compression_ratio)  # 192 (384*4/8)
```

## Binary Quantizer

Binary quantization is stateless (no training required).

```python
from grafeo import BinaryQuantizer

# Quantize (sign-based: 1 if value > 0, else 0)
embedding = [0.1, -0.2, 0.3, -0.4]
binary = BinaryQuantizer.quantize(embedding)

# Hamming distance (SIMD-accelerated)
binary_a = BinaryQuantizer.quantize(vec_a)
binary_b = BinaryQuantizer.quantize(vec_b)
distance = BinaryQuantizer.hamming_distance(binary_a, binary_b)

# Memory requirements
bytes_needed = BinaryQuantizer.bytes_needed(384)  # 48 bytes (384/8)
```

## Distance Functions

Standalone distance functions for direct computation:

```python
from grafeo import (
    cosine_similarity,
    cosine_distance,
    euclidean_distance,
    dot_product,
    manhattan_distance,
    l2_norm,
    normalize,
)

vec_a = [0.1, 0.2, 0.3]
vec_b = [0.4, 0.5, 0.6]

# Similarities and distances
sim = cosine_similarity(vec_a, vec_b)   # 0.0 to 1.0
dist = cosine_distance(vec_a, vec_b)    # 1 - similarity
dist = euclidean_distance(vec_a, vec_b) # L2 distance
prod = dot_product(vec_a, vec_b)        # Inner product
dist = manhattan_distance(vec_a, vec_b) # L1 distance

# Utilities
norm = l2_norm(vec_a)                   # Vector magnitude
normalized = normalize(vec_a)           # Unit vector
```

## Storage Backends

### RAM Storage

```python
from grafeo import RamStorage

# Create in-memory storage
storage = RamStorage(dimensions=384)

# Operations
storage.insert(node_id=1, vector=[0.1] * 384)
vector = storage.get(node_id=1)           # Returns List[float] or None
exists = storage.contains(node_id=1)      # True
removed = storage.remove(node_id=1)       # True if existed
count = len(storage)                      # Number of vectors
dims = storage.dimensions                 # 384
memory = storage.memory_usage             # Bytes used
```

### Memory-Mapped Storage

```python
from grafeo import MmapStorage

# Create new storage file
storage = MmapStorage.create("vectors.bin", dimensions=384)

# Or open existing
storage = MmapStorage.open("vectors.bin")

# Configure cache
storage = storage.with_cache_limit(10000)  # LRU cache size

# Operations (same as RamStorage)
storage.insert(node_id=1, vector=[0.1] * 384)
vector = storage.get(node_id=1)

# Persistence
storage.flush()                           # Ensure written to disk
size = storage.file_size                  # File size in bytes
storage.clear_cache()                     # Free cache memory
```

## Complete Example

```python
import grafeo
from grafeo import ProductQuantizer, MmapStorage

# Setup database
db = grafeo.GrafeoDB()

# Load embeddings from your ML model
def get_embedding(text: str) -> list[float]:
    # Your embedding model here
    return model.encode(text).tolist()

# Store documents with embeddings
documents = [
    {"title": "Python Basics", "content": "..."},
    {"title": "Machine Learning", "content": "..."},
    # ... more documents
]

for doc in documents:
    embedding = get_embedding(doc["content"])
    db.execute(
        "INSERT (:Document {title: $title, embedding: $embedding})",
        {"title": doc["title"], "embedding": embedding}
    )

# Train quantizer for efficient storage
all_embeddings = [get_embedding(doc["content"]) for doc in documents[:1000]]
quantizer = ProductQuantizer.train(
    vectors=all_embeddings,
    num_subvectors=8,
    num_centroids=256,
    iterations=10
)

# Search
query = "How do I learn programming?"
query_embedding = get_embedding(query)

result = db.execute("""
    MATCH (d:Document)
    RETURN d.title, cosine_similarity(d.embedding, $query) AS score
    ORDER BY score DESC
    LIMIT 5
""", {"query": query_embedding})

for row in result:
    print(f"{row['d.title']}: {row['score']:.3f}")
```

## SIMD Support

Check available SIMD acceleration:

```python
import grafeo

# Returns dict with available SIMD features
simd = grafeo.simd_support()
print(simd)
# {'avx2': True, 'avx512': False, 'sse4': True, 'neon': False}
```

SIMD is automatically used for:

- Distance computation (cosine, euclidean, dot product)
- Binary quantization hamming distance
- Scalar quantization operations
