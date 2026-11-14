---
title: grafeo-langchain
description: LangChain integration for GrafeoDB with graph store, vector store and Graph RAG retrieval.
---

# grafeo-langchain

LangChain integration that provides graph store and vector store implementations backed by GrafeoDB. Build knowledge graphs and Graph RAG pipelines with no servers or Docker required.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-langchain){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-langchain/){ .md-button }

## Overview

grafeo-langchain provides two main components:

- **GrafeoGraphStore** - Store and query LLM-extracted knowledge graph triples
- **GrafeoGraphVectorStore** - Combined vector + graph store with Graph RAG retrieval

Both use GrafeoDB's embedded database directly - no intermediate servers needed.

## Installation

```bash
uv add grafeo-langchain
# or
pip install grafeo-langchain
```

Requires Python 3.12+ and grafeo >= 0.4.

## Quick Start

### Knowledge Graph Store

```python
from grafeo_langchain import GrafeoGraphStore
from langchain_core.documents import Document

store = GrafeoGraphStore()

# Add knowledge graph triples
store.upsert_triplet(("Alix", "KNOWS", "Gus"))
store.upsert_triplet(("Gus", "WORKS_AT", "Acme"))

# Query with GQL or Cypher
result = store.query("MATCH (a)-[:KNOWS]->(b) RETURN a, b")
```

### Graph Vector Store (Graph RAG)

```python
from grafeo_langchain import GrafeoGraphVectorStore
from langchain_openai import OpenAIEmbeddings

store = GrafeoGraphVectorStore(
    embedding=OpenAIEmbeddings(),
    db_path="./my-graph.db",
)

# Add documents with graph links
docs = [
    Document(
        page_content="Alix is an engineer at Acme.",
        metadata={"__graph_links__": [{"kind": "bidir", "tag": "MENTIONS", "id": "alix"}]},
    ),
]
store.add_documents(docs)

# Retrieval modes
results = store.similarity_search("engineer", k=5)
results = store.traversal_search("engineer", k=5, depth=2)
results = store.mmr_traversal_search("engineer", k=5, depth=2)
```

## Features

### GrafeoGraphStore

- Stores LLM-extracted triples as native Grafeo graph elements
- Supports GQL and Cypher query languages
- Schema introspection and refresh
- Graph document ingestion with optional source linking

### GrafeoGraphVectorStore

- LangChain `VectorStore` interface with graph traversal
- Native HNSW vector search with configurable embeddings
- Three retrieval modes:
    - `similarity_search()` - Standard vector similarity
    - `traversal_search()` - Vector search + multi-hop graph traversal
    - `mmr_traversal_search()` - MMR-diversified graph-enhanced retrieval
- Explicit graph links via `__graph_links__` metadata

## Requirements

- Python 3.12+
- grafeo >= 0.4
- langchain-core

## License

Apache-2.0
