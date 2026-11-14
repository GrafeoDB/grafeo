---
title: grafeo-memory
description: AI memory layer for LLM applications, powered by GrafeoDB.
---

# grafeo-memory

AI memory layer for LLM applications. Extract facts, entities and relations from conversations and persist them in a GrafeoDB graph with vector embeddings for semantic search.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-memory){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-memory/){ .md-button }

## Overview

grafeo-memory provides a `MemoryManager` that orchestrates an **extract -> search -> reconcile -> execute** loop:

1. **Extract** facts, entities and relations from text using an LLM
2. **Search** existing memories for duplicates or conflicts
3. **Reconcile** via LLM to decide ADD / UPDATE / DELETE / NONE
4. **Execute** mutations against the GrafeoDB graph

This keeps a persistent, deduplicated memory graph that grows and evolves over conversations.

## Installation

```bash
uv add grafeo-memory[openai]     # OpenAI
uv add grafeo-memory[mistral]    # Mistral
uv add grafeo-memory[anthropic]  # Anthropic
uv add grafeo-memory[all]        # all providers
```

Or with pip:

```bash
pip install grafeo-memory[openai]
```

Requires Python 3.12+, grafeo >= 0.5.1 and pydantic-ai.

## Quick Start

### OpenAI

```python
from openai import OpenAI
from grafeo_memory import MemoryManager, MemoryConfig, OpenAIEmbedder

embedder = OpenAIEmbedder(OpenAI())
config = MemoryConfig(db_path="./memory.db", user_id="alix")

with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
    # Add memories from conversations
    memory.add("I work at Acme Corp as a data scientist")
    memory.add("My favorite language is Python")

    # Semantic search
    results = memory.search("Where does the user work?")
    for r in results:
        print(r.text, r.score)

    # Reconciliation detects contradiction and updates
    memory.add("I switched to a machine learning engineer role at Acme")

    # Get all memories
    all_memories = memory.get_all()
```

### Mistral

```python
from mistralai import Mistral
from grafeo_memory import MemoryManager, MemoryConfig, MistralEmbedder

embedder = MistralEmbedder(Mistral())
config = MemoryConfig(db_path="./memory.db", user_id="alix")

with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
    memory.add("I work at Acme Corp as a data scientist")
    results = memory.search("Where does the user work?")
```

## Features

### Memory Management

- **Automatic deduplication** via LLM-powered reconciliation
- **Semantic search** using vector embeddings (HNSW index)
- **Graph search** via entity extraction and graph traversal
- **Multi-user support** with `user_id` isolation
- **Change history** per memory with full audit trail
- **Topology boost** (opt-in): re-rank search results by graph connectivity
- **Importance scoring** (opt-in): composite scoring with recency, frequency and importance
- **Memory summarization**: consolidate old memories into fewer, richer entries
- **Procedural memory**: separate memory type for instructions, preferences and rules
- **Episodic memory**: memory type for interaction events and reasoning context
- **Persistent or in-memory** storage modes
- **Built-in MCP server**: expose the memory API to AI agents via `grafeo-memory-mcp`
- **OpenTelemetry**: opt-in instrumentation for tracing LLM calls

### Graph Structure

Memories are stored as a rich graph:

- `:Memory` nodes with `text`, `embedding` and metadata properties
- `:Entity` nodes extracted from text (people, organizations, places, etc.)
- `:HAS_ENTITY` edges linking memories to their entities
- `:RELATION` edges between entities (e.g., "works at", "knows")
- `:HAS_HISTORY` edges linking memories to their change history
- `:DERIVED_FROM` edges linking summary memories to originals

### LLM Integration

- **pydantic-ai** model strings for any supported provider (OpenAI, Anthropic, Mistral, Groq, Google)
- Built-in **`OpenAIEmbedder`** and **`MistralEmbedder`**
- **Protocol-based** `EmbeddingClient` for custom embedding providers
- Structured extraction and reconciliation via pydantic-ai Agents

## API Reference

### MemoryManager

```python
MemoryManager(
    model: str,                        # pydantic-ai model string, e.g. "openai:gpt-4o-mini"
    config: MemoryConfig | None = None,
    *,
    embedder: EmbeddingClient,
)
```

Use as a context manager. Multiple sessions in the same process are supported.

Methods:

- `add(messages, user_id=None, ..., memory_type="semantic")` → `AddResult`: extract and store memories
- `search(query, user_id=None, k=10, ..., memory_type=None)` → `SearchResponse`: semantic + graph search
- `update(memory_id, text)` → `MemoryEvent`: update a memory's text directly
- `get_all(user_id=None, memory_type=None)` → `list[SearchResult]`: retrieve all memories
- `delete(memory_id)` → `bool`: delete a memory
- `delete_all(user_id=None)` → `int`: delete all memories for a user
- `summarize(user_id=None, ...)` → `AddResult`: consolidate old memories
- `history(memory_id)` → `list[HistoryEntry]`: get change history
- `set_importance(memory_id, importance)` → `bool`: set importance score

### Return Types

- **`AddResult`**: list subclass of `MemoryEvent`, with `.usage` for LLM token counts
- **`SearchResponse`**: list subclass of `SearchResult`, with `.usage` for LLM token counts
- **`MemoryEvent`**: `.action` (ADD/UPDATE/DELETE/NONE), `.memory_id`, `.text`, `.old_text`
- **`SearchResult`**: `.memory_id`, `.text`, `.score`, `.user_id`, `.metadata`, `.relations`, `.memory_type`
- **`HistoryEntry`**: `.event`, `.old_text`, `.new_text`, `.timestamp`, `.actor_id`, `.role`

Iterate results directly:

```python
for event in memory.add("text"):
    print(event.action, event.text)

for result in memory.search("query"):
    print(result.text, result.score)
```

### MemoryConfig

```python
MemoryConfig(
    db_path: str | None = None,              # None for in-memory
    user_id: str = "default",                # Default user scope
    embedding_dimensions: int = 1536,        # Embedding dimensions
    similarity_threshold: float = 0.7,       # Reconciliation threshold
    enable_importance: bool = False,         # Composite scoring
    enable_topology_boost: bool = False,     # Graph-connectivity re-ranking
    topology_boost_factor: float = 0.2,      # Topology boost strength
    consolidation_protect_threshold: float = 0.0,  # Protect hub memories from summarize
    instrument: bool = False,               # OpenTelemetry instrumentation
)
```

## MCP Server

grafeo-memory includes a built-in MCP server that exposes the memory API to AI agents. Unlike [grafeo-mcp](grafeo-mcp.md) which wraps the raw database, `grafeo-memory-mcp` wraps the high-level memory operations (extract, reconcile, search, summarize).

```bash
uv add grafeo-memory[mcp]
# or: pip install grafeo-memory[mcp]
```

Add to Claude Desktop config:

```json
{
  "mcpServers": {
    "grafeo-memory": {
      "command": "grafeo-memory-mcp",
      "env": {
        "GRAFEO_MEMORY_MODEL": "openai:gpt-4o-mini",
        "GRAFEO_MEMORY_DB": "./memory.db"
      }
    }
  }
}
```

### Tools

`memory_add`, `memory_add_batch`, `memory_search`, `memory_update`, `memory_delete`, `memory_delete_all`, `memory_list`, `memory_summarize`, `memory_history`

### Resources

- `memory://config` - current configuration
- `memory://stats` - memory count and database info

## Requirements

- Python 3.12+
- grafeo >= 0.5.1
- pydantic-ai-slim

## License

Apache-2.0
