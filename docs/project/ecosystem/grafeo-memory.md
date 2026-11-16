---
title: grafeo-memory
description: Memory for agents and LLM applications, in one Grafeo file, with any LLM.
---

# grafeo-memory

Memory for agents and LLM applications, in one Grafeo file. grafeo-memory extracts facts, entities and relations from conversations, reconciles them with what it already knows, and stores them as a graph with vector embeddings and a history of every change. It runs inside the Python process with any LLM that [pydantic-ai](https://ai.pydantic.dev) supports, with no server or Docker to set up.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-memory){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-memory/){ .md-button }

## Overview

Every `add()` runs a reconciliation loop:

1. **Extract** facts, entities and relations from the text (LLM).
2. **Search** the existing memories for related facts (vector and graph).
3. **Reconcile** the new facts with them: ADD, UPDATE, DELETE or NONE (LLM).
4. **Execute** the decisions against the Grafeo database.

The result is a deduplicated memory graph that follows what a user says over time.

## Installation

```bash
uv add "grafeo-memory[mistral]"      # Mistral models and embeddings
uv add "grafeo-memory[anthropic]"    # Anthropic models
uv add "grafeo-memory[openai]"       # OpenAI models and embeddings, also local models through Ollama
uv add "grafeo-memory[mcp,mistral]"  # the MCP server, with the provider extra of its model
uv add "grafeo-memory[all]"          # every LLM provider
```

grafeo-memory 0.4 requires Python 3.13+ and grafeo 0.6 (`>=0.6.2,<0.7`), which is installed with it. On Python 3.12, stay on grafeo-memory 0.3.

## Quick Start

Set `MISTRAL_API_KEY` first. Richard has just left Hooli to start Pied Piper:

```python
from mistralai.client import Mistral
from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder

embedder = MistralEmbedder(Mistral())
config = MemoryConfig(db_path="./pied-piper.grafeo", user_id="richard")

with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
    for event in memory.add(
        "I just left Hooli to start Pied Piper, a compression company built on middle-out. "
        "We work out of the Hacker Hostel in Palo Alto",
        speaker="Richard",
    ):
        print(event.action.value, event.text)
    # for example:
    # add Richard left Hooli to start Pied Piper, a compression company built on middle-out.
    # add Pied Piper operates out of the Hacker Hostel in Palo Alto.

    memory.add("We moved Pied Piper out of the Hacker Hostel into our own office in Mountain View", speaker="Richard")
    # for example: update Pied Piper operates out of their own office in Mountain View.

    for result in memory.search("Where is Pied Piper based?", k=2):
        print(round(result.score, 2), result.text)
    # for example:
    # 0.96 Pied Piper operates out of their own office in Mountain View.
    # 0.85 Richard left Hooli to start Pied Piper, a compression company built on middle-out.
```

The exact wording depends on the model. `user_id` partitions the store and never reaches the model; `speaker` names who is talking (without it, facts say "the user"), and `source="notes"` marks notes or documents, which have no speaker.

### Other models

The model is a pydantic-ai model string; the embedder is separate, so a model without its own embeddings pairs with `MistralEmbedder` or a custom embedder.

```python
# A local model through Ollama (the [openai] extra; OLLAMA_BASE_URL=http://localhost:11434/v1)
MemoryManager("ollama:ministral-3:3b", config, embedder=embedder)

# Anthropic (the [anthropic] extra), with Mistral embeddings
MemoryManager("anthropic:claude-sonnet-4-5-20250929", config, embedder=MistralEmbedder(Mistral()))

# OpenAI (the [openai] extra)
from openai import OpenAI
from grafeo_memory import OpenAIEmbedder

MemoryManager("openai:gpt-4o-mini", config, embedder=OpenAIEmbedder(OpenAI()))
```

| Embedder | Default model | Extra |
| --- | --- | --- |
| `MistralEmbedder` | `mistral-embed` | `[mistral]` |
| `OpenAIEmbedder` | `text-embedding-3-small` | `[openai]` |

Both take an optional `model`. Any other embedding source, a local one included, implements the `EmbeddingClient` protocol:

```python
class LocalEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [...]  # call your embedding model

    @property
    def dimensions(self) -> int:
        return 1024  # your model's output size
```

The vector index takes its size from the embedder's `dimensions`.

## Users and sessions

`user_id` on a call is a partition, not a permission. A server that acts for one user at a time uses a view, which refuses calls for another user and reaches only that user's memories by id:

```python
gilfoyle = memory.for_user("gilfoyle")  # shares the database and caches
dinesh = memory.for_user("dinesh")

built = gilfoyle.add("I built Son of Anton, an AI that fixes bugs by deleting the code", speaker="Gilfoyle")
dinesh.add("Gilfoyle is my rival", speaker="Dinesh")

dinesh.search("Son of Anton")  # Dinesh's own memories only: "Dinesh has a rival named Gilfoyle."
dinesh.delete(built[0].memory_id)  # False: Gilfoyle's memory is out of reach
dinesh.search("Son of Anton", user_id="gilfoyle")  # ValueError: the view is bound to dinesh
```

grafeo-memory does not authenticate anyone: the application checks who the user is and passes that id.

Sessions group the turns of one conversation. An anonymous session keeps its turns in memory only and never writes long-term memory; binding it to a user stores its turns for that user:

```python
sid = memory.start_session(anonymous=True)  # a visitor
memory.add("I want to put my money into a compression company", session_id=sid)
context = memory.assemble("compression", sid)  # the session window, nothing long-term

russ = memory.for_user("russ")
russ.start_session(sid)  # Russ signs in: the session joins russ, one way, and its turns are stored
russ.end_session(sid)
```

## Features

- **Deduplication** by LLM reconciliation, so a fact that changes is updated instead of stored twice
- **Search** by vector similarity and graph context; by default a search makes no model call
- **Change history** per memory, kept after a delete as an audit trail
- **Memory types**: semantic (the default), procedural (instructions, preferences, rules) and episodic (events)
- **Summarization** of old memories into fewer, richer entries
- **Opt-in scoring**: topology boost by graph connectivity, and importance with recency and frequency
- **Decision models** (opt-in): judgments in the write path can run on a decision model instead of, or before, the LLM
- **Built-in MCP server** (`grafeo-memory-mcp`) and a CLI (`grafeo-memory`)
- **OpenTelemetry** instrumentation (opt-in)

## Graph structure

Memories are ordinary graph data in the Grafeo file:

- `:Memory` nodes with the text, its embedding and metadata
- `:Entity` nodes for the people, organizations and places the text names, linked by `:HAS_ENTITY` edges from memories and by `:RELATION` edges between entities
- `:DERIVED_FROM` edges from a summary to the memories it replaces, and `:SUPERSEDES` edges from a newer fact to an older one
- `:History` nodes with the change history, keyed by memory id and not connected by edges, so they survive a delete and leave graph algorithm scores unchanged

## Persistence

```python
config = MemoryConfig(db_path="./memory.grafeo")  # a database file
config = MemoryConfig()  # in memory, gone when the process ends
```

A `db_path` is always a single file. While it is open, Grafeo keeps its write-ahead log in a `<path>.wal/` directory next to it and removes it at close. A `db_path` must not be an existing directory. One process at a time can open a database for writing: close the application, the CLI or the MCP server before another opens the same path.

### Upgrading from grafeo 0.5

grafeo-memory 0.3 and later run on grafeo 0.6, which stores databases in a new format. The first open with grafeo 0.6 migrates a grafeo 0.5 database at the same path, and keeps the old one next to it as `<path>.pre-0.6`. grafeo 0.5 cannot open the migrated database, so stop every process that uses it and back it up first. To go back, stop the grafeo 0.6 processes, move the migrated file away, rename the `.pre-0.6` copy to the original path, and run `uv add "grafeo-memory<0.3" "grafeo<0.6"`; changes made after the migration are only in the migrated file. Memories need no migration step of their own. See [Upgrading to 0.6](../../get-started/upgrading/0.6.md) and the [grafeo-memory README](https://github.com/GrafeoDB/grafeo-memory#upgrading-grafeo-05-to-06).

## API Reference

### MemoryManager

```python
MemoryManager(model, config=None, *, embedder, reranker=None, db=None)
```

`model` is a pydantic-ai model string such as `"mistral:mistral-small-latest"`; `db` is an open `grafeo.GrafeoDB` to share. Use it as a context manager.

| Method | Returns | What it does |
| --- | --- | --- |
| `add(messages, user_id=None, session_id=None, ..., speaker=None, source="chat", memory_type="semantic")` | `AddResult` | Extract and store memories |
| `search(query, user_id=None, k=10, ...)` | `SearchResponse` | Vector and graph search |
| `for_user(user_id)` | a view | Act for one user only |
| `start_session(session_id=None, *, anonymous=False)`, `assemble(query, session_id, budget)`, `end_session(session_id)` | | Sessions |
| `update(memory_id, text)` | `MemoryEvent` | Change a memory's text |
| `get_all(user_id=None, memory_type=None)` | `list[SearchResult]` | All memories |
| `delete(memory_id)`, `delete_all(user_id=None)` | `bool`, `int` | Delete memories (history is kept) |
| `summarize(user_id=None, ...)` | `AddResult` | Consolidate old memories |
| `compile()` | `CompileResult` | Offline pass: entity merges, graph metrics, community summaries |
| `history(memory_id)` | `list[HistoryEntry]` | Change history, oldest first |
| `set_importance(memory_id, importance)` | `bool` | Set an importance score |

`AddResult` and `SearchResponse` are lists of `MemoryEvent` and `SearchResult` with a `.usage` for token counts, so they can be iterated directly.

### MemoryConfig

| Setting | Default | Meaning |
| --- | --- | --- |
| `db_path` | `None` | Database file; `None` keeps it in memory |
| `user_id` | `"default"` | Default user scope |
| `reconciliation_threshold` | `0.3` | Minimum cosine similarity for an existing memory to be weighed in `add()` |
| `embedding_dimensions` | `1536` | Used only when the embedder does not report its size |
| `enable_importance`, `enable_topology_boost` | `False` | Opt-in scoring |
| `history_retention` | grafeo's default | Grafeo history for time travel: `"off"`, `"forever"`, a `timedelta` or seconds |
| `decision_model` | `None` | A decision model for the write path's judgments |
| `cache` | `False` | Cache LLM, embedding and decision calls |
| `instrument` | `False` | OpenTelemetry instrumentation |

The [README](https://github.com/GrafeoDB/grafeo-memory#memoryconfig) lists every setting.

## MCP server

`grafeo-memory-mcp` exposes the memory API to AI agents: extract, reconcile, search, summarize. Unlike [grafeo-mcp](grafeo-mcp.md), which gives agents the raw database, it works with memories. Give it an absolute database path, since a client starts it in a working directory of its own choosing. Without `GRAFEO_MEMORY_MODEL` it uses `openai:gpt-4o-mini`.

### Claude Code

```bash
claude mcp add grafeo-memory \
  -e GRAFEO_MEMORY_MODEL=mistral:mistral-small-latest \
  -e GRAFEO_MEMORY_DB=/home/me/.grafeo/memory.grafeo \
  -e MISTRAL_API_KEY=... \
  -- grafeo-memory-mcp
```

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "grafeo-memory": {
      "command": "grafeo-memory-mcp",
      "env": {
        "GRAFEO_MEMORY_MODEL": "mistral:mistral-small-latest",
        "GRAFEO_MEMORY_DB": "/Users/me/.grafeo/memory.grafeo",
        "MISTRAL_API_KEY": "..."
      }
    }
  }
}
```

Keys and settings can also come from a file named by `GRAFEO_MEMORY_ENV_FILE`. The server runs over stdio by default; `grafeo-memory-mcp sse` and `grafeo-memory-mcp streamable-http` start the other transports.

### Tools

`memory_add`, `memory_add_batch`, `memory_search`, `memory_update`, `memory_delete`, `memory_delete_all`, `memory_list`, `memory_summarize`, `memory_history`, `memory_open_decisions`, `memory_resolve_decision`, `memory_manage_roots`, `memory_assemble`, `memory_add_instruction`, `memory_session_start` and `memory_session_end`. Every tool acts for its `user_id` argument, or else the server's default user (`GRAFEO_MEMORY_USER`), and reaches that user's memories only.

### Settings

| Variable | Default | Meaning |
| --- | --- | --- |
| `GRAFEO_MEMORY_MODEL` | `openai:gpt-4o-mini` | pydantic-ai model string |
| `GRAFEO_MEMORY_DB` | in memory | Database file; use an absolute path |
| `GRAFEO_MEMORY_USER` | `default` | Default user |
| `GRAFEO_MEMORY_ENV_FILE` | none | A `.env` file with keys and settings |
| `MISTRAL_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | none | Provider keys, never logged |

The [README](https://github.com/GrafeoDB/grafeo-memory#environment-variables) lists the cache, limit and logging settings.

## Requirements

- Python 3.13+
- grafeo 0.6 (`>=0.6.2,<0.7`)
- pydantic-ai-slim, with the extra of each model provider used

## License

Apache-2.0
