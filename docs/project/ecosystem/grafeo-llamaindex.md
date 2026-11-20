---
title: grafeo-llamaindex
description: LlamaIndex PropertyGraphStore backed by GrafeoDB, with vector search and knowledge graphs in one embedded file.
---

# grafeo-llamaindex

LlamaIndex integration that implements the `PropertyGraphStore` interface on GrafeoDB. Build knowledge graphs from documents or datasets, query them with Cypher or GQL, and run vector similarity search, all in a single `.grafeo` file with no servers or Docker.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-llamaindex){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-llamaindex/){ .md-button }

## Overview

grafeo-llamaindex provides two classes:

- **`GrafeoPropertyGraphStore`**: the full `PropertyGraphStore` interface, with `supports_structured_queries` and `supports_vector_queries` both `True`, so one embedded database holds the graph and its embeddings.
- **`GrafeoPGRetriever`**: a sub-retriever for `PropertyGraphIndex` that ranks triplets by vector similarity blended with PageRank from Grafeo.

## Installation

```bash
uv add grafeo-llamaindex
```

Requires Python 3.12+, `llama-index-core` 0.14 or later and grafeo 0.6 (`grafeo>=0.6.2,<0.7`), which is installed with it. For grafeo 0.5, use grafeo-llamaindex 0.2.0 together with `grafeo<0.6`: `uv add "grafeo-llamaindex==0.2.0" "grafeo<0.6"`.

## Quick Start

### The Silicon Valley graph

The quick start loads the Silicon Valley (Pied Piper) dataset from [grafeo.ai](https://grafeo.ai) and asks Mistral about it (other models below):

```bash
uv add grafeo-llamaindex "grafeo-datasets[grafeo]" llama-index-llms-mistralai llama-index-embeddings-mistralai
export MISTRAL_API_KEY=...
```

```python
from grafeo_datasets import silicon_valley
from llama_index.core import PropertyGraphIndex, Settings
from llama_index.core.graph_stores.types import EntityNode, Relation
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI
from grafeo_llamaindex import GrafeoPropertyGraphStore

Settings.llm = MistralAI(model="mistral-small-latest")
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")

graph_store = GrafeoPropertyGraphStore(
    db_path="./pied_piper.grafeo",
    embed_model=Settings.embed_model,  # reads 1024 dimensions from mistral-embed
)

# Pied Piper, its people, products and rivals: 64 entities and their relations
nodes = silicon_valley.nodes()
names = {n["key"]: n["name"] for n in nodes}
embeddings = Settings.embed_model.get_text_embedding_batch([f"{n['name']}: {n['description']}" for n in nodes])
graph_store.upsert_nodes(
    [
        EntityNode(
            name=n["name"], label=n["label"], properties={"description": n["description"], **n["props"]}, embedding=e
        )
        for n, e in zip(nodes, embeddings, strict=True)
    ]
)
graph_store.upsert_relations(
    [
        Relation(label=e["type"], source_id=names[e["source"]], target_id=names[e["target"]], properties=e["props"])
        for e in silicon_valley.edges()
    ]
)

index = PropertyGraphIndex.from_existing(property_graph_store=graph_store, use_async=False)
query_engine = index.as_query_engine(include_properties=True)
print(query_engine.query("Does Jian-Yang's fridge talk?"))
# for example: Yes, Jian-Yang's fridge has a screen and a voice.
```

The store keeps 147 relations of the dataset's 149 edges: a relation upsert updates the relation with the same source, label and target, so Richard Hendricks's three `WORKS_AT` edges to Pied Piper, one per role, become one relation that holds the last role.

Use a model alias such as `mistral-small-latest`: `llama-index-llms-mistralai` 0.12 checks model names against a fixed table and rejects dated IDs such as `mistral-small-2603` with `ValueError: Unknown model`. Without `Settings.llm` and `Settings.embed_model`, LlamaIndex falls back to OpenAI's models.

The graph is saved as you go. Reopen it later with the same embedding model:

```python
graph_store = GrafeoPropertyGraphStore(db_path="./pied_piper.grafeo", embed_model=Settings.embed_model)
index = PropertyGraphIndex.from_existing(
    property_graph_store=graph_store, embed_model=Settings.embed_model, use_async=False
)
```

`embed_model` makes the store embed one short text when it is created, to read the size of the model's embeddings. Without it, pass `embedding_dimensions` (default 1536). To try the store without an API key, run [`examples/mock_embedding_demo.py`](https://github.com/GrafeoDB/grafeo-llamaindex/blob/main/examples/mock_embedding_demo.py), or use a local model (below).

### Building from your own documents

`PropertyGraphIndex` extracts entities and relations from documents with the LLM:

```python
from llama_index.core import SimpleDirectoryReader

index = PropertyGraphIndex.from_documents(
    SimpleDirectoryReader("./data").load_data(),
    property_graph_store=graph_store,
    embed_kg_nodes=True,
    use_async=False,  # see the note below
)
```

**Why `use_async=False`**: with LlamaIndex's default `use_async=True`, `PropertyGraphIndex` embeds the chunks and the extracted entities in two separate `asyncio.run()` calls on the same embedding client. API clients built on `httpx` (Mistral, Anthropic, OpenAI, Ollama) keep a pooled connection from the first event loop, and the second call fails with `RuntimeError: Event loop is closed`, on Windows and Linux alike. This is a LlamaIndex issue that any graph store meets; `use_async=False` avoids it, and entity extraction still runs its LLM calls concurrently.

### Adding documents later

Entity extraction runs in its own `asyncio.run()` on every build and insert, whatever `use_async` is, so `index.insert(document)` with the LLM object the index was built with fails with `Event loop is closed` (the same LlamaIndex issue). An index keeps the extractors it was built with, so setting `Settings.llm` again does not help. Insert through an index over the same store with a new LLM object instead:

```python
from llama_index.core import Document
from llama_index.core.indices.property_graph import SimpleLLMPathExtractor


def inserting_index(graph_store):
    """An index over graph_store whose extractor has a new LLM client."""
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        kg_extractors=[SimpleLLMPathExtractor(llm=MistralAI(model="mistral-small-latest"))],
        embed_model=Settings.embed_model,
        embed_kg_nodes=True,
        use_async=False,
    )


inserting_index(graph_store).insert(Document(text="Jack Barker took over from Gavin Belson as CEO of Hooli."))

for source, relation, target in graph_store.get_triplets(
    entity_names=["Hooli"], relation_names=["WAS_CEO_OF", "BECAME_CEO_OF"]
):
    print(source.id, relation.label, target.id)
# for example: Gavin Belson WAS_CEO_OF Hooli, Jack Barker BECAME_CEO_OF Hooli
```

The extracted "Gavin belson" and "Jack barker" (LlamaIndex's extractor capitalises names) join the dataset's Gavin Belson and Jack Barker, since names match ignoring case (`case_insensitive_names`). Call `inserting_index()` again for each later insert.

### Other models

Set `Settings.llm` and `Settings.embed_model` before creating the store; the rest of the quick start stays the same.

**A local model with Ollama** (no API key; run `ollama pull ministral-3:3b` and `ollama pull nomic-embed-text` first):

```bash
uv add grafeo-llamaindex llama-index-llms-ollama llama-index-embeddings-ollama
```

```python
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(model="ministral-3:3b", request_timeout=300)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")
```

**Anthropic**, which has no embedding model, so it is paired with Mistral's here:

```bash
uv add grafeo-llamaindex llama-index-llms-anthropic llama-index-embeddings-mistralai
```

```python
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.anthropic import Anthropic

Settings.llm = Anthropic(model="claude-opus-5-5")
Settings.embed_model = MistralAIEmbedding(model_name="mistral-embed")
```

**OpenAI**:

```bash
uv add grafeo-llamaindex llama-index-llms-openai llama-index-embeddings-openai
```

```python
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

Settings.llm = OpenAI(model="gpt-4.1-mini")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
```

### Troubleshooting

- **The graph has only `text_chunk` nodes**: entity extraction failed for every chunk. LlamaIndex's extractors log the error and carry on; pass `kg_extractors=[SimpleLLMPathExtractor(llm=Settings.llm, raise_on_error=True)]` (from `llama_index.core.indices.property_graph`) to `from_documents` to see it (a rejected model name, a missing API key, a rate limit).
- **`RuntimeError: Event loop is closed`**: while building an index, pass `use_async=False`; on a later `insert`, insert through a new index with a new LLM object (see [Adding documents later](#adding-documents-later)).
- **`PermissionError: Security Violation [pathsec.open]: refusing multiply-linked file ... nltk_cache ...`** while building from long documents on Linux: NLTK 3.10 refuses to open the data files LlamaIndex's sentence splitter reads when uv installed them as hard links, uv's default on Linux. Reinstall with copies: `UV_LINK_MODE=copy uv sync --reinstall`, and keep `UV_LINK_MODE=copy` set for later installs.
- **`ValueError` about embedding dimensions**: the store was created with other dimensions than the embedding model gives. Pass `embed_model=` to the store, or the model's size as `embedding_dimensions`.

### Querying the graph

Queries against the quick start's graph, with their results:

```python
graph_store.structured_query(
    "MATCH (:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(:AI {name: 'Anton'}) RETURN p.name AS builder"
)
# [{'builder': 'Bertram Gilfoyle'}]: Gilfoyle built both

graph_store.structured_query(
    "MATCH (i)-[e:INVESTED_IN]->(:Company {name: $company}) RETURN i.name AS investor, e.round AS round ORDER BY e.date",
    param_map={"company": "Pied Piper"},
)
# Erlich Bachman (Incubator stake), Peter Gregory (Seed), Russ Hanneman (Series A), Raviga Capital (Secondary)

graph_store.structured_query("MATCH (s:AI {name: 'Son of Anton'}) RETURN s.bug_fix_strategy AS strategy")
# [{'strategy': 'delete the code'}]
```

LlamaIndex's `TextToCypherRetriever` and `CypherTemplateRetriever` generate Cypher, which is the default:

```python
from llama_index.core.indices.property_graph import TextToCypherRetriever

retriever = TextToCypherRetriever(graph_store, llm=Settings.llm)  # or any LlamaIndex LLM
nodes = retriever.retrieve("Who invested in Pied Piper?")
```

Relation labels are stored upper case with spaces and hyphens as underscores (`"works at"` becomes `WORKS_AT`). `get_schema_str()` describes the labels, properties and relationship patterns for an LLM; LlamaIndex's `TextToCypherRetriever` writes Cypher from it. Building a graph with `SchemaLLMPathExtractor` (typed labels and allowed relations) gives the model real structure to query.

### Vector search

```python
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters, VectorStoreQuery

query = VectorStoreQuery(
    query_embedding=Settings.embed_model.get_query_embedding("a kitchen appliance that talks"), similarity_top_k=3
)
nodes, scores = graph_store.vector_query(query)  # for example: Jian-Yang's fridge first

commas = MetadataFilters(filters=[MetadataFilter(key="net_worth_commas", value=2, operator=FilterOperator.GTE)])
retriever = index.as_retriever(filters=commas)  # passed on to VectorContextRetriever: only Russ Hanneman qualifies
```

`vector_query` searches entity embeddings with Grafeo's HNSW index; text chunks are left out. Filters combine with AND, with the operators `==`, `!=`, `>`, `>=`, `<`, `<=`, `in` and `nin`, and narrow the search inside the index.

## Configuration

```python
GrafeoPropertyGraphStore(
    db_path=None,  # str | PathLike | None: a .grafeo file, or None for in-memory
    embed_model=None,  # BaseEmbedding | None: read embedding_dimensions from this model
    embedding_dimensions=None,  # int | None: 1536 without embed_model
    embedding_metric="cosine",  # "cosine", "euclidean", "dot_product" or "manhattan"
    dedup_threshold=None,  # float | None: cosine similarity at which a new entity merges into an existing one
    query_language="cypher",  # "cypher" or "gql": the language of structured_query
    case_insensitive_names=True,  # match entity names ignoring case on upsert and lookup
)
```

## PropertyGraphStore Interface

| Method | Description |
| --- | --- |
| `upsert_nodes(nodes)` | Insert or update `EntityNode` and `ChunkNode` objects |
| `upsert_relations(relations)` | Insert or update relations between existing nodes |
| `get(properties, ids)` | Nodes by ID or property filter |
| `get_triplets(...)` | `(source, relation, target)` triplets by entity name (ignoring case by default), ID, relation label or property |
| `get_rel_map(nodes, depth, limit, ignore_rels)` | Relations around seed nodes, in both directions |
| `delete(...)` | Nodes with their relations, or relations by label; deleting a chunk also deletes what was extracted only from it |
| `structured_query(query, param_map, language)` | Cypher (default) or GQL, or Gremlin for a query starting with `g.`; `param_map` values are query parameters; one markdown code fence around the query is removed first, for LLMs that fence their answer |
| `vector_query(query)` | HNSW search over entity embeddings |
| `get_schema_str()` | Node and relation properties and relationship patterns, for text-to-Cypher prompts |
| `persist(path)` | Save an in-memory store to a `.grafeo` file |

GraphQL, SPARQL and SQL/PGQ run through `store.client` (`execute_graphql`, `execute_sparql`, `execute_sql`).

## Graph Algorithms

`store.client.algorithms` gives access to Grafeo's 30+ built-in algorithms, such as PageRank, betweenness and closeness centrality, Louvain communities, connected components and shortest paths (Dijkstra, BFS).

## Upgrading from grafeo 0.5

grafeo-llamaindex 0.3 requires grafeo 0.6, which stores databases in a new format. A database written by grafeo-llamaindex 0.2 on grafeo 0.5 migrates on the first read-write open, and the old database is kept next to it as `<path>.pre-0.6`. Back it up first and stop every process that opens it with grafeo 0.5. To go back, delete the migrated file (and the `<path>.wal` directory next to it, if there is one), rename the `.pre-0.6` copy to the original path, and run `uv add "grafeo-llamaindex==0.2.0" "grafeo<0.6"`: grafeo-llamaindex 0.2 does not cap grafeo, so without the second pin it would install grafeo 0.6 and migrate the database again. See [Upgrading to 0.6](../../get-started/upgrading/0.6.md) and the [grafeo-llamaindex README](https://github.com/GrafeoDB/grafeo-llamaindex#upgrading-from-grafeo-05).

## Requirements

- Python 3.12+
- grafeo 0.6 (`>=0.6.2,<0.7`)
- llama-index-core 0.14 or later (`<1`)

## License

Apache-2.0
