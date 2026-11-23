---
title: grafeo-langchain
description: LangChain integration for GrafeoDB with a graph store, a vector store with graph traversal, and question answering over knowledge graphs.
---

# grafeo-langchain

LangChain integration backed by GrafeoDB: a knowledge graph store, a vector store with graph traversal for Graph RAG, and a question-answering chain. Everything runs inside the Python process, with no server, Docker or plugins to set up.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-langchain){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-langchain/){ .md-button }

## Overview

| Component | What it does |
| --- | --- |
| `GrafeoGraphStore` | Stores entities and relationships (LLM-extracted or your own) and runs GQL/Cypher queries on them |
| `GrafeoQAChain` | Answers questions in plain language: an LLM writes a GQL query, Grafeo runs it, the LLM answers |
| `GrafeoGraphVectorStore` | A LangChain `VectorStore` whose documents link to each other, for vector search followed by graph traversal |
| `GrafeoAdapter` | Connects the vector store to `langchain-graph-retriever` (Eager, BFS and MMR strategies) |

A store is in memory, or one `.grafeo` file when given a `db_path`. The examples follow Pied Piper, the compression startup from HBO's Silicon Valley, and use Mistral; any LangChain chat model and embeddings work.

## Installation

```bash
uv add grafeo-langchain

# GrafeoAdapter for langchain-graph-retriever
uv add "grafeo-langchain[retriever]"

# The Mistral models of the examples (set MISTRAL_API_KEY)
uv add langchain-mistralai
```

This page describes grafeo-langchain 0.3, which requires Python 3.12+, grafeo 0.6 (`>=0.6.2,<0.7`) and langchain-core 1.2+.

## Knowledge graph

```python
from langchain_core.documents import Document
from grafeo_langchain import GrafeoGraphStore, GraphDocument, Node, Relationship

richard = Node(id="Richard Hendricks", type="Person", properties={"indentation": "tabs"})
gilfoyle = Node(id="Bertram Gilfoyle", type="Person")
dinesh = Node(id="Dinesh Chugtai", type="Person")
pied_piper = Node(id="Pied Piper", type="Company")
son_of_anton = Node(id="Son of Anton", type="AI", properties={"bug_fix_strategy": "delete the code"})
graph_document = GraphDocument(
    nodes=[richard, gilfoyle, dinesh, pied_piper, son_of_anton],
    relationships=[
        Relationship(source=richard, target=pied_piper, type="FOUNDED"),
        Relationship(source=gilfoyle, target=pied_piper, type="WORKS_AT"),
        Relationship(source=dinesh, target=pied_piper, type="WORKS_AT"),
        Relationship(source=gilfoyle, target=dinesh, type="RIVAL_OF"),
        Relationship(source=gilfoyle, target=son_of_anton, type="BUILT"),
    ],
    source=Document(
        page_content="Richard Hendricks founded Pied Piper. Gilfoyle and Dinesh work there and are rivals. "
        "Gilfoyle built Son of Anton, an AI that fixes bugs by deleting the code.",
        id="episode-1",
    ),
)

store = GrafeoGraphStore(db_path="./pied-piper.grafeo")
store.add_graph_documents([graph_document], include_source=True)

store.query("MATCH (p:Person)-[:WORKS_AT]->(c) RETURN p.node_id AS person, c.node_id AS company ORDER BY person")
# [{"person": "Bertram Gilfoyle", "company": "Pied Piper"}, {"person": "Dinesh Chugtai", "company": "Pied Piper"}]
```

- Entity ids are stored in the `node_id` property. Ingestion is idempotent: adding the same graph documents again updates their properties instead of duplicating them.
- With `include_source=True`, each source document becomes a `SourceDocument` node, with its text and a `MENTIONED_IN` edge from each of its entities. Its `node_id` is `source_` followed by the document's id (`source_episode-1`).
- `query(..., read_only=True)` runs a query in a transaction that is always rolled back, and raises `ValueError` if the query wrote anything. Use it for queries written by an LLM or a user.
- `get_schema` describes the graph for an LLM prompt: each label and relationship type with its properties and their types, then the relationship patterns such as `(:Person)-[:BUILT]->(:AI)`. Patterns that differ only in a label share a line (`(:Company|Organization)-[:LOCATED_IN]->(:City)`), and the edges to source documents take one. `get_structured_schema` returns the same as a dict, with `node_props`, `rel_props` and `relationships` keys.

Any chat model with structured output can turn text into graph documents. The [README](https://github.com/GrafeoDB/grafeo-langchain#extracting-the-graph-with-an-llm) has a recipe with fixed entity and relation types, one name per entity across documents, and amounts as properties, tested with Mistral Small 4.

## Question answering

```python
from langchain_mistralai import ChatMistralAI
from grafeo_langchain import GrafeoQAChain

llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
chain = GrafeoQAChain.from_llm(llm, graph=store)
chain.invoke({"query": "Who built Son of Anton?"})
# for example: {"query": "Who built Son of Anton?", "result": "Bertram Gilfoyle built Son of Anton."}
```

- **Read-only by default**: a generated query that would write is rolled back and raises `ValueError`, so the graph stays unchanged. `allow_writes=True` runs it.
- **Failing queries are fixed**: a query that fails goes back to the LLM with grafeo's error, once by default (`max_retries`). A reply that is not a query gets the same retry, then raises `ValueError`.
- **Made for extracted graphs**: the query prompt matches names by a distinctive word (`CONTAINS 'hooli'`), leaves node labels out and lists alternative relationship types. A query that matches nothing gets one looser try (`retry_on_empty`, read-only chains only).
- **Options**: `top_k` (result rows the LLM sees, 10 by default), `return_direct`, `return_intermediate_steps`, `id_property`, separate `query_llm` and `qa_llm`, and custom prompts.
- **A LangChain runnable**: `invoke`, `ainvoke`, `batch`, `|` composition and callbacks work as for any chain. The input and output keys are those of `GraphCypherQAChain`.

Generated queries run under the database's query timeout, 30 seconds by default.

## The Silicon Valley dataset

`grafeo-datasets` ships the Pied Piper story as a graph of 64 nodes and 149 edges (`uv add "grafeo-datasets[grafeo]"`). Load it into a store and ask away:

```python
from grafeo_datasets import silicon_valley

pied_piper = GrafeoGraphStore()
silicon_valley.load(pied_piper.client)

# How is Son of Anton related to Anton? Gilfoyle built both:
pied_piper.query(
    "MATCH (:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(:AI {name: 'Anton'}) RETURN p.name AS builder"
)
# [{"builder": "Bertram Gilfoyle"}]

chain = GrafeoQAChain.from_llm(llm, graph=pied_piper)
chain.invoke({"query": "Does Jian-Yang's fridge talk?"})["result"]
# for example: "Yes, Jian-Yang's fridge talks."
chain.invoke({"query": "What did Jian-Yang build?"})["result"]
# for example: "Jian-Yang built Not Hotdog and SeeFood."
```

The dataset names its entities by a `name` property, not `node_id`. The chain reads that from the schema and matches on `name` (`id_property` chooses another). A question that needs a path the model has to work out, such as the Son of Anton one, is safer as a GQL query like the one above.

## Vector store with graph links

```python
from langchain_mistralai import MistralAIEmbeddings
from grafeo_langchain import GrafeoGraphVectorStore

store = GrafeoGraphVectorStore(embedding=MistralAIEmbeddings(model="mistral-embed"), db_path="./pied-piper-docs.grafeo")

store.add_texts(
    [
        "Son of Anton is an AI that fixes bugs by deleting the code. It once ordered 4000 pounds of meat.",
        "Anton is Gilfoyle's server, the machine that runs Pied Piper's network.",
        "Jian-Yang's fridge has a screen and a voice, and is on more of the house network than it needs.",
    ],
    metadatas=[{"kind": "AI", "__graph_links__": [{"target_id": "anton", "type": "TRAINED_ON"}]}, {"kind": "AI"}, {"kind": "AI"}],
    ids=["son-of-anton", "anton", "jian-yangs-fridge"],
)

store.similarity_search("Which AI deletes code?", k=2)
store.traversal_search("an AI that deletes code", k=1, depth=1)  # Son of Anton, then Anton through TRAINED_ON
store.mmr_traversal_search("Pied Piper's machines", k=2, depth=1)
store.as_retriever(search_type="mmr").invoke("smart appliances")
```

- The embedding dimensions are detected from the model.
- Graph links go in the `__graph_links__` metadata key: a list of `{"target_id", "type", "properties"}` dicts (`type` defaults to `LINKS_TO`). A link may point to a document later in the same call.
- Adding a text whose id is stored replaces that document and keeps its links. `delete(ids)` removes documents with their links.
- `similarity_search_with_score`, `max_marginal_relevance_search`, `get_by_ids` and every `as_retriever` search type work, as do the async variants.
- Search results carry `metadata["score"]`, a relevance from 0 to 1 (1 identical, 0.5 unrelated, 0 opposite), and `metadata["_depth"]`: 0 for a search hit, the number of hops for a linked document.
- Filters (`filter={"kind": "AI"}`) are exact-match and type-strict, and in the traversal searches they apply to linked documents too.

## Upgrading from 0.2

grafeo-langchain 0.3 runs on grafeo 0.6, which migrates a grafeo 0.5 database to its single-file format the first time it opens it, and keeps the old files as `<path>.pre-0.6`. grafeo 0.5 cannot open the migrated database, so back it up first. To go back, delete the migrated file (and the `<path>.wal` directory next to it, if there is one), rename the `.pre-0.6` copy to the original path, and run `uv add "grafeo-langchain<0.3" "grafeo<0.6"`: grafeo-langchain 0.2 does not cap grafeo, so without the second pin it would install grafeo 0.6 and migrate the database again. See [Upgrading to 0.6](../../get-started/upgrading/0.6.md); the [README](https://github.com/GrafeoDB/grafeo-langchain#upgrading-from-grafeo-05-grafeo-langchain-02) has the steps, and the [changelog](https://github.com/GrafeoDB/grafeo-langchain/blob/main/CHANGELOG.md) lists what else changes.

## Requirements

- Python 3.12+
- grafeo 0.6 (`>=0.6.2,<0.7`)
- langchain-core 1.2+
- langchain-graph-retriever 0.8+, for `GrafeoAdapter` (the `[retriever]` extra)
- langchain-mistralai and grafeo-datasets, for the examples only

## License

Apache-2.0
