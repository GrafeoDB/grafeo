# Ecosystem

The GrafeoDB ecosystem extends beyond the core database with AI/LLM integrations, visualization tools, benchmarking and browser support.

<div class="grid cards" markdown>

-   :material-server:{ .lg .middle } **grafeo-server**

    ---

    HTTP server & web UI for Grafeo: REST API, transactions, single binary (~40MB Docker image).

    [:octicons-arrow-right-24: Learn more](grafeo-server.md)

-   :material-chart-line:{ .lg .middle } **graph-bench**

    ---

    Comprehensive benchmark suite for comparing graph database performance across 8 engines.

    [:octicons-arrow-right-24: Learn more](graph-bench.md)

-   :material-vector-point:{ .lg .middle } **anywidget-vector**

    ---

    Interactive 3D vector visualization for embeddings in Python notebooks.

    [:octicons-arrow-right-24: Learn more](anywidget-vector.md)

-   :material-graph:{ .lg .middle } **anywidget-graph**

    ---

    Interactive graph visualization widget for Python notebooks with multi-backend support.

    [:octicons-arrow-right-24: Learn more](anywidget-graph.md)

-   :material-web:{ .lg .middle } **grafeo-web**

    ---

    Browser-based Grafeo via WebAssembly with IndexedDB persistence.

    [:octicons-arrow-right-24: Learn more](grafeo-web.md)

-   :material-link-variant:{ .lg .middle } **grafeo-langchain**

    ---

    LangChain integration for Grafeo: graph store, vector store and Graph RAG retrieval.

    [:octicons-arrow-right-24: Learn more](grafeo-langchain.md)

-   :material-llama:{ .lg .middle } **grafeo-llamaindex**

    ---

    LlamaIndex integration: PropertyGraphStore, vector search and knowledge graph construction.

    [:octicons-arrow-right-24: Learn more](grafeo-llamaindex.md)

-   :material-chip:{ .lg .middle } **grafeo-mcp**

    ---

    Model Context Protocol server: expose a graph database as tools for LLM agents.

    [:octicons-arrow-right-24: Learn more](grafeo-mcp.md)

-   :material-brain:{ .lg .middle } **grafeo-memory**

    ---

    AI memory layer for LLM applications: extract facts, deduplicate, semantic search, change history.

    [:octicons-arrow-right-24: Learn more](grafeo-memory.md)

-   :material-table-arrow-right:{ .lg .middle } **Arrow / Polars / pandas**

    ---

    Zero-copy bulk export via Arrow IPC for analytics workflows. `nodes_to_arrow()`, `nodes_to_polars()`, DuckDB integration.

    [:octicons-arrow-right-24: Arrow export tutorial](../tutorials/examples/arrow-export-example.md)

</div>
