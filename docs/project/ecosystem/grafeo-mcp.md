---
title: grafeo-mcp
description: Model Context Protocol server that gives AI agents a Grafeo database.
---

# grafeo-mcp

A [Model Context Protocol](https://modelcontextprotocol.io/) (MCP) server that gives AI agents a Grafeo database: GQL queries, time-travel reads, full-text and vector search, and graph algorithms. The server embeds the database, so there is no separate database server to run.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-mcp){ .md-button }
[:material-package-variant: PyPI](https://pypi.org/project/grafeo-mcp/){ .md-button }

## Installation

```bash
uv tool install grafeo-mcp   # install the grafeo-mcp command
uvx grafeo-mcp               # or run it without installing
```

grafeo-mcp 0.3 needs Python 3.12 or later and runs on grafeo 0.6 (`grafeo>=0.6.2,<0.7`), which installs with it. `grafeo-mcp --version` prints both versions, `grafeo-mcp --help` the options.

!!! warning "Coming from grafeo-mcp 0.2"
    grafeo 0.6 writes a new file format. The first read-write start migrates a grafeo 0.5.x database in place and keeps the original as `<path>.pre-0.6`; grafeo 0.5 cannot open the migrated database. Stop every process that uses the database and back it up first. The [upgrade steps](https://github.com/GrafeoDB/grafeo-mcp#upgrading-from-grafeo-mcp-02-grafeo-05) are in the README.

## Quick start

Set `GRAFEO_DB_PATH` to keep your data: without it the database lives in memory and is gone when the client stops the server.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "grafeo": {
      "command": "uvx",
      "args": ["grafeo-mcp"],
      "env": { "GRAFEO_DB_PATH": "/path/to/your/graph.grafeo" }
    }
  }
}
```

Claude Desktop often does not see your shell's `PATH`, so `uvx` (or the full path that `uv tool dir --bin` shows) is the safer command.

### Claude Code

```bash
claude mcp add grafeo --scope project -e GRAFEO_DB_PATH=./graph.grafeo -- grafeo-mcp
```

### VS Code / Copilot

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "grafeo": {
      "command": "grafeo-mcp",
      "env": { "GRAFEO_DB_PATH": "${workspaceFolder}/graph.grafeo" }
    }
  }
}
```

### HTTP transport

For clients that connect to a running server instead of starting one:

```bash
grafeo-mcp streamable-http                  # http://127.0.0.1:8000/mcp
grafeo-mcp streamable-http --port 9000      # another port
grafeo-mcp streamable-http --host 0.0.0.0   # every interface
```

All clients share one database. The server has no authentication: on an address other than loopback, anyone who reaches the port can read and change the database. Put it behind a proxy that authenticates, or set `GRAFEO_READ_ONLY=1`. The legacy `sse` transport takes the same options.

## Try it with the demo dataset

The Silicon Valley dataset from [grafeo.ai](https://grafeo.ai) holds Pied Piper's world: 64 nodes (people, companies, products, AIs) and 149 edges, with a MiniLM embedding (384 dimensions) and a cosine vector index for every label. Load it into a file and point the server at it:

```bash
uv run --with "grafeo-datasets[grafeo]" python -c "import grafeo; from grafeo_datasets import silicon_valley; db = grafeo.GrafeoDB('pied-piper.grafeo'); silicon_valley.load(db, vectors=True); db.close()"
claude mcp add pied-piper -e GRAFEO_DB_PATH=./pied-piper.grafeo -- grafeo-mcp
```

Questions an agent can answer, and the call it makes (node IDs as in a fresh load):

| Ask                                       | Call                                                                                                                                                  | Answer                                            |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| How is Son of Anton related to Anton?     | `execute_gql`: `MATCH (s:AI {name: 'Son of Anton'})<-[:BUILT]-(p:Person)-[:BUILT]->(a:AI {name: 'Anton'}) RETURN p.name AS builder`                   | Bertram Gilfoyle built both                       |
| Does Jian-Yang's fridge talk?             | `execute_gql` with `params={"name": "Jian-Yang's fridge"}`: `MATCH (f:AI {name: $name})<-[:OWNS]-(p:Person) RETURN f.talks AS talks, p.name AS owner` | `talks: true`, owned by Jian-Yang                 |
| How does Son of Anton fix bugs?           | `execute_gql`: `MATCH (s:AI {name: 'Son of Anton'}) RETURN s.bug_fix_strategy AS fix`                                                                 | "delete the code"                                 |
| How many commas does Russ have?           | `execute_gql`: `MATCH (r:Person {name: 'Russ Hanneman'}) RETURN r.net_worth_commas AS commas`                                                         | 3                                                 |
| What is most like Jian-Yang's fridge?     | `vector_search(label="AI", property="embedding", like_node_id=30, k=2)`                                                                               | Anton (distance 0.574), Fiona (0.760)             |
| Who is the most central person?           | `pagerank(label="Person", top_k=1)`                                                                                                                   | Richard Hendricks                                 |
| How is Richard connected to Son of Anton? | `dijkstra(source_id=51, target_id=58)`                                                                                                                | Richard Hendricks, Bertram Gilfoyle, Son of Anton |

`like_node_id` searches near a node's stored vector, so an agent without its own embedding model can still ask what is alike. A query that misses says why: `MATCH (a:Ai) RETURN a.name` returns `"hint": "Not in the graph: label :Ai (did you mean :AI?)."`. Add `GRAFEO_READ_ONLY=1` to explore without changing the file.

## Configuration

| Environment variable       | Default                                       | Description                                                                                  |
| -------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------------- |
| `GRAFEO_DB_PATH`           | in memory (lost when the server stops)        | Path to the database file, such as `graph.grafeo`; created if nothing exists there           |
| `GRAFEO_READ_ONLY`         | off                                           | `1`, `true`, `yes` or `on` disables every write                                              |
| `GRAFEO_HISTORY_RETENTION` | the stored setting (`off` for a new database) | History kept for `query_as_of` and `get_node_as_of`: `off`, `forever` or a number of seconds |

A database is always one file. A database open for writing is locked: no other process can open it, not even read-only. Several `GRAFEO_READ_ONLY=1` servers can share a file while no process has it open for writing. A bad setting or a database that cannot be opened stops the server with one `grafeo-mcp: error: ...` line.

## Tools

grafeo-mcp has 25 tools. Agents start with `graph_info`; the server's instructions tell them so.

| Group        | Tools                                                                                                                                                                               |
| ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Query        | `execute_gql`: one GQL statement per call (Cypher syntax such as `CREATE`, `MERGE`, `OPTIONAL MATCH` runs as written), with `params` and write counters                             |
| Time travel  | `query_as_of`, `get_node_as_of`                                                                                                                                                     |
| Graph        | `graph_info`, `create_node`, `create_edge`, `get_node`, `update_node`, `delete_node`, `update_edge`, `delete_edge`, `get_neighbors`, `search_nodes_by_label`                        |
| Batch import | `batch_import`: up to 500 nodes and 1000 edges, all or nothing                                                                                                                      |
| Full-text    | `create_text_index`, `search_text`                                                                                                                                                  |
| Vector       | `create_vector_index`, `vector_search`, `mmr_search`, `vector_graph_search`: from a query vector, or from a node's stored vector with `like_node_id`                                |
| Algorithms   | `pagerank` and `betweenness_centrality` (with an optional `label` filter), `dijkstra` (edge direction ignored unless `direction` says otherwise), `louvain`, `connected_components` |

A reading `execute_gql` query that finds nothing, or returns only nulls, gets a hint naming the labels, edge types or properties that do not exist, such as `Not in the graph: label :Persn (did you mean :Person?).` A path pattern written straight into `WHERE` or `DELETE` gets a hint with the GQL form. Embeddings show as `<vector: 384 dims>` wherever a node appears. Every failure is one `Error: <tool>: ...` message, flagged `isError` over MCP, with a hint where the next step is not obvious.

## Resources and prompts

- Resources: `graph://schema` (labels with sample values, edge type patterns, indexes) and `graph://stats` (counts, memory and disk sizes); template `graph://nodes/{node_id}` (one node and its connections).
- Prompts: `explore_graph`, `knowledge_extraction`, `graph_analysis`, `similarity_search`.

## See also

- **[grafeo-memory](grafeo-memory.md)** includes an MCP server (`grafeo-memory-mcp`) for its memory API: extract, reconcile, search, summarize. Use it for AI memory management rather than raw graph access.

## License

Apache-2.0
