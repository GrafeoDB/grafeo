# grafeo-server

Standalone HTTP server and web UI for the Grafeo graph database.

[:octicons-mark-github-16: GitHub](https://github.com/GrafeoDB/grafeo-server){ .md-button }
[:material-docker: Docker Hub](https://hub.docker.com/r/grafeodb/grafeo-server){ .md-button }

## Overview

grafeo-server wraps the Grafeo engine in a REST API, turning it from an embeddable library into a standalone database server. Pure Rust, single binary, ~20MB Docker image.

- **REST API** with auto-commit and explicit transaction modes
- **Multi-language queries**: GQL, Cypher, GraphQL via dedicated endpoints
- **Web UI** for interactive query exploration
- **ACID transactions** with session-based lifecycle
- **In-memory or persistent**: omit data directory for ephemeral, set it for durable storage

## Quick Start

### Docker

```bash
docker compose up -d
```

Server at `http://localhost:7474`. Web UI at `http://localhost:7474/ui`.

### Binary

```bash
grafeo-server --data-dir ./mydata    # persistent
grafeo-server                        # in-memory
```

## API Endpoints

### Query (auto-commit)

| Endpoint | Language | Example |
|----------|----------|---------|
| `POST /query` | GQL (default) | `{"query": "MATCH (p:Person) RETURN p.name"}` |
| `POST /cypher` | Cypher | `{"query": "MATCH (n) RETURN count(n)"}` |
| `POST /graphql` | GraphQL | `{"query": "{ Person { name age } }"}` |

```bash
curl -X POST http://localhost:7474/query \
  -H "Content-Type: application/json" \
  -d '{"query": "MATCH (p:Person) RETURN p.name, p.age"}'
```

### Transactions

```bash
# Begin
SESSION=$(curl -s -X POST http://localhost:7474/tx/begin | jq -r .session_id)

# Execute
curl -X POST http://localhost:7474/tx/query \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: $SESSION" \
  -d '{"query": "INSERT (:Person {name: '\''Alice'\''})"}'

# Commit (or POST /tx/rollback)
curl -X POST http://localhost:7474/tx/commit \
  -H "X-Session-Id: $SESSION"
```

### Health

```bash
curl http://localhost:7474/health
```

## Configuration

Environment variables (prefix `GRAFEO_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFEO_HOST` | `0.0.0.0` | Bind address |
| `GRAFEO_PORT` | `7474` | Bind port |
| `GRAFEO_DATA_DIR` | _(none)_ | Persistence directory (omit for in-memory) |
| `GRAFEO_SESSION_TTL` | `300` | Transaction session timeout (seconds) |
| `GRAFEO_CORS_ORIGINS` | _(none)_ | Comma-separated allowed origins |
| `GRAFEO_LOG_LEVEL` | `info` | Tracing log level |

CLI flags override environment variables.

## When to Use

| Use Case | Recommendation |
|----------|----------------|
| Multi-client access over HTTP | grafeo-server |
| Embedded in your application | [grafeo](https://github.com/GrafeoDB/grafeo) (library) |
| Browser-only, no backend | [grafeo-web](https://github.com/GrafeoDB/grafeo-web) (WASM) |

## Requirements

- Docker (recommended) or Rust toolchain for building from source

## License

AGPL-3.0
