# Docker

The simplest way to run grafeo-server. All cloud deployment guides build on these images.

## Quick Start

```bash
# In-memory (ephemeral)
docker run -p 7474:7474 grafeo/grafeo-server

# Persistent storage
docker run -p 7474:7474 -v grafeo-data:/data grafeo/grafeo-server
```

Studio UI available at `http://localhost:7474/studio/`.

## Docker Compose

```yaml
services:
  grafeo:
    image: grafeo/grafeo-server:latest
    ports:
      - "7474:7474"
    volumes:
      - grafeo-data:/data
    environment:
      - GRAFEO_SESSION_TTL=300
      - GRAFEO_LOG_FORMAT=json
    restart: unless-stopped

volumes:
  grafeo-data:
```

```bash
docker compose up -d
```

## Production Example

```bash
docker run -p 7474:7474 \
  grafeo/grafeo-server:full \
  --data-dir /data \
  --tls-cert /certs/cert.pem \
  --tls-key /certs/key.pem \
  --auth-token $API_TOKEN \
  --rate-limit 1000 \
  --cors-origins "https://app.example.com" \
  --log-format json
```

## Health Check

All HTTP tiers expose a health endpoint:

```bash
curl http://localhost:7474/health
```

```json
{
  "status": "ok",
  "features": {
    "languages": ["gql", "cypher", "sparql", "gremlin", "graphql", "sql-pgq"],
    "engine": ["parallel", "wal", "spill", "mmap"],
    "server": ["gwp"]
  }
}
```

## Building from Source

```bash
# Standard tier
docker build --target standard -t grafeo-server:standard .

# Full tier
docker build --target full -t grafeo-server:full .

# GWP tier (~7 MB)
docker build --target gwp -t grafeo-server:gwp .

# Bolt tier (~8 MB)
docker build --target bolt -t grafeo-server:bolt .
```

For full tier comparison, environment variables, and API docs, see [grafeo-server](../ecosystem/grafeo-server.md).
