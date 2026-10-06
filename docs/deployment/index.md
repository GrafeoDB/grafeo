# Deployment

Grafeo-server can be deployed anywhere Docker runs. This section covers cloud-specific guides for running a production Grafeo instance on managed serverless infrastructure.

## Deployment Options

| Option | Best For | Scale-to-Zero | Setup Effort |
|--------|----------|---------------|-------------|
| [Docker](docker.md) | Self-hosted, on-premise, VMs | No | Low |
| [Azure](azure.md) | Azure Container Apps | Yes | Medium |
| [AWS](aws.md) | App Runner or Fargate | Partial | Medium |
| [GCP](gcp.md) | Cloud Run | Yes | Medium |
| [Kubernetes](kubernetes.md) | ASK, EKS, GKE, self-managed | No (pod-level) | Higher |

## Choosing a Tier

All deployment guides use the same Docker images. Pick the tier that matches your workload:

| Tier | Tag | Transport | Use Case |
|------|-----|-----------|----------|
| **gwp** | `grafeo-server:gwp` | gRPC :7688 | Sidecar, CI/CD, edge |
| **bolt** | `grafeo-server:bolt` | Bolt :7687 | Neo4j driver compatibility |
| **standard** | `grafeo-server:latest` | HTTP :7474 | General purpose, Studio UI |
| **full** | `grafeo-server:full` | All protocols | Production with auth, TLS, AI/search |

See [grafeo-server](../ecosystem/grafeo-server.md) for detailed tier comparison, environment variables, and API reference.

## Common Configuration

All deployments support the same environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRAFEO_DATA_DIR` | _(none)_ | Persistence path (omit for in-memory) |
| `GRAFEO_QUERY_TIMEOUT` | `30` | Query timeout in seconds |
| `GRAFEO_SESSION_TTL` | `300` | Transaction session timeout in seconds |
| `GRAFEO_LOG_FORMAT` | `pretty` | `pretty` or `json` (use `json` for cloud logging) |
| `GRAFEO_AUTH_TOKEN` | _(none)_ | Bearer token (full tier only) |

Full configuration reference: [Configuration](../getting-started/configuration.md)
