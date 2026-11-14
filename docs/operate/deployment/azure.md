# Azure Container Apps

Run grafeo-server on Azure Container Apps with scale-to-zero. Idle instances cost nothing (only storage).

## Architecture

```
Internet → Container Apps (consumption plan) → Blob Storage / Managed Disk
```

- **Compute**: Container Apps consumption plan (~$0.000012/vCPU-second, scales to zero)
- **Storage**: Azure Blob Storage (~$0.018/GB/month) or Managed Disk for lower latency
- **Networking**: built-in ingress with TLS termination

## Prerequisites

- Azure CLI (`az`) installed and authenticated
- A resource group (examples use `grafeo-rg` in `westeurope`)

## Deploy

### 1. Create the Container Apps Environment

```bash
az containerapp env create \
  --name grafeo-env \
  --resource-group grafeo-rg \
  --location westeurope
```

### 2. Create a Storage Account (for persistence)

```bash
az storage account create \
  --name grafeostorage \
  --resource-group grafeo-rg \
  --location westeurope \
  --sku Standard_LRS

az storage share create \
  --name grafeo-data \
  --account-name grafeostorage
```

### 3. Mount Storage to Environment

```bash
az containerapp env storage set \
  --name grafeo-env \
  --resource-group grafeo-rg \
  --storage-name grafeo-storage \
  --azure-file-account-name grafeostorage \
  --azure-file-account-key $(az storage account keys list --account-name grafeostorage --query '[0].value' -o tsv) \
  --azure-file-share-name grafeo-data \
  --access-mode ReadWrite
```

### 4. Deploy grafeo-server

```bash
az containerapp create \
  --name grafeo-server \
  --resource-group grafeo-rg \
  --environment grafeo-env \
  --image grafeo/grafeo-server:full \
  --target-port 7474 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi \
  --env-vars \
    GRAFEO_DATA_DIR=/data \
    GRAFEO_LOG_FORMAT=json \
    GRAFEO_AUTH_TOKEN=secretref:auth-token \
  --secrets auth-token=$GRAFEO_AUTH_TOKEN
```

Key flags:

- `--min-replicas 0`: enables scale-to-zero (no cost when idle)
- `--max-replicas 3`: auto-scales under load
- `--ingress external`: creates a public HTTPS endpoint

### 5. Get the Endpoint

```bash
az containerapp show \
  --name grafeo-server \
  --resource-group grafeo-rg \
  --query properties.configuration.ingress.fqdn \
  -o tsv
```

Your Grafeo instance is available at `https://<fqdn>/`.

## Scale-to-Zero Behavior

Container Apps scales to zero when no requests arrive for the configured cooldown period (default: 300 seconds). On the next request, cold start takes 1-3 seconds depending on the tier.

To keep the instance warm for lower latency, set `--min-replicas 1`.

## Cost Estimate

| Usage | Compute | Storage (10 GB) | Total |
|-------|---------|-----------------|-------|
| Idle (scale-to-zero) | $0.00 | ~$0.18 | ~$0.18/mo |
| Light (1 hr/day queries) | ~$1.30 | ~$0.18 | ~$1.48/mo |
| Moderate (8 hrs/day) | ~$10.40 | ~$0.18 | ~$10.58/mo |

## Monitoring

```bash
# View logs
az containerapp logs show \
  --name grafeo-server \
  --resource-group grafeo-rg

# View metrics
az monitor metrics list \
  --resource $(az containerapp show --name grafeo-server --resource-group grafeo-rg --query id -o tsv) \
  --metric Requests
```

## Updating

```bash
az containerapp update \
  --name grafeo-server \
  --resource-group grafeo-rg \
  --image grafeo/grafeo-server:full
```

Container Apps performs a rolling update with zero downtime.
