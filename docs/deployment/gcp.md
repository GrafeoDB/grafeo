# Google Cloud Run

Run grafeo-server on Cloud Run with automatic scaling and scale-to-zero. The most straightforward serverless option of the three major clouds.

## Architecture

```
Internet → Cloud Run (auto-scaled, scale-to-zero) → GCS / Filestore
```

- **Compute**: Cloud Run (~$0.000024/vCPU-second, scales to zero)
- **Storage**: Cloud Storage (~$0.020/GB/month) or Filestore (~$0.20/GB/month)
- **Networking**: built-in HTTPS with managed certificates

## Prerequisites

- Google Cloud CLI (`gcloud`) installed and authenticated
- A GCP project with billing enabled

## Deploy

### 1. Deploy to Cloud Run

```bash
gcloud run deploy grafeo-server \
  --image grafeo/grafeo-server:full \
  --port 7474 \
  --region europe-west4 \
  --min-instances 0 \
  --max-instances 3 \
  --cpu 1 \
  --memory 2Gi \
  --set-env-vars "GRAFEO_LOG_FORMAT=json" \
  --allow-unauthenticated
```

Key flags:

- `--min-instances 0`: enables scale-to-zero
- `--max-instances 3`: auto-scales under load
- `--allow-unauthenticated`: makes the endpoint public (use `--no-allow-unauthenticated` + IAM for private access)

### 2. Get the Endpoint

```bash
gcloud run services describe grafeo-server \
  --region europe-west4 \
  --format 'value(status.url)'
```

Your instance is at `https://grafeo-server-xxx.a.run.app`.

## Persistence

Cloud Run containers are ephemeral by default. For durable storage:

### Option A: Cloud Storage Volume Mount

```bash
# Create a bucket
gsutil mb -l europe-west4 gs://grafeo-data-$PROJECT_ID

# Deploy with volume mount (Cloud Run v2)
gcloud run deploy grafeo-server \
  --image grafeo/grafeo-server:full \
  --port 7474 \
  --region europe-west4 \
  --min-instances 0 \
  --max-instances 3 \
  --cpu 1 \
  --memory 2Gi \
  --set-env-vars "GRAFEO_DATA_DIR=/data,GRAFEO_LOG_FORMAT=json" \
  --add-volume name=grafeo-data,type=cloud-storage,bucket=grafeo-data-$PROJECT_ID \
  --add-volume-mount volume=grafeo-data,mount-path=/data
```

### Option B: Filestore (NFS)

Lower latency, higher cost. Requires a VPC connector:

```bash
# Create Filestore instance
gcloud filestore instances create grafeo-nfs \
  --zone europe-west4-a \
  --file-share name=grafeo_data,capacity=10GB \
  --network name=default

# Create VPC connector for Cloud Run
gcloud compute networks vpc-access connectors create grafeo-connector \
  --region europe-west4 \
  --network default \
  --range 10.8.0.0/28
```

Then deploy with `--vpc-connector grafeo-connector` and NFS volume mount.

## Authentication

### Public with API Key (via Grafeo)

Use the `GRAFEO_AUTH_TOKEN` environment variable (full tier):

```bash
gcloud run deploy grafeo-server \
  --image grafeo/grafeo-server:full \
  --set-env-vars "GRAFEO_AUTH_TOKEN=$API_TOKEN,GRAFEO_LOG_FORMAT=json" \
  ...
```

### Private with IAM

```bash
gcloud run deploy grafeo-server \
  --no-allow-unauthenticated \
  ...

# Grant access to specific users/service accounts
gcloud run services add-iam-policy-binding grafeo-server \
  --region europe-west4 \
  --member "user:dev@example.com" \
  --role "roles/run.invoker"
```

Clients authenticate with `gcloud auth print-identity-token` in the `Authorization` header.

## Cost Estimate

| Usage | Compute | Storage (10 GB GCS) | Total |
|-------|---------|---------------------|-------|
| Idle (scale-to-zero) | $0.00 | ~$0.20 | ~$0.20/mo |
| Light (1 hr/day queries) | ~$2.60 | ~$0.20 | ~$2.80/mo |
| Moderate (8 hrs/day) | ~$20.70 | ~$0.20 | ~$20.90/mo |

## Monitoring

```bash
# View logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=grafeo-server" \
  --limit 50

# Stream logs
gcloud run services logs tail grafeo-server --region europe-west4
```

Cloud Run automatically exports metrics to Cloud Monitoring: request count, latency, instance count, CPU/memory utilization.

## Updating

```bash
gcloud run deploy grafeo-server \
  --image grafeo/grafeo-server:full \
  --region europe-west4
```

Cloud Run performs a rolling update. The previous revision stays available until the new one passes health checks.
