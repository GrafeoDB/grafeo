# AWS (App Runner / Fargate)

Run grafeo-server on AWS with two options: App Runner (simpler, scale-to-zero) or Fargate (more control).

## Option 1: AWS App Runner (Recommended)

Simplest path. Supports scale-to-zero, automatic TLS, and managed networking.

### Architecture

```
Internet → App Runner (auto-scaled) → EFS / S3
```

- **Compute**: App Runner (~$0.064/vCPU-hour, pauses when idle)
- **Storage**: EFS (~$0.30/GB/month) or S3 (~$0.023/GB/month)

### Prerequisites

- AWS CLI (`aws`) installed and authenticated
- ECR repository or public Docker Hub image

### Deploy

#### 1. Push Image to ECR (Optional)

If using a private registry:

```bash
aws ecr create-repository --repository-name grafeo-server

aws ecr get-login-password | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

docker tag grafeo/grafeo-server:full $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/grafeo-server:full
docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/grafeo-server:full
```

#### 2. Create the Service

```bash
aws apprunner create-service \
  --service-name grafeo-server \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "grafeo/grafeo-server:full",
      "ImageRepositoryType": "ECR_PUBLIC",
      "ImageConfiguration": {
        "Port": "7474",
        "RuntimeEnvironmentVariables": {
          "GRAFEO_DATA_DIR": "/data",
          "GRAFEO_LOG_FORMAT": "json",
          "GRAFEO_AUTH_TOKEN": "'$GRAFEO_AUTH_TOKEN'"
        }
      }
    },
    "AutoDeploymentsEnabled": false
  }' \
  --instance-configuration '{
    "Cpu": "1 vCPU",
    "Memory": "2 GB"
  }' \
  --health-check-configuration '{
    "Protocol": "HTTP",
    "Path": "/health",
    "Interval": 10,
    "Timeout": 5,
    "HealthyThreshold": 1,
    "UnhealthyThreshold": 3
  }'
```

#### 3. Get the Endpoint

```bash
aws apprunner describe-service \
  --service-arn $SERVICE_ARN \
  --query 'Service.ServiceUrl' \
  --output text
```

Your instance is at `https://<service-url>/`.

### Persistence with EFS

App Runner supports EFS volumes for durable storage:

```bash
# Create EFS filesystem
aws efs create-file-system --creation-token grafeo-data --tags Key=Name,Value=grafeo-data

# Associate with App Runner VPC connector (required for EFS)
aws apprunner create-vpc-connector \
  --vpc-connector-name grafeo-vpc \
  --subnets $SUBNET_IDS \
  --security-groups $SG_ID
```

---

## Option 2: ECS Fargate

More control over networking, placement, and task configuration. No scale-to-zero (minimum 1 task).

### Architecture

```
ALB → ECS Fargate Task → EFS / S3
```

### Deploy with Fargate

#### 1. Task Definition

```json
{
  "family": "grafeo-server",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "grafeo",
      "image": "grafeo/grafeo-server:full",
      "portMappings": [
        { "containerPort": 7474, "protocol": "tcp" }
      ],
      "environment": [
        { "name": "GRAFEO_DATA_DIR", "value": "/data" },
        { "name": "GRAFEO_LOG_FORMAT", "value": "json" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -sf http://localhost:7474/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/grafeo-server",
          "awslogs-region": "eu-west-1",
          "awslogs-stream-prefix": "grafeo"
        }
      }
    }
  ]
}
```

#### 2. Create Service

```bash
aws ecs create-service \
  --cluster grafeo \
  --service-name grafeo-server \
  --task-definition grafeo-server \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration '{
    "awsvpcConfiguration": {
      "subnets": ["subnet-xxx"],
      "securityGroups": ["sg-xxx"],
      "assignPublicIp": "ENABLED"
    }
  }'
```

## Cost Estimate

| Setup | Compute | Storage (10 GB EFS) | Total |
|-------|---------|---------------------|-------|
| App Runner (idle, paused) | ~$0.00 | ~$3.00 | ~$3.00/mo |
| App Runner (1 hr/day) | ~$1.90 | ~$3.00 | ~$4.90/mo |
| Fargate (always on, 1 task) | ~$46.00 | ~$3.00 | ~$49.00/mo |

App Runner is significantly cheaper for intermittent workloads due to pause-when-idle.

## Updating

```bash
# App Runner
aws apprunner update-service \
  --service-arn $SERVICE_ARN \
  --source-configuration '{
    "ImageRepository": {
      "ImageIdentifier": "grafeo/grafeo-server:full",
      "ImageRepositoryType": "ECR_PUBLIC"
    }
  }'

# Fargate
aws ecs update-service --cluster grafeo --service grafeo-server --force-new-deployment
```
