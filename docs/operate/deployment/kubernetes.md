# Kubernetes

Deploy grafeo-server on any Kubernetes cluster: ASK, EKS, GKE, or self-managed.

## Quick Start

### Deployment

```yaml
# grafeo-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grafeo-server
  labels:
    app: grafeo
spec:
  replicas: 1
  selector:
    matchLabels:
      app: grafeo
  template:
    metadata:
      labels:
        app: grafeo
    spec:
      containers:
        - name: grafeo
          image: grafeo/grafeo-server:full
          ports:
            - containerPort: 7474
              name: http
          env:
            - name: GRAFEO_DATA_DIR
              value: /data
            - name: GRAFEO_LOG_FORMAT
              value: json
          volumeMounts:
            - name: data
              mountPath: /data
          resources:
            requests:
              cpu: 500m
              memory: 512Mi
            limits:
              cpu: "2"
              memory: 2Gi
          livenessProbe:
            httpGet:
              path: /health
              port: 7474
            initialDelaySeconds: 5
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /health
              port: 7474
            initialDelaySeconds: 3
            periodSeconds: 10
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: grafeo-data
```

### Persistent Volume Claim

```yaml
# grafeo-pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: grafeo-data
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

### Service

```yaml
# grafeo-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: grafeo-server
spec:
  selector:
    app: grafeo
  ports:
    - name: http
      port: 7474
      targetPort: 7474
  type: ClusterIP
```

### Apply

```bash
kubectl apply -f grafeo-pvc.yaml
kubectl apply -f grafeo-deployment.yaml
kubectl apply -f grafeo-service.yaml
```

## Full Tier with All Protocols

Expose HTTP, Bolt, and GWP ports:

```yaml
ports:
  - containerPort: 7474
    name: http
  - containerPort: 7687
    name: bolt
  - containerPort: 7688
    name: gwp
```

```yaml
# Service with all ports
spec:
  ports:
    - name: http
      port: 7474
      targetPort: 7474
    - name: bolt
      port: 7687
      targetPort: 7687
    - name: gwp
      port: 7688
      targetPort: 7688
```

## Ingress

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: grafeo-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt
spec:
  tls:
    - hosts:
        - grafeo.example.com
      secretName: grafeo-tls
  rules:
    - host: grafeo.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: grafeo-server
                port:
                  number: 7474
```

## Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: grafeo-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: grafeo-server
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

## Cloud-Specific Notes

### ASK (Azure)

```bash
az ask create --resource-group grafeo-rg --name grafeo-cluster --node-count 2
az ask get-credentials --resource-group grafeo-rg --name grafeo-cluster
```

Uses Azure Disk for PVCs by default.

### EKS (AWS)

```bash
eksctl create cluster --name grafeo-cluster --region eu-west-1 --nodes 2
```

Install the EBS CSI driver for persistent volumes.

### GKE (Google Cloud)

```bash
gcloud container clusters create grafeo-cluster --zone europe-west4-a --num-nodes 2
gcloud container clusters get-credentials grafeo-cluster --zone europe-west4-a
```

Uses Persistent Disk for PVCs by default.
