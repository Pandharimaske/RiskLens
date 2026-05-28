# RiskLens Model Serving - Deployment Guide

## Overview
This guide explains how to deploy the RiskLens model serving API using Docker.

## Prerequisites
- Docker (19.03+)
- Docker Compose (1.25+)
- Model artifacts in `artifacts/` directory
- Trained model files:
  - `artifacts/calibrated_model.pkl`
  - `artifacts/preprocessor.pkl`
  - `artifacts/feature_names.pkl`

## Quick Start

### 1. Build Docker Image
\`\`\`bash
docker build -t risklens-api:latest .
\`\`\`

### 2. Run Container
\`\`\`bash
docker-compose up -d
\`\`\`

### 3. Verify Service
\`\`\`bash
curl http://localhost:8000/health
\`\`\`

## API Endpoints

### Health Check
\`\`\`bash
curl http://localhost:8000/health
\`\`\`

Response:
\`\`\`json
{
  "status": "healthy",
  "timestamp": "2026-05-27T14:30:00",
  "model_loaded": true,
  "model_version": "1.0.0",
  "uptime_seconds": 120
}
\`\`\`

### Model Info
\`\`\`bash
curl http://localhost:8000/model-info
\`\`\`

### Single Prediction
\`\`\`bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "id": "CUST_001",
    "gender": "Male",
    "age": 35,
    "driving_license": 1,
    "region_code": 28,
    "previously_insured": 0,
    "vehicle_age": "1-2 Year",
    "vehicle_damage": "No",
    "annual_premium": 40000,
    "policy_sales_channel": 26,
    "vintage": 200
  }'
\`\`\`

### Batch Predictions
\`\`\`bash
curl -X POST http://localhost:8000/predict-batch \
  -F "file=@batch_data.csv"
\`\`\`

## Environment Variables

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `WORKERS`: Number of worker processes (default: 4)
- `LOG_LEVEL`: Logging level (default: INFO)
- `DECISION_THRESHOLD`: Decision threshold (default: 0.0400)
- `MAX_BATCH_SIZE`: Maximum batch size (default: 10000)

## Performance Tuning

### Single Prediction
- Latency: ~50-100ms per request
- Throughput: ~100-200 requests/second

### Batch Prediction
- Latency: ~1-2 seconds for 10,000 records
- Throughput: ~5,000-10,000 records/second

## Production Deployment

### Kubernetes
\`\`\`yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: risklens-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: risklens-api
  template:
    metadata:
      labels:
        app: risklens-api
    spec:
      containers:
      - name: api
        image: risklens-api:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
\`\`\`

### AWS ECS
1. Push image to ECR: `aws ecr push risklens-api:latest`
2. Create ECS task definition with image URI
3. Deploy to ECS service with load balancer
4. Configure auto-scaling based on CPU/memory metrics

### Azure Container Instances
\`\`\`bash
az container create \
  --resource-group myResourceGroup \
  --name risklens-api \
  --image risklens-api:latest \
  --ports 8000 \
  --cpu 1 \
  --memory 2
\`\`\`

## Monitoring & Logging

### Docker Logs
\`\`\`bash
docker-compose logs -f risklens-api
\`\`\`

### Metrics
- Container memory usage: `docker stats`
- Request latency: API logs include request timestamps
- Error rates: Check container logs for HTTP errors

## Security Considerations

1. **Network Security**
   - Run behind reverse proxy (nginx, Apache)
   - Use HTTPS/TLS for all connections
   - Implement rate limiting

2. **Model Security**
   - Store model artifacts in secure location
   - Use read-only volumes for model files
   - Enable audit logging

3. **Input Validation**
   - Pydantic validates all inputs
   - Max batch size enforced
   - Request timeout protection

## Troubleshooting

### Model Not Loading
\`\`\`
Check artifact files exist in artifacts/ directory
\`\`\`

### High Memory Usage
\`\`\`
Reduce WORKERS environment variable
Reduce MAX_BATCH_SIZE
\`\`\`

### Slow Predictions
\`\`\`
Increase WORKERS
Use batch endpoint for multiple predictions
Check host system resources
\`\`\`

## Scaling

### Horizontal Scaling
- Use load balancer (ALB, NLB, nginx)
- Deploy multiple container replicas
- Each replica uses same model artifacts

### Vertical Scaling
- Increase CPU/memory allocation
- Increase WORKERS for more parallelism
- Monitor CPU and memory limits

## Version Management

Track model versions by:
1. Tagging image: `risklens-api:v1.0.0`
2. Storing model metadata in artifacts
3. Logging model version in API responses

## Next Steps

- **Monitoring**: Set up Prometheus/Grafana for metrics
- **Logging**: Configure ELK stack or CloudWatch for centralized logging
- **CI/CD**: Integrate with Jenkins/GitHub Actions for automated deployment
- **API Documentation**: Access Swagger docs at `http://localhost:8000/docs`
