# KataForge API Reference

## Overview

KataForge provides a comprehensive RESTful API for martial arts technique analysis, built with FastAPI. The API supports real-time analysis, batch processing, coach management, authentication, and more.

## API Base URL

```
http://localhost:8000  # Development
https://api.kataforge.com  # Production
```

## Authentication

### API Key Authentication

```bash
# Set API key in header
curl -X POST \
  http://localhost:8000/analyze \
  -H "X-API-Key: your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "technique.mp4", "coach": "nagato"}'
```

### JWT Authentication

```bash
# Login to get JWT token
curl -X POST \
  http://localhost:8000/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user&password=pass"

# Use JWT token
curl -X POST \
  http://localhost:8000/analyze \
  -H "Authorization: Bearer your_jwt_token" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "technique.mp4", "coach": "nagato"}'
```

## Endpoints

### Health Check

**GET /** - Health check endpoint

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-28T12:00:00Z",
  "gpu_available": true,
  "models_loaded": ["graphsage", "form_assessor"]
}
```

### Technique Analysis

**POST /analyze** - Analyze a single technique

```bash
curl -X POST \
  http://localhost:8000/analyze \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "http://example.com/technique.mp4",
    "coach": "nagato",
    "technique": "roundhouse",
    "llm_backend": "ollama",
    "verbose": true
  }'
```

**Request Body:**
```json
{
  "video_url": "string (required)",  # URL or base64 encoded video
  "coach": "string (required)",      # Coach identifier
  "technique": "string",             # Technique name
  "llm_backend": "string",           # "ollama" or "llamacpp"
  "verbose": "boolean",              # Detailed analysis
  "device": "string"                 # "auto", "cpu", "cuda", "rocm"
}
```

**Response:**
```json
{
  "id": "analysis_12345",
  "status": "completed",
  "video": "http://example.com/technique.mp4",
  "coach": "nagato",
  "technique": "roundhouse",
  "timestamp": "2026-01-28T12:00:00Z",
  "metrics": {
    "overall_score": 8.5,
    "aspect_scores": {
      "speed": 8.2,
      "force": 8.7,
      "timing": 7.9,
      "balance": 8.8,
      "coordination": 8.4
    },
    "biomechanics": {
      "max_speed": 4.8,
      "peak_force": 1200.5,
      "mean_power": 850.2,
      "kinetic_chain_efficiency": 88.5
    }
  },
  "corrections": [
    "Improve hip rotation timing - initiate rotation earlier",
    "Maintain better center of gravity throughout the technique"
  ],
  "recommendations": [
    "Practice shadowboxing for 10 minutes daily",
    "Focus on chambering technique before strikes",
    "Add plyometric exercises to improve explosive speed"
  ],
  "processing_time": 2.45
}
```

### Batch Analysis

**POST /batch_analyze** - Analyze multiple techniques

```bash
curl -X POST \
  http://localhost:8000/batch_analyze \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "videos": [
      {
        "url": "technique1.mp4",
        "coach": "nagato",
        "technique": "roundhouse"
      },
      {
        "url": "technique2.mp4",
        "coach": "nagato",
        "technique": "teep"
      }
    ],
    "parallel": true,
    "max_workers": 4
  }'
```

**Request Body:**
```json
{
  "videos": [
    {
      "url": "string (required)",
      "coach": "string (required)",
      "technique": "string",
      "metadata": "object"
    }
  ],
  "parallel": "boolean",
  "max_workers": "integer",
  "device": "string"
}
```

**Response:**
```json
{
  "batch_id": "batch_67890",
  "status": "completed",
  "results": [
    {
      "video": "technique1.mp4",
      "analysis_id": "analysis_12345",
      "status": "completed",
      "score": 8.5,
      "processing_time": 2.45
    },
    {
      "video": "technique2.mp4",
      "analysis_id": "analysis_12346",
      "status": "completed",
      "score": 7.8,
      "processing_time": 2.10
    }
  ],
  "summary": {
    "total_videos": 2,
    "completed": 2,
    "failed": 0,
    "avg_score": 8.15,
    "total_time": 4.55
  }
}
```

### Coach Management

**GET /coaches** - List all coaches

```bash
curl -X GET \
  http://localhost:8000/coaches \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "coaches": [
    {
      "id": "nagato",
      "name": "Nagato",
      "style": "Muay Thai",
      "rank": "Champion",
      "years_experience": 15,
      "techniques": ["roundhouse", "teep", "elbow"],
      "created_at": "2026-01-01T00:00:00Z",
      "updated_at": "2026-01-28T12:00:00Z"
    },
    {
      "id": "sagat",
      "name": "Sagat",
      "style": "Muay Thai",
      "rank": "Master",
      "years_experience": 25,
      "techniques": ["tiger_knee", "tiger_uppercut"],
      "created_at": "2026-01-02T00:00:00Z",
      "updated_at": "2026-01-28T12:00:00Z"
    }
  ],
  "total": 2
}
```

**GET /coaches/{coach_id}** - Get coach details

```bash
curl -X GET \
  http://localhost:8000/coaches/nagato \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "id": "nagato",
  "name": "Nagato",
  "style": "Muay Thai",
  "rank": "Champion",
  "years_experience": 15,
  "teachers": ["Master Toddy"],
  "techniques": ["roundhouse", "teep", "elbow"],
  "teaching_philosophy": {
    "focus": "Precision and power",
    "style": "Aggressive counter-attacking",
    "specialty": "Clinch work and knee strikes"
  },
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-28T12:00:00Z"
}
```

**POST /coaches** - Create a new coach

```bash
curl -X POST \
  http://localhost:8000/coaches \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "new_coach",
    "name": "New Coach",
    "style": "Muay Thai",
    "rank": "Instructor",
    "years_experience": 5,
    "techniques": ["roundhouse", "teep"]
  }'
```

**Request Body:**
```json
{
  "id": "string (required)",
  "name": "string (required)",
  "style": "string (required)",
  "rank": "string",
  "years_experience": "integer",
  "teachers": "array",
  "techniques": "array",
  "teaching_philosophy": "object"
}
```

**Response:**
```json
{
  "id": "new_coach",
  "name": "New Coach",
  "style": "Muay Thai",
  "rank": "Instructor",
  "years_experience": 5,
  "techniques": ["roundhouse", "teep"],
  "created_at": "2026-01-28T12:00:00Z",
  "updated_at": "2026-01-28T12:00:00Z"
}
```

**PUT /coaches/{coach_id}** - Update coach

```bash
curl -X PUT \
  http://localhost:8000/coaches/nagato \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Nagato Updated",
    "years_experience": 16
  }'
```

**Request Body:** (any coach field)

**Response:**
```json
{
  "id": "nagato",
  "name": "Nagato Updated",
  "style": "Muay Thai",
  "rank": "Champion",
  "years_experience": 16,
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-28T12:00:00Z"
}
```

**DELETE /coaches/{coach_id}** - Delete coach

```bash
curl -X DELETE \
  http://localhost:8000/coaches/nagato \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "success": true,
  "message": "Coach nagato deleted successfully"
}
```

### Model Management

**GET /models** - List available models

```bash
curl -X GET \
  http://localhost:8000/models \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "models": [
    {
      "name": "graphsage",
      "type": "technique_classifier",
      "version": "0.1.0",
      "description": "GraphSAGE model for technique classification",
      "input_dim": 132,
      "output_dim": 50,
      "hidden_dim": 128,
      "layers": 3,
      "accuracy": 0.91,
      "created_at": "2026-01-01T00:00:00Z",
      "updated_at": "2026-01-28T12:00:00Z"
    },
    {
      "name": "form_assessor",
      "type": "form_analysis",
      "version": "0.1.0",
      "description": "LSTM + Attention model for form assessment",
      "input_dim": 132,
      "lstm_units": 256,
      "lstm_layers": 2,
      "attention_heads": 8,
      "output_aspects": 5,
      "accuracy": 0.88,
      "created_at": "2026-01-02T00:00:00Z",
      "updated_at": "2026-01-28T12:00:00Z"
    }
  ],
  "total": 2
}
```

**POST /models/train** - Train a new model

```bash
curl -X POST \
  http://localhost:8000/models/train \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "graphsage",
    "coach": "nagato",
    "data_dir": "/path/to/training/data",
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "device": "cuda"
  }'
```

**Request Body:**
```json
{
  "model_type": "string (required)",  # "graphsage", "form_assessor", "style_encoder"
  "coach": "string (required)",
  "data_dir": "string (required)",
  "epochs": "integer",
  "batch_size": "integer",
  "learning_rate": "float",
  "device": "string",
  "checkpoint_dir": "string",
  "resume_from": "string"
}
```

**Response:**
```json
{
  "training_id": "train_12345",
  "status": "started",
  "model_type": "graphsage",
  "coach": "nagato",
  "epochs": 100,
  "batch_size": 16,
  "learning_rate": 0.001,
  "device": "cuda",
  "started_at": "2026-01-28T12:00:00Z",
  "estimated_completion": "2026-01-28T14:00:00Z"
}
```

**GET /models/train/{training_id}** - Get training status

```bash
curl -X GET \
  http://localhost:8000/models/train/train_12345 \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "training_id": "train_12345",
  "status": "in_progress",
  "model_type": "graphsage",
  "coach": "nagato",
  "progress": {
    "current_epoch": 45,
    "total_epochs": 100,
    "train_loss": 0.123,
    "val_loss": 0.145,
    "val_accuracy": 0.912,
    "best_accuracy": 0.921
  },
  "started_at": "2026-01-28T12:00:00Z",
  "estimated_completion": "2026-01-28T14:00:00Z",
  "logs": [
    {
      "timestamp": "2026-01-28T12:00:00Z",
      "message": "Training started",
      "level": "info"
    },
    {
      "timestamp": "2026-01-28T12:45:00Z",
      "message": "Epoch 45/100 completed",
      "level": "info"
    }
  ]
}
```

### System Information

**GET /system/info** - Get system information

```bash
curl -X GET \
  http://localhost:8000/system/info \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "app_name": "kataforge",
  "version": "0.1.0",
  "environment": "production",
  "debug": false,
  "gpu": {
    "available": true,
    "device": "cuda",
    "memory_total": 24576,
    "memory_used": 8192,
    "memory_free": 16384,
    "utilization": 33.3
  },
  "models": [
    {
      "name": "graphsage",
      "loaded": true,
      "version": "0.1.0"
    },
    {
      "name": "form_assessor",
      "loaded": true,
      "version": "0.1.0"
    }
  ],
  "coaches": 5,
  "analyses": 128,
  "uptime": "2d 3h 15m",
  "started_at": "2026-01-26T09:00:00Z"
}
```

**GET /system/health** - Get health status

```bash
curl -X GET \
  http://localhost:8000/system/health \
  -H "Authorization: Bearer your_token"
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-01-28T12:00:00Z",
  "components": {
    "database": "healthy",
    "gpu": "healthy",
    "models": "healthy",
    "llm_backend": "healthy",
    "storage": "healthy"
  },
  "warnings": [],
  "errors": []
}
```

### Metrics

**GET /metrics** - Get Prometheus metrics

```bash
curl -X GET \
  http://localhost:8000/metrics
```

**Response:**
```
# HELP kataforge_analyses_total Total number of analyses
# TYPE kataforge_analyses_total counter
kataforge_analyses_total{coach="nagato"} 45
kataforge_analyses_total{coach="sagat"} 32

# HELP kataforge_analysis_time Analysis processing time
# TYPE kataforge_analysis_time histogram
kataforge_analysis_time_bucket{le="0.5"} 10
kataforge_analysis_time_bucket{le="1.0"} 35
kataforge_analysis_time_bucket{le="2.0"} 60
kataforge_analysis_time_bucket{le="5.0"} 77
kataforge_analysis_time_bucket{le="+Inf"} 77
kataforge_analysis_time_sum 123.45
kataforge_analysis_time_count 77

# HELP kataforge_api_requests_total Total API requests
# TYPE kataforge_api_requests_total counter
kataforge_api_requests_total{endpoint="/analyze",method="POST"} 77
kataforge_api_requests_total{endpoint="/coaches",method="GET"} 15
kataforge_api_requests_total{endpoint="/system/health",method="GET"} 42

# HELP kataforge_gpu_memory_usage GPU memory usage
# TYPE kataforge_gpu_memory_usage gauge
kataforge_gpu_memory_usage 8192

# HELP kataforge_model_inference_time Model inference time
# TYPE kataforge_model_inference_time histogram
kataforge_model_inference_time_bucket{model="graphsage",le="0.01"} 5
kataforge_model_inference_time_bucket{model="graphsage",le="0.05"} 45
kataforge_model_inference_time_bucket{model="graphsage",le="0.1"} 70
kataforge_model_inference_time_bucket{model="graphsage",le="+Inf"} 77
kataforge_model_inference_time_sum 4.23
kataforge_model_inference_time_count 77
```

## Error Handling

### Error Responses

**400 Bad Request**
```json
{
  "error": {
    "code": "invalid_request",
    "message": "Invalid request parameters",
    "details": {
      "video_url": "This field is required",
      "coach": "This field is required"
    },
    "timestamp": "2026-01-28T12:00:00Z"
  }
}
```

**401 Unauthorized**
```json
{
  "error": {
    "code": "unauthorized",
    "message": "Authentication required",
    "timestamp": "2026-01-28T12:00:00Z"
  }
}
```

**403 Forbidden**
```json
{
  "error": {
    "code": "forbidden",
    "message": "Insufficient permissions",
    "timestamp": "2026-01-28T12:00:00Z"
  }
}
```

**404 Not Found**
```json
{
  "error": {
    "code": "not_found",
    "message": "Resource not found",
    "resource": "coach",
    "id": "unknown_coach",
    "timestamp": "2026-01-28T12:00:00Z"
  }
}
```

**429 Too Many Requests**
```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Rate limit exceeded",
    "limit": 100,
    "window": 60,
    "retry_after": 30,
    "timestamp": "2026-01-28T12:00:00Z"
  }
}
```

**500 Internal Server Error**
```json
{
  "error": {
    "code": "internal_error",
    "message": "Internal server error",
    "request_id": "req_12345",
    "timestamp": "2026-01-28T12:00:00Z"
  }
}
```

### Error Codes

| Code | Description |
|------|-------------|
| `invalid_request` | Invalid request parameters |
| `unauthorized` | Authentication required |
| `forbidden` | Insufficient permissions |
| `not_found` | Resource not found |
| `rate_limit_exceeded` | Rate limit exceeded |
| `internal_error` | Internal server error |
| `model_error` | Model inference error |
| `gpu_error` | GPU processing error |
| `storage_error` | Storage operation error |
| `validation_error` | Data validation error |
| `timeout_error` | Request timeout |
| `llm_error` | LLM processing error |

## Rate Limiting

The API enforces rate limiting to prevent abuse:

- **Default Limit**: 100 requests per 60 seconds
- **Burst Allowance**: 20 requests above normal rate
- **Authentication**: Required for most endpoints

**Rate Limit Headers:**
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 30
Retry-After: 30
```

## Authentication Methods

### API Key Authentication

```bash
# Set API key in header
curl -H "X-API-Key: your_api_key" \
  http://localhost:8000/analyze

# Or in query parameter
curl "http://localhost:8000/analyze?api_key=your_api_key"
```

### JWT Authentication

```bash
# Login to get token
curl -X POST \
  http://localhost:8000/auth/login \
  -d "username=user&password=pass"

# Use token
curl -H "Authorization: Bearer your_token" \
  http://localhost:8000/analyze
```

### Multiple API Keys

```bash
# Set multiple API keys
export DOJO_API_KEYS="key1,key2,key3"

# Or use API keys file
echo "key1" > api_keys.txt
echo "key2" >> api_keys.txt
export DOJO_API_KEYS_FILE="api_keys.txt"
```

## WebSocket API

### Real-time Analysis

**WS /ws/analyze** - Real-time analysis WebSocket

```javascript
const socket = new WebSocket('ws://localhost:8000/ws/analyze');

socket.onopen = () => {
  socket.send(JSON.stringify({
    action: 'start_analysis',
    video_url: 'technique.mp4',
    coach: 'nagato'
  }));
};

socket.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Analysis update:', data);
};

socket.onclose = () => {
  console.log('Connection closed');
};
```

**Messages:**

**Client → Server:**
```json
{
  "action": "start_analysis",
  "video_url": "technique.mp4",
  "coach": "nagato",
  "technique": "roundhouse"
}
```

**Server → Client:**
```json
{
  "type": "status",
  "status": "processing",
  "progress": 0.45,
  "message": "Extracting poses from video"
}

{
  "type": "biomechanics",
  "data": {
    "max_speed": 4.2,
    "peak_force": 1150.8
  }
}

{
  "type": "score",
  "data": {
    "overall": 8.2,
    "aspects": {
      "speed": 8.0,
      "force": 8.5
    }
  }
}

{
  "type": "correction",
  "data": {
    "text": "Improve hip rotation timing"
  }
}

{
  "type": "complete",
  "data": {
    "analysis_id": "analysis_12345",
    "score": 8.2,
    "processing_time": 2.45
  }
}
```

## API Client Libraries

### Python Client

```python
import httpx
from kataforge.client import KataForgeClient

# Initialize client
client = KataForgeClient(
    base_url="http://localhost:8000",
    api_key="your_api_key"
)

# Analyze technique
result = client.analyze(
    video_url="technique.mp4",
    coach="nagato",
    technique="roundhouse"
)

print(f"Score: {result['metrics']['overall_score']}")

# List coaches
coaches = client.list_coaches()
for coach in coaches:
    print(f"{coach['id']}: {coach['name']}")
```

### JavaScript Client

```javascript
const kataforge = require('kataforge-client');

// Initialize client
const client = new kataforge.Client({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your_api_key'
});

// Analyze technique
async function analyze() {
  const result = await client.analyze({
    videoUrl: 'technique.mp4',
    coach: 'nagato',
    technique: 'roundhouse'
  });
  
  console.log(`Score: ${result.metrics.overall_score}`);
}

// List coaches
async function listCoaches() {
  const coaches = await client.listCoaches();
  coaches.forEach(coach => {
    console.log(`${coach.id}: ${coach.name}`);
  });
}
```

## API Security

### Best Practices

1. **Use HTTPS** in production
2. **Rotate API keys** regularly
3. **Enable authentication** for all endpoints
4. **Use rate limiting** to prevent abuse
5. **Validate all inputs** on server side
6. **Sanitize outputs** to prevent injection
7. **Monitor API usage** for anomalies

### Security Headers

```
Content-Security-Policy: default-src 'self'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Strict-Transport-Security: max-age=31536000; includeSubDomains
Referrer-Policy: strict-origin-when-cross-origin
```

## API Versioning

```
GET /v1/analyze        # Version 1
GET /v2/analyze        # Version 2 (future)
```

**Headers:**
```
Accept: application/vnd.kataforge.v1+json
```

## API Development

### Running the API Server

```bash
# Start development server
kataforge serve --reload

# Start production server
kataforge serve --workers 4

# Start with custom settings
export DOJO_API_HOST=0.0.0.0
export DOJO_API_PORT=8000
export DOJO_API_WORKERS=4
kataforge serve
```

### API Configuration

```yaml
# config/api.yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  reload: false
  timeout: 30
  max_request_size: 104857600  # 100MB
  cors_origins: "http://localhost:3000,http://localhost:7860"
  auth_enabled: true
  rate_limit_enabled: true
  rate_limit_requests: 100
  rate_limit_window: 60
```

### Environment Variables

```bash
# API settings
export DOJO_API_HOST=0.0.0.0
export DOJO_API_PORT=8000
export DOJO_API_WORKERS=4
export DOJO_API_RELOAD=true
export DOJO_CORS_ORIGINS="http://localhost:3000"

# Security
export DOJO_AUTH_ENABLED=true
export DOJO_API_KEYS="key1,key2"
export DOJO_RATE_LIMIT_ENABLED=true
export DOJO_RATE_LIMIT_REQUESTS=100

# Start server
kataforge serve
```

## API Testing

### Using curl

```bash
# Test health endpoint
curl http://localhost:8000/

# Test authentication
curl -X POST http://localhost:8000/auth/login -d "username=user&password=pass"

# Test analysis with API key
curl -X POST http://localhost:8000/analyze \
  -H "X-API-Key: your_key" \
  -H "Content-Type: application/json" \
  -d '{"video_url": "technique.mp4", "coach": "nagato"}'
```

### Using Python

```python
import httpx

# Test API
async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
    # Health check
    response = await client.get("/")
    print(response.json())
    
    # Analyze technique
    response = await client.post(
        "/analyze",
        json={
            "video_url": "technique.mp4",
            "coach": "nagato"
        },
        headers={"X-API-Key": "your_key"}
    )
    print(response.json())
```

### Using Postman

1. Import OpenAPI specification
2. Set environment variables
3. Test endpoints with GUI
4. Save requests for reuse

## API Monitoring

### Prometheus Metrics

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'kataforge'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

### Grafana Dashboard

Import KataForge dashboard JSON to visualize:
- Request rates
- Processing times
- Error rates
- GPU utilization
- Model performance

## API Deployment

### Docker

```bash
# Build Docker image
nix build .#docker-cpu

# Run Docker container
docker run -p 8000:8000 kataforge:latest

# With environment variables
docker run \
  -p 8000:8000 \
  -e DOJO_API_KEYS="key1,key2" \
  -e DOJO_AUTH_ENABLED=true \
  kataforge:latest
```

### Kubernetes

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kataforge-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kataforge-api
  template:
    metadata:
      labels:
        app: kataforge-api
    spec:
      containers:
      - name: kataforge
        image: kataforge:latest
        ports:
        - containerPort: 8000
        env:
        - name: DOJO_API_KEYS
          valueFrom:
            secretKeyRef:
              name: kataforge-secrets
              key: api_keys
        - name: DOJO_AUTH_ENABLED
          value: "true"
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"
```

### Load Balancing

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: kataforge-service
spec:
  selector:
    app: kataforge-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## API Performance

### Optimization Tips

1. **Use GPU acceleration** for model inference
2. **Enable batch processing** for multiple videos
3. **Use async/await** for I/O operations
4. **Cache frequent queries** to database
5. **Optimize model sizes** for your hardware
6. **Use connection pooling** for database
7. **Enable compression** for large responses

### Performance Metrics

| Metric | Target |
|--------|-------|
| Request latency | < 500ms |
| Model inference | < 100ms |
| Concurrent users | 100+ |
| Throughput | 50+ req/sec |
| GPU utilization | 70-90% |

## API Examples

### Complete Workflow

```python
import httpx
import asyncio

async def analyze_workflow():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        
        # 1. Login
        login_response = await client.post(
            "/auth/login",
            data={"username": "user", "password": "pass"}
        )
        token = login_response.json()["access_token"]
        
        # 2. List coaches
        coaches_response = await client.get(
            "/coaches",
            headers={"Authorization": f"Bearer {token}"}
        )
        coaches = coaches_response.json()["coaches"]
        
        # 3. Analyze technique
        analysis_response = await client.post(
            "/analyze",
            json={
                "video_url": "technique.mp4",
                "coach": coaches[0]["id"],
                "technique": "roundhouse"
            },
            headers={"Authorization": f"Bearer {token}"}
        )
        analysis = analysis_response.json()
        
        print(f"Score: {analysis['metrics']['overall_score']}")
        print(f"Corrections: {analysis['corrections']}")
        
        # 4. Get system info
        system_response = await client.get(
            "/system/info",
            headers={"Authorization": f"Bearer {token}"}
        )
        system = system_response.json()
        print(f"GPU: {system['gpu']['device']}")

asyncio.run(analyze_workflow())
```

### Batch Processing

```python
import httpx
import asyncio

async def batch_analysis():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        
        videos = [
            {"url": "technique1.mp4", "coach": "nagato", "technique": "roundhouse"},
            {"url": "technique2.mp4", "coach": "nagato", "technique": "teep"},
            {"url": "technique3.mp4", "coach": "sagat", "technique": "tiger_knee"}
        ]
        
        response = await client.post(
            "/batch_analyze",
            json={
                "videos": videos,
                "parallel": True,
                "max_workers": 3
            },
            headers={"X-API-Key": "your_key"}
        )
        
        results = response.json()
        for result in results["results"]:
            print(f"{result['video']}: {result['score']}")

asyncio.run(batch_analysis())
```

## API Reference Implementation

The API is implemented in `kataforge/api/server.py` with the following structure:

```python
from fastapi import FastAPI, APIRouter, Depends, HTTPException
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import logging

# Create FastAPI app
app = FastAPI(title="KataForge API", version="0.1.0")

# Security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    # Verify API key
    if api_key not in get_api_keys():
        raise HTTPException(status_code=401, detail="Unauthorized")
    return api_key

# Models
class AnalysisRequest(BaseModel):
    video_url: str
    coach: str
    technique: str = None
    llm_backend: str = "ollama"
    verbose: bool = False
    device: str = "auto"

class AnalysisResponse(BaseModel):
    id: str
    status: str
    video: str
    coach: str
    metrics: dict
    corrections: list
    recommendations: list

# Routes
router = APIRouter(dependencies=[Depends(verify_api_key)])

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_technique(request: AnalysisRequest):
    # Process analysis
    result = await process_analysis(request)
    return result

@router.get("/coaches")
async def list_coaches():
    # List coaches
    coaches = get_coaches()
    return {"coaches": coaches}

# Include routes
app.include_router(router, prefix="/api/v1")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

## API Error Handling

```python
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": "http_error",
                "message": exc.detail,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "Internal server error",
                "request_id": request.state.request_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "code": "validation_error",
                "message": "Validation failed",
                "details": exc.errors(),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }
    )
```

## API Rate Limiting

```python
from fastapi import Request, HTTPException
from fastapi.limits import RateLimiter
import redis.asyncio as redis

# Setup Redis for rate limiting
redis_client = redis.Redis(host="localhost", port=6379, db=0)

limiter = RateLimiter(
    redis_client,
    key_func=lambda req: req.client.host,
    rate="100/minute",
    burst=20
)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    try:
        await limiter.check(request)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Remaining"] = str(await limiter.get_remaining(request))
        return response
    except RateLimitExceeded:
        raise HTTPException(
            status_code=429,
            detail={
                "error": {
                    "code": "rate_limit_exceeded",
                    "message": "Rate limit exceeded",
                    "limit": 100,
                    "window": 60,
                    "retry_after": 30
                }
            }
        )
```

## Conclusion

The KataForge API provides a comprehensive interface for martial arts technique analysis with support for real-time processing, batch operations, coach management, and more. The API is designed for performance, security, and ease of use, with support for multiple authentication methods, rate limiting, and comprehensive error handling.

For best results:
- Use HTTPS in production
- Enable authentication for all endpoints
- Monitor API usage and performance
- Rotate API keys regularly
- Optimize for your specific hardware configuration
