# DMS API Gateway

A production-grade FastAPI gateway for the Railway Data Management System (DMS) backend.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [Health Endpoints](#health-endpoints)
  - [Abnormal Alerts](#abnormal-alerts-api)
  - [Container Identification](#container-identification-api)
  - [Signal Light Tracking](#signal-light-tracking-api)
  - [Ticket OCR](#ticket-ocr-api)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Development](#development)
- [Project Structure](#project-structure)

## Overview

The DMS API Gateway provides a robust, scalable interface for railway operation monitoring and management. It acts as a middleware layer between client applications and the DMS backend system, offering:

- Unified REST API for all railway operations
- Automatic retry with exponential backoff
- Circuit breaker pattern for fault tolerance
- Request validation and sanitization
- Comprehensive logging and monitoring

## Features

| Feature | Description |
|---------|-------------|
| **Abnormal Alerts** | Report operational abnormalities with photo evidence |
| **Container Identification** | Submit OCR-identified container information |
| **Signal Light Tracking** | Report signal light state changes |
| **Ticket OCR** | Parse grouping orders using optical character recognition |
| **Health Checks** | Kubernetes-ready liveness and readiness probes |
| **Circuit Breaker** | Automatic fault isolation for upstream failures |
| **Rate Limiting** | Configurable request rate limiting per client |
| **API Key Auth** | Optional API key authentication |

## Quick Start

### Prerequisites

- Python 3.11+
- pip or conda

### Installation

```bash
# Clone the repository
cd dms_api

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app.main:app --reload
```

### Access the API

- **API Base URL**: http://localhost:8000
- **Interactive Docs (Swagger)**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

---

## API Reference

### Response Format

All API responses follow a consistent structure:

```json
{
  "success": true,
  "message": "Success",
  "data": { ... },
  "request_id": "abc123",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

### Error Response Format

```json
{
  "success": false,
  "error": {
    "error_code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": { ... }
  },
  "request_id": "abc123"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 422 | Validation Error |
| 429 | Rate Limited |
| 500 | Internal Server Error |
| 502 | Upstream Service Error |
| 503 | Circuit Breaker Open |

---

## Health Endpoints

### GET /health

Basic health check for load balancers and monitoring.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "environment": "production",
  "timestamp": "2024-01-10T12:00:00Z",
  "checks": {
    "api": { "status": "ok" }
  }
}
```

### GET /health/live

Kubernetes liveness probe.

**Response:**
```json
{
  "status": "alive"
}
```

### GET /health/ready

Kubernetes readiness probe. Checks if the service is ready to accept traffic.

**Response:**
```json
{
  "ready": true,
  "checks": {
    "dms_backend": true
  }
}
```

### GET /health/detailed

Detailed health check with dependency status.

**Response:**
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "environment": "production",
  "checks": {
    "dms_backend": {
      "status": "ok",
      "url": "http://123.127.38.120:2010"
    },
    "configuration": {
      "status": "ok",
      "environment": "production",
      "debug": false
    }
  }
}
```

---

## Abnormal Alerts API

### POST /api/v1/abnormal

Report an abnormal condition detected during train operation.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `carbinNo` | string | Yes | Carriage number (e.g., "C12345") |
| `descr` | string | Yes | Alert description |
| `file` | file | Yes | Photo file (JPEG, PNG, etc.) |

**Example Request (cURL):**
```bash
curl -X POST "http://localhost:8000/api/v1/abnormal" \
  -F "carbinNo=C12345" \
  -F "descr=Unusual vibration detected" \
  -F "file=@photo.jpg"
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Alert created successfully",
  "data": {
    "id": 12345,
    "carbinNo": "C12345",
    "descr": "Unusual vibration detected"
  },
  "request_id": "abc123",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

---

### POST /api/v1/abnormal/condition

Report carriage condition anomalies (unlocked doors, foreign objects).

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `carbinNo` | string | Yes | Carriage number |
| `abnormalType` | integer | Yes | Abnormality type: `1` = Unlocked, `2` = Foreign object |
| `isAbnormal` | integer | Yes | Status: `0` = Normal, `1` = Abnormal |
| `descr` | string | No | Description of the abnormality |
| `file` | file | No | Optional photo evidence |

**Abnormal Types:**

| Value | Description |
|-------|-------------|
| `1` | UNLOCKED - Lock not properly secured (未落锁) |
| `2` | FOREIGN_OBJECT - Foreign object detected (有异物) |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/abnormal/condition" \
  -F "carbinNo=C12345" \
  -F "abnormalType=1" \
  -F "isAbnormal=1" \
  -F "descr=Door lock appears damaged" \
  -F "file=@photo.jpg"
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Condition reported successfully",
  "data": {
    "id": 12346,
    "carbinNo": "C12345",
    "abnormalType": 1,
    "isAbnormal": 1,
    "descr": "Door lock appears damaged"
  },
  "request_id": "abc124",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

---

## Container Identification API

### POST /api/v1/container

Submit OCR-identified container information.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `trainNo` | string | Yes | Train number (车号) |
| `carbinNo` | string | Yes | Carriage number (车厢编号) |
| `contrainerNo` | string | Yes | Container number (集装箱编号) |
| `file` | file | Yes | Identification photo |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/container" \
  -F "trainNo=T001" \
  -F "carbinNo=C12345" \
  -F "contrainerNo=CSLU1234567" \
  -F "file=@container.jpg"
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Container recorded",
  "data": {
    "id": 12347,
    "trainNo": "T001",
    "carbinNo": "C12345",
    "contrainerNo": "CSLU1234567"
  },
  "request_id": "abc125",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

---

## Signal Light Tracking API

### POST /api/v1/signal/change

Report a signal light state change with photographic evidence.

**Content-Type:** `multipart/form-data`

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Signal light photo |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/signal/change" \
  -F "file=@signal.jpg"
```

**Response (201 Created):**
```json
{
  "success": true,
  "message": "Signal change recorded",
  "data": {
    "id": 12348,
    "status": "processed"
  },
  "request_id": "abc126",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

---

## Ticket OCR API

### POST /api/v1/ticket/parse

Parse a grouping order ticket using OCR.

**Content-Type:** `multipart/form-data`

**Supported File Formats:**
- Images: JPEG, PNG, BMP, GIF
- Documents: PDF

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | Ticket image or document |

**Example Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ticket/parse" \
  -F "file=@ticket.jpg"
```

**Response (200 OK):**
```json
{
  "success": true,
  "message": "Ticket parsed successfully",
  "data": {
    "ASSIGN_ID": 12349,
    "planNo": "PLN001",
    "trainNo": "T12345",
    "trainType": "货车",
    "stock": "1",
    "seq": "001",
    "oilType": "柴油",
    "emptyCapacity": "25.5",
    "changeLength": "1.0",
    "loadCapacity": "60.0",
    "container1": "CSLU1234567",
    "container2": "CSLU7654321",
    "startStation": "北京",
    "destStation": "上海",
    "carryType": "钢材",
    "descr": "普通货物",
    "ticketNo": "TK20240110001"
  },
  "request_id": "abc127",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

**Extracted Fields:**

| Field | Description |
|-------|-------------|
| `ASSIGN_ID` | Record ID |
| `planNo` | Plan sequence number (计划序号) |
| `trainNo` | Train number (车号) |
| `trainType` | Train type (车种) |
| `stock` | Track number (股道) |
| `seq` | Sequence number (序号) |
| `oilType` | Oil type (油种) |
| `emptyCapacity` | Empty weight (自重) |
| `changeLength` | Conversion length (换长) |
| `loadCapacity` | Load capacity (载重) |
| `container1` | Container 1 (集装箱1) |
| `container2` | Container 2 (集装箱2) |
| `startStation` | Origin station (发站) |
| `destStation` | Destination station (到站) |
| `carryType` | Cargo type (品名) |
| `descr` | Notes (记事) |
| `ticketNo` | Ticket number (票据号) |

---

## Configuration

Configuration is managed via environment variables with the `DMS_` prefix.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_APP_NAME` | DMS API Gateway | Application name |
| `DMS_ENVIRONMENT` | development | Environment: development, staging, production |
| `DMS_DEBUG` | false | Enable debug mode |
| `DMS_HOST` | 0.0.0.0 | Server host |
| `DMS_PORT` | 8000 | Server port |
| `DMS_WORKERS` | 1 | Number of workers |

### DMS Backend Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_DMS_BASE_URL` | http://123.127.38.120:2010 | DMS backend API URL |
| `DMS_DMS_TIMEOUT_CONNECT` | 5.0 | Connection timeout (seconds) |
| `DMS_DMS_TIMEOUT_READ` | 30.0 | Read timeout (seconds) |
| `DMS_DMS_MAX_RETRIES` | 3 | Maximum retry attempts |
| `DMS_DMS_RETRY_BACKOFF` | 1.0 | Retry backoff factor |

### Circuit Breaker Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_CIRCUIT_BREAKER_ENABLED` | true | Enable circuit breaker |
| `DMS_CIRCUIT_BREAKER_THRESHOLD` | 5 | Failure threshold |
| `DMS_CIRCUIT_BREAKER_TIMEOUT` | 30.0 | Reset timeout (seconds) |

### Security Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_API_KEY_ENABLED` | false | Enable API key authentication |
| `DMS_API_KEYS` | | Comma-separated API keys |
| `DMS_CORS_ORIGINS` | * | Allowed CORS origins |

### Rate Limiting Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_RATE_LIMIT_ENABLED` | false | Enable rate limiting |
| `DMS_RATE_LIMIT_REQUESTS` | 100 | Requests per window |
| `DMS_RATE_LIMIT_WINDOW` | 60 | Window in seconds |

### Logging Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_LOG_LEVEL` | INFO | Log level: DEBUG, INFO, WARNING, ERROR |
| `DMS_LOG_FORMAT` | json | Log format: json, text |
| `DMS_LOG_FILE` | | Optional log file path |

### File Upload Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `DMS_MAX_UPLOAD_SIZE` | 52428800 | Maximum upload size (50MB) |
| `DMS_ALLOWED_EXTENSIONS` | .jpg,.jpeg,.png,.gif,.bmp,.pdf | Allowed file extensions |

### Example .env File

```env
# Application
DMS_ENVIRONMENT=production
DMS_DEBUG=false

# DMS Backend
DMS_DMS_BASE_URL=http://123.127.38.120:2010
DMS_DMS_TIMEOUT_READ=30.0
DMS_DMS_MAX_RETRIES=3

# Circuit Breaker
DMS_CIRCUIT_BREAKER_ENABLED=true
DMS_CIRCUIT_BREAKER_THRESHOLD=5

# Security
DMS_API_KEY_ENABLED=true
DMS_API_KEYS=key1,key2,key3

# Rate Limiting
DMS_RATE_LIMIT_ENABLED=true
DMS_RATE_LIMIT_REQUESTS=100

# Logging
DMS_LOG_LEVEL=INFO
DMS_LOG_FORMAT=json
```

---

## Deployment

### Docker

```bash
# Build the image
docker build -t dms-api-gateway .

# Run the container
docker run -d \
  -p 8000:8000 \
  -e DMS_ENVIRONMENT=production \
  -e DMS_DMS_BASE_URL=http://dms-backend:2010 \
  --name dms-api \
  dms-api-gateway
```

### Docker Compose

```bash
# Start production environment
docker-compose up -d dms-api

# Start development environment
docker-compose --profile dev up -d dms-api-dev
```

### Kubernetes

Example deployment configuration:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: dms-api-gateway
spec:
  replicas: 3
  selector:
    matchLabels:
      app: dms-api
  template:
    metadata:
      labels:
        app: dms-api
    spec:
      containers:
      - name: dms-api
        image: dms-api-gateway:latest
        ports:
        - containerPort: 8000
        env:
        - name: DMS_ENVIRONMENT
          value: production
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
```

---

## Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_api/test_endpoints.py -v
```

### Code Quality

```bash
# Format code
black app tests

# Lint code
ruff check app tests

# Type checking
mypy app
```

---

## Project Structure

```
dms_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Pydantic Settings configuration
│   ├── dependencies.py      # Dependency injection
│   ├── api/
│   │   ├── health.py        # Health check endpoints
│   │   └── v1/
│   │       ├── router.py    # API v1 router
│   │       ├── abnormal.py  # Abnormal alerts endpoints
│   │       ├── container.py # Container identification endpoints
│   │       ├── signal.py    # Signal light endpoints
│   │       └── ticket.py    # Ticket OCR endpoints
│   ├── core/
│   │   ├── exceptions.py    # Custom exception classes
│   │   ├── logging.py       # Structured logging setup
│   │   ├── middleware.py    # Request/response middleware
│   │   └── security.py      # API key validation
│   ├── repositories/
│   │   ├── base.py          # Base repository class
│   │   └── dms.py           # DMS backend client
│   ├── schemas/
│   │   ├── base.py          # Base Pydantic schemas
│   │   ├── common.py        # Common types and enums
│   │   ├── abnormal.py      # Abnormal alert schemas
│   │   ├── container.py     # Container schemas
│   │   ├── signal.py        # Signal light schemas
│   │   └── ticket.py        # Ticket OCR schemas
│   └── services/
│       ├── base.py          # Base service class
│       ├── abnormal.py      # Abnormal alert business logic
│       ├── container.py     # Container business logic
│       ├── signal.py        # Signal light business logic
│       └── ticket.py        # Ticket OCR business logic
├── tests/
│   └── test_api/
│       ├── conftest.py      # Test fixtures
│       └── test_endpoints.py # API endpoint tests
├── requirements.txt         # Python dependencies
├── pytest.ini               # Pytest configuration
├── Dockerfile               # Docker build configuration
├── docker-compose.yml       # Docker Compose configuration
└── README.md                # This file
```

---

## Response Headers

All responses include the following headers:

| Header | Description |
|--------|-------------|
| `X-Request-ID` | Unique request identifier for tracing |
| `X-Response-Time` | Request processing time |
| `X-Content-Type-Options` | Security header (nosniff) |
| `X-Frame-Options` | Security header (DENY) |
| `X-XSS-Protection` | Security header |
| `Referrer-Policy` | Security header |

---

## License

Copyright 2024 DMS API Team. All rights reserved.

---

## Support

For issues and feature requests, please contact the development team or open an issue in the repository.
