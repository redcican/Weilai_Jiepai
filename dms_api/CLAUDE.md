# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Start development server
uvicorn app.main:app --reload

# Run all tests
pytest

# Run specific test file
pytest tests/test_api/test_endpoints.py -v

# Run tests with coverage
pytest --cov=app --cov-report=html
```

## Architecture Overview

This is a FastAPI gateway for a Railway Data Management System (DMS). It includes local OCR capability using CnOCR and can optionally proxy requests to an upstream DMS backend.

### Layer Structure

```
Request → API Router → Service → Repository/OCR → DMS Backend
```

- **API Layer** (`app/api/`): FastAPI routers handle HTTP requests, form data parsing, file uploads
- **Service Layer** (`app/services/`): Business logic, validation, response building
- **Repository Layer** (`app/repositories/dms.py`): HTTP client to upstream DMS backend with retry and circuit breaker
- **OCR Layer** (`app/ocr/`): Local CnOCR-based table extraction for railway tickets

### Key Patterns

**Dependency Injection**: All dependencies flow through `app/dependencies.py`. Services depend on repositories, routers depend on services. Override `get_repository` in tests to inject mocks.

**Schemas**: Pydantic v2 models in `app/schemas/`. Use `serialize_by_alias=True` for camelCase JSON output. All responses extend `ResponseSchema[T]` from `base.py`.

**Configuration**: Pydantic Settings in `app/config.py`. All settings use `DMS_` prefix for environment variables.

**Error Handling**: Custom exceptions in `app/core/exceptions.py` are caught by handlers in `main.py` and converted to standardized JSON responses.

### OCR Integration

The OCR module (`app/ocr/`) provides local table extraction for railway tickets:

- **CnOCREngine** (`engine.py`): Singleton wrapper around CnOCR with ONNX backend
- **TableOCRProcessor** (`processor.py`): Extracts structured table data from images
- **OCRService** (`services/ocr.py`): High-level service for ticket parsing

The ticket service uses local OCR by default and falls back to DMS backend if OCR fails or is disabled. Configuration:
- `DMS_OCR_ENABLED`: Enable/disable local OCR (default: true)
- `DMS_OCR_ENHANCE_IMAGE`: Apply image enhancement before OCR (default: true)
- `DMS_OCR_FALLBACK_TO_DMS`: Fallback to DMS backend on OCR failure (default: true)

### API Endpoints

All business endpoints are under `/api/v1/`:
- `POST /api/v1/abnormal` - Abnormal alerts with photo (multipart/form-data)
- `POST /api/v1/abnormal/condition` - Carriage condition reports
- `POST /api/v1/container` - Container identification with photo
- `POST /api/v1/signal/change` - Signal light changes with photo
- `POST /api/v1/ticket/parse` - OCR ticket parsing (uses local OCR)

Health endpoints: `/health`, `/health/live`, `/health/ready`, `/health/detailed`

### Testing

Tests use `pytest-asyncio` with `AsyncClient`. The `conftest.py` sets up:
- `test_settings`: Overrides config with test values
- `mock_repository`: AsyncMock of DMSRepository
- `app`: FastAPI app with dependency overrides
- `client`: httpx AsyncClient for making requests

To mock the repository, override `get_repository` from `app.dependencies`, not `get_dms_repository` from `app.repositories.dms`.

### OCR Dependencies

The OCR engine requires:
- `cnocr>=2.2.0` - Chinese OCR engine
- `onnxruntime>=1.16.0` - ONNX inference runtime
- `Pillow>=10.0.0` - Image processing
- `numpy>=1.24.0` - Array operations

Models are downloaded automatically on first use (~200MB for densenet_lite_136-gru + ch_PP-OCRv3_det).
