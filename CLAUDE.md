# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Railway Data Management System (DMS) — a suite of tools for extracting structured data from railway/logistics document images via OCR and a FastAPI gateway that proxies operations to an upstream DMS backend.

The project is written in Chinese-context railway domain. Key domain terms: 股道 (track), 车种 (train type), 车号 (train number), 集装箱 (container), 编组单 (grouping order/ticket), 信号灯 (signal light).

## Repository Layout

Three independent modules at the root, each with its own dependencies:

| Module | Purpose |
|--------|---------|
| `OCR/` | Standalone table OCR extractor using PaddleOCR (multi-engine: paddle, tesseract, easyocr) |
| `OCR_CnOCR/` | CnOCR-based table extractor (ONNX backend, lighter weight, optimized for Chinese) |
| `dms_api/` | FastAPI gateway — REST API with local OCR + upstream DMS backend proxy |
| `API_test/` | DMS backend API documentation (DMS_API.docx) |

## Build & Run Commands

### dms_api (FastAPI Gateway)

```bash
cd dms_api

# Install
pip install -r requirements.txt

# Dev server
uvicorn app.main:app --reload

# Tests
pytest
pytest tests/test_api/test_endpoints.py -v
pytest --cov=app --cov-report=html

# Docker
docker-compose up -d dms-api              # production
docker-compose --profile dev up -d dms-api-dev  # development (port 8001)
```

### OCR_CnOCR (Standalone CnOCR)

```bash
cd OCR_CnOCR
pip install -r requirements.txt

# Process a folder of images
python table_ocr_cnocr.py ../OCR/fig -o ./output_cnocr

# With verbose output
python table_ocr_cnocr.py ../OCR/fig -r -v -o ./output_cnocr
```

### OCR (PaddleOCR multi-engine)

```bash
cd OCR
pip install paddlepaddle paddleocr

# Single image
python table_ocr_solution.py image.jpg -o output.json

# Folder batch
python table_ocr_solution.py ./images/ -o ./output/ --combined merged.json
```

## Architecture: dms_api

```
Request → Middleware → API Router → Service → Repository (httpx) → DMS Backend
                                      ↓
                                  OCR Module (local CnOCR, for ticket parsing)
```

### Layers

- **API** (`app/api/v1/`): FastAPI routers — form-data parsing, file uploads. Each domain has its own router file (abnormal, container, signal, ticket).
- **Services** (`app/services/`): Business logic. Each service takes a `DMSRepository` via constructor injection.
- **Repository** (`app/repositories/dms.py`): Single `DMSRepository` class wrapping httpx `AsyncClient`. Implements retry with exponential backoff and a circuit breaker. This is the only place that talks to the upstream DMS backend.
- **OCR** (`app/ocr/`): `CnOCREngine` (singleton, ONNX backend) → `TableOCRProcessor` → `OCRService`. Used by `TicketService` for local ticket parsing; falls back to DMS backend on failure.
- **Schemas** (`app/schemas/`): Pydantic v2 models. Responses extend `ResponseSchema[T]` from `base.py`. Use `serialize_by_alias=True` for camelCase output.
- **Config** (`app/config.py`): Pydantic Settings with `DMS_` env prefix. All config via environment variables.

### Dependency Injection

All wiring is in `app/dependencies.py`. Services get `DMSRepository` via `get_repository`. In tests, override `get_repository` (not `get_dms_repository`) to inject mocks.

### Key Config Env Vars

All prefixed with `DMS_`. Critical ones: `DMS_DMS_BASE_URL` (upstream backend), `DMS_OCR_ENABLED`, `DMS_OCR_FALLBACK_TO_DMS`, `DMS_CIRCUIT_BREAKER_ENABLED`.

### Testing

- pytest with `asyncio_mode = auto` (see `pytest.ini`)
- `conftest.py` provides: `test_settings`, `mock_repository` (AsyncMock), `app` (with dependency overrides), `client` (httpx AsyncClient)
- Run from `dms_api/` directory: `pytest`

## Architecture: OCR Modules

Both `OCR/table_ocr_solution.py` and `OCR_CnOCR/table_ocr_cnocr.py` follow the same pattern:
1. Load image → enhance (contrast/sharpness) → OCR recognition → group text boxes into rows by Y-coordinate (tolerance-based clustering) → extract structured table data
2. Output JSON with `status`, `message`, `metadata`, and `table data` array
3. Only extract rows starting with sequence numbers; skip headers and malformed rows

The OCR module inside `dms_api/app/ocr/` reuses the CnOCR approach but is adapted for in-process use (accepts bytes, singleton engine).

## Conda Environment

This project uses the `ml` conda environment:

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate ml
# Or directly: /home/cican/miniconda3/envs/ml/bin/python
```
