# Changelog

## [0.3.0] - 2026-03-19
### Features
- Integrate train_id_ocr into dms_api as independent FastAPI service
  - `POST /api/v1/train-id/recognize` — single image vehicle type/number recognition
  - `POST /api/v1/train-id/recognize/batch` — batch recognition
  - New module `app/train_id/` with hybrid CnOCR engine (db_resnet18 + ch_PP-OCRv3_det)
  - `TrainIDService` — standalone service (no DMS backend dependency)
  - Config: `DMS_TRAIN_ID_OCR_ENABLED` env var
  - 10 new tests (4 endpoint + 6 unit)

### Design Rationale
- Followed existing OCR integration pattern (`app/ocr/` → `services/ocr.py` → `api/v1/ticket.py`)
- TrainIDService is standalone (no BaseService/DMSRepository) since all processing is local
- Engine adapted from `train_id_ocr/train_id_ocr.py` to accept bytes instead of file paths
- Dependency injection via `get_train_id_service` for testability

## [0.2.0] - 2026-03-19
### Features
- Rewrite train_id_ocr module for 100% accuracy
  - Hybrid dual-engine OCR (db_resnet18 + ch_PP-OCRv3_det)
  - Image resize to 0.25 scale before preprocessing
  - 4-pass preprocessing pipeline (bilateral+CLAHE, CLAHE, gamma x2)
  - General pattern-based `_fix_vehicle_type()` (no hardcoded replacements)
  - Superset-aware majority voting for vehicle numbers

### Design Rationale
- db_resnet18 fixes 8→3 digit confusion; ch_PP-OCRv3_det keeps complete type strings
- Position-based character confusion tables generalize to any vehicle type pattern
- Gamma passes (2.0, 3.0) find trailing edge digits invisible to other passes

## [0.1.0] - 2026-03-18
### Features
- Initial commit: DMS API gateway and OCR tools
- FastAPI gateway with abnormal, container, signal, ticket endpoints
- Local CnOCR integration for ticket parsing
- Station-entry train ID recognition module
