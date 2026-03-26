# Changelog

## [0.4.0] - 2026-03-26
### Features
- **Two table type support** for OCR recognition:
  - **Type 1** (站存车打印): ~16 columns, vehicle type/number in separate columns. Track number output as `{"股道": "4"}` first element.
  - **Type 2** (集装箱编组单): ~5 columns with slash vehicle/number (e.g. `C70E/1805776`) and container numbers (e.g. `TBJU3216534`).
- Auto-detection based on slash-vehicle patterns (unique to Type 2)
- `table_type` field added to all OCR output (both standalone and API)
- Type 2 post-processing: infer missing sequence numbers, cap at 5 columns, filter fragment rows, clean leading/trailing OCR noise
- API output changed to raw table data arrays — no column-name mapping, `tableType` field added

### Design Rationale
- Slash-vehicle pattern (`C70E/1805776`) is the sole detection signal — container patterns alone cause false positives (Type 1 has cargo reference codes like `JHSX4535071` that resemble container numbers)
- Type 2 MAX_COLS=5 cap removes OCR hallucinations (e.g. "车海发公司") without hardcoding specific noise strings
- Missing sequence number inference uses position tracking (next_seq counter), with MIN_INFERRED=4 to filter fragment rows
- API schema changed from `TicketData` (named fields) to `OCRTableData` (raw arrays) because column structure differs by type. Old `TicketData` kept for DMS backend compatibility

### Files Changed
- `OCR_CnOCR/table_ocr_cnocr.py` — standalone: added `detect_table_type()`, `_extract_type2()`, modified `extract_table_data()` to branch by type
- `dms_api/app/ocr/utils.py` — added `detect_table_type()`, `is_page_footer()`, `extract_type2()`
- `dms_api/app/ocr/processor.py` — rewrote `process()` with type branching, removed column mapping methods
- `dms_api/app/ocr/models.py` — added `table_type` to `OCRResult`
- `dms_api/app/schemas/ticket.py` — new `OCRTableData` schema, `TicketParseResponse` wraps it
- `dms_api/app/services/ocr.py` — simplified to return `OCRResult` directly
- `dms_api/app/services/ticket.py` — uses `OCRTableData`, both local and DMS paths

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
