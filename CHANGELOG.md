# Changelog

## [0.6.1] - 2026-03-27
### Features
- **Type 2 pattern-based column assignment** — 集装箱编组单 values are now classified by pattern instead of position
  - Slash-vehicle (e.g., C70E/1721133) → ID1
  - Container number (e.g., TBJU3216534) → ID2, then ID3
  - Chinese text (e.g., 漳平) → 地点
  - Digits at position 0 → 序
  - Garbage/noise (e.g., `\y`, single letters) → skipped
  - Fragment rows (seq only, no data) → filtered out
  - Corrupted seq digits (e.g., `寸` for 4, `o` for 6) → inferred from previous row

### Design Rationale
- Positional assignment failed because OCR can miss values or insert garbage, shifting everything: `漳平` at position 3 → ID3 instead of 地点; `\y` garbage at position 0 → all subsequent values shift right
- Pattern-based classification is invariant to missing/extra values — each value is assigned by what it looks like, not where it appears
- Regex patterns `_SLASH_VEHICLE_RE`, `_CONTAINER_RE`, `_CHINESE_RE` handle all observed data types

### Files Changed
- `OCR_CnOCR/table_ocr_cnocr.py` — added `_classify_type2_row()`, rewrote `_extract_type2()` post-processing
- `dms_api/app/ocr/utils.py` — same changes (added `_classify_type2_row()`, rewrote `extract_type2()`)

## [0.6.0] - 2026-03-27
### Features
- **Type 1 column-boundary extraction** — 站存车打印 tables now output 16-key dicts with proper column names instead of raw arrays
  - Columns: 股道, 序, 车种, 油种, 车号, 自重, 换长, 载重, 到站, 品名, 记事, 发站, 篷布, 票据号, 属性, 收货人
  - Uses header row x-positions to define column boundaries, then assigns each OCR box to its column by x-coordinate overlap
  - Proportional character-width splitting for merged header text (e.g. "股道序车种油种" → 4 separate column centers)
- **Multi-row header merging** — handles documents where the header spans two OCR rows (e.g. left-side columns on a separate line)
- Changes applied to both standalone OCR (`OCR_CnOCR/table_ocr_cnocr.py`) and API (`dms_api/app/ocr/`)

### Design Rationale
- Header x-positions are the only reliable signal for column boundaries — OCR boxes from data rows merge unpredictably across columns
- Proportional character-width estimation works because CnOCR uses monospace-like bounding boxes for Chinese text
- Secondary header detection (≥1 keyword) with adjacency check avoids false positives while catching split headers

### Files Changed
- `OCR_CnOCR/table_ocr_cnocr.py` — added `_extract_type1_columns()`, `aggregate_to_box_rows()`, `_parse_column_centers()`, `_build_column_boundaries()`, `_box_row_to_dict()`, `_is_secondary_header_row()`; `extract_table_data()` Type 1 branch now delegates to `_extract_type1_columns()`
- `dms_api/app/ocr/utils.py` — added same functions (`extract_type1_columns()`, `aggregate_to_box_rows()`, column boundary helpers)
- `dms_api/app/ocr/processor.py` — simplified `process()` to use `extract_type1_columns()` for Type 1, removed old `_extract_type1()` method

## [0.5.0] - 2026-03-26
### Features
- **Batch image upload** for ticket OCR — `POST /api/v1/ticket/parse` now accepts multiple images in a single request
  - Response wraps all results in one `data` array with `filename` per item
  - Single `message`/`status` at top level (e.g. "识别成功, 2/2 张图片")
- **Swagger UI multi-file select** — custom `/docs` page with JS patch so users can select multiple files in one dialog (Ctrl/Shift+Click) instead of adding items one by one
- **UTF-8 charset fix** — middleware adds `charset=utf-8` to all JSON responses, fixing garbled Chinese in browsers
- **OpenAPI schema patch** — converts `contentMediaType` to `format: binary` for Swagger UI 5 file upload compatibility

### Design Rationale
- Swagger UI 5 doesn't support OpenAPI 3.1's `contentMediaType` for file inputs — patching the schema to use `format: binary` is the standard workaround
- Swagger UI only reads one file per `<input>` even with `multiple` attribute — a fetch interceptor rebuilds FormData with all selected files before the request is sent
- `charset=utf-8` in Content-Type is needed because some browsers default to system locale encoding for `application/json` without explicit charset

### Notes & Caveats
- Custom `/docs` page replaces FastAPI's built-in Swagger UI (only in dev/debug mode)
- The fetch interceptor is a Swagger UI workaround — programmatic clients (curl, httpx) send multiple files natively

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
