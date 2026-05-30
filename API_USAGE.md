# 识别接口调用说明

## 概述

识别服务支持两种输入格式：

1. **JPEG/PNG 图片** —— 浏览器/客户端上传的标准图像文件
2. **原始像素数据** —— 工业相机（海康 GigE Vision）返回的 Bayer/Mono 裸像素流

调用方通过传入不同参数，自动选择解码方式：
- 只传 `image_bytes` → 按 JPEG/PNG 解码
- 同时传 `pixel_type` + `width` + `height` → 按原始像素解码

**GPU/CPU 自动适配：** 服务启动时自动检测 CUDA。现场有 RTX 3060 等显卡时自动启用 GPU（OCR ~50ms/帧）；无显卡时自动回退 CPU，并启用 4 进程并发工作池（OCR ~200ms/帧 × 4 并发）。

**多摄像头并发（4 路）：** 单图接口新增 `cam_id` 参数。传入后启用智能帧过滤——同一摄像头 150ms 内的重复帧自动跳过并返回缓存结果，减少无效 OCR 调用。

两个接口（车厢号、板车）的调用方式完全一致。

---

## 车厢号识别

### 接口

```python
async def recognize_image(
    image_bytes: bytes,
    filename: str = "unknown",
    pixel_type: int | None = None,
    width: int | None = None,
    height: int | None = None,
    cam_id: str | None = None,
) -> TrainIDData
```

### 参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_bytes` | `bytes` | 是 | 图像字节流 |
| `filename` | `str` | 否 | 文件名（仅用于日志） |
| `pixel_type` | `int` | 否 | 像素格式代码，传了则按原始像素解码 |
| `width` | `int` | 否 | 图像宽度（像素），传了则按原始像素解码 |
| `height` | `int` | 否 | 图像高度（像素），传了则按原始像素解码 |
| `cam_id` | `str` | 否 | **摄像头标识**（如 `cam1`、`cam2`），用于智能帧过滤降采样 |

> **规则**：`pixel_type`、`width`、`height` 三个参数**同时传**才走原始像素解码；**任何一个不传**都走 JPEG/PNG 解码。

> **`cam_id` 作用**：多摄像头场景下，同一摄像头 150ms 内的重复帧会被跳过并返回缓存结果。建议 4 路摄像头分别传入固定标识（如 `cam1`~`cam4`）。

### 返回值 `TrainIDData`

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | 空挡标记，`"########"` 表示空挡帧，`""` 表示正常帧 |
| `vehicleType` | `str` | 车种（如 `C70E`、`X70`） |
| `vehicleNumber` | `str` | 车号（如 `49 31846`） |
| `confidence` | `float` | 平均置信度（0.0 ~ 1.0） |
| `container` | `str` | 集装箱箱号（如 `TBJU881313`），无则为空 |
| `containerConfidence` | `float` | 集装箱识别置信度 |

### 调用示例

**HTTP API（JPEG/PNG）：**
```bash
curl -X POST "http://localhost:8000/api/v1/train-id/recognize" \
  -F "image=@frame_0001.jpg"
```

**HTTP API（多摄像头，带 cam_id）：**
```bash
# cam1 ~ cam4 分别传入固定标识，启用帧过滤降采样
curl -X POST "http://localhost:8000/api/v1/train-id/recognize" \
  -F "image=@frame_0001.jpg" \
  -F "cam_id=cam1"
```

**服务层内部调用（JPEG/PNG）：**
```python
image_bytes = await image.read()  # UploadFile 读取的 JPEG/PNG
result = await service.recognize_image(image_bytes, image.filename)
```

**服务层内部调用（原始 Bayer 像素）：**
```python
image_bytes = bytes(user_buffer)          # 从相机 SDK 复制的原始像素流
pixel_type = stFrameInfo.enPixelType      # 如 0x0108000A (BayerGB8)
width = stFrameInfo.nWidth                # 如 2448
height = stFrameInfo.nHeight              # 如 2048

result = await service.recognize_image(
    image_bytes, filename, pixel_type, width, height, cam_id="cam1"
)
```

---

## 板车识别

### 接口

```python
async def recognize_flatcar_image(
    image_bytes: bytes,
    filename: str = "unknown",
    pixel_type: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> FlatcarData
```

### 参数

与 `recognize_image` 基本一致，`cam_id` 暂不支持（板车处理器未接入帧过滤）。

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `image_bytes` | `bytes` | 是 | 图像字节流 |
| `filename` | `str` | 否 | 文件名（仅用于日志） |
| `pixel_type` | `int` | 否 | 像素格式代码 |
| `width` | `int` | 否 | 图像宽度 |
| `height` | `int` | 否 | 图像高度 |

### 返回值 `FlatcarData`

| 字段 | 类型 | 说明 |
|------|------|------|
| `type` | `str` | 空挡标记，`"########"` 表示空挡帧，`""` 表示正常帧 |
| `vehicleType` | `str` | 车型（如 `X70`、`C70E`） |
| `vehicleNumber` | `str` | 车号（如 `5240903`） |
| `confidence` | `float` | 平均置信度（0.0 ~ 1.0） |

### 调用示例

**HTTP API（JPEG/PNG）：**
```python
image_bytes = await image.read()
result = await service.recognize_flatcar_image(image_bytes, image.filename)
```

**队列消费端（原始 Bayer 像素）：**
```python
result = await service.recognize_flatcar_image(
    image_bytes, filename, pixel_type, width, height
)
```

---

## 批量识别

### 接口

```python
async def recognize_batch(
    images: list[tuple[bytes, str]],
    cam_id: str | None = None,
) -> list[TrainIDBatchItem]
```

### 说明

内部使用 `asyncio.gather()` **并行处理**多张图片，不是串行循环。批量请求的总耗时 ≈ 单张耗时（而非单张 × 数量）。

### 调用示例

**HTTP API（批量上传，同一摄像头）：**
```bash
curl -X POST "http://localhost:8000/api/v1/train-id/recognize/batch" \
  -F "images=@frame_0001.jpg" \
  -F "images=@frame_0002.jpg" \
  -F "images=@frame_0003.jpg" \
  -F "cam_id=cam1"
```

**服务层内部调用：**
```python
images = [(bytes1, "1.jpg"), (bytes2, "2.jpg"), (bytes3, "3.jpg")]
results = await service.recognize_batch(images, cam_id="cam1")
```

---

## 空挡标记说明

车厢号和板车的返回值中都包含 `type` 字段：

| 场景 | `type` 值 | 含义 |
|------|----------|------|
| 正常帧（车厢本体） | `""` | 可以正常识别车种/车号 |
| 空挡帧（车厢连接处） | `"########"` | 两节车厢之间的空隙，OCR 结果可能为空 |

调用方可根据 `type == "########"` 判断当前是否为空挡，用于时序分析或跳过无效帧。

---

## 多摄像头并发架构

现场 4 路摄像头同时推图时，服务内部架构：

```
FastAPI (单进程 uvicorn)
    │
    ├── 请求1 (cam1) ──→ run_in_executor ──→ OCR Worker Pool
    ├── 请求2 (cam2) ──→ run_in_executor ──→ OCR Worker Pool
    ├── 请求3 (cam3) ──→ run_in_executor ──→ OCR Worker Pool
    └── 请求4 (cam4) ──→ run_in_executor ──→ OCR Worker Pool
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    ▼                           ▼                           ▼
            ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
            │ PaddleOCR     │           │ PaddleOCR     │           │ PaddleOCR     │
            │ (GPU/CPU)     │           │ (GPU/CPU)     │           │ (GPU/CPU)     │
            └───────────────┘           └───────────────┘           └───────────────┘
```

- **有 GPU（如 3060）**：单进程 GPU 模式，OCR ~50ms，总帧耗时 ~300ms，基本满足 3fps
- **无 GPU**：4 进程 CPU 并发，OCR ~200ms × 4 并发，总处理能力 ~12-20fps
- **帧过滤**：传入 `cam_id` 后，同一摄像头 150ms 内重复帧自动跳过（不消耗 OCR 资源）

---

## 已支持的像素格式

原始像素解码当前支持以下格式：

| 格式 | 说明 |
|------|------|
| `BayerGB8` | 单通道 8bit Bayer（海康相机常用） |
| `Mono8` | 单通道 8bit 灰度 |
| `RGB8` | 三通道 8bit RGB |
| `BGR8` | 三通道 8bit BGR |

具体 `pixel_type` 值由相机 SDK 的 `enPixelType` 字段提供，调用方直接透传即可。
