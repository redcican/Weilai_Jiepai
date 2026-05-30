# 信号灯检测模块集成变更说明

> 本文档面向 dms_api 集成方与现场运维，说明 `signal_detect.py` 接入后的接口变更。**只描述调用方式与返回格式，不写实现细节。**

---

## 一、变更概述

`dms_api` 的信号灯检测模块已从原有 Engine 的全图自动分析，切换为 `Light_signal/signal_detect.py` 的 **ROI 定点检测**。

核心变化：
- 新增 `camera_type` 参数，用于匹配预配置的 ROI 和信号中心点
- 返回值新增 `confidence`（置信度）和 `scores`（各颜色像素分数）
- 检测精度提升，消除灰墙面/夕阳反光等背景误报

---

## 二、API 接口

### 路由

```
POST /api/v1/signal-light/detect/batch
```

### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | `list[UploadFile]` | **是** |  surveillance camera 图片（JPEG/PNG/BMP） |
| `camera_type` | `str` | 否 | 摄像头标识，用于查 ROI 配置。可选值：`front_signal`、`rear_signal`、`exit_signal` |

**调用示例：**

```bash
# 指定 camera_type（推荐，精度最高）
curl -X POST "http://localhost:8000/api/v1/signal-light/detect/batch" \
  -F "files=@exit_01.jpg" \
  -F "camera_type=exit_signal"

# 不指定 camera_type（兼容旧调用方）
curl -X POST "http://localhost:8000/api/v1/signal-light/detect/batch" \
  -F "files=@exit_01.jpg"
```

> **建议**：现场固定摄像头务必传 `camera_type`，否则回退到全图检测，精度下降。

---

## 三、返回格式

### 响应结构

```json
{
  "success": true,
  "message": "识别完成, 1 张图片",
  "data": [
    {
      "filename": "exit_01.jpg",
      "color": "红色",
      "confidence": 0.995,
      "scores": {
        "red": 220,
        "blue": 0,
        "white": 1.2
      }
    }
  ],
  "request_id": "xxx",
  "timestamp": "2026-05-23T05:30:00Z"
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `filename` | `str` | 原始文件名 |
| `color` | `str` | 检测颜色：`红色` / `白色` / `蓝色` / `未知` |
| `confidence` | `float` | 置信度（0.0 ~ 1.0），新增字段 |
| `scores` | `dict` | 各颜色 blob 面积分数，新增字段 |

> **兼容性说明**：`confidence` 和 `scores` 为新增字段，旧客户端忽略即可，不影响解析。

---

## 四、配置文件

### 文件位置

```
dms_api/app/signal_light/signal_config.json
```

### 内容示例

```json
{
  "front_signal": {
    "roi": [710, 255, 840, 370]
  },
  "rear_signal": {
    "roi": [400, 200, 640, 440],
    "signal_center": [550, 280]
  },
  "exit_signal": {
    "roi": [230, 20, 330, 120],
    "signal_center": [266, 88]
  }
}
```

- `roi`：检测区域，坐标基于 **1280×720**，运行时按实际图片自动缩放
- `signal_center`：信号灯发光中心，用于蓝/白消歧校验
- 支持中文 key 和英文 alias

### 部署注意事项

- `signal_config.json` 必须和 `signal_detect.py` 放在**同一目录**（`dms_api/app/signal_light/`）
- 若现场更换摄像头或调整角度，需同步更新此文件中的 ROI 坐标

---

## 五、文件变更清单

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `dms_api/app/api/v1/signal_light.py` | 修改 | 新增 `camera_type` 可选参数 |
| `dms_api/app/services/signal_light.py` | 重写 | 去掉 Engine 调用，直接接入 `signal_detect` |
| `dms_api/app/schemas/signal_light.py` | 修改 | `SignalLightItem` 新增 `confidence` + `scores` |
| `dms_api/app/signal_light/signal_detect.py` | 新增（复制） | 来自 `Light_signal/signal_detect.py` |
| `dms_api/app/signal_light/signal_config.json` | 新增（复制） | 来自 `Light_signal/signal_config.json` |

---

## 六、现场调试

如需查看 ROI 框和 blob 标注图，可在 `Light_signal/` 目录下直接运行命令行工具：

```bash
cd Light_signal
python signal_detect.py ./exit_signal/ --debug
```

标注图会输出到 `./debug/` 目录，黄框为 ROI，红/蓝/白圈为对应颜色 blob。
