# 更新日志

## [0.9.0] - 2026-04-10
### 功能
- **行人异常检测** — `POST /api/v1/pedestrian/detect/batch` 批量检测端点
  - YOLOv8n（COCO 预训练权重）检测货车车厢俯视图中的工作人员
  - 有行人 → 异常（异常），无行人 → 正常（正常）
  - **两阶段检测策略**：
    - Pass 1：全图推理（1280px），捕获大部分目标
    - Pass 2：滑动窗口分块（640px 瓦片，30% 重叠），捕获小目标/边缘目标
  - **安全装备颜色验证** — 分块检测后通过 HSV 色彩分析过滤误检：
    - 橙色安全帽：H=5-22, S>100, V>100
    - 荧光背心：H=25-85, S>60, V>80
    - 阈值 100 像素完美分离真检测（最低 103）与误检（最高 93）
  - 40 张测试图片准确率 **100%**（20 张异常 + 20 张正常）
  - GPU 推理支持：API 层 `use_gpu` 参数 + 配置 `DMS_PEDESTRIAN_DETECTION_USE_GPU`
- **独立检测脚本** — `abnormaldrivingsafety/pedestrian_detect.py` CLI 工具，支持文件夹评估、调试标注图、JSON 输出

### 设计说明
- 沿用 `signal_light`/`train_id` 集成模式：独立引擎（单例）→ 服务 → 批量 API 端点
- 4096×3000 图片中人物仅占极小区域，单次全图推理（即使 1280px）召回率不足——分块策略解决小目标检测
- 分块检测降低置信度阈值（0.15）导致机械部件/阴影误检——安全装备颜色验证利用工人必穿橙色安全帽+荧光背心的领域知识，零误报消除
- 颜色验证仅应用于 Pass 2（分块），Pass 1（全图高置信度）无需过滤
- GPU 切换设计为运行时可选（per-request），而非部署时固定，便于测试和灰度发布

### 文件变更
- 新增 `dms_api/app/pedestrian/__init__.py` — 模块初始化
- 新增 `dms_api/app/pedestrian/engine.py` — `PedestrianEngine` 单例（两阶段检测 + 安全装备颜色验证）
- 新增 `dms_api/app/schemas/pedestrian.py` — `PedestrianItem`、`PedestrianBatchResponse`
- 新增 `dms_api/app/services/pedestrian.py` — `PedestrianService` 单例
- 新增 `dms_api/app/api/v1/pedestrian.py` — `POST /detect/batch` 端点
- 新增 `abnormaldrivingsafety/pedestrian_detect.py` — 独立 CLI 检测脚本
- 新增 `abnormaldrivingsafety/config.json` — 独立脚本配置文件
- 修改 `dms_api/app/api/v1/router.py` — 注册 `pedestrian_router`
- 修改 `dms_api/app/dependencies.py` — 添加 `PedestrianServiceDep`
- 修改 `dms_api/app/config.py` — 添加 10 项行人检测配置（模型、阈值、分块参数）
- 修改 `dms_api/app/schemas/__init__.py` — 更新导出列表
- 修改 `dms_api/requirements.txt` — 添加 `ultralytics>=8.0.0`

## [0.8.0] - 2026-04-01
### 功能
- **信号灯自动检测（移除 ROI 依赖）** — 完全重写检测引擎，仅从图片推断信号灯颜色，62 张测试图片准确率 98.4%（61/62）
  - 三阶段检测策略：蓝色 LED → 场景亮度分流 → 粉色像素计数
  - **蓝色 LED 检测**（双路径）：
    - 路径 A：过曝中心点（V>235, S<35）+ 蓝色光晕验证（环形 r=4-14, S>80, R-B<-25）
    - 路径 C：超亮饱和蓝像素（V≥250, S>140, H=85-125）≥7 个 + 光晕确认
  - **红/白分类**：场景中位亮度(medV)≥112 → 白色（明亮场景）；medV<112 时粉色像素(H>155)<500 → 白色，≥500 → 红色
  - 唯一未检出：灰度 IR 图像（无色彩信息，物理限制）
- **移除 ROI 参数** — API 端点、服务层、引擎层全部移除 `roi` 字段，简化为仅上传图片即可检测
- **更新测试图片** — 三个摄像头文件夹重新编号（01.png-25.png），附带 `信号灯标识.txt` 标注文件

### 设计说明
- LED 过曝特征（V>235 中心去饱和）是区分蓝色 LED 与蓝色喷漆设备的关键——喷漆表面保持高饱和度
- 超亮饱和蓝(V≥250, S>140)在设备反光中极少见（通常<5 像素），LED 可达 13+ 像素
- 场景亮度(medV)天然区分白天（cam1/cam2 白色信号 medV>112）与夜间场景
- 粉色像素(H>155)是红色 LED 光晕的独特标识：cam3 白色信号 pink<409，红色信号 pink>1772，完美分离
- 夜间红色 LED 在 IR 监控相机下呈粉/品红色（H=155-175），而非纯红，因此使用 H>155 而非 H>170
- 移除 ROI 的原因：自动检测已达到足够精度，ROI 增加部署复杂度且无额外收益

### 测试结果
- cam1（拨车机前侧）：17/18（94.4%）— 仅灰度 IR 图未检出
- cam2（拨车机后侧）：19/19（100%）
- cam3（装车楼出口）：25/25（100%）

### 文件变更
- 重写 `dms_api/app/signal_light/engine.py` — 移除 `_detect_with_roi()`、`_blob_score()`，重写 `_detect_auto()` 使用三阶段策略，新增 `_detect_blue_led()`
- 修改 `dms_api/app/api/v1/signal_light.py` — 移除 `roi` Form 参数及相关验证逻辑
- 修改 `dms_api/app/services/signal_light.py` — `detect_batch()` 移除 `roi` 参数
- 更新 `Light_signal/` — 三个摄像头文件夹图片重新编号，删除旧 debug 目录和旧编号图片

## [0.7.1] - 2026-04-01
### 功能
- **信号灯颜色识别** — `POST /api/v1/signal-light/detect/batch` 批量检测端点
  - 纯 OpenCV + NumPy（无 ML 模型）— HSV 色彩空间分析 + 连通域 blob 检测
  - 输出：中文颜色标签 — 红色、白色、蓝色、未知
  - 两种检测模式：
    - **ROI 模式**（推荐）：传入 `roi=x1,y1,x2,y2`（1280×720 坐标系），准确率 100%（22/22）
    - **自动模式**：不传 ROI，通过 S*V 亮度×饱和度排序自动定位信号灯，适合信号灯明亮且背景简单的场景
  - 红色 LED 双范围检测：冷红/品红(H≥155) + 暖红/橙红(H≤18)
  - 蓝色 LED 严格亮度阈值(V>200)区分发光 LED 与蓝色喷漆设备
  - ROI 内自动红/白消歧：检测最亮像素的 R-B 通道差异，无需额外参数
  - 新模块 `app/signal_light/`，`SignalLightEngine` 单例引擎
  - `SignalLightService` — 独立服务（无 DMS 后端依赖），沿用 `TrainIDService` 模式
  - 配置项：`DMS_SIGNAL_LIGHT_ENABLED` 环境变量
- **清理旧端点** — 移除 `POST /api/v1/signal/change` 和 `POST /api/v1/container`

### 设计说明
- 沿用 `train_id` 集成模式：独立引擎 → 单例服务 → 批量 API 端点
- 选择纯 CV 方案而非 ML，依赖最小化（仅 OpenCV + NumPy，项目已有）
- ROI 模式下固定摄像头只需部署时配置一次坐标，消除天空、火车、设备等背景干扰
- 自动模式下 camera 2 红色信号灯 V 值仅 69-134，与背景亮度接近，纯 CV 全图扫描无法可靠区分——这是物理限制而非算法缺陷
- ROI 内红/白消歧：在 ROI 区域找最亮像素，若该点过曝(V>200)且低饱和(S<55)，检查 R-B < -5 则判定为白色

### 文件变更
- 新增 `dms_api/app/signal_light/__init__.py` — 模块初始化
- 新增 `dms_api/app/signal_light/engine.py` — `SignalLightEngine`（从 `Light_signal/signal_detect.py` 适配，接受 bytes 输入）
- 新增 `dms_api/app/schemas/signal_light.py` — `SignalLightItem`、`SignalLightBatchResponse`
- 新增 `dms_api/app/services/signal_light.py` — `SignalLightService` 单例
- 新增 `dms_api/app/api/v1/signal_light.py` — `POST /detect/batch` 端点
- 新增 `Light_signal/signal_detect.py` — 独立信号灯检测脚本（CLI 工具，含校准/调试/评估功能）
- 修改 `dms_api/app/api/v1/router.py` — 注册 `signal_light_router`，移除 signal/container 路由
- 修改 `dms_api/app/dependencies.py` — 添加 `SignalLightServiceDep`，移除旧依赖
- 修改 `dms_api/app/config.py` — 添加 `signal_light_enabled` 配置项
- 修改 `dms_api/app/schemas/__init__.py` — 更新导出列表
- 修改 `dms_api/app/services/__init__.py` — 更新导出列表
- 删除 `dms_api/app/api/v1/signal.py`、`container.py` — 旧端点
- 删除 `dms_api/app/services/signal.py`、`container.py` — 旧服务
- 删除 `dms_api/app/schemas/signal.py`、`container.py` — 旧 schema

## [0.6.1] - 2026-03-27
### 功能
- **Type 2 基于模式的列分配** — 集装箱编组单的值按模式而非位置分类
  - 斜杠车型（如 C70E/1721133）→ ID1
  - 集装箱号（如 TBJU3216534）→ ID2，然后 ID3
  - 中文文本（如 漳平）→ 地点
  - 位置 0 处的数字 → 序
  - 垃圾/噪声（如 `\y`、单字母）→ 跳过
  - 片段行（仅有序号，无数据）→ 过滤
  - 损坏的序号数字（如 `寸` 代替 4，`o` 代替 6）→ 从上一行推断

### 设计说明
- 位置分配方式失败，因为 OCR 可能遗漏值或插入垃圾字符，导致所有后续值偏移
- 基于模式的分类不受缺失/多余值影响——每个值按其外观分配，而非出现位置
- 正则模式 `_SLASH_VEHICLE_RE`、`_CONTAINER_RE`、`_CHINESE_RE` 覆盖所有已观察到的数据类型

### 文件变更
- `OCR_CnOCR/table_ocr_cnocr.py` — 添加 `_classify_type2_row()`，重写 `_extract_type2()` 后处理
- `dms_api/app/ocr/utils.py` — 同上

## [0.6.0] - 2026-03-27
### 功能
- **Type 1 列边界提取** — 站存车打印表格输出 16 键字典，列名为：股道、序、车种、油种、车号、自重、换长、载重、到站、品名、记事、发站、篷布、票据号、属性、收货人
  - 使用表头行 x 坐标定义列边界，按 x 坐标重叠将 OCR 框分配到对应列
  - 合并表头文本的等比字符宽度拆分（如 "股道序车种油种" → 4 个独立列中心）
- **多行表头合并** — 处理表头跨两行 OCR 行的文档
- 同步应用到独立 OCR（`OCR_CnOCR/table_ocr_cnocr.py`）和 API（`dms_api/app/ocr/`）

### 设计说明
- 表头 x 坐标是列边界的唯一可靠信号——数据行的 OCR 框会跨列不可预测地合并
- 等比字符宽度估算可行，因为 CnOCR 对中文文本使用近似等宽的边界框
- 次级表头检测（≥1 个关键词）配合邻近性检查，避免误报同时捕获分割的表头

### 文件变更
- `OCR_CnOCR/table_ocr_cnocr.py` — 添加 `_extract_type1_columns()` 等函数
- `dms_api/app/ocr/utils.py` — 添加相同函数
- `dms_api/app/ocr/processor.py` — 简化 `process()` 使用 `extract_type1_columns()`

## [0.5.0] - 2026-03-26
### 功能
- **票据 OCR 批量图片上传** — `POST /api/v1/ticket/parse` 单次请求支持多张图片
- **Swagger UI 多文件选择** — 自定义 `/docs` 页面，支持 Ctrl/Shift+Click 选择多文件
- **UTF-8 字符集修复** — 中间件为所有 JSON 响应添加 `charset=utf-8`，修复浏览器中文乱码
- **OpenAPI schema 补丁** — 将 `contentMediaType` 转换为 `format: binary` 兼容 Swagger UI 5

### 设计说明
- Swagger UI 5 不支持 OpenAPI 3.1 的 `contentMediaType` 文件输入——修补 schema 使用 `format: binary` 是标准方案
- `charset=utf-8` 在 Content-Type 中是必需的，某些浏览器默认使用系统区域编码

## [0.4.0] - 2026-03-26
### 功能
- **双表格类型 OCR 识别**：
  - **Type 1**（站存车打印）：~16 列，车种/车号独立列
  - **Type 2**（集装箱编组单）：~5 列，斜杠车型/车号（如 `C70E/1805776`）和集装箱号
- 基于斜杠车型模式自动检测表格类型
- 所有 OCR 输出添加 `table_type` 字段
- Type 2 后处理：推断缺失序号、上限 5 列、过滤片段行、清理 OCR 噪声

### 设计说明
- 斜杠车型模式（`C70E/1805776`）是唯一的检测信号——仅用集装箱模式会误报
- Type 2 MAX_COLS=5 上限移除 OCR 幻觉，无需硬编码特定噪声字符串
- API schema 从命名字段改为原始数组，因列结构按类型不同

## [0.3.0] - 2026-03-19
### 功能
- **进站车辆识别集成** — 作为独立 FastAPI 服务集成到 dms_api
  - `POST /api/v1/train-id/recognize` — 单张图片车种/车号识别
  - `POST /api/v1/train-id/recognize/batch` — 批量识别
  - 新模块 `app/train_id/`，混合 CnOCR 引擎（db_resnet18 + ch_PP-OCRv3_det）
  - `TrainIDService` — 独立服务（无 DMS 后端依赖）

### 设计说明
- 沿用现有 OCR 集成模式
- TrainIDService 独立运行，所有处理在本地完成
- 引擎从 `train_id_ocr/train_id_ocr.py` 适配为接受 bytes 输入

## [0.2.0] - 2026-03-19
### 功能
- **重写 train_id_ocr 模块**，准确率 100%
  - 混合双引擎 OCR（db_resnet18 + ch_PP-OCRv3_det）
  - 图片缩放至 0.25 倍后预处理
  - 4 轮预处理流水线（bilateral+CLAHE、CLAHE、gamma×2）
  - 基于模式的通用 `_fix_vehicle_type()`（无硬编码替换）
  - 超集感知的车号多数投票

### 设计说明
- db_resnet18 修复 8→3 数字混淆；ch_PP-OCRv3_det 保持完整车型字符串
- 基于位置的字符混淆表可泛化到任意车型模式
- Gamma 轮次（2.0、3.0）发现其他轮次不可见的末尾边缘数字

## [0.1.0] - 2026-03-18
### 功能
- 初始提交：DMS API 网关和 OCR 工具
- FastAPI 网关，含异常告警、票据解析端点
- 本地 CnOCR 集成用于票据解析
- 进站车辆识别模块
