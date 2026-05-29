# 更新日志

## [0.10.6] - 2026-05-29

### 功能
- **89 种标准铁路货车车型白名单** — `train_id_ocr_paddle.py` 新增 `_CHINA_RAIL_VEHICLE_TYPES`
  - 覆盖 C 系列（C50/C70E/C80 等）、P 系列、G 系列、X 系列、K 系列等 89 种标准车型
  - 误识从 18 段非标准字符串降至 0 段
  - 新增 `_is_vehicle_type_pattern()` 进行白名单精确匹配 + 近似匹配
- **车型近似匹配** — 单字符删除修复，如 `C701E` → `C70E`
  - 候选在白名单中时，允许删除一个字符修复常见漏框/误识
- **空挡检测 v3 优化** — `flatcar_gap_detector.py` 增加车钩反光辅助特征
  - 中央 45%-55% 窄带亮度比 > 1.55 且高亮像素 1.5%-8% → coupler_score=1.0
  - `min_gap_score` 从 6 提至 7，误检从 172 帧(21.9%)降至 19 帧(2.4%)
- **板车空挡模式** — `FlatcarGapDetector` 支持 `vehicle_type='flatcar'`
  - 亮度/标准差/边缘阈值针对板车暗光场景重新校准
  - 板车及格线 5 分 vs 标准模式 7 分
- **GPU 自动检测** — `get_ocr_processor()` 默认自动检测 CUDA 可用性
  - 内联 `_check_gpu_available()`，有 CUDA 则自动启用 GPU，无则回退 CPU
  - CLI 新增 `--gpu`/`--cpu` 强制开关（互斥），默认不传时自动检测
  - 解决现场 3060 有 GPU 但代码硬编码 CPU 导致的性能瓶颈
- **全局单例缓存** — `get_ocr_processor()` 避免每次 API 调用重复初始化 PaddleOCR
  - 首次初始化 ~0.8s，后续直接返回缓存实例
  - 支持多配置共存（CPU/GPU × 增强模式），用配置字符串作为 key

### 修复
- **空挡检测硬编码** — `_process_img()` 取消 `is_gap=False` 硬编码，真正调用 `FlatcarGapDetector`
  - 之前代码写了检测逻辑但从未执行，空挡帧仍走完整 OCR 浪费性能
- **`is_train_param()` 误杀车型** — 移除宽泛的 `'t'` 关键词匹配
  - 改为 `\d+t` 精确匹配（如 `70t`），避免 `CTOE` 等车型含 `T` 被误过滤为参数
- **车型纠错增强** — `_fix_vehicle_type()` 新增 T→7、O→0 直接映射
  - `CTOE` → `C70E`，命中白名单即返回，无需逐字符修复链
- **标准车型补全** — 白名单新增 `C70EH`、`C70BH`

### 重构
- **`get_ocr_processor()` 签名** — `use_gpu: bool = False` → `use_gpu: Optional[bool] = None`
  - `None` 表示自动检测，`True`/`False` 表示强制指定

### 代码清理
- 删除 `train_id_ocr/gap_detector_v2.py`（未使用的旧版空挡检测器）
- 移除 `train_id_ocr_paddle.py` 中未使用的 `Counter` 导入
- 更新 `.gitignore`

### 文件变更
- 修改 `train_id_ocr/train_id_ocr_paddle.py` — 白名单、近似匹配、车型纠错、空挡检测启用、GPU 自动检测、全局单例
- 修改 `train_id_ocr/flatcar_gap_detector.py` — 车钩反光特征、板车空挡模式、阈值优化
- 删除 `train_id_ocr/gap_detector_v2.py`

## [0.10.5] - 2026-05-23

### 修复
- **信号灯检测阈值校准** — `Light_signal/signal_detect.py` 全面收紧 HSV 阈值，消除灰墙面/反光假阳性
  - `white_mask`: `(sat < 100, val > 220)` → `(sat < 50, val > 230)`，消除无灯场景误判为 white
  - `warm_red`: `sat > 70` → `sat > 80`，`val > 90` → `val > 150`，减少夕阳/车厢反光误报
  - `cool_red`: `sat > 40` → `sat > 80`，`val > 60` → `val > 150`，避免白灯 Hue≈173 被误判为 red
  - red 触发阈值: `≥ 10` → `≥ 25`，提高红色判定门槛
  - `_find_blobs` max_area: `2000` → `30000`，保留大体积蓝色 LED blob（约 27,000 像素）
- **Blue → White BGR 消歧** — 当 HSV 将白灯误判为 blue 时（Hue≈90-110 重叠区），检查 `signal_center` 附近 BGR 值
  - 若 `peak_v > 200` 且 `abs(R-B) < 30`（白色 LED 近似中性灰），回退为 white
  - 解决夜间/白天白灯被 blue mask 截获的问题
- **Red → White BGR 消歧** — 当 HSV 将白灯误判为 red 时，检查 `signal_center` 附近
  - 若 `peak_v > 200`、`mean_s < 50` 且 `(R-B) < -5`，回退为 white
- **CONFIG_FILE 路径修复** — 从相对当前工作目录改为 `Path(__file__).parent / "signal_config.json"`，避免 CWD 不同时找不到配置
- **UTF-8 编码显式声明** — `load_config()` / `save_config()` 添加 `encoding="utf-8"`，修复中文配置键读取异常
- **DEFAULT_CONFIG 同步** — 将 `exit_signal` 和 `rear_signal` 的 ROI / `signal_center` 与 `signal_config.json` 保持一致
  - `exit_signal`: ROI `[520, 330, 630, 400]` → `[230, 20, 330, 120]`，`signal_center` `[573, 370]` → `[266, 88]`
  - `rear_signal`: ROI `[468, 228, 628, 388]` → `[400, 200, 640, 440]`，`signal_center` `[548, 308]` → `[550, 280]`

### 文件变更
- 修改 `Light_signal/signal_detect.py`
- 新增 `Light_signal/signal_config.json`（未纳入 Git 跟踪）

## [0.10.4] - 2026-05-21

### 功能
- **工业相机原始像素解码** — 新增 `decode_raw_image()` 支持海康 GigE Vision 原始 Bayer/Mono 数据
  - `PixelType_Gvsp_BayerGB8`（单通道 8bit）→ OpenCV BGR
  - `PixelType_Gvsp_Mono8` → `COLOR_GRAY2BGR`
  - `PixelType_Gvsp_RGB8/BGR8` → BGR
  - `RESOLUTION_BAYER_MAP` 根据分辨率自动选择正确 Bayer 模式：
    - 2448×2048 → `COLOR_BayerRG2BGR`（115/117 摄像头）
    - 4096×3000 → `COLOR_BayerGR2BGR`（116/118 摄像头）
  - 解决海康 `BayerGB8` 与 OpenCV Bayer 常量命名不匹配问题（经 4 组实际数据验证）
- **车厢号识别支持原始像素** — `PaddleOCRProcessor` 新增 `process_raw_bytes(image_bytes, pixel_type, width, height)`
  - 与 `process_bytes()` 共用同一套 `_process_img()` OCR 逻辑
  - 输出格式不变（`ImageResult` 结构完全一致）
- **板车识别支持原始像素** — `FlatcarBottomProcessor` / `FlatcarImageProcessor` 新增 `process_raw_bytes()`
  - 同样共用 `_process_img()` 底部区域 OCR 逻辑
  - 输出格式不变
- **板车空挡检测集成** — `FlatcarBottomProcessor` 初始化时加载 `FlatcarGapDetector`
  - 空挡帧输出 `type: "########"`，正常帧输出 `type: ""`
  - 与车厢号空挡检测输出格式统一
  - 传统 CV 方案：中心 ROI + 亮度/边缘/方差/列极值差 8 特征评分
  - 阈值经 1227 帧实际数据校准，暗光空挡不漏检，金属结构不误判
- **Service 层新增 raw 入口** — `TrainIDService` 新增 `recognize_raw_image()` 供队列消费端调用
  - 提取 `_build_train_id_data()` 公共方法，避免 `recognize_image` / `recognize_raw_image` 重复代码

### 文件变更
- 修改 `dms_api/app/train_id/utils.py` — 新增 `decode_raw_image()`、`RESOLUTION_BAYER_MAP`、海康像素格式常量
- 修改 `dms_api/app/train_id/flatcar_image_processor.py` — 新增 `process_raw_bytes()`
- 修改 `train_id_ocr/run_bottom_merge_ocr.py` — 集成 `FlatcarGapDetector`，新增 `process_raw_bytes()` + `_process_img()`
- 修改 `train_id_ocr/train_id_ocr_paddle.py` — 新增 `process_raw_bytes()` + `_process_img()`
- 修改 `dms_api/app/services/train_id.py` — 新增 `recognize_raw_image()` + `_build_train_id_data()`，`recognize_flatcar_image` 读取 `type`
- 修改 `dms_api/app/schemas/train_id.py` — `FlatcarData` 新增 `type` 字段
- 新增 `train_id_ocr/flatcar_gap_detector.py` — `FlatcarGapDetector` 板车空挡检测器（生产级传统 CV）

## [0.10.3] - 2026-05-18

### 功能
- **API 返回集装箱箱号** — `/recognize` 和 `/recognize/batch` 响应新增 `container` 和 `containerConfidence` 字段
  - 从 `PaddleOCRProcessor` 上半区识别结果中取置信度最高的集装箱箱号
  - 无集装箱时返回空字符串 + 0.0 置信度
  - 单图接口和批量接口同步支持

### 修复
- **批量接口字段丢失** — `/recognize/batch` 构造 `TrainIDBatchItem` 时漏传 `container` 和 `containerConfidence`，已补上

### 文件变更
- 修改 `dms_api/app/schemas/train_id.py` — `TrainIDData`/`TrainIDBatchItem` 新增 `container` + `containerConfidence`
- 修改 `dms_api/app/services/train_id.py` — `recognize_image()` 提取最佳集装箱，`recognize_batch()` 透传
- 修改 `dms_api/app/api/v1/train_id.py` — 批量接口补上 `container`/`containerConfidence` 字段

## [0.10.2] - 2026-05-15

### 重构
- **API 与 CLI 统一** — `/recognize` 和 `/recognize/batch` 底层改为直接调用 `train_id_ocr_paddle.py` 的 `PaddleOCRProcessor`
  - 删除独立的 `PaddleImageProcessor`（`dms_api/app/train_id/paddle_image_processor.py`）
  - 保证 API 和 CLI 使用同一份核心识别代码，避免维护两套逻辑
  - `train_id_ocr_paddle.py` 新增 `process_bytes(image_bytes)` 方法供 API 调用

### 功能
- **空挡检测** — `/recognize` 和 `/recognize/batch` 响应新增 `type` 字段
  - `type: "########"` 表示空挡帧（车厢连接处）
  - `type: ""` 表示正常帧
  - 纯 CV 判断，不训练：中心 ROI + 四特征严格 AND（竖直边缘 + 低梯度 + 亮度 + 亮斑）
  - 新增 `dms_api/app/train_id/gap_detector.py` — `GapDetector` 空挡检测器
- **板车单图识别** — 新增 `POST /api/v1/train-id/recognize/flatcar` 端点
  - 底层使用 `run_bottom_merge_ocr.py` 的 `FlatcarBottomProcessor`
  - 底部区域（75%-100% 高度）+ LAB CLAHE 暗光增强 + 同行框拼接
  - 车型纠错覆盖 X70/X6K/C70E/C80 等
  - 车号优先取 7 位数字
  - 输出：`vehicleType` + `vehicleNumber` + `confidence`
- **GPU/CPU 自动兼容** — `PaddleOCRProcessor` 和 `FlatcarBottomProcessor` 默认尝试 GPU，失败自动回退 CPU
  - 支持 RTX 3060 等 Ampere 架构显卡
  - 无 CUDA 驱动或 GPU 初始化失败时无缝回退 CPU，不影响服务可用性

### 删除
- **移除 `/recognize/paddle` 端点** — 原集装箱+车种+车号独立接口，现由 `/recognize` 统一覆盖
- **移除 `/recognize/flatcar` 旧端点** — 原基于 `flatcar_image_processor.py` 的实现，现由新的 `/recognize/flatcar` 替换
- 删除 `train_id_ocr/batch_v6_flatcar.py`、`train_id_ocr/train_id_ocr_video_paddle_v5.py` 等旧文件

### 文件变更
- 修改 `train_id_ocr/train_id_ocr_paddle.py` — 新增 `process_bytes()`、GPU/CPU 自动切换
- 修改 `train_id_ocr/run_bottom_merge_ocr.py` — 新增 `FlatcarBottomProcessor` 类、GPU/CPU 自动切换
- 修改 `dms_api/app/api/v1/train_id.py` — 删除 `/recognize/paddle` 和旧 `/recognize/flatcar`，新增新的 `/recognize/flatcar`
- 修改 `dms_api/app/services/train_id.py` — 改为调用 `PaddleOCRProcessor` 和 `FlatcarBottomProcessor`
- 修改 `dms_api/app/schemas/train_id.py` — 删除 `PaddleImageData`/`FlatcarImageData`，新增 `FlatcarData`
- 修改 `dms_api/app/schemas/__init__.py` — 同步导出
- 新增 `dms_api/app/train_id/gap_detector.py` — 空挡检测器
- 新增 `train_id_ocr/train_id_ocr_paddle_backup.py` — 空挡检测集成前的原始备份

## [0.10.1] - 2026-05-14
### 功能
- **单图 PaddleOCR 识别** — `POST /api/v1/train-id/recognize/paddle` 单图识别端点
  - 基于 `train_id_ocr_video_paddle_v6.py` 改造为单图版 `train_id_ocr_paddle.py`
  - 复用 `PaddleOCREngine` 单例（`lang='en'`），GPU/CPU 自动切换
  - 上下分区策略：上半区（约55%）识别集装箱箱号，下半区识别铁路货车车种/车号
  - 输出：集装箱列表 + 车种列表 + 车号列表（简洁字符串列表，无置信度）
- **单图板车识别** — `POST /api/v1/train-id/recognize/flatcar` 板车单图识别端点
  - 基于 `flatcar_processor.py` 改造为单图版 `flatcar_image_processor.py`
  - 底部区域提取（75%-100% 高度），暗光预处理（LAB 空间 CLAHE）
  - 中文 PaddleOCR（`lang='ch'`），同行框拼接解决长数字串拆框问题
  - 车型纠错映射：`FLATCAR_CORRECTION` 覆盖 X70/X6K/C70E/C80 等
  - 输出：车型列表 + 车号列表（简洁字符串列表，无置信度）
- **视频接口下线（代码保留）** — `/recognize/video` 和 `/recognize/flatcar-video` 端点已注释移除
  - 视频相关处理代码保留在 `video_processor.py`、`flatcar_processor.py` 中（注释状态）
  - 日后如需恢复可直接取消注释

### 文件变更
- 新增 `train_id_ocr/train_id_ocr_paddle.py` — 单图版 PaddleOCR CLI 工具（视频代码注释保留）
- 新增 `dms_api/app/train_id/paddle_image_processor.py` — `PaddleImageProcessor` 单图处理核心
- 新增 `dms_api/app/train_id/flatcar_image_processor.py` — `FlatcarImageProcessor` 单图处理核心
- 修改 `dms_api/app/api/v1/train_id.py` — 新增 `/recognize/paddle` 和 `/recognize/flatcar`，注释移除 `/recognize/video` 和 `/recognize/flatcar-video`
- 修改 `dms_api/app/schemas/train_id.py` — 新增 `PaddleImageData`、`PaddleImageResponse`、`FlatcarImageData`、`FlatcarImageResponse`，注释移除视频相关 schema
- 修改 `dms_api/app/schemas/__init__.py` — 导出新增 schema
- 修改 `dms_api/app/services/train_id.py` — 新增 `recognize_paddle_image()`、`recognize_flatcar_image()`，注释移除视频相关方法
- 修改 `dms_api/app/train_id/__init__.py` — 导出 `PaddleImageProcessor`、`FlatcarImageProcessor`
- 修改 `dms_api/app/train_id/video_engine.py` — 新增 `ocr(img, cls=True)` 支持 numpy array 直接输入

## [0.10.0] - 2026-05-14
### 功能
- **视频车号识别** — `POST /api/v1/train-id/recognize/video` 视频识别端点
  - 基于 `train_id_ocr_video_paddle_v6.py` 改造集成到 dms_api
  - PaddleOCR 引擎（英文模型），GPU/CPU 自动切换
  - 上下分区策略：上半区识别集装箱箱号，下半区识别铁路货车车种/车号
  - 时序聚合：跨帧去重、碎片拼接、重叠合并
  - 参数：`interval_sec`（抽帧间隔，默认 0.5s）、`gap_sec`（聚合间隔，默认 3.0s）
  - 输出：集装箱列表 + 车种列表 + 车号列表（三套独立时序序列）
- **车板号视频识别** — `POST /api/v1/train-id/recognize/flatcar-video` 车板号识别端点
  - 基于 `run_bottom_merge_ocr.py` 改造集成到 dms_api
  - 底部 75%-100% 区域 ROI 裁剪，针对车板号喷涂位置优化
  - 中文 PaddleOCR（`lang='ch'`），支持中文车型字符识别
  - 同行框拼接：按 Y 坐标分行，同行内按 X 坐标排序合并，解决长数字串拆框问题
  - 车型纠错映射：`FLATCAR_CORRECTION` 覆盖 X70/X6K/C70E/C80 等常见车板型号
  - 车号重叠拼接：跨帧 2-3 框重叠合并，目标 7 位数字
  - 车种-车号时间窗口配对：按 `start_sec`~`end_sec` 重叠度匹配
  - 输出：`results` 数组，每条包含 `type`（车型）、`number`（车号）、`frames`、`avgConf`
- **PaddleOCR 引擎多语言支持** — `video_engine.py` 单例按 `(lang, use_gpu)` 组合缓存
  - `lang='en'`：集装箱/货车识别（英文数字+字母）
  - `lang='ch'`：车板号识别（中文字符+数字）

### 文件变更
- 新增 `dms_api/app/train_id/video_engine.py` — `PaddleOCREngine` 单例，支持 en/ch 双语言
- 新增 `dms_api/app/train_id/video_processor.py` — `VideoTrainIDProcessor`，集装箱+货车视频处理核心
- 新增 `dms_api/app/train_id/flatcar_processor.py` — `FlatcarVideoProcessor`，车板号视频处理核心
- 修改 `dms_api/app/api/v1/train_id.py` — 新增 `/recognize/video` 和 `/recognize/flatcar-video` 端点
- 修改 `dms_api/app/schemas/train_id.py` — 新增 `VideoTrainIDData`、`VideoTrainIDResponse`、`FlatcarVideoData`、`FlatcarVideoResponse`、`FlatcarItem`
- 修改 `dms_api/app/services/train_id.py` — 新增 `recognize_video()`、`recognize_flatcar_video()`、`get_video_processor()`、`get_flatcar_processor()`
- 修改 `dms_api/app/train_id/__init__.py` — 导出 `PaddleOCREngine`、`VideoTrainIDProcessor`、`VideoRecognitionResult`、`FlatcarVideoProcessor`、`FlatcarRecognitionResult`
- 修改 `dms_api/requirements.txt` — 新增 `paddlepaddle>=2.5.0`、`paddleocr>=2.7.0`

## [0.9.2] - 2026-04-26
### 功能
- **递归扫描子目录** — `signal_detect.py` 默认递归扫描所有包含媒体文件的目录
  - 支持 `night_signal/` 下的深层时间戳子目录（如 `front_signal/front_signal_20260424103529/`）
  - 新增 `_resolve_camera_config()` 向上遍历父目录匹配 ROI 配置，解决时间戳文件夹名无法匹配 config 的问题
- **视频检测支持** — 支持 `.mp4` / `.avi` / `.mov` 视频文件
  - 默认每秒采样 1 帧，超长视频自动加大间隔，上限 60 帧避免处理几小时的监控录像
  - 使用 `cap.set(CAP_PROP_POS_FRAMES)` 跳帧读取，避免逐帧解码大文件（处理速度从数百秒降至数十秒）
  - 视频级汇总输出：主导颜色、颜色分布、采样帧数、帧间隔
  - 视频结果展开显示帧级详情（Frame / Time / Predicted / Conf），不再只输出一行汇总
  - `--debug` 模式下为每个视频生成独立文件夹保存推理标注图（`debug/{folder}/{video}/f{frame}.jpg`）
- **重构核心检测接口** — 拆分 `detect_signal_color()` 为 `detect_signal_color_from_frame(img: np.ndarray)` + 文件读取包装器，图片和视频共用同一套检测逻辑

### 修复
- 排除 `_preview.jpg` 临时预览图混入检测流程
- 修复视频 debug 目录未自动创建的问题

### 文件变更
- 修改 `Light_signal/signal_detect.py` — 新增 `detect_signal_color_from_frame()`、`detect_signal_video()`、`_resolve_camera_config()`；重写 `evaluate_folder()` 支持视频；`main()` 默认递归扫描；`_discover_folders()` 递归发现媒体目录
- 重命名 `Light_signal/night_signal/` 下所有中文文件夹和文件为英文

## [0.9.1] - 2026-04-26
### 修复
- **信号灯检测中文路径修复** — 解决 OpenCV `cv2.imread()` 在 Windows 上无法读取含中文路径图片的问题
  - 将 `Light_signal/` 下四个中文文件夹重命名为英文：`front_signal`、`rear_signal`、`exit_signal`、`night_signal`
  - 同步修改 `signal_detect.py` 中 `DEFAULT_CONFIG` 的 key 为英文（`front_signal`、`rear_signal`、`exit_signal`）
  - 检测脚本现可正常读取图片并输出颜色识别结果（red/white/blue），不再全部 SKIP

### 文件变更
- 修改 `Light_signal/signal_detect.py` — `DEFAULT_CONFIG` 中文 key 改为英文
- 重命名 `Light_signal/拨车机前侧信号灯识别` → `Light_signal/front_signal`
- 重命名 `Light_signal/拨车机后侧信号灯` → `Light_signal/rear_signal`
- 重命名 `Light_signal/装车楼出口信号灯` → `Light_signal/exit_signal`
- 重命名 `Light_signal/夜晚信号灯` → `Light_signal/night_signal`

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
    - 荧光背心：H=25-85, S>60, V=80
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
