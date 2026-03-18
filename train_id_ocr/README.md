# Train ID OCR — 进站车辆识别

从进站摄像头图片中提取车种（vehicle type）和车号（vehicle number），使用 CnOCR + 多重预处理策略。

## 安装

```bash
pip install -r requirements.txt
```

## 使用

```bash
# OCR 识别
python train_id_ocr.py /path/to/image/folder -o ./output

# 详细输出
python train_id_ocr.py /path/to/image/folder -v

# 评估（需要 ground truth 文件）
python evaluate.py -p ./output -g ground_truth.text
```

## 方法

针对进站场景的暗光、模板字（stenciled text）图像，采用 **图像缩放 + 双检测引擎 + 四重预处理 + 智能合并** 策略：

### 预处理流水线

1. **图像缩放**：原始 4096×3000 图像缩放至 1024×750（0.25 倍），提升 OCR 准确率
2. **四重预处理 + 双检测引擎**：
   - **Bilateral+CLAHE** + db_resnet18 — 保边滤波 + 对比度增强，数字准确率最高
   - **CLAHE** + ch_PP-OCRv3_det — 通用对比度增强，车种检测更准确
   - **Gamma(2.0)+CLAHE** + db_resnet18 — 提亮暗区，发现微弱边缘数字
   - **Gamma(3.0)+CLAHE** + db_resnet18 — 最强提亮，发现极弱文字

### 合并策略

- **车种**：跨所有预处理结果多数投票，平局时优先选择伽马增强结果（字符对比度更高）
- **车号**：超集感知投票 — 当某个预处理发现了额外的边缘数字（如尾部的 "7"），自动选择更完整的结果

### OCR 后处理

采用通用的**位置感知字符修正**，无硬编码替换规则：
- 将文本分段为 `字母前缀 + 数字中段 + 字母后缀`
- 数字位置：O→0, I/l→1, S→5, A→4, G→6, T→7, B→8, Z→2
- 字母位置：0→O, 5→S 等反向映射
- 自动过滤铁路标志噪声（®、京、单字符低置信度等）

## 输出格式

每张图片生成一个 JSON 文件：

```json
{
  "file": "1.bmp",
  "vehicle_type": "C64K",
  "vehicle_number": "49 31846",
  "confidence": 0.5432
}
```

评估结果保存至 `evaluation.json`：

```json
{
  "summary": {
    "total_images": 17,
    "type_accuracy": 1.0,
    "number_digit_accuracy": 1.0
  },
  "details": [...]
}
```

## 当前精度

在 `进站_OCR/` 数据集（17 张可用图片）上的结果：

| 指标 | 结果 |
|------|------|
| 车种精确匹配 | **100%** (17/17) |
| 车号数字匹配 | **100%** (17/17) |
| 平均编辑距离（车种） | 0.00 |
| 平均编辑距离（车号） | 0.00 |
