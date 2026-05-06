# 表格图片OCR提取工具

## 📋 概述

本工具用于从表格图片中提取文字，将每一行转换为一条JSON记录。

## 🔧 安装依赖

### 方案1：PaddleOCR（推荐，中文识别最佳）
```bash
pip install paddlepaddle paddleocr
```

### 方案2：EasyOCR（多语言支持好）
```bash
pip install easyocr
```

### 方案3：Tesseract（本地运行，无需联网）
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim tesseract-ocr-chi-tra
pip install pytesseract pillow opencv-python

# macOS
brew install tesseract tesseract-lang
pip install pytesseract pillow opencv-python

# Windows
# 下载安装 https://github.com/UB-Mannheim/tesseract/wiki
pip install pytesseract pillow opencv-python
```

## 📂 文件说明

| 文件 | 说明 |
|------|------|
| `table_ocr_solution.py` | 完整解决方案，支持多引擎 |
| `table_ocr_tesseract.py` | Tesseract专用版本 |
| `table_ocr_improved.py` | 改进版（含预处理） |

## 🚀 使用方式

### 基本用法
```bash
python table_ocr_solution.py image.jpg -o output.json
```

### 指定OCR引擎
```bash
# 使用PaddleOCR（推荐中文）
python table_ocr_solution.py image.jpg --engine paddle -o output.json

# 使用Tesseract
python table_ocr_solution.py image.jpg --engine tesseract -o output.json
```

### 调整参数
```bash
# 指定表头行（从0开始）
python table_ocr_solution.py image.jpg --header-row 0

# 调整行聚合容差（像素）
python table_ocr_solution.py image.jpg --tolerance 20
```

### 文件夹批量处理
```bash
# 处理整个文件夹中的所有图片
python table_ocr_solution.py ./images/

# 处理文件夹，每张图片单独输出JSON
python table_ocr_solution.py ./images/ -o ./output/

# 处理文件夹，合并所有结果到一个JSON
python table_ocr_solution.py ./images/ --combined all_results.json

# 递归处理子文件夹
python table_ocr_solution.py ./images/ -r --combined all.json

# 同时单独输出和合并输出
python table_ocr_solution.py ./images/ -o ./output/ --combined merged.json
```

### 批量处理多个文件
```bash
python table_ocr_solution.py *.jpg --batch -o ./output/
```

### Python代码调用
```python
from table_ocr_solution import extract_table, TableOCRExtractor

# 简单调用
result = extract_table('image.jpg', 'output.json')
print(result['records'])

# 自定义表头
extractor = TableOCRExtractor(engine='paddle')
result = extractor.extract_with_custom_headers(
    'image.jpg',
    headers=['序号', '车种', '车号', '自重', '到站', '品名'],
    output_path='output.json'
)

# 批量处理多个文件
from table_ocr_solution import batch_extract
results = batch_extract(['img1.jpg', 'img2.jpg'], output_dir='./output/')

# 处理整个文件夹
from table_ocr_solution import extract_from_folder

# 基本用法
results = extract_from_folder('./images/')

# 带输出目录（每张图片生成单独JSON）
results = extract_from_folder('./images/', output_dir='./output/')

# 合并输出（所有结果合并到一个JSON）
results = extract_from_folder('./images/', combined_output='all_results.json')

# 递归处理子文件夹
results = extract_from_folder('./images/', recursive=True, combined_output='all.json')
```

## 📊 输出格式

```json
{
  "source_file": "image.jpg",
  "ocr_engine": "PaddleOCR",
  "total_rows": 50,
  "total_records": 46,
  "header_row": 0,
  "headers": ["股道", "序", "车种", "车号", ...],
  "raw_rows": [
    ["1", "C70", "1664442", ...],
    ...
  ],
  "records": [
    {"股道": "1", "序": "1", "车种": "C70", "车号": "1664442", ...},
    ...
  ]
}
```

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--engine` | OCR引擎 (auto/paddle/tesseract/easyocr) | auto |
| `--tolerance` | 行聚合容差（像素） | 15 |
| `--header-row` | 表头行索引 | 自动检测 |
| `--headers` | 自定义表头 | - |
| `--batch` | 批量处理模式 | False |
| `-r, --recursive` | 递归处理子文件夹 | False |
| `--combined` | 合并输出文件路径 | - |

## 📁 支持的图片格式

jpg, jpeg, png, bmp, tiff, tif, webp, gif

## 💡 最佳实践

### 1. 引擎选择
- **中文表格**：优先使用 PaddleOCR
- **多语言混合**：使用 EasyOCR
- **离线环境**：使用 Tesseract

### 2. 提高识别率
- 确保图片清晰，分辨率足够
- 表格线条清晰
- 避免倾斜和扭曲
- 适当调整 `--tolerance` 参数

### 3. 处理复杂表格
- 使用自定义表头避免表头识别错误
- 对于合并单元格，可能需要后处理
- 建议先检查 `raw_rows` 确认识别效果

## 🔍 常见问题

**Q: 识别率不高怎么办？**
- 尝试使用 PaddleOCR 引擎
- 检查图片质量
- 调整 tolerance 参数

**Q: 行数据错位怎么办？**
- 增大 tolerance 值（如 20-25）
- 检查表格是否有倾斜

**Q: 如何处理多页表格？**
- 将多页图片放入同一文件夹
- 使用文件夹模式：`python table_ocr_solution.py ./pages/ --combined result.json`
- 结果会自动合并，并标注来源文件

## 📝 示例：铁路站存车表格

```python
# 针对站存车打印表格的优化提取
RAILWAY_HEADERS = [
    "股道", "序", "车种", "油种", "车号", "自重", 
    "换长", "载重", "到站", "品名", "记事", 
    "发站", "篷布", "票据号"
]

extractor = TableOCRExtractor(engine='paddle', row_tolerance=12)
result = extractor.extract_with_custom_headers(
    'railway_table.jpg',
    headers=RAILWAY_HEADERS,
    skip_rows=2  # 跳过标题行
)
```

## 📜 许可证

MIT License
