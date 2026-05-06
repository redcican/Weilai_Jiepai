# 表格OCR提取工具 - CnOCR优化版

基于 CnOCR 的智能表格识别工具，支持从图片中自动提取表格数据和元数据。

## 功能特点

- 🚀 基于 CnOCR + ONNX 后端，识别速度快
- 🎯 自动检测表格结构（序列号列、车型列、车辆编号列）
- 🔧 智能修正常见OCR识别错误
- 📊 自动分离元数据和表格数据
- 🔄 支持批量处理图片文件夹
- 📝 输出标准 JSON 格式结果

## 1. 安装必要的依赖

### 1.1 系统要求

- Python 3.8+
- Linux / Windows / macOS

### 1.2 创建虚拟环境（推荐）

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate
```

### 1.3 安装 Python 依赖

```bash
# 安装核心依赖
pip install cnocr==2.3.0.2
pip install Pillow>=9.0.0
pip install numpy>=1.21.0
pip install onnxruntime>=1.12.0

# 或者使用 requirements.txt（推荐）
pip install -r requirements.txt
```

**requirements.txt 文件内容：**

```txt
cnocr==2.3.0.2
Pillow>=9.0.0
numpy>=1.21.0
onnxruntime>=1.12.0
```

### 1.4 用于 FastAPI 部署的额外依赖

```bash
# 安装 FastAPI 相关依赖
pip install fastapi>=0.109.0
pip install uvicorn[standard]>=0.27.0
pip install python-multipart>=0.0.6
pip install pydantic>=2.0.0
```

**完整的 requirements.txt（包含 FastAPI）：**

```txt
# OCR 核心依赖
cnocr==2.3.0.2
Pillow>=9.0.0
numpy>=1.21.0
onnxruntime>=1.12.0

# FastAPI 依赖
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
python-multipart>=0.0.6
pydantic>=2.0.0
```

### 1.5 验证安装

```bash
# 验证 CnOCR 安装
python -c "from cnocr import CnOcr; print('CnOCR installed successfully')"

# 验证 FastAPI 安装
python -c "import fastapi; print('FastAPI installed successfully')"
```

## 2. 如何执行代码

### 2.1 命令行使用

#### 基本使用

```bash
# 处理单个文件夹
python table_ocr_cnocr.py /path/to/image/folder

# 指定输出目录
python table_ocr_cnocr.py /path/to/image/folder -o ./output_cnocr

# 递归处理子文件夹
python table_ocr_cnocr.py /path/to/image/folder -r

# 自定义行聚合容差
python table_ocr_cnocr.py /path/to/image/folder --tolerance 25

# 详细输出模式
python table_ocr_cnocr.py /path/to/image/folder -v
```

#### 示例

```bash
# 示例1: 处理当前目录的 fig 文件夹
python table_ocr_cnocr.py ../OCR/fig -o ./output_cnocr

# 示例2: 递归处理所有子文件夹并显示详细日志
python table_ocr_cnocr.py ../OCR/fig -r -v -o ./output_cnocr

# 示例3: 处理单张图片所在文件夹
python table_ocr_cnocr.py ./test_images -o ./test_output
```

### 2.2 输出格式

输出的 JSON 文件格式如下：

```json
{
  "message": "识别成功",
  "status": "success",
  "metadata": {
    "股道": "1",
    "其他元数据": "值"
  },
  "table data": [
    ["1", "C70", "1234567", "10.5", "上海", "煤炭"],
    ["2", "NX70AF", "2345678", "12.3", "北京", "钢材"],
    ...
  ]
}
```

## 3. 如何部署在 FastAPI 中

### 3.2 启动 FastAPI 服务

# 方法1: 使用 uvicorn（推荐）
uvicorn api:app --host 0.0.0.0 --port 8000 --reload

# 方法2: 指定工作进程数
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 生产环境

```bash
# 使用 Gunicorn + Uvicorn Workers（推荐）
pip install gunicorn

gunicorn api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```


### 3.4 查看 API 文档

FastAPI 自动生成交互式 API 文档：

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3.5 Docker 部署（可选）

创建 `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY table_ocr_cnocr.py .
COPY api.py .

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建和运行：

```bash
# 构建镜像
docker build -t ocr-api:latest .

# 运行容器
docker run -d -p 8000:8000 --name ocr-service ocr-api:latest

# 查看日志
docker logs -f ocr-service
```

## 4. 项目结构

```
OCR_CnOCR/
├── README.md                    # 本文档
├── requirements.txt             # Python 依赖
├── table_ocr_cnocr.py          # 核心 OCR 处理脚本
├── output_cnocr/               # 输出目录（自动创建）
│   ├── fig1.json
│   ├── fig2.json
│   └── ...
dms_api/
│   ├── app/

```




