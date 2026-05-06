# OCR Table Extraction Project

## Project Goal
Extract structured tabular data from images of railway/logistics documents using OCR, focusing on efficiency and data quality.

## Core Requirements

### Data Filterin
- **Only extract rows starting with sequence numbers** (e.g., "1", "2", "70", etc.)
- **Ignore header rows and column names** - these are metadata, not data
- **Skip empty or malformed rows** - filter during extraction, not post-processing

### Performance Guidelines

#### Vectorized Operations
Prefer NumPy/Pandas vectorized operations over Python loops:

```python
# BAD: Python loop
results = []
for row in data:
    if row[0].isdigit():
        results.append(row)

# GOOD: Vectorized with NumPy/Pandas
import numpy as np
mask = np.array([r[0].isdigit() if r else False for r in data])
results = np.array(data)[mask]

# BETTER: Use pandas for structured data
import pandas as pd
df = pd.DataFrame(data)
df = df[df.iloc[:, 0].str.match(r'^\d+', na=False)]
```

#### Batch Processing
- Process multiple images in batches when possible
- Reuse OCR engine instance across images (avoid repeated initialization)
- Use multiprocessing for CPU-bound tasks, asyncio for I/O-bound tasks

```python
# Reuse engine instance
engine = PaddleEngine()  # Initialize once
for img in images:
    result = engine.recognize(img)  # Reuse
```

#### Memory Efficiency
- Stream large files instead of loading entirely into memory
- Use generators for processing large datasets
- Clear intermediate results when no longer needed

## OCR Engine Priority
1. **PaddleOCR (PP-OCRv5)** - Best accuracy for Chinese documents, GPU acceleration
2. **EasyOCR** - Good multilingual support, fallback option
3. **Tesseract** - Lightweight, last resort

## Output Format

### Minimal JSON Structure: example as below(仅做参考)
```json
{
"message":"编组单号识别成功",
"status": "success",
"table data":[
["1","C70/1664442","TBJU5915952","TBJU3444196","马尾"],
["2","C70/1621631","TBJU4966446","TBJU1748543","马尾"],
["3","C70E/1689740","TBJU1885422","TBJU1726122","马尾"]
]
}
```


# Dependencies
```
paddlepaddle-gpu>=2.5.0  # or paddlepaddle for CPU
paddleocr>=2.7.0
paddlex>=3.0.0
numpy>=1.24.0
pandas>=2.0.0
Pillow>=10.0.0
```
