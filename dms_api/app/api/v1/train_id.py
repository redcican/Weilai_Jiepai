"""
Train ID Recognition Endpoints

FastAPI router for train identification recognition.
Supports single image and batch processing using local CnOCR / PaddleOCR engines.
"""

import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from ...schemas.train_id import (
    TrainIDResponse,
    TrainIDBatchResponse,
    TrainIDBatchItem,
    PaddleImageResponse,
)
from ...services.train_id import get_train_id_service_singleton

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/train-id")


def _get_error_response(
    message: str,
    status_code: int = 500,
    endpoint: str = "",
) -> JSONResponse:
    """Create a standard error JSONResponse."""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "data": None,
            "endpoint": endpoint,
        },
    )


@router.post(
    "/recognize",
    response_model=TrainIDResponse,
    responses={
        200: {"description": "Train ID recognized successfully"},
        400: {"description": "Invalid image format"},
        422: {"description": "OCR processing failed"},
        503: {"description": "Engine not available"},
    },
    summary="识别单张列车图片（CnOCR 车种/车号）",
    description="""
    上传一张车站进站摄像头拍摄的列车图片，使用 CnOCR 引擎识别车种和车号信息。

    **适用场景：** 常规列车进站图片识别

    **返回数据：** 车种(vehicleType)、车号(vehicleNumber)、置信度(confidence)
    """,
)
async def recognize_train_id(
    image: UploadFile = File(..., description="车站进站摄像头图片文件"),
) -> TrainIDResponse | JSONResponse:
    """Recognize train vehicle type and number from a single image."""
    service = get_train_id_service_singleton()
    endpoint = "/api/v1/train-id/recognize"

    if not service.available:
        return _get_error_response(
            "Train ID engine not available",
            status_code=503,
            endpoint=endpoint,
        )

    try:
        image_bytes = await image.read()
        data = await service.recognize_image(image_bytes, image.filename)

        return TrainIDResponse(
            success=True,
            message="Train ID recognized successfully",
            data=data,
        )

    except Exception as e:
        logger.error(f"Train ID recognition error: {e}")
        return _get_error_response(str(e), endpoint=endpoint)


@router.post(
    "/recognize/batch",
    response_model=TrainIDBatchResponse,
    responses={
        200: {"description": "Batch processed successfully"},
        400: {"description": "Invalid image format"},
        422: {"description": "OCR processing failed"},
        503: {"description": "Engine not available"},
    },
    summary="批量识别列车图片（CnOCR 车种/车号）",
    description="""
    上传多张列车图片进行批量识别。

    **返回数据：** 包含每张图片的识别结果列表
    """,
)
async def recognize_train_id_batch(
    images: list[UploadFile] = File(..., description="车站进站摄像头图片文件列表"),
) -> TrainIDBatchResponse | JSONResponse:
    """Recognize train IDs from multiple images."""
    service = get_train_id_service_singleton()
    endpoint = "/api/v1/train-id/recognize/batch"

    if not service.available:
        return _get_error_response(
            "Train ID engine not available",
            status_code=503,
            endpoint=endpoint,
        )

    try:
        items = []
        for image in images:
            image_bytes = await image.read()
            data = await service.recognize_image(image_bytes, image.filename)
            items.append(TrainIDBatchItem(
                filename=image.filename,
                vehicleType=data.vehicle_type,
                vehicleNumber=data.vehicle_number,
                confidence=data.confidence,
            ))

        return TrainIDBatchResponse(
            success=True,
            message=f"Processed {len(items)} images",
            data=items,
        )

    except Exception as e:
        logger.error(f"Batch recognition error: {e}")
        return _get_error_response(str(e), endpoint=endpoint)


@router.post(
    "/recognize/paddle",
    response_model=PaddleImageResponse,
    responses={
        200: {"description": "PaddleOCR image recognized successfully"},
        400: {"description": "Invalid image format"},
        422: {"description": "OCR processing failed"},
        503: {"description": "Engine not available"},
    },
    summary="识别单张列车图片（PaddleOCR 集装箱+车种/车号）",
    description="""
    上传一张列车图片，使用 PaddleOCR 引擎识别集装箱箱号、车种和车号。

    **识别策略：**
    - 上半区域（约55%）：识别集装箱箱号
    - 下半区域（约45%）：识别车种和车号

    **适用场景：** 同时需要集装箱箱号和列车编号的场景
    """,
)
async def recognize_paddle_image(
    image: UploadFile = File(..., description="列车图片文件"),
) -> PaddleImageResponse | JSONResponse:
    """Recognize container IDs and train IDs from a single image using PaddleOCR."""
    service = get_train_id_service_singleton()
    endpoint = "/api/v1/train-id/recognize/paddle"

    if not service.paddle_available:
        return _get_error_response(
            "PaddleOCR image engine not available",
            status_code=503,
            endpoint=endpoint,
        )

    try:
        image_bytes = await image.read()
        data = await service.recognize_paddle_image(image_bytes, image.filename)

        return PaddleImageResponse(
            success=True,
            message="PaddleOCR image recognized successfully",
            data=data,
        )

    except Exception as e:
        logger.error(f"PaddleOCR image recognition error: {e}")
        return _get_error_response(str(e), endpoint=endpoint)


# ---------------------------------------------------------------------------
# Video recognition endpoints (deprecated, kept for reference)
# ---------------------------------------------------------------------------
# @router.post("/recognize/video", ...)
# @router.post("/recognize/flatcar-video", ...)
