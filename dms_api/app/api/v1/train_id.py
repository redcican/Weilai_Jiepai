"""
Train ID Recognition Endpoints

FastAPI router for train identification recognition.
Supports single image and batch processing using train_id_ocr_paddle.
"""

import logging

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse

from ...schemas.train_id import (
    TrainIDResponse,
    TrainIDBatchResponse,
    TrainIDBatchItem,
    FlatcarResponse,
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
    summary="识别单张列车图片（车种/车号）",
    description="""
    上传一张列车图片，使用 PaddleOCR 引擎识别车种和车号信息。

    **返回数据：** type(空挡标记)、车种(vehicleType)、车号(vehicleNumber)、置信度(confidence)
    """,
)
async def recognize_train_id(
    image: UploadFile = File(..., description="列车图片文件"),
    cam_id: str | None = Form(None, description="摄像头标识（如 cam1/cam2/cam3/cam4），用于帧过滤降采样"),
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
        data = await service.recognize_image(image_bytes, image.filename, cam_id=cam_id)

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
    summary="批量识别列车图片（车种/车号）",
    description="""
    上传多张列车图片进行批量识别。

    **返回数据：** 包含每张图片的识别结果列表
    """,
)
async def recognize_train_id_batch(
    images: list[UploadFile] = File(..., description="列车图片文件列表"),
    cam_id: str | None = Form(None, description="摄像头标识，批量图片来自同一摄像头时启用帧过滤"),
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
            data = await service.recognize_image(image_bytes, image.filename, cam_id=cam_id)
            items.append(TrainIDBatchItem(
                filename=image.filename,
                type=data.type,
                vehicleType=data.vehicle_type,
                vehicleNumber=data.vehicle_number,
                confidence=data.confidence,
                container=data.container,
                containerConfidence=data.container_confidence,
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
    "/recognize/flatcar",
    response_model=FlatcarResponse,
    responses={
        200: {"description": "Flatcar recognized successfully"},
        400: {"description": "Invalid image format"},
        422: {"description": "OCR processing failed"},
        503: {"description": "Engine not available"},
    },
    summary="识别单张板车图片（车型/车号）",
    description="""
    上传一张板车（车板号）图片，使用底部区域 OCR + 同行框拼接识别车型和车号。

    **识别策略：**
    - 底部区域（75%-100% 高度）：针对车板号喷涂位置优化
    - 暗光预处理：LAB 空间 CLAHE 增强
    - 同行框拼接：按 Y 坐标分行，同行内按 X 坐标排序合并

    **返回数据：** 车型(vehicleType)、车号(vehicleNumber)、置信度(confidence)
    """,
)
async def recognize_flatcar_image(
    image: UploadFile = File(..., description="板车图片文件"),
) -> FlatcarResponse | JSONResponse:
    """Recognize flatcar type and number from a single image."""
    service = get_train_id_service_singleton()
    endpoint = "/api/v1/train-id/recognize/flatcar"

    if not service.flatcar_available:
        return _get_error_response(
            "Flatcar engine not available",
            status_code=503,
            endpoint=endpoint,
        )

    try:
        image_bytes = await image.read()
        data = await service.recognize_flatcar_image(image_bytes, image.filename)

        return FlatcarResponse(
            success=True,
            message="Flatcar recognized successfully",
            data=data,
        )

    except Exception as e:
        logger.error(f"Flatcar recognition error: {e}")
        return _get_error_response(str(e), endpoint=endpoint)
