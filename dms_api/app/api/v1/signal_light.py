"""
Signal Light Detection API Endpoints

Batch endpoint for railway signal light color recognition (信号灯颜色识别).
"""

from fastapi import APIRouter, UploadFile, File, status
from typing import Annotated

from ...dependencies import SignalLightServiceDep, RequestIdDep
from ...schemas.signal_light import SignalLightBatchResponse

router = APIRouter(prefix="/signal-light", tags=["Signal Light Detection"])


@router.post(
    "/detect/batch",
    response_model=SignalLightBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch detect signal light colors",
    description="""
    Detect signal light color (红色/白色/蓝色) from surveillance camera images.

    Uses pure computer vision (HSV + scene analysis) — no ML models.
    Auto-detects signal color without requiring region coordinates.

    Supported formats: JPEG, PNG, BMP
    """,
    responses={
        200: {"description": "Detection completed"},
        400: {"description": "No valid images provided"},
    },
)
async def detect_signal_light_batch(
    service: SignalLightServiceDep,
    request_id: RequestIdDep,
    files: Annotated[list[UploadFile], File(description="Surveillance camera images")],
) -> SignalLightBatchResponse:
    """Batch detect signal light colors from surveillance camera images."""
    images = []
    for f in files:
        content = await f.read()
        images.append((content, f.filename or "unknown"))

    results = await service.detect_batch(images)

    response = SignalLightBatchResponse.ok(
        data=results,
        message=f"识别完成, {len(results)} 张图片",
    )
    response.request_id = request_id
    return response
