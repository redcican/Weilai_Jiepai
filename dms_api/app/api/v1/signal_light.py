"""
Signal Light Detection API Endpoints

Batch endpoint for railway signal light color recognition (信号灯颜色识别).
"""

from fastapi import APIRouter, UploadFile, File, Form, status
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

    Uses pure computer vision (HSV + blob analysis) — no ML models.

    **roi** (optional): signal light region as `x1,y1,x2,y2` in 1280×720
    coordinates. Providing ROI significantly improves accuracy for fixed cameras.
    Without ROI, the engine auto-detects via brightness/saturation ranking.

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
    roi: Annotated[str | None, Form(description="Signal region: x1,y1,x2,y2 (1280×720 coords)")] = None,
) -> SignalLightBatchResponse:
    """Batch detect signal light colors from surveillance camera images."""
    images = []
    for f in files:
        content = await f.read()
        images.append((content, f.filename or "unknown"))

    roi_list = None
    if roi:
        roi_list = [int(x.strip()) for x in roi.split(",")]

    results = await service.detect_batch(images, roi=roi_list)

    response = SignalLightBatchResponse.ok(
        data=results,
        message=f"识别完成, {len(results)} 张图片",
    )
    response.request_id = request_id
    return response
