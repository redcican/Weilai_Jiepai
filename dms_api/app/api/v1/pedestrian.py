"""
Pedestrian Detection API Endpoints

Batch endpoint for railway driving safety pedestrian anomaly detection (行人异常检测).
"""

from fastapi import APIRouter, UploadFile, File, Form, status
from typing import Annotated

from ...dependencies import PedestrianServiceDep, RequestIdDep
from ...schemas.pedestrian import PedestrianBatchResponse

router = APIRouter(prefix="/pedestrian", tags=["Pedestrian Detection"])


@router.post(
    "/detect/batch",
    response_model=PedestrianBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch detect pedestrians in images",
    description="""
    Detect pedestrians in freight carriage top-down camera images.

    Uses YOLOv8 with a two-pass strategy:
    1. Full-image detection at high resolution
    2. Sliding-window tiling fallback for small/edge targets

    If any person is detected → 异常
    No person detected → 正常

    **use_gpu**: Set to `true` to run inference on GPU (CUDA).
    Defaults to the server config (`DMS_PEDESTRIAN_DETECTION_USE_GPU`).

    Supported formats: JPEG, PNG, BMP
    """,
    responses={
        200: {"description": "Detection completed"},
        400: {"description": "No valid images provided"},
    },
)
async def detect_pedestrians_batch(
    service: PedestrianServiceDep,
    request_id: RequestIdDep,
    files: Annotated[list[UploadFile], File(description="Carriage camera images")],
    use_gpu: Annotated[bool, Form(description="Use GPU (CUDA) for inference")] = False,
) -> PedestrianBatchResponse:
    """Batch detect pedestrians in freight carriage images."""
    images = []
    for f in files:
        content = await f.read()
        images.append((content, f.filename or "unknown"))

    results = await service.detect_batch(images, use_gpu=use_gpu)

    response = PedestrianBatchResponse.ok(
        data=results,
        message=f"检测完成, {len(results)} 张图片",
    )
    response.request_id = request_id
    return response
