"""
Train ID Recognition API Endpoints

Endpoints for station-entry vehicle identification (车种/车号) recognition.
"""

from fastapi import APIRouter, UploadFile, File, status
from typing import Annotated

from ...dependencies import TrainIDServiceDep, RequestIdDep
from ...schemas.train_id import TrainIDResponse, TrainIDBatchResponse, TrainIDData, TrainIDBatchItem

router = APIRouter(prefix="/train-id", tags=["Train ID Recognition"])


@router.post(
    "/recognize",
    response_model=TrainIDResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize train ID from image",
    description="""
    Recognize vehicle type (车种) and vehicle number (车号) from a
    station-entry camera image.

    Supported file formats: JPEG, PNG, BMP, TIFF, WebP

    The OCR system uses multi-pass preprocessing with hybrid detection
    models to extract:
    - Vehicle type code (e.g. C64K, C70E, NX70)
    - Vehicle number (e.g. 49 31846)
    """,
    responses={
        200: {"description": "Train ID recognized successfully"},
        400: {"description": "Invalid file format"},
        422: {"description": "Recognition processing failed"},
    },
)
async def recognize_train_id(
    service: TrainIDServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Station-entry camera image")],
) -> TrainIDResponse:
    """Recognize vehicle type and number from a station-entry camera image."""
    image_bytes = await file.read()

    data = await service.recognize_image(image_bytes, file.filename or "unknown")

    response = TrainIDResponse.ok(data=data, message="Train ID recognized")
    response.request_id = request_id
    return response


@router.post(
    "/recognize/batch",
    response_model=TrainIDBatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch recognize train IDs",
    description="""
    Recognize vehicle type and number from multiple station-entry camera images.
    """,
    responses={
        200: {"description": "Batch recognition completed"},
        400: {"description": "Invalid file format"},
    },
)
async def recognize_train_id_batch(
    service: TrainIDServiceDep,
    request_id: RequestIdDep,
    files: Annotated[list[UploadFile], File(description="Station-entry camera images")],
) -> TrainIDBatchResponse:
    """Batch recognize vehicle type and number from multiple images."""
    images = []
    for f in files:
        content = await f.read()
        images.append((content, f.filename or "unknown"))

    results = await service.recognize_batch(images)

    response = TrainIDBatchResponse.ok(
        data=results,
        message=f"Processed {len(results)} images",
    )
    response.request_id = request_id
    return response
