"""
Train ID Recognition API Endpoints

Endpoints for station-entry vehicle identification (车种/车号) recognition.
Supports single-image, batch-image, and video-based recognition.
Also supports flatcar (车板号) video recognition.
"""

from fastapi import APIRouter, UploadFile, File, Form, status
from typing import Annotated

from ...dependencies import TrainIDServiceDep, RequestIdDep
from ...schemas.train_id import (
    TrainIDResponse,
    TrainIDBatchResponse,
    TrainIDData,
    TrainIDBatchItem,
    VideoTrainIDResponse,
    FlatcarVideoResponse,
)

router = APIRouter(prefix="/train-id", tags=["Train ID Recognition"])


# ------------------------------------------------------------------
# Image recognition (existing)
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# Video recognition (container + train)
# ------------------------------------------------------------------

@router.post(
    "/recognize/video",
    response_model=VideoTrainIDResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize train ID and containers from video",
    description="""
    Recognize container IDs (集装箱箱号) and railway train IDs (车种/车号)
    from a surveillance video.

    Uses PaddleOCR with temporal aggregation:
    - Upper half of frame: container IDs
    - Lower half of frame: train vehicle types and numbers

    Supported formats: MP4, AVI, MOV, MKV
    """,
    responses={
        200: {"description": "Video recognition completed"},
        400: {"description": "Invalid file format"},
        422: {"description": "Video processing failed"},
    },
)
async def recognize_train_id_video(
    service: TrainIDServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Surveillance video file")],
    interval_sec: Annotated[
        float,
        Form(description="Frame extraction interval in seconds (default: 0.5)")
    ] = 0.5,
    gap_sec: Annotated[
        float,
        Form(description="Temporal deduplication gap in seconds (default: 3.0)")
    ] = 3.0,
) -> VideoTrainIDResponse:
    """Recognize container IDs and train IDs from a video file."""
    video_bytes = await file.read()

    data = await service.recognize_video(
        video_bytes=video_bytes,
        filename=file.filename or "unknown",
        interval_sec=interval_sec,
        gap_sec=gap_sec,
    )

    response = VideoTrainIDResponse.ok(
        data=data,
        message=(
            f"识别完成: {data.container_count}个集装箱, "
            f"{data.train_type_count}个车种, {data.train_number_count}个车号, "
            f"处理{data.frames_processed}帧"
        ),
    )
    response.request_id = request_id
    return response


# ------------------------------------------------------------------
# Flatcar (车板号) video recognition
# ------------------------------------------------------------------

@router.post(
    "/recognize/flatcar-video",
    response_model=FlatcarVideoResponse,
    status_code=status.HTTP_200_OK,
    summary="Recognize flatcar (车板号) from video",
    description="""
    Recognize flatcar type and number (车板号) from a surveillance video.

    Features:
    - Processes only the bottom 75%-100% region of each frame
    - Uses Chinese PaddleOCR (ch_PP-OCRv3) for better Chinese character recognition
    - Row-wise box merging to handle split digit strings
    - Temporal aggregation with type-number pairing

    Supported flatcar types: X70, X6K, X2K, X2H, X4K, NX70, NX17, NX17B, C70, C70E, C80

    Supported formats: MP4, AVI, MOV, MKV
    """,
    responses={
        200: {"description": "Flatcar video recognition completed"},
        400: {"description": "Invalid file format"},
        422: {"description": "Video processing failed"},
    },
)
async def recognize_flatcar_video(
    service: TrainIDServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Surveillance video file")],
    interval_sec: Annotated[
        float,
        Form(description="Frame extraction interval in seconds (default: 0.05)")
    ] = 0.05,
    gap_sec: Annotated[
        float,
        Form(description="Temporal aggregation gap in seconds (default: 0.15)")
    ] = 0.15,
) -> FlatcarVideoResponse:
    """Recognize flatcar type and number from a video file."""
    video_bytes = await file.read()

    data = await service.recognize_flatcar_video(
        video_bytes=video_bytes,
        filename=file.filename or "unknown",
        interval_sec=interval_sec,
        gap_sec=gap_sec,
    )

    response = FlatcarVideoResponse.ok(
        data=data,
        message=(
            f"车板号识别完成: {data.type_count}个车型, "
            f"{data.number_count}个车号, "
            f"{len(data.results)}条合并结果, "
            f"处理{data.frames_processed}帧"
        ),
    )
    response.request_id = request_id
    return response
