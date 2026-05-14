"""
Train ID Recognition Schemas

Schemas for station-entry vehicle identification operations.
"""

from pydantic import Field
from .base import BaseSchema, ResponseSchema


class TrainIDData(BaseSchema):
    """
    Recognized train identification data.

    Contains vehicle type and number extracted from station-entry camera images.
    """

    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Vehicle type code (车种), e.g. C64K, C70E",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Vehicle number (车号), e.g. 49 31846",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score (0.0 - 1.0)",
    )


class TrainIDResponse(ResponseSchema[TrainIDData]):
    """Response for single image train ID recognition."""

    pass


class TrainIDBatchItem(BaseSchema):
    """Single item in a batch recognition response."""

    filename: str = Field(..., description="Original filename")
    vehicle_type: str = Field(
        default="",
        alias="vehicleType",
        description="Vehicle type code (车种)",
    )
    vehicle_number: str = Field(
        default="",
        alias="vehicleNumber",
        description="Vehicle number (车号)",
    )
    confidence: float = Field(
        default=0.0,
        description="Average OCR confidence score",
    )


class TrainIDBatchResponse(ResponseSchema[list[TrainIDBatchItem]]):
    """Response for batch train ID recognition."""

    pass


# ---------------------------------------------------------------------------
# Video recognition schemas
# ---------------------------------------------------------------------------

class VideoTrainIDData(BaseSchema):
    """Recognized data from video processing."""

    containers: list[str] = Field(
        default_factory=list,
        description="Container IDs (集装箱箱号)",
    )
    train_types: list[str] = Field(
        default_factory=list,
        alias="trainTypes",
        description="Train vehicle types (车种), e.g. C70E, C64K",
    )
    train_numbers: list[str] = Field(
        default_factory=list,
        alias="trainNumbers",
        description="Train vehicle numbers (车号)",
    )
    container_count: int = Field(
        default=0,
        alias="containerCount",
        description="Number of container sequences detected",
    )
    train_type_count: int = Field(
        default=0,
        alias="trainTypeCount",
        description="Number of train type sequences detected",
    )
    train_number_count: int = Field(
        default=0,
        alias="trainNumberCount",
        description="Number of train number sequences detected",
    )
    frames_processed: int = Field(
        default=0,
        alias="framesProcessed",
        description="Number of frames processed",
    )
    duration_sec: float = Field(
        default=0.0,
        alias="durationSec",
        description="Video duration in seconds",
    )


class VideoTrainIDResponse(ResponseSchema[VideoTrainIDData]):
    """Response for video train ID and container recognition."""

    pass


# ---------------------------------------------------------------------------
# Flatcar (车板号) video recognition schemas
# ---------------------------------------------------------------------------

class FlatcarItem(BaseSchema):
    """Single flatcar recognition entry with type and number."""

    type: str | None = Field(
        default=None,
        description="Flatcar type code (车型), e.g. X70, C70E, NX70",
    )
    number: str | None = Field(
        default=None,
        description="Flatcar number (车号), e.g. 5240903",
    )
    frames: int = Field(
        default=0,
        description="Number of frames this entry was detected in",
    )
    avg_conf: float = Field(
        default=0.0,
        alias="avgConf",
        description="Average confidence score",
    )


class FlatcarVideoData(BaseSchema):
    """Recognized data from flatcar video processing."""

    results: list[FlatcarItem] = Field(
        default_factory=list,
        description="List of flatcar type+number pairs",
    )
    type_count: int = Field(
        default=0,
        alias="typeCount",
        description="Number of type sequences detected",
    )
    number_count: int = Field(
        default=0,
        alias="numberCount",
        description="Number of number sequences detected (7-digit filtered)",
    )
    frames_processed: int = Field(
        default=0,
        alias="framesProcessed",
        description="Number of frames processed",
    )
    duration_sec: float = Field(
        default=0.0,
        alias="durationSec",
        description="Video duration in seconds",
    )


class FlatcarVideoResponse(ResponseSchema[FlatcarVideoData]):
    """Response for flatcar (车板号) video recognition."""

    pass
