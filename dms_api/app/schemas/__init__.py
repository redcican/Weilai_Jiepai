"""Pydantic v2 Schemas Module."""

from .base import BaseSchema, ResponseSchema, PaginatedResponse
from .common import (
    ResponseCode,
    AbnormalType,
    AbnormalStatus,
    HealthStatus,
    FileUpload,
)
from .abnormal import (
    AbnormalCreateRequest,
    AbnormalResponse,
    BadConditionCreateRequest,
    BadConditionResponse,
)
from .ticket import (
    TicketParseRequest,
    TicketData,
    TicketParseResponse,
)
from .train_id import (
    TrainIDData,
    TrainIDResponse,
    TrainIDBatchItem,
    TrainIDBatchResponse,
)
from .signal_light import (
    SignalLightItem,
    SignalLightBatchResponse,
)

__all__ = [
    # Base
    "BaseSchema",
    "ResponseSchema",
    "PaginatedResponse",
    # Common
    "ResponseCode",
    "AbnormalType",
    "AbnormalStatus",
    "HealthStatus",
    "FileUpload",
    # Abnormal
    "AbnormalCreateRequest",
    "AbnormalResponse",
    "BadConditionCreateRequest",
    "BadConditionResponse",
    # Ticket
    "TicketParseRequest",
    "TicketData",
    "TicketParseResponse",
    # Train ID
    "TrainIDData",
    "TrainIDResponse",
    "TrainIDBatchItem",
    "TrainIDBatchResponse",
    # Signal Light
    "SignalLightItem",
    "SignalLightBatchResponse",
]
