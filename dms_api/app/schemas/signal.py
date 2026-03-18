"""
Signal Light Schemas

Schemas for signal light state change reporting.
"""

from pydantic import Field
from .base import BaseSchema, ResponseSchema


class SignalChangeRequest(BaseSchema):
    """
    Request to report signal light state change.

    The image file is provided separately via multipart form data.
    This schema is for any additional metadata.
    """

    # Currently the API only requires the file
    # This schema can be extended for additional metadata
    pass


class SignalChangeData(BaseSchema):
    """Data returned after signal change report."""

    id: int | None = Field(default=None, description="Record ID")
    status: str | None = Field(default=None, description="Processing status")


class SignalChangeResponse(ResponseSchema[SignalChangeData | None]):
    """Response for signal light change report."""

    pass
