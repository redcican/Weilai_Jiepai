"""
Signal Light API Endpoints

Endpoints for signal light state change reporting.
"""

from fastapi import APIRouter, UploadFile, File, status
from typing import Annotated

from ...dependencies import SignalServiceDep, RequestIdDep
from ...schemas.signal import SignalChangeResponse

router = APIRouter(prefix="/signal", tags=["Signal Light"])


@router.post(
    "/change",
    response_model=SignalChangeResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report signal light change",
    description="""
    Report a signal light state change.

    Used by the AI recognition system to submit signal light
    state transitions with photographic evidence.

    The photo should capture the signal light clearly for verification.
    """,
    responses={
        201: {"description": "Signal change recorded"},
        400: {"description": "Invalid request data"},
        502: {"description": "Upstream service error"},
    },
)
async def report_signal_change(
    service: SignalServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Signal light photo")],
) -> SignalChangeResponse:
    """Submit signal light state change with photo."""
    response = await service.report_signal_change(file)
    response.request_id = request_id
    return response
