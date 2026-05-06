"""
Ticket Parsing API Endpoints

Endpoints for OCR ticket parsing operations.
Supports batch processing of multiple images.
"""

from fastapi import APIRouter, UploadFile, status

from ...dependencies import TicketServiceDep, RequestIdDep
from ...schemas.ticket import TicketParseResponse

router = APIRouter(prefix="/ticket", tags=["Ticket OCR"])


@router.post(
    "/parse",
    response_model=TicketParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse ticket images with OCR (batch)",
    description="""
    Parse one or more ticket images using OCR. Supports batch processing.

    Auto-detects table type per image:
    - **Type 1** (站存车打印): ~16 columns, vehicle type and number in separate columns.
      Output starts with `{"股道": "4"}` if track number is detected.
    - **Type 2** (集装箱编组单): ~5 columns with slash vehicle/number (e.g. C70E/1805776)
      and container numbers (e.g. TBJU3216534).

    Each result includes the filename and its own tableType.

    Supported formats: JPEG, PNG, BMP, PDF
    """,
    responses={
        200: {"description": "Tickets parsed successfully"},
        400: {"description": "Invalid file format"},
        422: {"description": "OCR processing failed"},
        502: {"description": "Upstream service error"},
    },
)
async def parse_ticket(
    service: TicketServiceDep,
    request_id: RequestIdDep,
    files: list[UploadFile],
) -> TicketParseResponse:
    """Parse ticket images using OCR and extract structured data."""
    response = await service.parse_tickets(files)
    response.request_id = request_id
    return response
