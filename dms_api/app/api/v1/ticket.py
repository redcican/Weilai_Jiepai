"""
Ticket Parsing API Endpoints

Endpoints for OCR ticket parsing operations.
Returns raw table data with table_type — no column-name mapping.
"""

from fastapi import APIRouter, UploadFile, File, status
from typing import Annotated

from ...dependencies import TicketServiceDep, RequestIdDep
from ...schemas.ticket import TicketParseResponse

router = APIRouter(prefix="/ticket", tags=["Ticket OCR"])


@router.post(
    "/parse",
    response_model=TicketParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse ticket with OCR",
    description="""
    Parse a grouping order (编组单) or station car list (站存车打印) using OCR.

    Auto-detects table type:
    - **Type 1** (站存车打印): ~16 columns, vehicle type and number in separate columns.
      Output starts with `{"股道": "4"}` if track number is detected.
    - **Type 2** (集装箱编组单): ~5 columns with slash vehicle/number (e.g. C70E/1805776)
      and container numbers (e.g. TBJU3216534).

    Returns raw table data arrays — no column-name mapping.

    Supported formats: JPEG, PNG, BMP, PDF
    """,
    responses={
        200: {"description": "Ticket parsed successfully"},
        400: {"description": "Invalid file format"},
        422: {"description": "OCR processing failed"},
        502: {"description": "Upstream service error"},
    },
)
async def parse_ticket(
    service: TicketServiceDep,
    request_id: RequestIdDep,
    file: Annotated[UploadFile, File(description="Ticket image or document")],
) -> TicketParseResponse:
    """Parse ticket document using OCR and extract structured data."""
    response = await service.parse_ticket(file)
    response.request_id = request_id
    return response
