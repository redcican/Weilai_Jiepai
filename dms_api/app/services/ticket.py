"""
Ticket Parsing Service

Business logic for OCR ticket parsing operations.
Uses local CnOCR engine for table extraction.
"""

from fastapi import UploadFile

from .base import BaseService
from .ocr import get_ocr_service, OCRService
from ..repositories.dms import DMSRepository
from ..schemas.ticket import TicketParseResponse, TicketData
from ..core.exceptions import DMSUpstreamError, DMSFileError


class TicketService(BaseService):
    """
    Service for ticket parsing operations.

    Uses local OCR engine for parsing, with optional fallback to DMS backend.
    """

    def __init__(
        self,
        repository: DMSRepository,
        ocr_service: OCRService | None = None,
        use_local_ocr: bool = True,
    ):
        super().__init__(repository)
        self._ocr_service = ocr_service or get_ocr_service()
        self._use_local_ocr = use_local_ocr

    async def parse_ticket(
        self,
        file: UploadFile,
    ) -> TicketParseResponse:
        """
        Parse ticket using OCR.

        Args:
            file: Ticket image or document

        Returns:
            TicketParseResponse with parsed data
        """
        # Validate file
        if not file.filename:
            raise DMSFileError(message="Filename is required")

        content_type = file.content_type or "application/octet-stream"
        file_content = await file.read()

        if not file_content:
            raise DMSFileError(message="File is empty", filename=file.filename)

        self._logger.info(
            f"Parsing ticket: filename={file.filename}, "
            f"size={len(file_content)}, type={content_type}",
        )

        # Try local OCR first
        if self._use_local_ocr and self._ocr_service.available:
            return await self._parse_with_local_ocr(file_content, file.filename)

        # Fallback to DMS backend
        return await self._parse_with_dms_backend(
            file_content, file.filename, content_type
        )

    async def _parse_with_local_ocr(
        self,
        file_content: bytes,
        filename: str,
    ) -> TicketParseResponse:
        """Parse ticket using local CnOCR engine."""
        self._logger.info(f"Using local OCR for {filename}")

        result = await self._ocr_service.parse_ticket_image(
            image_bytes=file_content,
            filename=filename,
        )

        if not result.get("success"):
            error_msg = result.get("error", "OCR processing failed")
            self._logger.warning(f"Local OCR failed: {error_msg}")

            # Try DMS backend as fallback
            if self._repository:
                self._logger.info("Falling back to DMS backend")
                return await self._parse_with_dms_backend(
                    file_content, filename, "image/jpeg"
                )

            raise DMSFileError(message=f"OCR failed: {error_msg}", filename=filename)

        # Create list of TicketData from all OCR rows
        ticket_list = self._ocr_service.create_ticket_data(result)

        if ticket_list is None:
            return TicketParseResponse.ok(
                data=[],
                message="Ticket parsed but data mapping incomplete",
            )

        return TicketParseResponse.ok(
            data=ticket_list,
            message=f"Ticket parsed successfully (local OCR), {len(ticket_list)} records",
        )

    async def _parse_with_dms_backend(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> TicketParseResponse:
        """Parse ticket using DMS backend."""
        self._logger.info(f"Using DMS backend for {filename}")

        result = await self._repository.parse_ticket(
            file_content=file_content,
            filename=filename,
            content_type=content_type,
        )

        if result.is_success:
            ticket_list = []
            if result.data:
                try:
                    items = result.data if isinstance(result.data, list) else [result.data]
                    for item in items:
                        ticket_list.append(TicketData.model_validate(item))
                except Exception as e:
                    self._logger.warning(f"Failed to parse ticket data: {e}")

            return TicketParseResponse.ok(
                data=ticket_list,
                message=result.msg or "Ticket parsed successfully (DMS backend)",
            )

        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to parse ticket: {result.msg}",
        )
