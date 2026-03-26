"""
Ticket Parsing Service

Business logic for OCR ticket parsing operations.
Uses local CnOCR engine for table extraction.
"""

from fastapi import UploadFile

from .base import BaseService
from .ocr import get_ocr_service, OCRService
from ..repositories.dms import DMSRepository
from ..schemas.ticket import TicketParseResponse, OCRTableData
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

        Returns raw table data with table_type (no column-name mapping).
        """
        if not file.filename:
            raise DMSFileError(message="Filename is required")

        file_content = await file.read()

        if not file_content:
            raise DMSFileError(message="File is empty", filename=file.filename)

        self._logger.info(
            f"Parsing ticket: filename={file.filename}, "
            f"size={len(file_content)}",
        )

        # Try local OCR first
        if self._use_local_ocr and self._ocr_service.available:
            return await self._parse_with_local_ocr(file_content, file.filename)

        # Fallback to DMS backend
        content_type = file.content_type or "application/octet-stream"
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

        if not result.is_success:
            self._logger.warning(f"Local OCR failed: {result.message}")

            # Try DMS backend as fallback
            if self._repository:
                self._logger.info("Falling back to DMS backend")
                return await self._parse_with_dms_backend(
                    file_content, filename, "image/jpeg"
                )

            raise DMSFileError(message=f"OCR failed: {result.message}", filename=filename)

        ocr_data = OCRTableData(
            table_type=result.table_type,
            metadata=result.metadata,
            table_data=result.table_data,
        )

        return TicketParseResponse.ok(
            data=ocr_data,
            message=f"识别成功 (type={result.table_type}), {len(result.table_data)} rows",
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
            # Wrap DMS backend response as Type 1 raw data
            table_data = result.data if isinstance(result.data, list) else [result.data] if result.data else []
            ocr_data = OCRTableData(
                table_type=1,
                metadata={},
                table_data=table_data,
            )
            return TicketParseResponse.ok(
                data=ocr_data,
                message=result.msg or "Ticket parsed successfully (DMS backend)",
            )

        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to parse ticket: {result.msg}",
        )
