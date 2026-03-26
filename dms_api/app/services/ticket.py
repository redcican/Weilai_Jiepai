"""
Ticket Parsing Service

Business logic for OCR ticket parsing operations.
Supports batch processing of multiple images.
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
    Supports batch processing of multiple images in a single request.
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

    async def parse_tickets(
        self,
        files: list[UploadFile],
    ) -> TicketParseResponse:
        """
        Parse one or more ticket images using OCR.

        Returns a single response with all results.
        """
        if not files:
            raise DMSFileError(message="No files provided")

        results: list[OCRTableData] = []

        for file in files:
            ocr_data = await self._parse_single(file)
            results.append(ocr_data)

        success_count = sum(1 for r in results if r.table_data)
        return TicketParseResponse.ok(
            data=results,
            message=f"识别成功, {success_count}/{len(results)} 张图片",
        )

    async def _parse_single(self, file: UploadFile) -> OCRTableData:
        """Parse a single file and return OCRTableData."""
        filename = file.filename or "unknown"

        if not file.filename:
            raise DMSFileError(message="Filename is required")

        file_content = await file.read()

        if not file_content:
            raise DMSFileError(message="File is empty", filename=filename)

        self._logger.info(f"Parsing ticket: filename={filename}, size={len(file_content)}")

        # Try local OCR first
        if self._use_local_ocr and self._ocr_service.available:
            return await self._parse_with_local_ocr(file_content, filename)

        # Fallback to DMS backend
        content_type = file.content_type or "application/octet-stream"
        return await self._parse_with_dms_backend(file_content, filename, content_type)

    async def _parse_with_local_ocr(
        self,
        file_content: bytes,
        filename: str,
    ) -> OCRTableData:
        """Parse ticket using local CnOCR engine."""
        self._logger.info(f"Using local OCR for {filename}")

        result = await self._ocr_service.parse_ticket_image(
            image_bytes=file_content,
            filename=filename,
        )

        if not result.is_success:
            self._logger.warning(f"Local OCR failed for {filename}: {result.message}")

            # Try DMS backend as fallback
            if self._repository:
                self._logger.info(f"Falling back to DMS backend for {filename}")
                return await self._parse_with_dms_backend(
                    file_content, filename, "image/jpeg"
                )

            raise DMSFileError(message=f"OCR failed: {result.message}", filename=filename)

        return OCRTableData(
            filename=filename,
            table_type=result.table_type,
            metadata=result.metadata,
            table_data=result.table_data,
        )

    async def _parse_with_dms_backend(
        self,
        file_content: bytes,
        filename: str,
        content_type: str,
    ) -> OCRTableData:
        """Parse ticket using DMS backend."""
        self._logger.info(f"Using DMS backend for {filename}")

        result = await self._repository.parse_ticket(
            file_content=file_content,
            filename=filename,
            content_type=content_type,
        )

        if result.is_success:
            table_data = result.data if isinstance(result.data, list) else [result.data] if result.data else []
            return OCRTableData(
                filename=filename,
                table_type=1,
                metadata={},
                table_data=table_data,
            )

        raise DMSUpstreamError(
            upstream_code=result.code,
            upstream_message=result.msg,
            message=f"Failed to parse ticket: {result.msg}",
        )
