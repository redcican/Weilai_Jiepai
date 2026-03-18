"""
Structured Logging Configuration

Provides JSON and text logging formatters with request context.
"""

import logging
import sys
import json
from datetime import datetime, timezone
from typing import Any
from contextvars import ContextVar

# Context variable for request ID
request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request ID if available
        request_id = request_id_ctx.get()
        if request_id:
            log_data["request_id"] = request_id

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add source location in debug mode
        if record.levelno <= logging.DEBUG:
            log_data["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_data, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """Colored text formatter for development."""

    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        # Add color
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname:8}{self.RESET}"

        # Add request ID prefix
        request_id = request_id_ctx.get()
        prefix = f"[{request_id[:8]}] " if request_id else ""

        # Format message
        formatted = super().format(record)
        return f"{prefix}{formatted}"


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: str | None = None,
) -> None:
    """
    Configure application logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('json' or 'text')
        log_file: Optional log file path
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(JSONFormatter())  # Always JSON for files
        root_logger.addHandler(file_handler)

    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with extra context."""

    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.get("extra", {})
        extra["extra_fields"] = self.extra
        kwargs["extra"] = extra
        return msg, kwargs
