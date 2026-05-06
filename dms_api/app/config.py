"""
Application Configuration

Uses pydantic-settings for type-safe configuration with environment variable support.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field, field_validator, AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.

    All settings can be overridden via environment variables with the DMS_ prefix.
    Example: DMS_DEBUG=true, DMS_DMS_BASE_URL=http://localhost:8080
    """

    model_config = SettingsConfigDict(
        env_prefix="DMS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="DMS API Gateway", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    environment: Literal["development", "staging", "production"] = Field(
        default="development", description="Environment name"
    )

    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, ge=1, le=65535, description="Server port")
    workers: int = Field(default=1, ge=1, description="Number of workers")

    # DMS Backend API
    dms_base_url: AnyHttpUrl = Field(
        default="http://123.127.38.120:2010",
        description="DMS backend API base URL"
    )
    dms_timeout_connect: float = Field(default=5.0, gt=0, description="Connection timeout")
    dms_timeout_read: float = Field(default=30.0, gt=0, description="Read timeout")
    dms_max_retries: int = Field(default=3, ge=0, description="Max retry attempts")
    dms_retry_backoff: float = Field(default=1.0, gt=0, description="Retry backoff factor")

    # Circuit Breaker
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
    circuit_breaker_threshold: int = Field(default=5, ge=1, description="Failure threshold")
    circuit_breaker_timeout: float = Field(default=30.0, gt=0, description="Reset timeout")

    # Security
    api_key_enabled: bool = Field(default=False, description="Enable API key authentication")
    api_keys: list[str] = Field(default_factory=list, description="Valid API keys")
    cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests: int = Field(default=100, ge=1, description="Requests per window")
    rate_limit_window: int = Field(default=60, ge=1, description="Window in seconds")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")
    log_file: str | None = Field(default=None, description="Log file path")

    # File Upload
    max_upload_size: int = Field(
        default=50 * 1024 * 1024,  # 50MB
        description="Maximum upload size in bytes"
    )
    allowed_extensions: list[str] = Field(
        default=[".jpg", ".jpeg", ".png", ".gif", ".bmp", ".pdf"],
        description="Allowed file extensions"
    )

    # OCR Configuration
    ocr_enabled: bool = Field(default=True, description="Enable local OCR processing")
    ocr_enhance_image: bool = Field(default=True, description="Enhance images before OCR")
    ocr_fallback_to_dms: bool = Field(default=True, description="Fallback to DMS backend if OCR fails")

    # Train ID OCR Configuration
    train_id_ocr_enabled: bool = Field(default=True, description="Enable train ID recognition")

    # Signal Light Detection Configuration
    signal_light_enabled: bool = Field(default=True, description="Enable signal light color detection")

    # Pedestrian Detection Configuration
    pedestrian_detection_enabled: bool = Field(default=True, description="Enable pedestrian detection")
    pedestrian_detection_model: str = Field(default="yolov8n.pt", description="YOLO model file name")
    pedestrian_detection_use_gpu: bool = Field(default=False, description="Use GPU (CUDA) for inference")
    pedestrian_detection_confidence: float = Field(default=0.25, ge=0.0, le=1.0, description="Primary confidence threshold")
    pedestrian_detection_iou: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    pedestrian_detection_imgsz: int = Field(default=1280, description="Primary inference image size")
    pedestrian_detection_tile_enabled: bool = Field(default=True, description="Enable sliding-window tiling fallback")
    pedestrian_detection_tile_size: int = Field(default=640, description="Tile size in pixels")
    pedestrian_detection_tile_overlap: float = Field(default=0.3, ge=0.0, lt=1.0, description="Tile overlap ratio")
    pedestrian_detection_tile_confidence: float = Field(default=0.15, ge=0.0, le=1.0, description="Tile confidence threshold")

    @field_validator("api_keys", mode="before")
    @classmethod
    def parse_api_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        return self.environment == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
