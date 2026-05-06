"""
DMS API Gateway

A production-grade FastAPI gateway for the DMS (Data Management System) backend.

Features:
- RESTful API endpoints for railway operations
- Automatic retry with exponential backoff
- Circuit breaker pattern for fault tolerance
- Request validation with Pydantic v2
- Comprehensive error handling and logging
- Health checks and metrics
- Rate limiting and security headers

Quick Start:
    uvicorn app.main:app --reload

API Documentation:
    http://localhost:8000/docs
"""

__version__ = "2.0.0"
__author__ = "DMS API Team"

from app.main import app, create_application

__all__ = [
    "__version__",
    "app",
    "create_application",
]
