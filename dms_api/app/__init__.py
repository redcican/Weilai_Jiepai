"""
DMS API Gateway Application

FastAPI-based API gateway for the DMS backend system.
"""

from .main import app, create_application

__all__ = ["app", "create_application"]
