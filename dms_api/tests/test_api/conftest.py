"""
FastAPI Test Configuration

Pytest fixtures for API testing.
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock
from httpx import AsyncClient, ASGITransport

from app.main import create_application
from app.config import Settings, get_settings
from app.repositories.dms import DMSRepository
from app.dependencies import get_repository
from app.schemas.common import ResponseCode, DMSUpstreamResponse

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    return Settings(
        app_name="DMS API Test",
        environment="development",
        debug=True,
        dms_base_url="http://test-dms:8080",
        circuit_breaker_enabled=False,
        rate_limit_enabled=False,
        api_key_enabled=False,
    )


@pytest.fixture
def mock_dms_response() -> DMSUpstreamResponse:
    """Create a mock successful DMS response."""
    return DMSUpstreamResponse(
        code=ResponseCode.SUCCESS,
        msg="Success",
        data=None,
    )


@pytest.fixture
def mock_repository(mock_dms_response) -> AsyncMock:
    """Create a mock DMS repository."""
    repo = AsyncMock(spec=DMSRepository)
    repo.save_abnormal.return_value = mock_dms_response
    repo.save_bad_condition.return_value = mock_dms_response
    repo.save_signal_change.return_value = mock_dms_response
    repo.save_container.return_value = mock_dms_response
    repo.parse_ticket.return_value = DMSUpstreamResponse(
        code=ResponseCode.SUCCESS,
        msg="Success",
        data={
            "ASSIGN_ID": 12345,
            "planNo": "PLN001",
            "trainNo": "T12345",
        },
    )
    repo.health_check.return_value = True
    return repo


@pytest.fixture
def app(test_settings, mock_repository):
    """Create test application with mocked dependencies."""
    application = create_application()

    # Override settings
    application.dependency_overrides[get_settings] = lambda: test_settings

    # Override repository - must override get_repository from dependencies module
    async def get_mock_repo():
        return mock_repository

    application.dependency_overrides[get_repository] = get_mock_repo

    return application


@pytest_asyncio.fixture
async def client(app) -> AsyncClient:
    """Create async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_image_content() -> bytes:
    """Create sample image content (minimal JPEG)."""
    return bytes([
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01,
        0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB, 0x00, 0x43,
        0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07, 0x07, 0x07, 0x09,
        0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B, 0x0B, 0x0C, 0x19, 0x12,
        0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E, 0x1D, 0x1A, 0x1C, 0x1C, 0x20,
        0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C, 0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29,
        0x2C, 0x30, 0x31, 0x34, 0x34, 0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32,
        0x3C, 0x2E, 0x33, 0x34, 0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01,
        0x00, 0x01, 0x01, 0x01, 0x11, 0x00, 0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01,
        0x00, 0x00, 0x3F, 0x00, 0x7F, 0xFF, 0xD9,
    ])
