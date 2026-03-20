"""
Train ID Recognition Endpoint Tests
"""

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from httpx import AsyncClient, ASGITransport

from app.main import create_application
from app.config import Settings, get_settings
from app.repositories.dms import DMSRepository
from app.dependencies import get_repository, get_train_id_service
from app.schemas.common import ResponseCode, DMSUpstreamResponse
from app.schemas.train_id import TrainIDData, TrainIDBatchItem
from app.services.train_id import TrainIDService


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        app_name="DMS API Test",
        environment="development",
        debug=True,
        dms_base_url="http://test-dms:8080",
        circuit_breaker_enabled=False,
        rate_limit_enabled=False,
        api_key_enabled=False,
        train_id_ocr_enabled=True,
    )


@pytest.fixture
def mock_repository() -> AsyncMock:
    repo = AsyncMock(spec=DMSRepository)
    repo.health_check.return_value = True
    return repo


@pytest.fixture
def mock_train_id_service() -> AsyncMock:
    service = AsyncMock(spec=TrainIDService)
    service.available = True
    service.recognize_image.return_value = TrainIDData(
        vehicleType="C64K",
        vehicleNumber="49 31846",
        confidence=0.85,
    )
    service.recognize_batch.return_value = [
        TrainIDBatchItem(
            filename="image1.bmp",
            vehicleType="C64K",
            vehicleNumber="49 31846",
            confidence=0.85,
        ),
        TrainIDBatchItem(
            filename="image2.bmp",
            vehicleType="C70E",
            vehicleNumber="160 1707",
            confidence=0.90,
        ),
    ]
    return service


@pytest.fixture
def app(test_settings, mock_repository, mock_train_id_service):
    application = create_application()
    application.dependency_overrides[get_settings] = lambda: test_settings

    async def get_mock_repo():
        return mock_repository

    application.dependency_overrides[get_repository] = get_mock_repo

    async def get_mock_train_id():
        return mock_train_id_service

    application.dependency_overrides[get_train_id_service] = get_mock_train_id

    return application


@pytest_asyncio.fixture
async def client(app) -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_image_content() -> bytes:
    """Minimal JPEG bytes."""
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


@pytest.mark.asyncio
class TestTrainIDEndpoints:
    """Tests for train ID recognition endpoints."""

    async def test_recognize_success(
        self,
        client: AsyncClient,
        mock_train_id_service: AsyncMock,
        sample_image_content: bytes,
    ):
        """Test single image recognition."""
        response = await client.post(
            "/api/v1/train-id/recognize",
            files={"file": ("station_01.bmp", sample_image_content, "image/bmp")},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["vehicleType"] == "C64K"
        assert data["data"]["vehicleNumber"] == "49 31846"
        assert data["data"]["confidence"] == 0.85
        assert data["message"] == "Train ID recognized"

        mock_train_id_service.recognize_image.assert_called_once()

    async def test_recognize_has_request_id(
        self,
        client: AsyncClient,
        sample_image_content: bytes,
    ):
        """Test that response includes request ID."""
        response = await client.post(
            "/api/v1/train-id/recognize",
            files={"file": ("test.jpg", sample_image_content, "image/jpeg")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "requestId" in data or "request_id" in data

    async def test_batch_recognize_success(
        self,
        client: AsyncClient,
        mock_train_id_service: AsyncMock,
        sample_image_content: bytes,
    ):
        """Test batch image recognition."""
        response = await client.post(
            "/api/v1/train-id/recognize/batch",
            files=[
                ("files", ("image1.bmp", sample_image_content, "image/bmp")),
                ("files", ("image2.bmp", sample_image_content, "image/bmp")),
            ],
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert len(data["data"]) == 2
        assert data["data"][0]["vehicleType"] == "C64K"
        assert data["data"][1]["vehicleType"] == "C70E"
        assert "Processed 2 images" in data["message"]

        mock_train_id_service.recognize_batch.assert_called_once()

    async def test_recognize_empty_result(
        self,
        client: AsyncClient,
        mock_train_id_service: AsyncMock,
        sample_image_content: bytes,
    ):
        """Test recognition with no detectable text."""
        mock_train_id_service.recognize_image.return_value = TrainIDData(
            vehicleType="",
            vehicleNumber="",
            confidence=0.0,
        )

        response = await client.post(
            "/api/v1/train-id/recognize",
            files={"file": ("dark.bmp", sample_image_content, "image/bmp")},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert data["data"]["vehicleType"] == ""
        assert data["data"]["vehicleNumber"] == ""


@pytest.mark.asyncio
class TestTrainIDUnit:
    """Unit tests for train ID engine internals."""

    def test_fix_vehicle_type_basic(self):
        """Test position-based OCR correction."""
        from app.train_id.utils import fix_vehicle_type
        assert fix_vehicle_type("C64K") == "C64K"
        assert fix_vehicle_type("C7OE") == "C70E"
        assert fix_vehicle_type("C6AK") == "C64K"

    def test_fix_vehicle_type_no_hardcode(self):
        """Verify no hardcoded string replacements — works on any pattern."""
        from app.train_id.utils import fix_vehicle_type
        assert fix_vehicle_type("NX7O") == "NX70"
        assert fix_vehicle_type("P6SK") == "P65K"

    def test_fix_vehicle_number(self):
        """Test digit cleaning."""
        from app.train_id.utils import fix_vehicle_number
        assert fix_vehicle_number("49 3l846") == "49 31846"
        assert fix_vehicle_number("l6O l7O7") == "160 1707"

    def test_is_vehicle_type_pattern(self):
        """Test type pattern detection with garbled-number rejection."""
        from app.train_id.utils import is_vehicle_type_pattern
        assert is_vehicle_type_pattern("C64K") is True
        assert is_vehicle_type_pattern("C70E") is True
        assert is_vehicle_type_pattern("12345") is False
        # Garbled numbers rejected
        assert is_vehicle_type_pattern("O7TO") is False

    def test_pick_best_type_majority(self):
        """Test majority voting for types."""
        from app.train_id.processor import _pick_best_type
        from app.train_id.models import TrainIDResult
        candidates = [
            TrainIDResult(vehicle_type="C64K"),
            TrainIDResult(vehicle_type="C70E"),
            TrainIDResult(vehicle_type="C64K"),
            TrainIDResult(vehicle_type="C64K"),
        ]
        assert _pick_best_type(candidates) == "C64K"

    def test_pick_best_number_superset(self):
        """Test superset-aware number voting."""
        from app.train_id.processor import _pick_best_number
        from app.train_id.models import TrainIDResult
        candidates = [
            TrainIDResult(vehicle_number="160 170"),
            TrainIDResult(vehicle_number="160 170"),
            TrainIDResult(vehicle_number="160 1707"),
        ]
        # Should prefer longer superset
        assert _pick_best_number(candidates) == "160 1707"
