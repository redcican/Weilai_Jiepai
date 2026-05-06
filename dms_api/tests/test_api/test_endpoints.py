"""
API Endpoint Tests

Tests for all FastAPI endpoints.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
class TestHealthEndpoints:
    """Tests for health check endpoints."""

    async def test_health_check(self, client: AsyncClient):
        """Test basic health check."""
        response = await client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "environment" in data

    async def test_liveness_probe(self, client: AsyncClient):
        """Test Kubernetes liveness probe."""
        response = await client.get("/health/live")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "alive"

    async def test_readiness_probe(self, client: AsyncClient):
        """Test Kubernetes readiness probe."""
        response = await client.get("/health/ready")
        assert response.status_code == 200

        data = response.json()
        assert "ready" in data
        assert "checks" in data


@pytest.mark.asyncio
class TestAbnormalEndpoints:
    """Tests for abnormal alert endpoints."""

    async def test_create_abnormal_success(
        self,
        client: AsyncClient,
        sample_image_content: bytes,
    ):
        """Test creating an abnormal alert."""
        response = await client.post(
            "/api/v1/abnormal",
            data={
                "carbinNo": "C12345",
                "descr": "Test alert",
            },
            files={"file": ("test.jpg", sample_image_content, "image/jpeg")},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["success"] is True
        assert "data" in data

    async def test_create_abnormal_missing_field(
        self,
        client: AsyncClient,
        sample_image_content: bytes,
    ):
        """Test validation error for missing field."""
        response = await client.post(
            "/api/v1/abnormal",
            data={
                "carbinNo": "C12345",
                # Missing descr
            },
            files={"file": ("test.jpg", sample_image_content, "image/jpeg")},
        )
        assert response.status_code == 422

    async def test_create_bad_condition_success(
        self,
        client: AsyncClient,
        sample_image_content: bytes,
    ):
        """Test reporting bad condition."""
        response = await client.post(
            "/api/v1/abnormal/condition",
            data={
                "carbinNo": "C12345",
                "abnormalType": "1",
                "isAbnormal": "1",
                "descr": "Lock issue",
            },
            files={"file": ("test.jpg", sample_image_content, "image/jpeg")},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["success"] is True

    async def test_create_bad_condition_no_file(
        self,
        client: AsyncClient,
    ):
        """Test reporting bad condition without file."""
        response = await client.post(
            "/api/v1/abnormal/condition",
            data={
                "carbinNo": "C12345",
                "abnormalType": "1",
                "isAbnormal": "0",
            },
        )
        assert response.status_code == 201


@pytest.mark.asyncio
class TestTicketEndpoints:
    """Tests for ticket parsing endpoints."""

    async def test_parse_ticket_success(
        self,
        client: AsyncClient,
        sample_image_content: bytes,
    ):
        """Test parsing ticket (batch)."""
        response = await client.post(
            "/api/v1/ticket/parse",
            files=[
                ("files", ("ticket.jpg", sample_image_content, "image/jpeg")),
            ],
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 1
        assert "filename" in data["data"][0]

    async def test_parse_ticket_batch(
        self,
        client: AsyncClient,
        sample_image_content: bytes,
    ):
        """Test parsing multiple tickets."""
        response = await client.post(
            "/api/v1/ticket/parse",
            files=[
                ("files", ("ticket1.jpg", sample_image_content, "image/jpeg")),
                ("files", ("ticket2.jpg", sample_image_content, "image/jpeg")),
            ],
        )
        assert response.status_code == 200

        data = response.json()
        assert data["success"] is True
        assert isinstance(data["data"], list)
        assert len(data["data"]) == 2
        assert data["data"][0]["filename"] == "ticket1.jpg"
        assert data["data"][1]["filename"] == "ticket2.jpg"


@pytest.mark.asyncio
class TestResponseHeaders:
    """Tests for response headers."""

    async def test_request_id_header(self, client: AsyncClient):
        """Test that X-Request-ID header is returned."""
        response = await client.get("/health")
        assert "X-Request-ID" in response.headers

    async def test_response_time_header(self, client: AsyncClient):
        """Test that X-Response-Time header is returned."""
        response = await client.get("/health")
        assert "X-Response-Time" in response.headers

    async def test_security_headers(self, client: AsyncClient):
        """Test security headers are present."""
        response = await client.get("/health")
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
