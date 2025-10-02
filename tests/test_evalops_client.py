"""
Unit tests for EvalOps API client.
"""

import os
from unittest.mock import patch

import pytest
import respx
from httpx import Response

from evalops_client import EvalOpsClient


@pytest.mark.asyncio
class TestEvalOpsClient:
    """Test suite for EvalOps client."""

    async def test_client_initialization_with_api_key(self):
        """Client initializes successfully with provided API key."""
        client = EvalOpsClient(api_key="test-key", api_url="https://test.evalops.dev")
        assert client.api_key == "test-key"
        assert client.base_url == "https://test.evalops.dev"
        await client.close()

    async def test_client_initialization_from_env(self):
        """Client reads API key from environment variable."""
        with patch.dict(os.environ, {"EVALOPS_API_KEY": "env-key"}):
            client = EvalOpsClient()
            assert client.api_key == "env-key"
            await client.close()

    async def test_client_initialization_missing_api_key(self):
        """Client raises ValueError when no API key is available."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                EvalOpsClient()

    @respx.mock
    async def test_create_test_run_success(self):
        """Successfully create a test run via API."""
        mock_response = {
            "data": {
                "id": "test-run-123",
                "testSuiteId": "suite-456",
                "status": "pending",
            },
            "jobId": "job-789",
        }

        route = respx.post("https://test.evalops.dev/api/v1/test-runs/").mock(
            return_value=Response(200, json=mock_response)
        )

        async with EvalOpsClient(api_key="test-key", api_url="https://test.evalops.dev") as client:
            result = await client.create_test_run(
                test_suite_id="suite-456",
                name="Test Run",
                config={"model": "gpt-4"},
            )

        assert route.called
        assert result["data"]["id"] == "test-run-123"
        request_body = route.calls.last.request.content
        assert b'"testSuiteId":"suite-456"' in request_body

    @respx.mock
    async def test_get_test_run_success(self):
        """Successfully retrieve a test run."""
        mock_response = {
            "data": {
                "id": "test-run-123",
                "status": "completed",
                "summary": {"passed": 8, "failed": 2},
            }
        }

        respx.get("https://test.evalops.dev/api/v1/test-runs/test-run-123").mock(
            return_value=Response(200, json=mock_response)
        )

        async with EvalOpsClient(api_key="test-key", api_url="https://test.evalops.dev") as client:
            result = await client.get_test_run("test-run-123")

        assert result["data"]["status"] == "completed"

    @respx.mock
    async def test_wait_for_test_run_completes(self):
        """Wait for test run polls until completion."""
        respx.get("https://test.evalops.dev/api/v1/test-runs/test-run-123").mock(
            side_effect=[
                Response(200, json={"data": {"id": "test-run-123", "status": "pending"}}),
                Response(200, json={"data": {"id": "test-run-123", "status": "running"}}),
                Response(200, json={"data": {"id": "test-run-123", "status": "completed"}}),
            ]
        )

        async with EvalOpsClient(api_key="test-key", api_url="https://test.evalops.dev") as client:
            result = await client.wait_for_test_run("test-run-123", poll_interval=0.01)

        assert result["data"]["status"] == "completed"

    @respx.mock
    async def test_submit_training_results(self):
        """Submit training results creates test run with correct metadata."""
        mock_response = {
            "data": {"id": "test-run-999", "status": "pending"},
            "jobId": "job-111",
        }

        route = respx.post("https://test.evalops.dev/api/v1/test-runs/").mock(
            return_value=Response(200, json=mock_response)
        )

        async with EvalOpsClient(api_key="test-key", api_url="https://test.evalops.dev") as client:
            result = await client.submit_training_results(
                test_suite_id="suite-123",
                round_number=2,
                model_checkpoint="tinker://checkpoint-uri",
                metrics={"accuracy": 0.85, "f1": 0.82},
                metadata={"base_model": "llama-8b"},
            )

        assert route.called
        request_body = route.calls.last.request.content
        assert b'"testSuiteId":"suite-123"' in request_body
        assert b'"round":2' in request_body
        assert b'"accuracy":0.85' in request_body
        assert b'"Training Round 2"' in request_body
        assert b'"tinker"' in request_body

    @respx.mock
    async def test_api_error_handling(self):
        """Client raises exception on API errors."""
        respx.post("https://test.evalops.dev/api/v1/test-runs/").mock(
            return_value=Response(500, json={"error": "Internal server error"})
        )

        async with EvalOpsClient(api_key="test-key", api_url="https://test.evalops.dev") as client:
            with pytest.raises(Exception):
                await client.create_test_run("suite-123")

    async def test_context_manager(self):
        """Client can be used as async context manager."""
        async with EvalOpsClient(api_key="test-key") as client:
            assert client.client is not None
