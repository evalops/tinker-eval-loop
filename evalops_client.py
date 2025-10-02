"""
EvalOps API Client for submitting evaluation results.

This module provides a Python SDK wrapper around the EvalOps REST API,
specifically designed to submit test runs and evaluation results from
automated fine-tuning pipelines.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx


class EvalOpsClient:
    """Client for interacting with the EvalOps evaluation platform."""

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize the EvalOps client.

        Args:
            api_url: Base URL of the EvalOps API. Defaults to EVALOPS_API_URL env var
                or https://api.evalops.dev.
            api_key: API key for authentication. Defaults to EVALOPS_API_KEY env var.
            timeout: Request timeout in seconds. Default 300s for long-running evaluations.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        self.base_url = (
            api_url
            or os.getenv("EVALOPS_API_URL")
            or "https://api.evalops.dev"
        )
        self.api_key = api_key or os.getenv("EVALOPS_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key required. Set EVALOPS_API_KEY environment variable or pass api_key parameter."
            )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def create_test_run(
        self,
        test_suite_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        visibility: str = "team",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new test run in EvalOps.

        Args:
            test_suite_id: ID of the test suite to execute.
            name: Optional name for the test run.
            description: Optional description.
            config: Configuration override (model, provider, temperature, etc.).
            visibility: Visibility level ("private", "team", "organization").
            tags: Optional list of tags for categorization.

        Returns:
            Response containing test run details and job ID.

        Raises:
            httpx.HTTPStatusError: If the API request fails.
        """
        payload = {
            "testSuiteId": test_suite_id,
            "visibility": visibility,
            "sharedWithTeams": [],
        }

        if name:
            payload["name"] = name
        if description:
            payload["description"] = description
        if config:
            payload["config"] = config
        if tags:
            payload["tags"] = tags

        response = await self.client.post("/api/v1/test-runs/", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_test_run(self, test_run_id: str) -> Dict[str, Any]:
        """
        Retrieve a test run by ID.

        Args:
            test_run_id: ID of the test run to retrieve.

        Returns:
            Test run details including status and results.
        """
        response = await self.client.get(f"/api/v1/test-runs/{test_run_id}")
        response.raise_for_status()
        return response.json()

    async def wait_for_test_run(
        self,
        test_run_id: str,
        poll_interval: float = 5.0,
        max_wait: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Poll the test run until it completes or fails.

        Args:
            test_run_id: ID of the test run to monitor.
            poll_interval: Seconds between poll requests.
            max_wait: Maximum seconds to wait before timing out. None = wait forever.

        Returns:
            Completed test run with final results.

        Raises:
            TimeoutError: If max_wait is exceeded.
            httpx.HTTPStatusError: If polling fails.
        """
        import asyncio

        start_time = datetime.now()

        while True:
            run = await self.get_test_run(test_run_id)
            data = run.get("data", {})
            status = data.get("status")

            if status in ["completed", "failed"]:
                return run

            if max_wait:
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed >= max_wait:
                    raise TimeoutError(
                        f"Test run did not complete within {max_wait}s"
                    )

            await asyncio.sleep(poll_interval)

    async def submit_training_results(
        self,
        test_suite_id: str,
        round_number: int,
        model_checkpoint: str,
        metrics: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Submit training round results to EvalOps as a test run.

        Args:
            test_suite_id: ID of the evaluation test suite.
            round_number: Current training round number.
            model_checkpoint: Path or URI to the model checkpoint.
            metrics: Evaluation metrics (e.g., {"accuracy": 0.85, "f1": 0.82}).
            metadata: Additional metadata about the training run.

        Returns:
            Created test run response.
        """
        run_name = f"Training Round {round_number}"
        run_metadata = {
            "source": "tinker-eval-loop",
            "checkpoint": model_checkpoint,
            "round": round_number,
            "metrics": metrics,
            **(metadata or {}),
        }

        config = {
            "model": model_checkpoint,
            "metadata": run_metadata,
        }

        return await self.create_test_run(
            test_suite_id=test_suite_id,
            name=run_name,
            description=f"Automated evaluation from Tinker fine-tuning round {round_number}",
            config=config,
            tags=["tinker", "automated", f"round-{round_number}"],
        )
