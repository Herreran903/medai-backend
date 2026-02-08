"""
HTTP Client for NER Microservices Communication.

This module provides an async HTTP client for the API gateway to communicate
with NER microservices (Transformer, BiLSTM, LLM). It implements:

- Request/response serialization using shared schemas
- Automatic retry with exponential backoff
- Error handling and meaningful exceptions
- Health check validation
- Connection pooling for performance

Architecture Context:
    This client is ONLY used when NER_MODE='remote' in the gateway configuration.
    It routes extraction requests from the gateway to the appropriate NER service
    based on the model parameter, maintaining the same interface as local extraction.

Usage:
    >>> from app.services.ner_client import NERClient
    >>> from app.config import get_settings
    >>>
    >>> async with NERClient(get_settings()) as client:
    ...     result = await client.predict(
    ...         model="lstm",
    ...         text="Paciente con FiO2 60%"
    ...     )
    >>> print(result["entities"])
    [{"type": "FIO2", "text": "FiO2 60%", "start": 13, "end": 21, "code": None}]

See Also:
    - :mod:`app.services.pipeline` for integration with extraction pipeline
    - :mod:`app.config` for NER service URL configuration
    - :mod:`shared.schemas` for request/response contracts
"""

import asyncio
import logging
from typing import Dict, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import Settings

logger = logging.getLogger(__name__)


# ==============================================================================
# Custom Exceptions
# ==============================================================================


class NERServiceError(Exception):
    """Base exception for NER service communication errors."""

    pass


class NERServiceUnavailableError(NERServiceError):
    """Raised when NER service is unreachable (network error, service down)."""

    pass


class NERServiceTimeoutError(NERServiceError):
    """Raised when NER service request exceeds timeout."""

    pass


class NERServiceResponseError(NERServiceError):
    """Raised when NER service returns an error response (4xx, 5xx)."""

    pass


# ==============================================================================
# NER HTTP Client
# ==============================================================================


class NERClient:
    """
    Async HTTP client for NER microservices communication.

    This client provides a high-level interface for the gateway to interact
    with NER microservices, handling all HTTP communication details including
    retries, timeouts, and error handling.

    Features:
        - Automatic service URL resolution based on model name
        - Exponential backoff retry for transient failures
        - Connection pooling for efficient HTTP reuse
        - Context manager support for proper resource cleanup
        - Structured error responses

    Attributes:
        settings: Application settings with service URLs and timeouts
        _client: Underlying httpx AsyncClient instance (initialized in __aenter__)

    Example:
        >>> async with NERClient(settings) as client:
        ...     # Check service health
        ...     is_ready = await client.health_check("transformer")
        ...
        ...     # Extract entities
        ...     result = await client.predict(
        ...         model="transformer",
        ...         text="Paciente con FiO2 60%",
        ...         model_variant="roberta"
        ...     )
    """

    def __init__(self, settings: Settings):
        """
        Initialize NER client with application settings.

        Args:
            settings: Application settings containing service URLs and timeouts
        """
        self.settings = settings
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """
        Context manager entry: initialize HTTP client.

        Creates an httpx AsyncClient with configured timeout and connection limits.
        Connection pooling allows efficient reuse of HTTP connections across requests.

        Returns:
            NERClient: Self for context manager usage
        """
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.settings.ner_request_timeout),
            limits=httpx.Limits(
                max_connections=100,  # Total connection pool size
                max_keepalive_connections=20,  # Persistent connections
            ),
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit: close HTTP client and cleanup resources.

        Ensures all connections are properly closed, even in case of exceptions.
        """
        if self._client:
            await self._client.aclose()

    def _get_service_url(self, model: str) -> str:
        """
        Map model name to NER service URL.

        Args:
            model: Model identifier (transformer, lstm, llm)

        Returns:
            str: Service base URL (e.g., "http://ner-transformer:8001")

        Raises:
            ValueError: If model identifier is not recognized
        """
        url_map = {
            "transformer": self.settings.ner_transformer_url,
            "lstm": self.settings.ner_lstm_url,
            "llm": self.settings.ner_llm_url,
        }

        if model not in url_map:
            raise ValueError(
                f"Unknown model: {model}. Valid models: {list(url_map.keys())}"
            )

        return url_map[model]

    @retry(
        stop=stop_after_attempt(3),  # Max 3 attempts (configurable via settings)
        wait=wait_exponential(multiplier=1, min=2, max=10),  # 2s, 4s, 8s backoff
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError)
        ),  # Only retry transient errors
        reraise=True,  # Re-raise exception after all retries exhausted
    )
    async def predict(
        self,
        model: str,
        text: str,
        *,
        model_variant: Optional[str] = None,
        normalize: bool = False,
        systems: Optional[List[str]] = None,
        restrict_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Call NER service /predict endpoint with automatic retry.

        Sends an extraction request to the appropriate NER microservice and
        returns the structured entities. Automatically retries on timeout or
        network errors with exponential backoff.

        Args:
            model: Model identifier (transformer, lstm, llm)
            text: Input clinical text to process
            model_variant: Optional model variant (roberta, gpt; both fixed)
            normalize: Whether to apply UMLS normalization
            systems: Target coding systems for normalization
            restrict_types: Entity types to include (empty = all types)

        Returns:
            Dict with "entities" (List[Dict]) and "meta" (Dict) keys

        Raises:
            NERServiceUnavailableError: Service not reachable after retries
            NERServiceTimeoutError: Request timeout after retries
            NERServiceResponseError: Service returned error response (4xx, 5xx)
            RuntimeError: Client not initialized (use as context manager)

        Example:
            >>> result = await client.predict(
            ...     model="transformer",
            ...     text="Paciente con FiO2 60%",
            ...     model_variant="roberta"
            ... )
            >>> print(len(result["entities"]))
            2
        """
        if not self._client:
            raise RuntimeError(
                "NERClient not initialized. Use as async context manager: "
                "async with NERClient(settings) as client: ..."
            )

        service_url = self._get_service_url(model)
        endpoint = f"{service_url}/predict"

        # Build request payload (shared.schemas.NERRequest format)
        payload = {
            "text": text,
            "config": {
                "model_variant": model_variant,
                "normalize": normalize,
                "systems": systems or [],
                "restrict_types": restrict_types or [],
            },
        }

        try:
            logger.debug(
                f"Calling NER service: {endpoint} (text_length={len(text)}, variant={model_variant})"
            )
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()

            result = response.json()
            logger.debug(
                f"NER service response: {len(result.get('entities', []))} entities"
            )
            return result

        except httpx.TimeoutException as e:
            logger.error(
                f"NER service timeout: {model} at {service_url} (timeout={self.settings.ner_request_timeout}s)"
            )
            raise NERServiceTimeoutError(
                f"Request to {model} service timed out after {self.settings.ner_request_timeout}s"
            ) from e

        except httpx.NetworkError as e:
            logger.error(
                f"NER service network error: {model} at {service_url} - {type(e).__name__}: {e}"
            )
            raise NERServiceUnavailableError(
                f"Cannot reach {model} service at {service_url}"
            ) from e

        except httpx.HTTPStatusError as e:
            logger.error(
                f"NER service error: {model} returned {e.response.status_code} - {e.response.text[:200]}"
            )

            # Try to extract error details from response
            try:
                error_body = e.response.json()
                error_detail = error_body.get("error", {}).get("message", str(e))
            except Exception:
                error_detail = e.response.text[:200]

            raise NERServiceResponseError(
                f"{model} service error ({e.response.status_code}): {error_detail}"
            ) from e

    async def health_check(self, model: str) -> bool:
        """
        Check if NER service is healthy and ready to accept requests.

        Calls the service's /readyz endpoint to verify:
        - Service is running
        - Model is loaded
        - Service is ready to process requests

        Args:
            model: Model identifier (transformer, lstm, llm)

        Returns:
            bool: True if service is healthy and ready, False otherwise

        Example:
            >>> is_ready = await client.health_check("transformer")
            >>> if is_ready:
            ...     result = await client.predict(model="transformer", text="...")
        """
        if not self._client:
            logger.warning("Health check called but client not initialized")
            return False

        try:
            service_url = self._get_service_url(model)
            endpoint = f"{service_url}/readyz"

            response = await self._client.get(endpoint, timeout=5.0)
            is_healthy = response.status_code == 200

            if is_healthy:
                logger.debug(f"{model} service is healthy")
            else:
                logger.warning(
                    f"{model} service not ready: {response.status_code} - {response.text[:100]}"
                )

            return is_healthy

        except Exception as e:
            logger.warning(f"Health check failed for {model}: {type(e).__name__}: {e}")
            return False
