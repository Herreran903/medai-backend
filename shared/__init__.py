"""Shared module for MedAI microservices."""

from shared.schemas import (
    Entity,
    ErrorDetail,
    ErrorResponse,
    NERConfig,
    NERRequest,
    NERResponse,
)

__all__ = [
    "Entity",
    "ErrorDetail",
    "ErrorResponse",
    "NERConfig",
    "NERRequest",
    "NERResponse",
]

__version__ = "1.0.0"
