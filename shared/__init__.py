"""Shared module for MedAI microservices."""

from shared.schemas import (
    Code,
    Entity,
    ErrorDetail,
    ErrorResponse,
    NERConfig,
    NERRequest,
    NERResponse,
)

__all__ = [
    "Code",
    "Entity",
    "ErrorDetail",
    "ErrorResponse",
    "NERConfig",
    "NERRequest",
    "NERResponse",
]

__version__ = "1.0.0"
