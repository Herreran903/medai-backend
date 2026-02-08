"""
Shared Pydantic schemas for MedAI microservices.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """Clinical entity extracted from medical text."""

    type: str = Field(..., description="Entity type (e.g., FIO2, PEEP, DX)")
    text: str = Field(..., description="Exact text span extracted")
    start: Optional[int] = Field(default=None, ge=0, description="Start character offset")
    end: Optional[int] = Field(default=None, ge=0, description="End character offset")
    code: Optional[str] = Field(
        default=None,
        description="Normalized value extracted from the entity text (if available)",
    )


# ==============================================================================
# NER Service Request/Response Schemas
# ==============================================================================


class NERConfig(BaseModel):
    """Configuration options for NER extraction."""

    model_variant: Optional[str] = Field(default=None, description="Model variant")
    normalize: bool = Field(default=False, description="Apply UMLS normalization")
    systems: List[str] = Field(default_factory=list, description="Target coding systems")
    restrict_types: List[str] = Field(default_factory=list, description="Entity types filter")


class NERRequest(BaseModel):
    """Request payload for NER extraction."""

    text: str = Field(..., description="Clinical text to process", min_length=1)
    config: Optional[NERConfig] = Field(default_factory=NERConfig)


class NERResponse(BaseModel):
    """Response payload from NER extraction."""

    entities: List[Entity] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Error Schemas
# ==============================================================================


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None)


class ErrorResponse(BaseModel):
    """Error response payload."""

    error: ErrorDetail
