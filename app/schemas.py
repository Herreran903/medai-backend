# app/schemas.py
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Code(BaseModel):
    system: str
    code: str
    display: Optional[str] = None
    score: Optional[float] = None
    source: Optional[str] = None


class Entity(BaseModel):
    type: str
    text: str
    score: Optional[float] = None
    code: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None
    codes: List[Code] = Field(default_factory=list)


class ExtractResponse(BaseModel):
    text: str
    entities: List[Entity] = Field(default_factory=list)
    meta: Dict[str, Any] = {}


class BatchItem(BaseModel):
    filename: str
    entities: List[Entity] = Field(default_factory=list)
    meta: Dict[str, Any] = {}
