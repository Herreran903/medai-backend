from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    type: str
    text: str
    score: Optional[float] = None
    code: Optional[str] = None
    start: Optional[int] = None
    end: Optional[int] = None


class ExtractResponse(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    meta: Dict[str, Any] = {}


class BatchItem(BaseModel):
    filename: str
    entities: List[Entity] = Field(default_factory=list)
    meta: Dict[str, Any] = {}
