from typing import Any, Dict, List, Literal, Optional

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
    meta: Dict[str, Any] = Field(default_factory=dict)


class BatchItem(BaseModel):
    filename: str
    entities: List[Entity] = Field(default_factory=list)
    meta: Dict[str, Any] = Field(default_factory=dict)


# -------- NUEVOS (para ACKs) --------


class ExtractAck(BaseModel):
    id: str
    stored: bool
    url: Optional[str] = None
    filename: Optional[str] = None
    episode_id: Optional[str] = None
    note_date: Optional[str] = None
    entity_count: Optional[int] = None
    result: Optional[ExtractResponse] = None


class BatchAckItem(BaseModel):
    filename: str
    id: Optional[str] = None
    stored: bool
    entity_count: Optional[int] = None
    url: Optional[str] = None
    error: Optional[str] = None


class BatchAckResponse(BaseModel):
    items: List[BatchAckItem]
