"""Validated response schemas for evidence-grounded answers."""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str = Field(min_length=1)
    snippet: str = Field(min_length=1)
    source_id: str | None = None


class Answer(BaseModel):
    text: str = Field(min_length=1)
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
