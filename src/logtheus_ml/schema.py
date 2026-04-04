from typing import Any

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    attributes: dict[str, Any] = Field(default_factory=dict)
    low_confidence_attributes: dict[str, Any] = Field(default_factory=dict)
    attribute_confidence: dict[str, Any] = Field(default_factory=dict)
    message: str
    confidence: float
    model_version: str
