from typing import Any

from pydantic import BaseModel, Field


class PredictionResult(BaseModel):
    attributes: dict[str, Any] = Field(default_factory=dict)
    unknown_attributes: dict[str, Any] = Field(default_factory=dict)
    message: str
    confidence: float
    model_version: str
    parser_strategy: str = "ml_token_classifier"
