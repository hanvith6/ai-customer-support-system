"""
Structured response schemas for the AI Customer Support System.

Inspired by the travel-planner-ai project (ai-plan--main.zip) which uses
structured JSON schemas to enforce consistent AI response formats via
OpenAI function calling. This module applies the same pattern using
Pydantic models to validate and structure chatbot pipeline output.

These schemas serve as the contract between the NLP pipeline and the API
layer, ensuring every response conforms to a predictable structure.
"""

from pydantic import BaseModel, Field


class EntitySchema(BaseModel):
    """Schema for a single named entity extracted from user input."""
    entity: str = Field(description="Entity label (e.g. PER, ORG, LOC, MISC)")
    word: str = Field(description="The matched text span")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")
    start: int = Field(ge=0, description="Character start offset")
    end: int = Field(ge=0, description="Character end offset")


class SentimentSchema(BaseModel):
    """Schema for sentiment analysis result."""
    label: str = Field(description="Sentiment label: POSITIVE, NEGATIVE, or UNKNOWN")
    score: float = Field(ge=0.0, le=1.0, description="Confidence score")


class ChatResponseSchema(BaseModel):
    """
    Structured response schema for the /chat endpoint.

    Enforces a consistent JSON contract similar to how the travel-planner-ai
    project uses OpenAI function schemas to guarantee structured output.
    All NLP pipeline results are validated through this schema before
    being returned to the client.
    """
    intent: str = Field(description="Classified intent tag")
    confidence: float = Field(ge=0.0, le=1.0, description="Intent classification confidence")
    entities: list[EntitySchema] = Field(default_factory=list, description="Extracted named entities")
    sentiment: SentimentSchema = Field(description="User message sentiment")
    response: str = Field(min_length=1, description="Bot response text")


class ChatRequestSchema(BaseModel):
    """Schema for incoming chat requests with validation."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User message text",
    )
