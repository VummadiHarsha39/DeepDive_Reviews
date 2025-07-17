from pydantic import BaseModel, RootModel # Import RootModel
from typing import Dict, List, Literal, Optional

# --- Request Schemas ---
class ReviewInput(BaseModel):
    """Schema for a single review text input."""
    review_text: str

class BatchReviewInput(BaseModel):
    """Schema for a batch of review texts input."""
    reviews: List[str]

# --- Response Schemas ---
# For Pydantic v2+, use RootModel for single-value models like this
class EmotionOutput(RootModel[Dict[str, float]]):
    """Schema for the emotion classification results. Represents a dictionary of emotion scores."""
    pass # No need for __root__ field in RootModel

class AnalysisResult(BaseModel):
    """Schema for the full analysis result of a single review."""
    verdict: Literal["real", "fake", "unknown"] # Use Literal for specific string values
    confidence: float # Probability score for the verdict
    emotions: EmotionOutput # Nested emotion results
    explanation: str # Human-readable explanation for the verdict
    customer_intent_summary: str # 1-2 line summary of customer's main point

class BatchAnalysisResponse(BaseModel):
    """Schema for the response of a batch analysis."""
    results: List[AnalysisResult] # List of individual analysis results