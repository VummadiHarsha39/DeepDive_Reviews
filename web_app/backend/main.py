from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from typing import List

# Import your schemas for request/response validation
from .schemas import ReviewInput, BatchReviewInput, AnalysisResult, BatchAnalysisResponse

# Import your core prediction service
from src.services.prediction_service import ReviewAnalysisService

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Emotional Counterfeit Detector API",
    description="API for detecting fake reviews, classifying emotions, and summarizing customer intent.",
    version="0.1.0",
)

# --- CORS Middleware ---
# This is crucial for allowing your frontend (running on a different domain/port) to access your backend.
# For local development, "*" is fine. For actual deployment, replace with specific frontend URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- Global Variable for Prediction Service ---
# This service will be initialized once when the FastAPI app starts up.
review_analysis_service: ReviewAnalysisService = None

# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """
    Event handler that runs when the FastAPI application starts up.
    Used to load the ML models into memory once.
    """
    global review_analysis_service
    print("FastAPI application startup: Initializing ReviewAnalysisService (loading ML models)...")
    try:
        # Set MLflow tracking URI for model loading (ensure it matches your running mlflow ui)
        os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000" 
        
        # Instantiate the service with explicit stage for DeceptionDetector
        # This parameter controls which stage to look for in MLflow Registry.
        review_analysis_service = ReviewAnalysisService(deception_model_stage="Production") # CORRECTED TO "PRODUCTION"
        
        print("ReviewAnalysisService initialized. Models loaded.")
    except Exception as e:
        print(f"ERROR: Failed to initialize ReviewAnalysisService during startup: {e}")
        # Raise an HTTPException to fail startup clearly
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service not ready: {e}"
        )

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    if review_analysis_service:
        return {"status": "ok", "message": "Service is ready"}
    else:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Service not initialized")

# --- Single Review Analysis Endpoint ---
@app.post("/analyze_review", response_model=AnalysisResult)
async def analyze_single_review(review_input: ReviewInput):
    """
    Analyzes a single review text for authenticity, emotions, and provides explanations.
    """
    if not review_analysis_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not loaded yet.")

    try:
        result = review_analysis_service.analyze_review(review_input.review_text)
        return result
    except Exception as e:
        print(f"Error analyzing single review: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Analysis failed: {e}")

# --- Batch Review Analysis Endpoint ---
@app.post("/batch_analyze_reviews", response_model=BatchAnalysisResponse)
async def analyze_batch_reviews(batch_input: BatchReviewInput):
    """
    Analyzes multiple review texts for authenticity, emotions, and provides explanations.
    """
    if not review_analysis_service:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Models not loaded yet.")

    results: List[AnalysisResult] = []
    for review_text in batch_input.reviews:
        try:
            result = review_analysis_service.analyze_review(review_text)
            results.append(result)
        except Exception as e:
            # For batch, log error but continue with other reviews
            print(f"Error analyzing review in batch '{review_text[:50]}...': {e}")
            results.append(AnalysisResult(
                verdict="unknown",
                confidence=0.0,
                emotions={},
                explanation=f"Error: {e}",
                customer_intent_summary="Analysis failed."
            ))
    
    return {"results": results}

# --- For local development ---
if __name__ == "__main__":
    # Ensure MLflow tracking URI is set for local testing
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000" 
    
    print("Starting FastAPI server locally...")
    # --reload enables hot-reloading (restarts server on code changes)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)