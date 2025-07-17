import os
import numpy as np
import pandas as pd
import torch
import mlflow
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel 
from mlflow.transformers import load_model as load_hf_model # To load Hugging Face models from MLflow
from mlflow.pytorch import load_model as load_pytorch_model # To load custom PyTorch models
from torch.utils.data import DataLoader, TensorDataset
from src.data.preprocessor import clean_text
from src.models.deception_detector import DeceptionModel # Import your custom DeceptionModel class
from src.explanation.explanation_generator import generate_explanation, analyze_emotional_patterns # Import explanation logic
from typing import Dict, Any, List

# All cloud-related imports (fsspec, gcsfs, tempfile, shutil) have been removed from this local-only version.

class ReviewAnalysisService:
    """
    Service to perform end-to-end analysis of a review:
    1. Clean text.
    2. Classify emotions.
    3. Detect real/fake.
    4. Generate explanations and customer intent summary.
    """
    def __init__(self, 
                 mlflow_tracking_uri: str = "http://127.0.0.1:5000",
                 emotion_model_name: str = "EmotionClassifier",
                 deception_model_name: str = "DeceptionDetector",
                 emotion_model_stage: str = "Production", 
                 deception_model_stage: str = "Production", # CORRECTED: This is now explicitly "Production"
                 max_len: int = 128
                 ):
        
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.emotion_model_name = emotion_model_name
        self.deception_model_name = deception_model_name
        self.emotion_model_stage = emotion_model_stage
        self.deception_model_stage = deception_model_stage
        self.max_len = max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emotion_model = None
        self.emotion_tokenizer = None
        self.deception_model = None
        self.deception_tokenizer = None
        self.emotion_labels = []

        # For pure local operation, models are always loaded from local MLflow registry
        mlflow.set_tracking_uri(self.mlflow_tracking_uri) 
        self._load_models() 

    def _load_models(self):
        """Loads Emotion and Deception models from local MLflow registry."""
        
        print("Attempting to load models from local MLflow registry...")
        try:
            # Load Emotion Classifier from local MLflow
            emotion_model_uri = f"models:/{self.emotion_model_name}/{self.emotion_model_stage}"
            loaded_emotion_mlflow_model = load_hf_model(emotion_model_uri)
            self.emotion_model = loaded_emotion_mlflow_model.model
            self.emotion_tokenizer = loaded_emotion_mlflow_model.tokenizer
            self.emotion_model.to(self.device)
            self.emotion_model.eval()
            print("Emotion Classifier loaded successfully from local MLflow registry.")

            # Load Deception Detector (custom PyTorch model) from local MLflow
            deception_model_uri = f"models:/{self.deception_model_name}/{self.deception_model_stage}"
            self.deception_model = load_pytorch_model(deception_model_uri)
            self.deception_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.deception_model.to(self.device)
            self.deception_model.eval()
            print("Deception Detector loaded successfully from local MLflow registry.")
            
            self.emotion_labels = [ # Ensure emotion_labels are set
                "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
                "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
                "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "neutral",
                "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"
            ]
            
        except Exception as e:
            print(f"FATAL ERROR: Failed to load ML models from local MLflow registry: {e}")
            raise Exception(f"Service failed to start: Could not load ML models from local MLflow. Error: {e}")

    def _get_emotion_probabilities(self, text: str) -> np.ndarray:
        """Helper to get emotion probabilities for a single text."""
        cleaned_text = clean_text(text)
        if not cleaned_text:
            return np.zeros(len(self.emotion_labels))

        encoding = self.emotion_tokenizer.encode_plus(
            cleaned_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Move inputs to the same device as the model
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.emotion_model(input_ids=input_ids, attention_mask=attention_mask)
        
        emotion_probs = torch.sigmoid(outputs.logits).cpu().numpy().flatten()
        torch.cuda.empty_cache() # Clear GPU memory after emotion inference
        return emotion_probs

    def analyze_review(self, review_text: str) -> Dict[str, Any]:
        """
        Performs full analysis of a review.
        """
        # 1. Clean the input text
        cleaned_review_text = clean_text(review_text)
        if not cleaned_review_text:
            return {
                "verdict": "unknown",
                "confidence": 0.0,
                "emotions": {label: 0.0 for label in self.emotion_labels},
                "explanation": "Review text is empty or unreadable after cleaning.",
                "customer_intent_summary": "No discernible intent due to empty text."
            }

        # 2. Get Emotion Probabilities
        emotion_probs_array = self._get_emotion_probabilities(cleaned_review_text)
        emotion_output = {label: float(prob) for label, prob in zip(self.emotion_labels, emotion_probs_array)}

        # 3. Prepare for Deception Detection
        # The text for deception detection is the cleaned text
        deception_encoding = self.deception_tokenizer.encode_plus(
            cleaned_review_text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Move inputs to the same device as the model
        deception_input_ids = deception_encoding['input_ids'].to(self.device)
        deception_attention_mask = deception_encoding['attention_mask'].to(self.device)
        emotion_features_tensor = torch.tensor([emotion_probs_array], dtype=torch.float).to(self.device) # Ensure 2D tensor

        # 4. Get Deception Prediction
        with torch.no_grad():
            outputs = self.deception_model(
                input_ids=deception_input_ids,
                attention_mask=deception_attention_mask,
                emotion_features=emotion_features_tensor
            )
            logits = outputs["logits"] # Access logits from the returned dict
            verdict_prob = torch.sigmoid(logits).item()

        verdict = "fake" if verdict_prob > 0.5 else "real"
        confidence = verdict_prob if verdict == "fake" else (1 - verdict_prob)

        # 5. Generate Explanation and Customer Intent Summary
        explanation_output = generate_explanation(
            review_text=review_text, # Pass original text for better context in explanation
            verdict=verdict,
            confidence=confidence,
            emotion_probs=emotion_probs_array,
            emotion_names=self.emotion_labels
        )
        
        torch.cuda.empty_cache() # Clear GPU memory after full analysis
        return {
            "verdict": verdict,
            "confidence": confidence,
            "emotions": emotion_output,
            "explanation": explanation_output['explanation'],
            "customer_intent_summary": explanation_output['customer_intent_summary']
        }

# Example usage (this part will only run if you execute prediction_service.py directly)
if __name__ == "__main__":
    # Ensure MLflow tracking URI is set for local testing
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:5000" 

    print("Initializing Review Analysis Service...")
    try:
        # For pure local testing, GCS_BUCKET_NAME is not needed, models load from local MLflow
        service = ReviewAnalysisService(deception_model_stage="Production") 
        print("Service initialized successfully.")

        # Test Review 1: Example of a potentially fake, overly positive review
        review1 = "This product is absolutely amazing! Best purchase ever! I love love love it! Highly recommend to everyone."
        print(f"\n--- Analyzing Review 1 ---")
        print(f"Review: '{review1}'")
        result1 = service.analyze_review(review1)
        print(f"Verdict: {result1['verdict']} (Confidence: {result1['confidence']:.2f})")
        print(f"Emotions: {', '.join([f'{k}: {v:.2f}' for k,v in result1['emotions'].items() if v > 0.1])}")
        print(f"Explanation: {result1['explanation']}")
        print(f"Summary: {result1['customer_intent_summary']}")

        # Test Review 2: Example of a genuine, critical review
        review2 = "The service was really slow and disappointing. The food arrived cold. I am quite upset with the experience."
        print(f"\n--- Analyzing Review 2 ---")
        print(f"Review: '{review2}'")
        result2 = service.analyze_review(review2)
        print(f"Verdict: {result2['verdict']} (Confidence: {result2['confidence']:.2f})")
        print(f"Emotions: {', '.join([f'{k}: {v:.2f}' for k,v in result2['emotions'].items() if v > 0.1])}")
        print(f"Explanation: {result2['explanation']}")
        print(f"Summary: {result2['customer_intent_summary']}")

        # Test Review 3: Genuine, neutral review
        review3 = "Product arrived on time. Functions as described in the manual. No issues encountered so far."
        print(f"\n--- Analyzing Review 3 ---")
        print(f"Review: '{review3}'")
        result3 = service.analyze_review(review3)
        print(f"Verdict: {result3['verdict']} (Confidence: {result3['confidence']:.2f})")
        print(f"Emotions: {', '.join([f'{k}: {v:.2f}' for k,v in result3['emotions'].items() if v > 0.1])}")
        print(f"Explanation: {result3['explanation']}")
        print(f"Summary: {result3['customer_intent_summary']}")

    except Exception as e:
        print(f"Error during service initialization or analysis: {e}")
        print("Please ensure MLflow server is running and models are registered/staged correctly.")