import numpy as np
import re
from typing import Dict, List

def analyze_emotional_patterns(emotion_probs: np.ndarray, emotion_names: List[str], review_text: str) -> Dict[str, any]:
    """
    Analyzes emotional patterns for cues of potential fakery or typical emotional expression.
    """
    analysis = {}

    # 1. Dominant Emotion
    if len(emotion_probs) > 0 and len(emotion_names) == len(emotion_probs):
        dominant_emotion_idx = np.argmax(emotion_probs)
        analysis['dominant_emotion'] = emotion_names[dominant_emotion_idx]
        analysis['dominant_emotion_score'] = emotion_probs[dominant_emotion_idx]
    else:
        analysis['dominant_emotion'] = 'none'
        analysis['dominant_emotion_score'] = 0.0

    # 2. Emotional Diversity (using Shannon Entropy)
    # Add a small epsilon to avoid log(0) if any probability is zero
    epsilon = 1e-9
    normalized_probs = (emotion_probs + epsilon) / (emotion_probs.sum() + len(emotion_probs) * epsilon)
    analysis['emotional_diversity'] = -np.sum(normalized_probs * np.log2(normalized_probs))
    
    # 3. Exaggeration/Flatness Score (simple heuristics)
    # High score for one emotion, low diversity
    if analysis['dominant_emotion_score'] > 0.8 and analysis['emotional_diversity'] < 1.5: # Thresholds can be fine-tuned
        analysis['is_emotionally_exaggerated_or_flat'] = True
    else:
        analysis['is_emotionally_exaggerated_or_flat'] = False

    # 4. Repetitive Language Check (simple heuristic)
    analysis['is_repetitive'] = has_repetitive_phrasing(review_text)

    return analysis

def has_repetitive_phrasing(text: str, ngram_range: tuple = (2,3), threshold: float = 0.05) -> bool:
    """
    Checks for repetitive phrasing using n-grams.
    A text is considered repetitive if the most frequent n-gram (within range)
    appears more often than a certain 'threshold' percentage of all n-grams.
    """
    from collections import Counter

    words = text.lower().split()
    if len(words) < ngram_range[0]:
        return False # Too short to form ngrams

    all_ngrams = []
    for n in range(ngram_range[0], ngram_range[1] + 1):
        ngrams = [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    
    if not all_ngrams:
        return False

    ngram_counts = Counter(all_ngrams)
    if not ngram_counts:
        return False
        
    most_common_ngram_count = ngram_counts.most_common(1)[0][1]
    
    # Check if the most common n-gram constitutes a high percentage of all n-grams
    if most_common_ngram_count / len(all_ngrams) > threshold:
        return True
    return False


def generate_explanation(
    review_text: str,
    verdict: str, # "real" or "fake"
    confidence: float, # 0.0 to 1.0
    emotion_probs: np.ndarray,
    emotion_names: List[str],
    accuracy_metric_for_fake: float = 0.5 # A baseline F1 or Recall for 'fake' from Deception Detector
) -> Dict[str, str]:
    """
    Generates a human-readable explanation for the real/fake verdict
    and a summary of customer intent.
    """
    explanation_parts = []
    summary_parts = []

    # --- 1. Emotional Pattern Analysis ---
    emotion_analysis = analyze_emotional_patterns(emotion_probs, emotion_names, review_text)
    
    dominant_emotion = emotion_analysis['dominant_emotion']
    dominant_emotion_score = emotion_analysis['dominant_emotion_score']
    emotional_diversity = emotion_analysis['emotional_diversity']
    is_emotionally_exaggerated_or_flat = emotion_analysis['is_emotionally_exaggerated_or_flat']
    is_repetitive = emotion_analysis['is_repetitive']

    # --- 2. Verdict Explanation ---
    if verdict == 'fake':
        explanation_parts.append(f"This review is **likely fake** with {confidence:.1%} confidence.")
        # Rules for 'fake' explanation
        if is_emotionally_exaggerated_or_flat:
            explanation_parts.append(
                f"A key suspicious sign is the **overwhelming {dominant_emotion} emotion ({dominant_emotion_score:.1%})** combined with **low emotional diversity ({emotional_diversity:.2f})**, a pattern often seen in inauthentic reviews."
            )
        if is_repetitive:
            explanation_parts.append(
                "The text also contains **repetitive phrasing**, which can indicate generated or copied content."
            )
        if len(explanation_parts) == 1: # If only default verdict was added, add generic
             explanation_parts.append("The system detected subtle inconsistencies in its emotional and linguistic patterns compared to genuine reviews.")
    else: # Verdict is 'real'
        explanation_parts.append(f"This review appears **genuine** with {confidence:.1%} confidence.")
        if dominant_emotion_score > 0.6 and emotional_diversity > 1.5:
            explanation_parts.append(
                f"It expresses a clear and diverse emotional range, with a strong presence of {dominant_emotion} ({dominant_emotion_score:.1%})."
            )
        elif dominant_emotion != 'none':
            explanation_parts.append(f"It primarily conveys {dominant_emotion} with a score of {dominant_emotion_score:.1%}.")
        else:
            explanation_parts.append("Its emotional and linguistic patterns are consistent with authentic user feedback.")


    # --- 3. Customer Intent/Conveyance Summary (Basic, Rule-Based) ---
    # This is a simplified version. For advanced use, an LLM would be ideal.
    summary_parts.append("The customer ")
    if verdict == 'real' and dominant_emotion_score > 0.5:
        if dominant_emotion in ['joy', 'gratitude', 'love', 'excitement', 'optimism', 'pride', 'admiration', 'amusement', 'relief']:
            summary_parts.append(f"is very positive and likely satisfied, conveying {dominant_emotion} regarding their experience.")
        elif dominant_emotion in ['anger', 'disgust', 'annoyance', 'disapproval', 'sadness', 'grief', 'remorse', 'disappointment', 'nervousness', 'fear']:
            summary_parts.append(f"is negative and likely dissatisfied, expressing {dominant_emotion} about their experience.")
        elif dominant_emotion in ['curiosity', 'confusion', 'surprise', 'realization']:
            summary_parts.append(f"is exploring or reacting to new information, conveying {dominant_emotion} about a specific aspect.")
        elif dominant_emotion == 'caring':
            summary_parts.append(f"is showing empathy or concern, conveying a supportive tone.")
        elif dominant_emotion == 'neutral':
            summary_parts.append(f"is providing factual or balanced feedback.")
        else: # Fallback for less common or mixed emotions
            summary_parts.append("is providing feedback with a mix of emotions.")
    else: # If fake or no strong dominant emotion
        # For a fake review, we might summarize what it *tries* to convey
        # This is where a more advanced LLM would shine for interpreting intent.
        if "positive" in explanation_parts[0].lower() and "overwhelming joy" in "".join(explanation_parts).lower():
            summary_parts.append("attempts to convey extreme positive sentiment, likely regarding product quality.")
        elif "negative" in explanation_parts[0].lower() and "dissatisfied" in "".join(explanation_parts).lower():
            summary_parts.append("attempts to convey strong negative sentiment, likely regarding a problem or issue.")
        else:
            summary_parts.append("is attempting to provide feedback.")
    
    # Simple length limit to ensure 1-2 lines
    full_summary = "".join(summary_parts).strip()
    if len(full_summary) > 150: # Adjust character limit for "1-2 lines"
        full_summary = full_summary[:147] + "..." # Truncate if too long

    return {
        "explanation": " ".join(explanation_parts).strip(),
        "customer_intent_summary": full_summary
    }

# Example usage (this part will only run if you execute explanation_generator.py directly)
if __name__ == "__main__":
    # Example Emotion Names (from GoEmotions)
    emotion_names_list = ["admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", 
                          "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment", 
                          "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "neutral", 
                          "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"]

    # --- Test Case 1: Likely Fake, Exaggerated Joy, Repetitive ---
    print("--- Test Case 1: Exaggerated Fake ---")
    text1 = "This is the best product ever! I love it so much. Best product! Amazing amazing amazing. Simply the best."
    emotions1 = np.zeros(len(emotion_names_list))
    emotions1[emotion_names_list.index('joy')] = 0.95
    emotions1[emotion_names_list.index('love')] = 0.1 # Some secondary emotion
    
    explanation_output1 = generate_explanation(
        review_text=text1,
        verdict="fake",
        confidence=0.92,
        emotion_probs=emotions1,
        emotion_names=emotion_names_list
    )
    print("Explanation:", explanation_output1['explanation'])
    print("Summary:", explanation_output1['customer_intent_summary'])
    print("-" * 50)

    # --- Test Case 2: Likely Real, Mixed Emotions ---
    print("--- Test Case 2: Genuine, Mixed Emotions ---")
    text2 = "The delivery was fast, which I appreciate, but the product was slightly damaged. Feeling disappointed."
    emotions2 = np.zeros(len(emotion_names_list))
    emotions2[emotion_names_list.index('gratitude')] = 0.4
    emotions2[emotion_names_list.index('disappointment')] = 0.6
    emotions2[emotion_names_list.index('neutral')] = 0.2
    
    explanation_output2 = generate_explanation(
        review_text=text2,
        verdict="real",
        confidence=0.85,
        emotion_probs=emotions2,
        emotion_names=emotion_names_list
    )
    print("Explanation:", explanation_output2['explanation'])
    print("Summary:", explanation_output2['customer_intent_summary'])
    print("-" * 50)

    # --- Test Case 3: Likely Real, Neutral/Factual ---
    print("--- Test Case 3: Genuine, Factual ---")
    text3 = "The device operates as described. Battery life is approximately 8 hours. No issues encountered."
    emotions3 = np.zeros(len(emotion_names_list))
    emotions3[emotion_names_list.index('neutral')] = 0.8
    
    explanation_output3 = generate_explanation(
        review_text=text3,
        verdict="real",
        confidence=0.95,
        emotion_probs=emotions3,
        emotion_names=emotion_names_list
    )
    print("Explanation:", explanation_output3['explanation'])
    print("Summary:", explanation_output3['customer_intent_summary'])
    print("-" * 50)

    # --- Test Case 4: Short, No Strong Emotion ---
    print("--- Test Case 4: Short, No Strong Emotion ---")
    text4 = "It's good."
    emotions4 = np.zeros(len(emotion_names_list))
    emotions4[emotion_names_list.index('neutral')] = 0.4 # Just a little neutral
    
    explanation_output4 = generate_explanation(
        review_text=text4,
        verdict="real",
        confidence=0.6,
        emotion_probs=emotions4,
        emotion_names=emotion_names_list
    )
    print("Explanation:", explanation_output4['explanation'])
    print("Summary:", explanation_output4['customer_intent_summary'])
    print("-" * 50)