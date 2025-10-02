<artifact identifier="minimal-readme" type="text/markdown" title="Minimal DeepDive README">
# DeepDive - Emotional Counterfeit Detector
Detects fake reviews by analyzing emotional patterns and linguistic cues in user-generated content.

[![Project Status](https://img.shields.io/badge/Status-Local%20Functional-brightgreen?style=for-the-badge)](https://github.com/VummadiHarsha39/emotional-counterfeit-detector__NLP__Application_for-Review_Analysis.git)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging%20Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFBA18?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/index)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14%2B-black?style=for-the-badge&logo=next.js&logoColor=white)](https://nextjs.org/)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.x-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)](https://tailwindcss.com/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0091DA?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Contact](https://img.shields.io/badge/Contact-@VummadiHarsha39-2ea44f?style=for-the-badge&logo=github)](https://github.com/VummadiHarsha39)

---

##  Project Overview

In the current digital environment, identifying authentic feedback amidst cleverly crafted fake content poses a significant challenge.

This initiative delves deeper than basic spam detection by examining the subtle emotional cues and language trends found in reviews and social media content, offering clear insights and identifying actionable customer intent.

![Structure of our Project](Images/Chart.png)

###  Problem Solved

Companies and consumers struggle to identify genuine online reviews. Fake reviews often appear normal but possess subtle, 'off' emotional patterns or repetitive language that betray their inauthenticity. This system aims to expose these subtle cues.

### 🚀 Key Features & Deliverables

* **Authenticity Classification:** Accurately classifies reviews as "Likely Real" or "Likely Fake" with a quantifiable confidence score.
* **Detailed Emotional Profiling:** Extracts and quantifies a diverse range of human emotions (e.g., joy, anger, confusion, curiosity) expressed in the text.
* **Explainable AI (XAI):** Provides human-readable explanations for the authenticity verdict, detailing *why* a review seems real or fake based on emotional and linguistic analysis.

---

## Project Demo

Click below to watch a clear methodolgy explanation and short demo of backend and frontend of the project:

[![Video](https://vumbnail.com/1102336040.jpg)](https://vimeo.com/1102336040)


### ⚙️ How It Works (Operational Flow)

1.  **User Input:** User enters review text on the **Frontend** (`http://localhost:3000`).
2.  **API Request:** Frontend sends the review via HTTP `POST` to the **FastAPI Backend API** (`http://localhost:8000/analyze_review`).
3.  **Backend Initialization (One-time on startup):**
    * FastAPI application starts.
    * Initializes the **Prediction Service** (`ReviewAnalysisService`).
    * **Prediction Service** connects to the **local MLflow Model Registry** and loads both the **Emotion Classifier** and **Deception Detector** models into memory (leveraging GPU for speed).
    * FastAPI begins listening for requests.
4.  **Review Analysis (Per Request):**
    * **Backend API** receives the review.
    * Calls `ReviewAnalysisService.analyze_review()` which orchestrates the ML pipeline:
        * Text is cleaned (`src/data/preprocessor.py`).
        * Cleaned text is passed to the **Emotion Classifier**, which predicts 28 emotion probabilities.
        * Original text + Emotion Probabilities are passed to the **Deception Detector**, which predicts "Real"/"Fake" and confidence.
        * All analysis results are passed to the **Explanation Generator**, which crafts a human-readable explanation and a customer intent summary.
    * **Backend API** returns a structured JSON response.
5.  **Display Results:** The **Frontend** receives the JSON, dynamically updates the UI to show the verdict, emotion chart, explanation, and intent summary.

---


This detailed overview should serve as your definitive guide to the project. It outlines every step, every component, and every major challenge you've overcome. You have built a truly impressive system.

If you want a concise explanation of this project, please feel free to checkout the article about this project on medium, https://medium.com/@vummadiharsha123/summer-project-3-nlp-application-to-detect-emotional-counterfeit-of-a-customer-review-e4e583626f96.
