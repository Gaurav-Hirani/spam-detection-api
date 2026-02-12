# **🛡️ Spam SMS Classifier & Detection System**

A high-performance Machine Learning project designed to classify SMS messages as **Spam** or **Ham (Not Spam)**. This project features a robust NLP pipeline and a hybrid detection engine that combines Machine Learning with rule-based heuristics to ensure maximum accuracy.

## **🚀 Overview**

Spam messages are not just annoying; they are often the first step in phishing attacks. This project provides a complete end-to-end solution:

1. **Exploratory Data Analysis (EDA):** Deep dive into the spam.csv dataset.  
2. **Preprocessing:** Advanced text cleaning (Stemming, Stopword removal, Punctuation removal).  
3. **Model Selection:** Benchmark of 10+ classifiers to find the optimal balance of Accuracy and Precision.  
4. **Web Deployment:** A FastAPI-based web dashboard for real-time message testing.

## **📊 Model Performance Benchmarks**

During the development phase (documented in SpamDectection (1).ipynb), multiple models were tested. While many achieved high accuracy, **Precision** was prioritized to avoid "False Positives" (marking important personal messages as spam).

| Model | Accuracy | Precision |
| :---- | :---- | :---- |
| **Multinomial Naive Bayes (MNB)** | **97.1%** | **100%** |
| Extra Trees Classifier (ETC) | 97.4% | 99.2% |
| Random Forest (RF) | 97.0% | 99.1% |
| Support Vector Machine (SVC) | 97.2% | 93.9% |
| Logistic Regression (LR) | 94.4% | 86.5% |

**Selected Model:** MultinomialNB was chosen for the final deployment due to its perfect precision score (![][image1]), ensuring that no legitimate message is incorrectly flagged as spam.

## **🛠️ Tech Stack**

* **Language:** Python 3.x  
* **Machine Learning:** Scikit-Learn  
* **NLP:** NLTK (Natural Language Toolkit)  
* **Backend:** FastAPI & Uvicorn  
* **Frontend:** HTML/CSS (Jinja2 Templates)  
* **Deployment:** Vercel / Cloud Ready

## **🧩 Hybrid Detection Logic**

Unlike standard ML classifiers, the main.py implementation uses a **Hybrid Engine**:

* **Vectorization:** TF-IDF Vectorizer converts text into numerical features.  
* **ML Prediction:** The MNB model calculates the probability of spam.  
* **Heuristic Override:** A custom rule-based filter checks for suspicious link patterns (e.g., bit.ly, tinyurl, .com) to catch modern phishing attempts that might bypass traditional filters.

## **📂 Project Structure**

* main.py: FastAPI application logic and prediction routes.  
* model.pkl: The trained Multinomial Naive Bayes model.  
* vectorizer.pkl: The TF-IDF vectorizer object.  
* spam.csv: The dataset used for training and testing.  
* SpamDectection (1).ipynb: The complete research, EDA, and model benchmarking notebook.  
* index.html: The clean, responsive UI for the classifier.
 

**Author:** \[Gaurav Hirani\]

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAXCAYAAAD+4+QTAAABLklEQVR4Xu2SMUoDURCGY+EFRFCWZd+unfWeIDdI4x1iRFFJ5wUscg2xSBMIWHoHwYBNSKGIooWtKdRv5OXxMpkEVxAs9oMhvG/+nXlLttGo+fc457p5nu9rvwqeaVJ31Cd1ofvf0Likpj4k1dGZZZA9pT5mZy7YlhlxZoFfLJH8rnYsO4/dHFWWMKhl3Rr3bvlAlSXkrq1huInlAxWXvFnDcCPLB/ySA+0tfHZhGO7G8gFpZll2qL0F2UdrGO7W8gG/5Eh7C7f8PxlbPiBNvppj7S3InlnD3E++LpacaC/whntFUWzFTvJpmm5oRw1jF0iSZNMHeroHa743d0POT9RkduaC25Ipy3I9zkmwT71QD9S9/32mpnGONxngurETcK8+fyULyO3oTE3N3/AFcyZqfRmUup4AAAAASUVORK5CYII=>
