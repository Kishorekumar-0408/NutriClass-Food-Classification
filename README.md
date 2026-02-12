# NutriClass: Food Classification Using Nutritional Data

## 📌 Project Overview

NutriClass is a multi-class machine learning classification system that predicts the exact food name based on nutritional attributes such as calories, protein, fats, carbohydrates, sugar, fiber, and other dietary features.

The system is designed for strict diet planning scenarios where a precise food recommendation is required.

---

## 🎯 Objectives

- Benchmark multiple traditional ML models
- Compare feature extraction strategies (with and without PCA)
- Analyze model performance using evaluation metrics
- Provide business-ready food prediction capability

---

## 🧠 Models Implemented

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting
- XGBoost

---

## 📊 Results Summary

- Best Model: **Gradient Boosting**
- Accuracy Achieved: **~99%**
- Balanced Precision, Recall, and F1-score across all classes
- PCA comparison showed no significant improvement over original features

### 🔍 Insight

Standard preprocessing (scaling + one-hot encoding) combined with Gradient Boosting achieved the highest performance. Dimensionality reduction using PCA did not significantly improve accuracy, indicating strong separability in the original feature space.

---

## 📈 Visual Performance Metrics

The project includes:

- Model Accuracy Comparison Chart
- Confusion Matrix for Best Model
- Feature Importance Visualization
- PCA vs Non-PCA Performance Comparison

These visualizations provide insights into model behavior and feature contribution.

---

## 💼 Business Use Case

This system can be integrated into:

- Smart diet planning applications
- Nutrition tracking platforms
- Health monitoring systems
- Meal planning tools

It ensures strict food classification based on user-defined nutritional requirements.

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn

---
## 📊 Model Performance Dashboard

![Dashboard](results/dashboard.png)

## 📂 Project Structure

## How to Run
1. Place dataset in `data/raw/`
2. Install dependencies:
   pip install -r requirements.txt
3. Run:
   python -m src.main

