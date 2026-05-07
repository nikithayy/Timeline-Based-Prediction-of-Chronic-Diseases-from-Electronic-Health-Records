# Multi-Disease Timeline Prediction using ML + RAG

## Overview

This project builds an end-to-end healthcare analytics pipeline to predict chronic disease risk and onset timelines using Electronic Health Record (EHR) data. The system combines ETL, machine learning, and a Retrieval-Augmented Generation (RAG) framework to provide explainable and clinically meaningful predictions.

Diseases Predicted:

* Diabetes
* Hypertension
* Heart Disease

The project predicts:

1. Whether a disease is likely to occur (Classification)
2. When the disease may occur (Regression)
3. Similar patient cases for explainability (RAG)

---

# Project Architecture

```text
Raw EHR Data
     ↓
ETL Pipeline (PySpark)
     ↓
Feature Engineering
     ↓
Machine Learning Models
 ├── Classification Models
 └── Regression Models
     ↓
RAG Retrieval System
     ↓
Explainable Predictions
```

---

# Technologies Used

## Data Engineering

* PySpark
* Pandas
* NumPy

## Machine Learning

* Scikit-learn
* XGBoost
* LightGBM
* Imbalanced-learn (SMOTE)

## Retrieval & Explainability

* SentenceTransformers
* FAISS

## Visualization

* Matplotlib
* Seaborn

---

# Dataset

The project uses synthetic EHR data generated from Synthea.

Main CSV files:

* patients.csv
* conditions.csv
* observations.csv
* medications.csv
* encounters.csv

---

# ETL Pipeline

File:

```text
etl_fhir.py
```

The ETL pipeline performs:

## 1. Data Loading

Loads raw healthcare CSV files using PySpark.

## 2. Patient Demographics

Extracts:

* Age
* Gender
* Race

## 3. Feature Engineering

Creates features such as:

* Prior condition count
* Medication count
* Medication cost
* Encounter count
* Average lab values

## 4. Temporal Processing

Uses a cutoff date:

```text
2019-01-01
```

* Data before cutoff → Features
* Data after cutoff → Labels

This prevents data leakage.

## 5. Disease Timeline Labels

Calculates:

* days_to_diabetes
* days_to_hypertension
* days_to_heart_disease

## 6. Output

Generates:

```text
data/processed/features_timeline.csv
```

---

# Machine Learning Pipeline

File:

```text
train_model.py
```

---

# Classification Models

Goal:
Predict whether a disease will occur.

## Models Used

* Logistic Regression
* Random Forest Classifier

## Processing Steps

1. Feature Encoding
2. Standard Scaling
3. Train-Test Split
4. SMOTE Oversampling
5. Model Training
6. Evaluation

## Metrics

* Accuracy
* ROC-AUC

---

# Regression Models

Goal:
Predict the timeline until disease onset.

## Models Used

* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor

## Processing Steps

1. Remove invalid values
2. Remove outliers
3. Oversample positive cases
4. Log-transform targets
5. Train regression models
6. Reverse transform predictions

## Metric

* Mean Absolute Error (MAE)

---

# Regularization Techniques

## Random Forest

* max_depth
* min_samples_leaf
* max_features
* Ensemble averaging

## XGBoost

* learning_rate
* max_depth
* subsample
* colsample_bytree

## LightGBM

* learning_rate
* num_leaves
* subsample
* colsample_bytree

---

# RAG-Based Explainability

File:

```text
rag.py
```

The RAG system retrieves similar patients to explain predictions.

## Workflow

```text
Patient Input
      ↓
ML Prediction
      ↓
Convert Patient → Text
      ↓
Sentence-BERT Embeddings
      ↓
FAISS Vector Search
      ↓
Retrieve Similar Patients
      ↓
MMR Re-ranking
      ↓
Generate Explanation
```

---

# Embedding Model

Model Used:

```text
all-MiniLM-L6-v2
```

Converts patient information into dense vector embeddings.

---

# FAISS Retrieval

FAISS performs fast vector similarity search to retrieve top-k similar patients.

Similarity Metric:

* L2 Distance

---

# MMR (Maximal Marginal Relevance)

MMR improves retrieval diversity.

Formula:

```text
score = λ * relevance - (1 - λ) * diversity
```

This balances:

* Relevance to query
* Diversity among retrieved patients

---

# Evaluation Metrics

## Classification

* Accuracy
* ROC-AUC

## Regression

* Mean Absolute Error (MAE)

## Retrieval

* Precision@K
* Recall@K

---

# Saved Models

Generated in:

```text
models/saved_models/
```

Saved artifacts:

* Classification models
* Regression models
* Feature lists
* Scalers

---

# Project Structure

```text
project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── plots/
│
├── models/
│   └── saved_models/
│
├── etl_fhir.py
├── train_model.py
├── rag.py
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
cd <project-folder>
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1: Run ETL

```bash
python etl_fhir.py
```

## Step 2: Train Models

```bash
python train_model.py
```

## Step 3: Run RAG System

```bash
python rag.py
```

---

# Example Output

```text
=== Predicted Risk ===
Diabetes: 0.72 (High)
Hypertension: 0.44 (Moderate)
Heart Disease: 0.20 (Low)

=== Similar Patients ===
Patient 12 | Diabetes occurred
Patient 45 | Similar history
```

---

# Key Contributions

* Multi-disease prediction system
* Timeline-based disease modeling
* ETL pipeline for longitudinal EHR data
* Integrated ML + RAG explainability
* Similar patient retrieval using embeddings and FAISS

---

# Future Improvements

* Real-world EHR integration
* More diseases
* LLM-powered explanations
* Clinical deployment
* Deep learning architectures

---

# Authors

Developed as part of a healthcare AI and predictive analytics Capstone project by Srinikitha Yendru advised by Prof. M. Mustafa Rafique.
