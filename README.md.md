# MLOps Assignment: Supervised & Unsupervised Machine Learning (End-to-End)

## 1. Assignment Overview
This assignment is designed to assess your understanding and practical application of **MLOps concepts** using both **Supervised** and **Unsupervised Machine Learning** models. You will build, version, train, evaluate, deploy, and document machine learning models following industry-grade MLOps practices.

You are required to:
- Build **one supervised ML model**
- Build **one unsupervised ML model**
- Apply **MLOps best practices** (experiment tracking, versioning, reproducibility)
- Deploy at least **one model** using **Flask, Streamlit, or Django**
- Provide **screenshots and documentation** as evidence

This assignment is submitted via **GitHub Classroom**.

---

## 2. Learning Objectives
By completing this assignment, you should demonstrate:
- Understanding of supervised vs unsupervised learning
- Ability to structure ML projects professionally
- Knowledge of MLOps lifecycle (data → training → evaluation → deployment → monitoring)
- Model versioning and reproducibility
- Basic ML deployment skills

---

## 3. Dataset Requirements
You may choose **public datasets** (Kaggle, UCI, sklearn datasets).

### Supervised Dataset Requirements
- Must include labeled data
- Classification **or** regression task
- Minimum 1,000 rows recommended

Examples:
- Customer churn
- Credit risk
- House price prediction

### Unsupervised Dataset Requirements
- No target variable
- Suitable for clustering or anomaly detection

Examples:
- Customer segmentation
- Transaction behavior
- Image or text embeddings

---

## 4. Project Structure (Mandatory)
Your GitHub repository **must** follow this structure:

```
mlops-assignment/
│── data/
│   ├── raw/
│   └── processed/
│
│── notebooks/
│   ├── eda_supervised.ipynb
│   └── eda_unsupervised.ipynb
│
│── src/
│   ├── data_preprocessing.py
│   ├── train_supervised.py
│   ├── train_unsupervised.py
│   ├── evaluate.py
│   └── utils.py
│
│── models/
│   ├── supervised_model.pkl
│   └── unsupervised_model.pkl
│
│── app/
│   ├── app.py   # Flask / Streamlit / Django entry
│   └── templates/ (if Flask/Django)
│
│── requirements.txt
│── README.md
│── screenshots/
│── .gitignore
```

---

## 5. Supervised Learning Task (40 Marks)

### Requirements
- Choose **one supervised algorithm**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - SVM
- Perform:
  - Data cleaning
  - Feature engineering
  - Train-test split

### Evaluation Metrics
- Classification:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Regression:
  - RMSE
  - MAE
  - R²

### Deliverables
- Trained model saved in `/models`
- Evaluation results logged
- Notebook + Python script

---

## 6. Unsupervised Learning Task (30 Marks)

### Requirements
- Choose **one unsupervised technique**:
  - K-Means
  - DBSCAN
  - Hierarchical Clustering
  - Isolation Forest

### Evaluation
- Elbow Method / Silhouette Score
- Cluster visualization

### Deliverables
- Trained model saved
- Clear interpretation of clusters/anomalies

---

## 7. MLOps Requirements (20 Marks)

You must demonstrate **at least 4** of the following:

- Experiment tracking (MLflow recommended)
- Model versioning
- Reproducibility (random seeds)
- Modular code
- Environment management (`requirements.txt`)
- Logging
- Data version awareness

---

## 8. Deployment (Mandatory – 10 Marks)

### Deployment Options
Choose **ONE**:
- Flask API
- Streamlit App
- Django Web App

### Deployment Features
- Upload or input data
- Return prediction
- Display confidence score / cluster label

### Evidence Required
Add **screenshots** of:
- Running web app
- Prediction output
- API response (if Flask)

Save screenshots inside `/screenshots/`.

---

## 9. README Documentation (Mandatory)
Your `README.md` must include:

- Project overview
- Dataset description
- Model choices
- How to run the project
- Deployment instructions
- Screenshots embedded

---

## 10. Submission Instructions

1. Accept assignment via **GitHub Classroom**
2. Push all code to your repository
3. Ensure repository is **public or classroom-accessible**
4. Final commit before deadline

---

## 11. Grading Rubric

| Component | Marks |
|--------|------|
| Supervised ML | 40 |
| Unsupervised ML | 30 |
| MLOps Practices | 20 |
| Deployment | 10 |
| **Total** | **100** |

---

## 12. Bonus (Optional)
- Dockerize the application (+5)
- CI/CD with GitHub Actions (+5)
- Model monitoring concept explanation (+5)

---

## 13. Academic Integrity
- Individual work unless stated otherwise
- Plagiarism results in disqualification
- Cite all external sources

---

**End of Assignment**

