# Gradient Boosting & XGBoost — Credit Card Fraud Detection

**Course:** ECOM198 Machine Learning for Finance
**University:** Queen Mary University of London
**Year:** 2026

---

## About This Project

This project builds a fraud detection system using Gradient Boosting 
and XGBoost on 200,000+ real credit card transactions. The core 
challenge is extreme class imbalance — only 0.25% of transactions 
are fraudulent, which makes standard accuracy completely misleading.

The full pipeline was implemented and compared in both R and Python 
to understand how the same algorithm behaves differently across 
frameworks — and what that means for real-world model evaluation.

---

## What's Inside

- Theory — Full explanation of gradient boosting from scratch
- Step-by-step worked examples — Regression and classification
- R Code — XGBoost implementation using the xgboost R package
- Python Code — XGBoost v2.0 implementation with scale_pos_weight
- EDA — Exploratory data analysis of the fraud dataset
- Results — Model evaluation using confusion matrix, recall, F1-score
- Comparison — XGBoost vs Logistic Regression

---

## Dataset

- Source: Kaggle Credit Card Fraud Detection
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 201,670 transactions
- 492 fraud cases (0.25%)
- Features V1–V28 are PCA-transformed
- Time and Amount are untransformed
- Class: 1 = fraud, 0 = normal

---

## Results

### R — XGBoost

| Metric | Value |
|---|---|
| Accuracy | 99.91% |
| Fraud caught | 68 / 98 |
| Fraud missed | 30 |
| Recall (fraud) | 69% |
| Kappa | 0.786 |

### Python — XGBoost v2.0 (scale_pos_weight = 408)

| Metric | Value |
|---|---|
| Accuracy | 99.95% |
| Fraud caught | 79 / 98 |
| Fraud missed | 19 |
| Recall (fraud) | 81% |
| False positives | 3 |

### Python — Logistic Regression (baseline comparison)

| Metric | Value |
|---|---|
| Recall (fraud) | 67% |
| F1-score (fraud) | 0.74 |

XGBoost outperformed Logistic Regression on every 
fraud-specific metric.

---

## Key Findings

Running the same model in R and Python produced different results.
The difference came from how each framework handles random splitting
and the internal mechanics of XGBoost across languages.

Python with scale_pos_weight properly set gave more realistic 
results. The R results showed near-perfect accuracy which on 
heavily imbalanced data is a warning sign, not a success signal.

The most important lesson — perfect accuracy does not mean a 
perfect model. In fraud detection, missing 19 fraud cases versus 
missing 30 is not a small technical difference. It represents 
real transactions and real consequences.

---

## Technologies

- R 4.5.1 — xgboost, caret, dplyr
- Python 3.14 — xgboost 2.0, scikit-learn, pandas, matplotlib

---

## How to Run

### R
```r
install.packages(c("xgboost", "caret", "dplyr"))
source("mid term assigment.R")
```

### Python
```bash
pip install xgboost scikit-learn pandas matplotlib
python MIDTERM.PY
```

---

## References

- Friedman, J.H. (2001) Greedy function approximation:
  A gradient boosting machine. Annals of Statistics, 29(5)
- Chen, T. and Guestrin, C. (2016) XGBoost: A scalable
  tree boosting system. KDD 2016
- Hastie, T., Tibshirani, R. and Friedman, J. (2009)
  The Elements of Statistical Learning. Springer
- Kaggle Dataset:
  https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## Author

Neeraj Sharma — ECOM198 Machine Learning for Finance 2026
