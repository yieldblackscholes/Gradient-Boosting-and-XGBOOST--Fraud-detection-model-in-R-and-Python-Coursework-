# Gradient-Boosting-and-XGBOOST--Fraud-detection-model-in-R-and-Python-Coursework-
# Gradient Boosting & XGBoost — Credit Card Fraud Detection

ECOM198 Machine Learning for Finance  
University:Queen Mary University of London  
Year:2026

## About This Project

This project explores Gradient Boosting and XGBoost algorithms 
applied to credit card fraud detection. It was developed as part 
of the ECOM198 midterm coursework at Queen Mary University of London.

The dataset contains 201,670 real credit card transactions from 
European cardholders in September 2013, with only 0.244% being 
fraudulent — making it a highly imbalanced classification problem.

## What's Inside

- heory— Full explanation of gradient boosting from scratch
- Step-by-step worked examples — Regression and classification
- R Code — XGBoost implementation using the xgboost R package
- Python Code — XGBoost implementation for comparison
- EDA — Exploratory data analysis of the fraud dataset
- Results — Model evaluation using confusion matrix and recall

## Dataset

- Source: Kaggle Credit Card Fraud Detection
- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 201,670 transactions | 492 fraud cases | 28 PCA features

## Technologies Used

- R (xgboost, caret, dplyr)
- Python (xgboost, sklearn, pandas)

## Key Results

- XGBoost successfully classified fraudulent transactions
- Recall was prioritised as the key metric — missing fraud 
  is more costly than a false alarm
- Class imbalance handled using scale_pos_weight parameter

## References

- Friedman, J.H. (2001) Greedy function approximation: 
  A gradient boosting machine. Annals of Statistics, 29(5)
- Chen, T. and Guestrin, C. (2016) XGBoost: A scalable 
  tree boosting system. KDD 2016
- Hastie, T., Tibshirani, R. and Friedman, J. (2009) 
  The Elements of Statistical Learning. Springer

## Author

Neeraj SHarma  — QMUL ECOM198 | Machine Learning for Finance 2026
