# STEP 1: import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from xgboost import XGBClassifier


# STEP 2: load data
data = pd.read_csv("credit-sample.csv")

print(data.head())
print(data.shape)


# STEP 3: check fraud vs normal
print(data['Class'].value_counts())
print(data['Class'].value_counts(normalize=True) * 100)


# STEP 4: check missing values
print(data.isna().sum().sum())


# STEP 5: histogram
plt.hist(data['Amount'])
plt.title("Amount Distribution")
plt.show()


# STEP 6: define X and y
X = data.drop('Class', axis=1)
y = data['Class']


# STEP 7: split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=123,
    stratify=y
)


# STEP 8: train model
model = XGBClassifier(
    n_estimators=100,
    objective='binary:logistic',
    eval_metric='logloss'
)

model.fit(X_train, y_train)


# STEP 9: predict probability
prob = model.predict_proba(X_test)[:, 1]


# STEP 10: convert to 0/1
final_pred = (prob > 0.5).astype(int)


# STEP 11: evaluate
print(confusion_matrix(y_test, final_pred))
print(classification_report(y_test, final_pred))