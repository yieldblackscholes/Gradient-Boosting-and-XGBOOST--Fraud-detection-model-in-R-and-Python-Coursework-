#install and run packages
import pandas as pd          # used to read and handle dataset
import numpy as np           # used for numerical operations
import matplotlib.pyplot as plt   # used to create graphs
import seaborn as sns        # used for advanced plots
from sklearn.model_selection import train_test_split   # split data into train/test
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc   # model evaluation
from sklearn.linear_model import LogisticRegression   # logistic regression model
import xgboost as xgb        # XGBoost model

#load the csv file
data = pd.read_csv("D:/ML FOR F IN R/MidTerm coursework/credit-sample.csv")  # load dataset
Data = data   # copy dataset into new variable

#class distribution check (imbalanced data check)
print(Data['Class'].value_counts())  
# count how many normal (0) and fraud (1)

print(Data['Class'].value_counts(normalize=True)*100)  
# convert counts into percentage

Data['Class'].value_counts().plot(kind='bar', color=['darkred','navy'])  
# bar chart of class distribution
plt.title("Class Distribution")
plt.show()

#Amount distribution check
print(Data['Amount'].describe())  
# shows mean, median, max, etc.

plt.hist(Data['Amount'], bins=100, color='darkred')  
# histogram of amount
plt.title("Distribution of Transaction Amounts")
plt.show()

plt.hist(np.log1p(Data['Amount']), bins=50, color='darkred')  
# log transformation to reduce skew
plt.title("Distribution of log(Amount + 1)")
plt.show()

sns.boxplot(x='Class', y='Amount', data=Data)  
# compare amount for fraud vs normal
plt.title("Transaction Amount: Non-Fraud vs Fraud")
plt.show()

#Time distribution check
plt.hist(Data['Time'], bins=100, color='darkred')  
# overall time distribution
plt.title("Distribution of Transaction Time")
plt.show()

plt.hist(Data[Data['Class']==0]['Time'], bins=50, color='blue')  
# time distribution for normal transactions
plt.title("Time — Non-Fraud Transactions")
plt.show()

plt.hist(Data[Data['Class']==1]['Time'], bins=50, color='red')  
# time distribution for fraud transactions
plt.title("Time — Fraudulent Transactions")
plt.show()

#Analysis of differences in V1–V28
fraud = Data[Data['Class']==1]  
# select only fraud rows

normal = Data[Data['Class']==0]  
# select only normal rows

V_Columns = [f"V{i}" for i in range(1,29)]  
# create list of feature names V1 to V28

mean_fraud = fraud[V_Columns].mean()  
# average of each feature for fraud

mean_normal = normal[V_Columns].mean()  
# average of each feature for normal

V_Comparison = pd.DataFrame({
    "Feature": V_Columns,
    "Mean_Fraud": round(mean_fraud,3),
    "Mean_Normal": round(mean_normal,3),
    "Difference": round(mean_fraud-mean_normal,3)
})
# create table comparing fraud vs normal

print(V_Comparison)

plt.bar(V_Columns, V_Comparison["Difference"],
        color=['navy' if x>0 else 'darkred' for x in V_Comparison["Difference"]])
# plot difference between fraud and normal
plt.title("Mean Difference V1–V28")
plt.xticks(rotation=90)
plt.show()

#Correlation Heatmap
sample_Data1 = Data.sample(5000, random_state=2026)[V_Columns]  
# take random sample of data

corr_matrix = sample_Data1.corr()  
# calculate correlation

sns.heatmap(corr_matrix, cmap="coolwarm")  
# plot heatmap
plt.title("Correlation between V1–V28")
plt.show()

# ==================== STEP 2 ====================
Data['Class'] = Data['Class'].astype(int)  
# convert class to integer (0 and 1)

X = Data.drop(columns=['Class','X'], errors='ignore')  
# remove target column from features

Y = Data['Class']  
# target variable

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2026
)
# split data into train (80%) and test (20%)

#scale_pos_weight
scale_pos_weight = (Y_train==0).sum()/(Y_train==1).sum()  
# handle imbalance (fraud is rare)

print(scale_pos_weight)

# ==================== STEP 3 ====================
model = xgb.XGBClassifier(
    objective='binary:logistic',   # binary classification
    eval_metric='auc',             # evaluation metric
    scale_pos_weight=scale_pos_weight,   # imbalance handling
    max_depth=6,                   # tree depth
    learning_rate=0.1,             # learning speed
    subsample=0.8,                 # row sampling
    colsample_bytree=0.8,          # column sampling
    n_estimators=100,              # number of trees
    random_state=2026
)

model.fit(X_train, Y_train)  
# train model

# ==================== STEP 4 ====================
XGB_Pred = model.predict_proba(X_test)[:,1]  
# predicted probability of fraud

XGB_Class = (XGB_Pred>0.5).astype(int)  
# convert probability to 0 or 1

print(confusion_matrix(Y_test, XGB_Class))  
# confusion matrix

print(classification_report(Y_test, XGB_Class))  
# precision, recall, f1-score

#ROC Curve
fpr, tpr, _ = roc_curve(Y_test, XGB_Pred)  
# calculate ROC

roc_auc = auc(fpr, tpr)  
# calculate AUC

plt.plot(fpr, tpr, color='navy')  
# plot ROC
plt.title("ROC Curve - XGBoost")
plt.show()

#Logistic Regression
Logistic_Model = LogisticRegression(max_iter=1000)  
# create logistic model

Logistic_Model.fit(X_train, Y_train)  
# train logistic model

Logistic_Pred = Logistic_Model.predict_proba(X_test)[:,1]  
# probability prediction

Logistic_Class = (Logistic_Pred>0.5).astype(int)  
# convert to class

print(confusion_matrix(Y_test, Logistic_Class))  
# confusion matrix

print(classification_report(Y_test, Logistic_Class))  
# evaluation report

#ROC Comparison
fpr2, tpr2, _ = roc_curve(Y_test, Logistic_Pred)  
# ROC for logistic

roc_auc2 = auc(fpr2, tpr2)  
# AUC for logistic

plt.plot(fpr, tpr, label="XGBoost")  
plt.plot(fpr2, tpr2, label="Logistic")
plt.legend()
plt.title("ROC Comparison")
plt.show()