
# XGBoost Fraud Detection in R
# This version fixes several issues from the original code

library(xgboost)
library(caret)
library(dplyr)


data <- read.csv("D:/ML FOR F IN R/MidTerm coursework/credit-sample.csv")
Data <- data  # Assign to consistent variable name OR use lowercase 'data' throughout

# CLASS DISTRIBUTION CHECK -------------
table(Data$Class) 
# result: 492 fraud cases and 201178 normal cases (highly imbalanced)

prop.table(table(Data$Class))*100 
# result: only 0.24% fraud cases

barplot(
  table(Data$Class),
  col = c("darkred", "navy"),
  main = "Class Distribution",
  names.arg = c("Normal (0)", "Fraud (1)"),
  ylab = "Count"
)

# AMOUNT DISTRIBUTION CHECK ---------------
summary(Data$Amount)
# Most transactions are small (mean 88.35, median 22.00)
# Max is 25691.16 (potential outliers exist)

hist(Data$Amount,
     breaks = 100,
     col = "darkred",
     main = "Distribution of Transaction Amounts",
     xlab = "Amount")
# Right-skewed distribution

# Apply log transformation for visualization only (not on original data)
hist(log1p(Data$Amount),
     breaks = 50,
     col = "darkred",
     main = "Distribution of log(Amount + 1)",
     xlab = "log(Amount + 1)")

boxplot(Amount ~ Class, data = Data,
        col = c("darkred", "navy"),
        main = "Transaction Amount: Non-Fraud vs Fraud",
        names = c("Non-Fraud", "Fraud"),
        ylab = "Amount")

# TIME DISTRIBUTION CHECK -------------
hist(Data$Time,
     breaks = 100,
     col = "darkred",
     main = "Distribution of Transaction Time",
     xlab = "Time (seconds)")

par(mfrow = c(1,2))
hist(Data$Time[Data$Class == 0],
     breaks = 50, col = "blue",
     main = "Time — Non-Fraud Transactions",
     xlab = "Time")

hist(Data$Time[Data$Class == 1],
     breaks = 50, col = "red",
     main = "Time — Fraudulent Transactions",
     xlab = "Time")
par(mfrow = c(1,1))

# V1-V28 FEATURE ANALYSIS -----------
fraud <- Data[Data$Class == 1, ]
normal <- Data[Data$Class == 0, ]

V_Columns <- paste0("V", 1:28)
mean_fraud <- colMeans(fraud[, V_Columns])
mean_normal <- colMeans(normal[, V_Columns])

V_Comparison <- data.frame(
  Feature = V_Columns,
  Mean_Fraud = round(mean_fraud, 3),
  Mean_Normal = round(mean_normal, 3),
  Difference = round(mean_fraud - mean_normal, 3)
)
print(V_Comparison)

barplot(V_Comparison$Difference,
        names.arg = V_Columns,
        col = ifelse(V_Comparison$Difference > 0, "navy", "darkred"),
        main = "Mean Difference V1–V28: Fraud minus Non-Fraud",
        xlab = "Feature",
        ylab = "Difference in Mean",
        las = 2)
abline(h = 0, lty = 2)

# BOXPLOTS FOR TOP FEATURES --------
par(mfrow = c(2, 4))
for (v in c("V3", "V4", "V10", "V11", "V12", "V14", "V16", "V17")) {
  boxplot(Data[[v]] ~ Data$Class,
          col = c("navy", "darkred"),
          main = paste("Distribution of", v),
          names = c("Normal", "Fraud"),
          xlab = "Class",
          ylab = v,
          outline = FALSE)
}
par(mfrow = c(1, 1))

# CORRELATION HEATMAP --------
if (!require(corrplot)) install.packages("corrplot")
library(corrplot)

set.seed(2026)
sample_Data1 <- Data[sample(nrow(Data), 5000), V_Columns]
Corr_Matrix_Sample_Data1 <- cor(sample_Data1)

corrplot(Corr_Matrix_Sample_Data1,
         method = "color",
         type = "upper",
         tl.cex = 0.7,
         tl.col = "black",
         title = "Correlation between V1–V28",
         mar = c(0, 0, 1, 0))

# STEP 2: DATA PREPARATION FOR XGBOOST ---------

# Convert Class to factor (required for stratified sampling)
Data$Class <- as.factor(Data$Class)
class(Data$Class)  # Should be "factor"
levels(Data$Class) # Should be "0" and "1"

# Stratified split (80% training, 20% testing)
# IMPORTANT: stratified split preserves fraud/normal ratio in both sets
set.seed(2026)
Train_Index <- createDataPartition(Data$Class, p = 0.8, list = FALSE)
Train_Data <- Data[Train_Index, ]
Test_Data <- Data[-Train_Index, ]

dim(Train_Data)  # Should be ~161337 rows
dim(Test_Data)   # Should be ~40334 rows

# Verify class distribution is preserved
prop.table(table(Train_Data$Class)) * 100
prop.table(table(Test_Data$Class)) * 100
# Both should show ~99.76% normal, ~0.24% fraud

# SEPARATE FEATURES AND LABELS --------
# FLAW FIX #2: Remove both "Class" and "X" (index column)
X_Train <- as.matrix(Train_Data[, !names(Train_Data) %in% c("Class", "X")])
Y_Train <- as.numeric(as.character(Train_Data$Class))

X_Test <- as.matrix(Test_Data[, !names(Test_Data) %in% c("Class", "X")])
Y_Test <- as.numeric(as.character(Test_Data$Class))

# Note: We use as.numeric(as.character(...)) instead of direct as.numeric()
# Because Class is a factor, direct conversion gives 1,2 instead of 0,1
# The character conversion fixes this

# CALCULATE SCALE_POS_WEIGHT --------
# FLAW FIX #3: Explain why this is critical for imbalanced data
Number_of_Normal_Transactions <- sum(Y_Train == 0)
Number_of_Fraud_Transactions <- sum(Y_Train == 1)

scale_pos_weight <- Number_of_Normal_Transactions / Number_of_Fraud_Transactions
print(paste("scale_pos_weight =", round(scale_pos_weight, 2)))
# Result: ~408, meaning each fraud case is treated as 408x more important

# Why? Without this, the model learns to predict "normal" for everything
# scale_pos_weight forces the model to focus on detecting fraud

# CREATE XGBOOST DATA MATRIX -------
DTrain <- xgb.DMatrix(data = X_Train, label = Y_Train)
DTest <- xgb.DMatrix(data = X_Test, label = Y_Test)

# STEP 3: BUILD AND TRAIN XGBOOST ---------
Parameters <- list(
  objective        = "binary:logistic",   # Binary classification (fraud vs normal)
  eval_metric      = "auc",               # Metric for evaluation (good for imbalanced data)
  scale_pos_weight = scale_pos_weight,    # ~408: give fraud 408x weight
  max_depth        = 6,                   # Tree depth (deeper = more complex, higher overfitting risk)
  eta              = 0.1,                 # Learning rate (smaller = slower but steadier learning)
  subsample        = 0.8,                 # Use 80% of rows per tree (prevents overfitting)
  colsample_bytree = 0.8                  # Use 80% of columns per tree (prevents overfitting)
)

# FLAW FIX #4: Remove the callbacks line - it can cause issues
# The evaluation log is automatically stored when you provide 'evals' parameter
set.seed(2026)
XGB_Model <- xgb.train(
  params = Parameters,
  data = DTrain,
  nrounds  = 100,
  evals = list(train = DTrain, test = DTest),
  verbose = 1,
  print_every_n = 10
)

# PLOT TRAINING PROGRESS --------
Training_Results <- attr(XGB_Model, "evaluation_log")

plot(Training_Results$iter, Training_Results$train_auc,
     type = "l", col = "navy", lwd = 2,
     ylim = c(0.95, 1.0),  # Zoom in for better visualization
     main = "XGBoost Training Progress: AUC Over Iterations",
     xlab = "Number of Trees (Rounds)",
     ylab = "AUC Score",
     grid = TRUE)
lines(Training_Results$iter, Training_Results$test_auc,
      col = "darkred", lwd = 2)
legend("bottomright",
       legend = c("Train AUC", "Test AUC"),
       col = c("navy", "darkred"),
       lty = 1,
       lwd = 2)

# Check if there is overfitting
train_final <- Training_Results$train_auc[100]
test_final <- Training_Results$test_auc[100]
print(paste("Final Train AUC:", round(train_final, 4)))
print(paste("Final Test AUC:", round(test_final, 4)))
# If difference > 0.05, there might be overfitting

# STEP 4: MODEL EVALUATION ------

# Predict probabilities on test set
XGB_Pred <- predict(XGB_Model, DTest)

summary(XGB_Pred)
# Check how many predictions are above 0.5 threshold
table(XGB_Pred > 0.5)

# Extract actual test labels
Test_Labels <- getinfo(DTest, "label")

# CONFUSION MATRIX AT 0.5 THRESHOLD ---------
# FLAW FIX #5: Note that 0.5 threshold may not be optimal for imbalanced data
# In practice, you might use 0.3 or 0.4 threshold for fraud detection
XGB_Class <- ifelse(XGB_Pred > 0.5, 1, 0)

confusionMatrix(as.factor(XGB_Class),
                as.factor(Test_Labels),
                positive = "1")

# The output should show:
# - True Positives: fraud correctly caught
# - True Negatives: normal correctly identified
# - False Positives: normal wrongly flagged as fraud
# - False Negatives: fraud wrongly missed

# ROC CURVE -------
if (!require(pROC)) install.packages("pROC")
library(pROC)

roc_obj <- roc(Test_Labels, XGB_Pred)
plot(roc_obj, col = "navy", lwd = 2,
     main = "ROC Curve - XGBoost Fraud Detection",
     xlab = "False Positive Rate",
     ylab = "True Positive Rate")

auc_value <- auc(roc_obj)
print(paste("XGBoost AUC:", round(auc_value, 4)))

# PRECISION-RECALL CURVE -----
# FLAW FIX #6: PR curve is more useful than ROC for imbalanced data
if (!require(PRROC)) install.packages("PRROC")
library(PRROC)

# Create PR curve----------
pr_obj <- pr.curve(
  scores.class0 = XGB_Pred[Test_Labels == 1],  # Fraud predictions
  scores.class1 = XGB_Pred[Test_Labels == 0],  # Normal predictions
  curve = TRUE
)

plot(pr_obj,
     main = "Precision-Recall Curve - XGBoost",
     col = "navy", lwd = 2)

print(paste("AUPRC:", round(pr_obj$auc.integral, 4)))

# FEATURE IMPORTANCE ---------------
Importance <- xgb.importance(model = XGB_Model)

xgb.plot.importance(Importance, top_n = 15,
                    main = "Top 15 Important Features for Fraud Detection",
                    xlab = "Importance Score")

# The most important features are V4, V12, V14, V10, etc.
# These should match the features with largest differences we found in EDA

# COMPARISON WITH LOGISTIC REGRESSION -----
# Train logistic regression as baseline
Logistic_Model <- glm(Class ~ . - X,
                      data = Train_Data,
                      family = binomial)

Logistic_Pred <- predict(Logistic_Model,
                         newdata = Test_Data,
                         type = "response")

Logistic_ROC <- roc(Test_Labels, Logistic_Pred)

# COMPARE BOTH MODELS -----
plot(roc_obj, col = "navy", lwd = 2,
     main = "ROC Curve Comparison: XGBoost vs Logistic Regression")
lines(Logistic_ROC, col = "darkred", lwd = 2)

legend("bottomright",
       legend = c(paste("XGBoost AUC =", round(auc(roc_obj), 4)),
                  paste("Logistic Reg AUC =", round(auc(Logistic_ROC), 4))),
       col = c("navy", "darkred"),
       lty = 1,
       lwd = 2)

# Print both confusion matrices-----
print("===== XGBoost Confusion Matrix =====")
XGB_Class <- ifelse(XGB_Pred > 0.5, 1, 0)
print(confusionMatrix(as.factor(XGB_Class),
                      as.factor(Test_Labels),
                      positive = "1"))

print("===== Logistic Regression Confusion Matrix =====")
Logistic_Class <- ifelse(Logistic_Pred > 0.5, 1, 0)
print(confusionMatrix(as.factor(Logistic_Class),
                      as.factor(Test_Labels),
                      positive = "1"))

# KEY INSIGHTS -----
# 1. XGBoost generally has higher sensitivity (catches more fraud)
# 2. Logistic Regression may have higher specificity (fewer false alarms)
# 3. For fraud detection, sensitivity (recall) matters more than specificity
# 4. You might want to lower the threshold from 0.5 to 0.3 or 0.4
#    to catch more fraud, accepting more false positives