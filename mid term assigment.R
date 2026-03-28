#install and run packages------


library(xgboost)
library(caret)
library(dplyr)

#load the csv file and check how many columns and rows are in the file then show it.

data <- read.csv("D:/ML FOR F IN R/MidTerm coursework/credit-sample.csv")

head(data) #see the first row
str(data)  #check the data types
dim(data)  #rows and columns 

# check how many fraud vs normal transactions we have----
table(data$Class)

# convert to percentage so we can see how imbalanced it is
prop.table(table(data$Class)) * 100

# check if any missing values exist in the whole data-set
sum(is.na(data))

# check basic statistics of Amount (min, max, mean etc.)-----
summary(data$Amount)

# plot histogram to see how Amount is distributed----
hist(data$Amount,
     main = "Transaction Amount Distribution",   # title of graph
     xlab = "Amount",                            # x-axis label
     col = "Darkblue")                          # color of bars

# check summary of Time----
summary(data$Time)

# histogram of Time----
hist(data$Time,
     main = "Transaction Time Distribution",
     xlab = "Time",
     col = "lightgreen")



#Building XGBOOST Model-------

# separate input features (X) and target variable (y)

X <- data[, -31]   # take all columns EXCEPT Class
y <- data$Class    # take only Class column (target)

#Train-Test Split-----

# fix randomness so that every time you run the code,
set.seed(999)   # for reproducibility (same result every time)

# fix randomness so that every time you run the code,
# you get the SAME train-test split (important for reproducibility)
set.seed(123)


# load caret package (used for data splitting)
library(caret)

# create index (80% training, keeps class balance)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)

# training features
X_train <- X[trainIndex, ]

# testing features (remaining 20%)
X_test  <- X[-trainIndex, ]

# training labels
y_train <- y[trainIndex]

# testing labels
y_test  <- y[-trainIndex]


# convert training data to matrix (required for xgboost)
train_matrix <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)

# convert testing data to matrix
test_matrix  <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)


# Set Parameters -------

# define model settings (how model will learn)
params <- list(
  objective = "binary:logistic",   # 0/1 classification (fraud or not)
  eval_metric = "logloss"          # error measure (lower = better)
)


# TRAIN MODEL -----

# train xgboost model using training data
model <- xgboost(
  x = as.matrix(X_train),   # input features (all columns except target)
  y = y_train,              # actual answers (0 = normal, 1 = fraud)
  nrounds = 100             # number of trees (model learns 100 times)
)


# Predict (Probability)-----

# use trained model to predict fraud probability on test data
prob <- predict(model, as.matrix(X_test))   # output will be values between 0 and 1


# convert probability into final prediction (0 or 1)------
final_pred <- ifelse(prob > 0.5, 1, 0)   # if >50% → fraud else normal


# evaluate model performance------

confusionMatrix(as.factor(final_pred), as.factor(y_test))