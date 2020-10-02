rm(list=ls())

library(mlbench)
library(class)

# Loading dataset

?PimaIndiansDiabetes

data("PimaIndiansDiabetes")
data <- PimaIndiansDiabetes
head(data)

sum(is.na(data))

summary(data)
str(data)

diabetes <- data$diabetes

normalize <- function(x) {
  num <- x - min(x)
  denom <- max(x) - min(x)
  return(num/denom)
}

dim(data)

mode(data)

lapply(data[,1:8], normalize)
data <- as.data.frame(lapply(data[,1:8], normalize))
head(data)

summary(data)

data$diabetes <- diabetes

head(data)

# Spliting dataset
train <- data[1:600,]
test <- data[601:768,]

head(train)
head(test)

########################################################################################
## Build a kNN model (with k = 10) on the training dataset in R to predict the diabetes 
## (pos or neg). So here we will consider "diabetes" as Class variable. Then test the 
## model on the testing dataset. Calculate accuracy and error rate.

cl <- train$diabetes # defining class - predictor variable

# Knn with k = 10
model <- knn(train[-9], test[-9], cl, k = 10)
model # predicted value on test data set

length(model)
test$diabetes

Accuracy <- mean(model == test$diabetes)
Accuracy # 0.7678571

# Even number k can give different accuracy at each iteration.

model <- knn(train[-9], test[-9], cl, k = 11)
Accuracy <- mean(model == test$diabetes)
Accuracy

####################################################################
# What happens when i change the sample?
####################################################################

set.seed(2000)
library(caTools)
split = sample.split(Y = data$diabetes, SplitRatio = .8)
split #t and f

# mean of y variable in train and test data and the main data will be approx same
traindata = data[split,]
testdata = data[!split,]

cl <- traindata$diabetes

model <- knn(traindata[-9], testdata[-9], cl, k = 11)
Accuracy <- mean(model == testdata$diabetes)
Accuracy

############################################################################
## Perform k-fold validation (with k = 10) on PimaIndiansDiabetes data.

# K fold with k = 10

library(caret)

control <- trainControl(method = "cv", number = 10, classProbs=TRUE,
                        summaryFunction = twoClassSummary)

# will compute the sensitivity, specificity and area under the ROC curve

fit_knn <- train(diabetes ~ ., data=PimaIndiansDiabetes, method="knn",
                 metric="ROC", trControl=control)

fit_knn

# k-Nearest Neighbors 
# 
# 768 samples
# 8 predictors
# 2 classes: 'neg', 'pos' 
# 
# No pre-processing
# Resampling: Cross-Validated (10 fold) 
# Summary of sample sizes: 691, 692, 691, 691, 691, 691, ... 
# Resampling results across tuning parameters:
#   
#   k  ROC        Sens   Spec     
#   5  0.7424900  0.818  0.5145299
#   7  0.7609103  0.836  0.5256410
#   9  0.7751909  0.842  0.5445869
# 
# ROC was used to select the optimal model using the largest value.
# The final value used for the model was k = 9.






