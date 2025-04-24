# Practical Homework 1: Youth Drug use Revised

#Load Package
library(tree)
library(randomForest)
library(gbm)
library(dplyr)

#Load dataset
load("C:/Users/eokoloko/Downloads/youth_data.Rdata")

youth_experience_cols
substance_cols
demographic_cols


#DATA PROCESSING FOR PROBLEM 1

# Select relevant variables i wanted to explore for this problem.
df_binary <- df[, c("MRJFLAG", "IRSEX", "NEWRACE2", "POVERTY3", "AVGGRADE", "RLGIMPT")]

# Have to convert to factors with proper labels based on the code book
# IRSEX: 1 = Male, 2 = Female
df_binary$IRSEX <- factor(df_binary$IRSEX, levels = c(1, 2), labels = c("Male", "Female"))
# NEWRACE2: 1–7 = Race categories
df_binary$NEWRACE2 <- factor(df_binary$NEWRACE2, levels = 1:7,
                             labels = c("NonHisp White", "NonHisp Black", "NonHisp Native Am",
                                        "NonHisp HI/PI", "NonHisp Asian", "NonHisp >1 Race", "Hispanic"))
# POVERTY3: 1–3 = Income group
df_binary$POVERTY3 <- factor(df_binary$POVERTY3, levels = 1:3,
                             labels = c("Living in Poverty", "Income <= 2x Poverty", "Income > 2x Poverty"))
# AVGGRADE: 1 = D or lower, 2 = A/B/C
df_binary$AVGGRADE <- factor(df_binary$AVGGRADE, levels = c(1, 2),
                             labels = c("D or Lower", "A/B/C"))
# RLGIMPT: 1 = Disagree, 2 = Agree (religious beliefs influence life decisions)
df_binary$RLGIMPT <- factor(df_binary$RLGIMPT, levels = c(1, 2),
                            labels = c("Disagree", "Agree"))

# Make the target a factor for the binary classification
df_binary$MRJFLAG <- factor(df_binary$MRJFLAG, levels = c(0, 1), labels = c("No", "Yes"))

# Drop any rows with missing data
df_binary <- na.omit(df_binary)

# Check that i have done it correctly
str(df_binary)
table(df_binary$MRJFLAG)


# PROBLEM 1 MODEL IMPLEMENTATION
#Binary Classsifiation - Predicting if a youth has or has not used marijuana before.

#Train test split
set.seed(1)
binary_train_ind <- sample(1:nrow(df_binary), size = 0.8 * nrow(df_binary))
binary_train <- df_binary[binary_train_ind, ]
binary_test <- df_binary[-binary_train_ind, ]

# Fit decision tree
binary_tree <- tree(MRJFLAG ~ ., data = binary_train)


# Predict on test set with unpruned tree
unpruned_pred <- predict(binary_tree, binary_test, type = "class")

# make a confusion matrix for unpruned tree
unpruned_conf_matrix <- table(Predicted = unpruned_pred, Actual = binary_test$MRJFLAG)
cat("Unpruned Tree Confusion Matrix:\n")
print(unpruned_conf_matrix)

# get the test error rate for unpruned tree
unpruned_error <- 1 - sum(diag(unpruned_conf_matrix)) / sum(unpruned_conf_matrix)
cat("Unpruned Tree Test Error Rate:", round(unpruned_error * 100, 2), "%\n\n")


# Now perform Cross-validation to choose best tree size 
cv.binary_train <- cv.tree(binary_tree, FUN = prune.misclass)

# Plot the cross-validation results
plot(cv.binary_train$size, cv.binary_train$dev, type = "b",
     xlab = "Tree Size", ylab = "CV Error", main = "CV Error vs Tree Size")

# Best size from CV
best_size <- cv.binary_train$size[which.min(cv.binary_train$dev)]
cat("Best Tree Size (from CV):", best_size, "\n\n")

# Now Prune the tree
prune.binary_tree <- prune.misclass(binary_tree, best = best_size)

# Make prediction on test set with pruned tree
pruned_pred <- predict(prune.binary_tree, binary_test, type = "class")

# Make a confusion matrix for pruned tree
pruned_conf_matrix <- table(Predicted = pruned_pred, Actual = binary_test$MRJFLAG)
cat("Pruned Tree Confusion Matrix:\n")
print(pruned_conf_matrix)

# get the test error rate for pruned tree
pruned_error <- 1 - sum(diag(pruned_conf_matrix)) / sum(pruned_conf_matrix)
cat("Pruned Tree Test Error Rate:", round(pruned_error * 100, 2), "%\n\n")


# Overall summary of both unpruned and pruned tree:
cat("Summary of Unpruned Tree:\n")
print(summary(binary_tree))

cat("\nSummary of Pruned Tree:\n")
print(summary(prune.binary_tree))

# My tree seems to have been to simple so I will expand the dataset a bit

## CREATING A LARGER DATASET
target_var <- "MRJFLAG"
selected_vars <- c(youth_experience_cols, demographic_cols, target_var)
df_binary_expanded <- df[, selected_vars]

# Replace placeholder values with NA
placeholders <- c(97, 98, 99, 7, 9, 6, 996, 997, 998, 999)
df_binary_expanded  <- df_binary_expanded %>%
  mutate(across(everything(), ~ replace(., . %in% placeholders, NA)))

# Convert all predictors to factors
predictor_cols <- setdiff(names(df_binary_expanded), target_var)

# Convert MRJFLAG to factor
df_expanded$MRJFLAG <- factor(df_expanded$MRJFLAG, levels = c(0, 1), labels = c("No", "Yes"))


df_binary_expanded <- df_binary_expanded %>%
  mutate(across(all_of(predictor_cols), as.factor))

# Drop all the rows that have missing data in the target column
df_binary_expanded  <- df_binary_expanded  %>% drop_na(MRJFLAG)

str(df_binary_expanded)




## MODEL IMPLEMENTATION 2

# Fit decision tree
set.seed(7)
binary_train_ind <- sample(1:nrow(df_binary_expanded), size = 0.8 * nrow(df_binary_expanded))
binary_train <- df_binary_expanded[binary_train_ind, ]
binary_test <- df_binary_expanded[-binary_train_ind, ]


binary_tree2 <- tree(MRJFLAG ~ ., data = df_binary_expanded)
unpruned_pred <- predict(binary_tree2, binary_test, type = "class")
unpruned_conf_matrix <- table(Predicted = unpruned_pred, Actual = binary_test$MRJFLAG)
cat("Unpruned Tree Confusion Matrix:\n")
print(unpruned_conf_matrix)
unpruned_error <- 1 - sum(diag(unpruned_conf_matrix)) / sum(unpruned_conf_matrix)
cat("Unpruned Tree Test Error Rate:", round(unpruned_error * 100, 2), "%\n\n")
cv.binary_train <- cv.tree(binary_tree2, FUN = prune.misclass)
plot(cv.binary_train$size, cv.binary_train$dev, type = "b",
     xlab = "Tree Size", ylab = "CV Error", main = "CV Error vs Tree Size")
best_size <- cv.binary_train$size[which.min(cv.binary_train$dev)]
cat("Best Tree Size (from CV):", best_size, "\n\n")
prune.binary_tree2 <- prune.misclass(binary_tree2, best = best_size)
pruned_pred <- predict(prune.binary_tree2, binary_test, type = "class")
pruned_conf_matrix <- table(Predicted = pruned_pred, Actual = binary_test$MRJFLAG)
cat("Pruned Tree Confusion Matrix:\n")
print(pruned_conf_matrix)
pruned_error <- 1 - sum(diag(pruned_conf_matrix)) / sum(pruned_conf_matrix)
cat("Pruned Tree Test Error Rate:", round(pruned_error * 100, 2), "%\n\n")
cat("Summary of Unpruned Tree:\n")
print(summary(binary_tree))
cat("\nSummary of Pruned Tree:\n")
print(summary(prune.binary_tree))

# I still got the exact same results this suggests that it didnt need to be pruned all along.


#PROBLEM 2
#Multi-Variable Classifiation - Are people with at least one parent (a father) more prone to drug use?

#Data cleaning and preprocessing 

# Hand pick variables
vars_problem2 <- c("IFATHER", "MRJFLAG", "ALCFLAG", "TOBFLAG", "IRSEX", "NEWRACE2")

# Subset and rename the dataset
df_mvc <- df[, vars_problem2]

# Replace placeholders with NA
placeholders <- c(97, 98, 99, 7, 9, 6, 996, 997, 998, 999)
df_mvc[] <- lapply(df_mvc, function(x) replace(x, x %in% placeholders, NA))

# Make variables a factor
df_mvc$IFATHER <- factor(df_mvc$IFATHER,
                         levels = c(1, 2, 3),
                         labels = c("Father in HH", "No Father in HH", "Don't Know"))

df_mvc$MRJFLAG <- factor(df_mvc$MRJFLAG, levels = c(0, 1), labels = c("No", "Yes"))
df_mvc$ALCFLAG <- factor(df_mvc$ALCFLAG, levels = c(0, 1), labels = c("Never Used", "Ever Used"))
df_mvc$TOBFLAG <- factor(df_mvc$TOBFLAG, levels = c(0, 1), labels = c("No Use", "Ever Used"))

df_mvc$IRSEX <- factor(df_mvc$IRSEX, levels = c(1, 2), labels = c("Male", "Female"))

df_mvc$NEWRACE2 <- factor(df_mvc$NEWRACE2,
                          levels = 1:7,
                          labels = c("NonHisp White", "NonHisp Black", "NonHisp Native Am",
                                     "NonHisp HI/PI", "NonHisp Asian", "NonHisp >1 Race", "Hispanic"))

# Drop rows with missing values
df_mvc <- na.omit(df_mvc)

# Train/Test Split
set.seed(42)
train_idx <- sample(1:nrow(df_mvc), size = 0.8 * nrow(df_mvc))
train_df <- df_mvc[train_idx, ]
test_df <- df_mvc[-train_idx, ]

# Model 1: Random Forest (mtry = 2)
rf_m2 <- randomForest(IFATHER ~ .-IFATHER, data = train_df, mtry = 2, importance = TRUE)
pred_m2 <- predict(rf_m2, newdata = test_df)
cat("\nRandom Forest (mtry = 2)\n")
print(table(Predicted = pred_m2, Actual = test_df$IFATHER))
cat("Accuracy:", round(mean(pred_m2 == test_df$IFATHER) * 100, 2), "%\n")
varImpPlot(rf_m2, main = "Importance (mtry = 2)")

# Model 2: Random Forest (mtry = 3) 
rf_m3 <- randomForest(IFATHER ~ .-IFATHER, data = train_df, mtry = 3, importance = TRUE)
pred_m3 <- predict(rf_m3, newdata = test_df)
cat("\nRandom Forest (mtry = 3)\n")
print(table(Predicted = pred_m3, Actual = test_df$IFATHER))
cat("Accuracy:", round(mean(pred_m3 == test_df$IFATHER) * 100, 2), "%\n")
varImpPlot(rf_m3, main = "Importance (mtry = 3)")

# Model 3: Random Forest (mtry = 4) 
rf_m4 <- randomForest(IFATHER ~ .-IFATHER, data = train_df, mtry = 4, importance = TRUE)
pred_m4 <- predict(rf_m4, newdata = test_df)
cat("\nRandom Forest (mtry = 4)\n")
print(table(Predicted = pred_m4, Actual = test_df$IFATHER))
cat("Accuracy:", round(mean(pred_m4 == test_df$IFATHER) * 100, 2), "%\n")
varImpPlot(rf_m4, main = "Importance (mtry = 4)")

# Model 4: Bagging Model (mtry = all predictors)
bagging_model <- randomForest(IFATHER ~ .-IFATHER, data = train_df, mtry = ncol(train_df) - 1, ntree = 500, importance = TRUE)
pred_bag <- predict(bagging_model, newdata = test_df)
cat("\nBagging Model\n")
print(table(Predicted = pred_bag, Actual = test_df$IFATHER))
cat("Accuracy:", round(mean(pred_bag == test_df$IFATHER) * 100, 2), "%\n")
varImpPlot(bagging_model, main = "Importance (Bagging)")



# PROBLEM 3

# Regression Problem- Predicting the number of days alcohol was used in the past year

# I would like to build a gradient boosting model 
# Target variable is iralcfy' alcohol frequency past year (1-365)

# Select relevant predictors and avoiding alcohol frequency vars to prevent data leakage
selected_vars <- c(
  "IRALCFY", "IRMJFY", "TOBFLAG", "MRJFLAG", "IRSEX", "NEWRACE2", "POVERTY3", "INCOME",
  "IMOTHER", "IFATHER", "AVGGRADE", "RLGIMPT", "YOFIGHT2", "YOSELL2", "YOSTOLE2",
  "YOHGUN2", "ARGUPAR", "PRPROUD2", "FRDMJMON", "ANYEDUC3", "RLGDCSN"
)

df_reg <- df[, selected_vars]

# Replace placeholder values with NA
placeholders <- c(97, 98, 99, 7, 9, 6, 996, 997, 998, 999)
df_reg[] <- lapply(df_reg, function(x) replace(x, x %in% placeholders, NA))

# Drop NAs and filter out IRALCFY <= 0 so it doesnt distort the results
df_reg <- na.omit(df_reg)
df_reg <- df_reg[df_reg$IRALCFY > 0, ]

# Log transform the target to make it more meaningful
df_reg$LogIRALCFY <- log(df_reg$IRALCFY)

# Implement boosting model
set.seed(4)
boosting_index <- sample(1:nrow(df_reg), size = 0.8 * nrow(df_reg))
boosting_train <- df_reg[boosting_index, ]
boosting_test <- df_reg[-boosting_index, ]

lambdas <- c(0.001, 0.01, 0.05, 0.1, 0.3)
train_mse <- numeric(length(lambdas))
test_mse <- numeric(length(lambdas))

for (i in 1:length(lambdas)) {
  boosting_model <- gbm(
    IRALCFY ~ . -LogIRALCFY,
    data = boosting_train,
    distribution = "gaussian",
    n.trees = 1000,
    shrinkage = lambdas[i],
    verbose = FALSE
  )
  
  # Training MSE
  train_preds <- predict(boosting_model, boosting_train, n.trees = 1000)
  train_mse[i] <- mean((train_preds - boosting_train$IRALCFY)^2)

  # Testing MSE
  test_preds <- predict(boosting_model, boosting_test, n.trees = 1000)
  test_mse[i] <- mean((test_preds - boosting_test$IRALCFY)^2)

}

par(mfrow = c(1, 2))

plot(lambdas, train_mse, type = "b", xlab = "Shrinkage (Lambda)",
     ylab = "Training MSE", main = "Training MSE vs Lambda")

plot(lambdas, test_mse, type = "b", xlab = "Shrinkage (Lambda)",
     ylab = "Test MSE", main = "Test MSE vs Lambda")

# The best shrinkage for test set
best_lambda <- lambdas[which.min(test_mse)]
cat("Best lambda based on test MSE:", best_lambda, "\n")
train_mse[i]
test_mse[i]






