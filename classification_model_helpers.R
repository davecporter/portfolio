## CLASSIFICATION MODEL COMPARATOR FUNCTIONS
# These functions are primarily used in `classification_model_comparisons.Rmd`, 
# `ISLR_ex_4_classification.Rmd` and `ISLR_ex_5_resampling.Rmd`

library(dplyr)
library(tidyr)
library(purrr)
library(class)
library(MASS)
select <- dplyr::select

## GENERATE CONFUSION MATRIX
# Accepts predictions or probabilities to generate confusion matrix with multiple definitions for errors 
# Outputs: precision, recall, f1 score, accuracy and null error rate
# Arguments:
# predicted   vector of binomial predictions or probabilities
# actual      vector of actual binomial classifications to compare predictions to
# p_cutoff    probability cutoff for classification if probabilities are passed as `predicted`
# n_digits    number of digits to show in the output
# aka         boolean input to show alternative error type names

confusion_matrix <- function(predicted, actual, p_cutoff=0.5, n_digits=2, aka=TRUE){  
  contrasts <- levels(actual)
  if(is.numeric(predicted) == TRUE) predicted <- ifelse(predicted < p_cutoff, contrasts[1], contrasts[2])
  
  # confusion matrix
  conf_matrix <- table(factor(predicted, levels = contrasts), actual) 
  prop_matrix_row <- prop.table(conf_matrix, margin=1)
  prop_matrix_col <- prop.table(conf_matrix, margin=2)
  conf_matrix_m <- addmargins(conf_matrix)
  prop_matrix_row_m <- addmargins(prop_matrix_row, margin=2)
  prop_matrix_col_m <- addmargins(prop_matrix_col, margin=1)
  
  # terms
  accuracy <- mean(predicted == actual) # fraction correct predictions
  precision <- prop_matrix_row[2,2] # positive predictive value
  npv <- prop_matrix_row[1,1] # negative predictive value
  sensitivity <- prop_matrix_col[2,2] # true positive rate
  specificity <- prop_matrix_col[1,1] # true negative rate
  f1_score_0 <- 2 * (npv * specificity) / (npv + specificity)
  f1_score_1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
  null_error_rate <- max(mean(actual == contrasts[1]),
                         mean(actual == contrasts[2])) # error rate when predicting majority class
  
  summary <- data.frame(accuracy, null_error_rate, p_cutoff) %>% mutate_all(list(~round(., n_digits)))
  
  table <- data.frame(contrast = c(0,1), class = contrasts,
                      precision = c(precision, npv),
                      recall = c(specificity, sensitivity),
                      f1_score = c(f1_score_0, f1_score_1)) %>%
    mutate_if(is.numeric, list(~round(., n_digits)))
  
  if(aka == TRUE) {
    table <- table %>%
      mutate(precision = paste(precision, c("neg. predictive value", "pos. predictive value")),
             recall = paste(recall, c("selectivity, true neg. rate", "specificity, true pos. rate ")))
  }
  
  list(conf_matrix_m, prop_matrix_row_m, prop_matrix_col_m, summary, table)
}

## RUN CLASSIFICATION MODEL
# Runs a choice of logistic regression, linear discriminant analysis, quadratic discriminant analysis or k nearest neighbours
# Outputs `confusion_martix()` from `helpers.R`
# Arguments:
# df          dataframe for analysis
#             has to take form of one column binary response as factor and p continuous predictors
# response    string of response variable name
# predictors  vector of string(s) for predictor variable names, "." uses all in `df`
# method      choice of "glm", "lda", "qda", "knn"
# tt_split    fraction of `df` to use as training set
# k           k coefficient for knn
# scaled      boolean value to scale predictors for knn
# seed        set seed, defaults to random

run_model <- function(df, response, predictors=".", method, tt_split=0.7, k=1, scaled=TRUE, seed=round(runif(1)*1e6)){
  set.seed(seed)
  # df <- df %>% mutate_("df[[response]]" = as.factor("df[[response]]"))
  train <- df %>% sample_frac(tt_split)
  test <- anti_join(df, train)
  
  formula <- paste("train[[response]] ~", paste(predictors, collapse="+")) %>% as.formula()
  if(method == "glm") {
    model_fit <- glm(formula, data = train %>% select(-response), family = "binomial")
    pred <- predict(model_fit, newdata = test %>% select(-response), type="response")
  } else if(method == "lda") {
    model_fit <- lda(formula, train %>% select(-response))
    pred <- predict(model_fit, newdata = test %>% select(-response))$class
  } else if(method == "qda") {
    model_fit <- qda(formula, train %>% select(-response))
    pred <- predict(model_fit, newdata = test %>% select(-response))$class
  } else if(method == "knn") {
    
    if(scaled == TRUE) {
      df_scaled <- df %>% select_if(is.numeric) %>% # scale numeric data for knn
        scale() %>% as.data.frame() %>% # removes all non-numeric data
        bind_cols(data.frame(response = df[[response]])) # puts responses back to df
      set.seed(seed)
      train_knn <- df_scaled %>% sample_frac(tt_split)
      test_knn <- anti_join(df_scaled, train_knn)
    } else {
      train_knn <- train
      test_knn <- test
    }
    pred <- knn(train_knn %>% select(-response), test_knn %>% select(-response), train_knn %>% pull(response), k = k)
  }
  confusion_matrix(pred, test %>% pull(response))
}

## GET ACCURACY OF MODEL
# Output data frame of accuracies of multiple classification model runs from `run_model()` above
# Arguments:
# df          dataframe for analysis
#             has to take form of one column binary response as factor and p continuous predictors
# response    string of response variable name to pass to `run_model()`
# predictors  vector of string(s) of predictor variable names to pass to `run_model()`, "." uses all in `df`
# method      classification model choice of "glm", "lda", "qda", "knn" to pass to `run_model()`
# n_trials    number of trials/accuracies to output
# tt_split    fraction of `df` to use as training set to pass to `run_model()`
# k           k coefficient for knn to pass to `run_model()`
# scaled      boolean value to scale predictors for knn to pass to `run_model()`

get_accuracy <- function(df, response, predictors=".", method, n_trials=10, tt_split=0.7, k=1, scaled=TRUE){ 
  accuracy <- list()
  for(i in 1:n_trials){
    result <- run_model(df, response, predictors, method, tt_split, k, scaled)
    accuracy[i] <- result[[4]]["accuracy"]
  }
  data.frame(accuracy = unlist(accuracy))
}

## COMPARE CLASSIFICATION METHOD
# Compares accuracies of glm, lda, qda and knn using `run_model()` and `get_accuracy()` above
# Outputs data frame of mean accuracies for each method and boxplot of accuracy distribution
# Arguments:
# df          dataframe for analysis
#             has to take form of one column binary response as factor and p continuous predictors
# response    string of response variable name to pass to `get_accuracy()` and `run_model()`
# predictors  vector of string(s) of predictor variable names to pass to `get_accuracy()` and `run_model()`, 
#             "." uses all in `df`
# method      classification model choice of "glm", "lda", "qda", "knn" to pass to `get_accuracy()` and `run_model()`
# n_trials    number of trials/accuracies to consider
# tt_split    fraction of `df` to use as training set to pass to `get_accuracy()` and `run_model()`
# k           k coefficient for knn to pass to `get_accuracy()` and `run_model()`
# scaled      boolean value to scale predictors for knn to pass to `get_accuracy()` and `run_model()`

compare_class_method <- function(df, response, predictors=".", methods=c("glm", "lda", "qda", "knn"), 
                                 n_trials=10, tt_split=0.7, k=1, scaled=TRUE){
  acc <- data.frame(method = methods, df %>% nest()) %>%
    mutate(model = map2(data, method, ~get_accuracy(., response, predictors, method=.y, n_trials, tt_split, k, scaled))) %>%
    unnest(model)
  acc_df <- acc %>% group_by(method) %>% summarise(mean_acc = mean(accuracy))
  plot <- ggplot(acc, aes(method, accuracy)) + geom_boxplot()
  list(acc_df, plot)
}

## OPTIMISE KNN k COEFFICIENT
# Outputs chart of knn accuracy vs ascending values for k
# Arguments:
# df          dataframe for analysis
#             has to take form of one column binary response as factor and p continuous predictors
# response    string of response variable name
# predictors  vector of string(s) for predictor variable names, "." uses all in `df`
# tt_split    fraction of `df` to use as training set
# nk_max      max k coefficient 
# scaled      boolean value to scale predictors for knn
# seed        set seed, defaults to random        

optimise_k <- function(df, response, predictors=".", tt_split=0.7, nk_max=50, 
                       scaled=TRUE, n_trials=30, seed=round(runif(1)*1e6)){
  accuracy <- list()
  for(ki in 1:nk_max){
    ki_result <- run_model(df, response, predictors, method = "knn", tt_split, k = ki, scaled, seed)
    accuracy[ki] <- ki_result[[4]]["accuracy"]
  }
  
  k_acc <- data.frame(k = 1:nk_max, accuracy = unlist(accuracy))
  ggplot(k_acc, aes(k, accuracy)) + geom_line() + geom_point()
}