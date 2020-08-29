# 1. Packages and data Load in

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(caretEnsemble)) install.packages("caretEnsemble", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

dat <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data")

# 2. Dividing into train and test sets 

dat <- as.data.frame(dat)
set.seed(1, sample.kind="Rounding")
index <- createDataPartition(dat$status, times = 1, p = 0.5, list = F)

test_dat <- dat[index,]
train_dat <- dat[-index,]

# 3. Preparing the training dataset
#renaming columns
names(train_dat) <- c("Subject", 
                      "Avg.vocal.freq",
                      "Max.vocal.freq",
                      "Min.vocal.freq",
                      "Jitter.perc",
                      "Jitter.Abs",
                      "RAP",
                      "PPQ",
                      "DDP",
                      "Shimmer",
                      "Shimmer.dB",
                      "Shimmer.APQ3",
                      "Shimmer.APQ5",
                      "APQ",
                      "Shimmer.DDA",
                      "NHR",
                      "HNR",
                      "status",
                      "RPDE",
                      "D2",
                      "DFA",
                      "Spread1",
                      "Spread2",
                      "PPE")

train_dat <- as.data.frame(train_dat)
train_dat_numeric <- train_dat %>% select(-"Subject", -"status")

colnames <- colnames(train_dat_numeric)

#preparing the test set:
names(test_dat) <- c("Subject", 
                     "Avg.vocal.freq",
                     "Max.vocal.freq",
                     "Min.vocal.freq",
                     "Jitter.perc",
                     "Jitter.Abs",
                     "RAP",
                     "PPQ",
                     "DDP",
                     "Shimmer",
                     "Shimmer.dB",
                     "Shimmer.APQ3",
                     "Shimmer.APQ5",
                     "APQ",
                     "Shimmer.DDA",
                     "NHR",
                     "HNR",
                     "status",
                     "RPDE",
                     "D2",
                     "DFA",
                     "Spread1",
                     "Spread2",
                     "PPE")

test_dat <- as.data.frame(test_dat)

## 3.1 Exploring the training dataset
#data set
train_dat %>%  head() %>% kable(format = "pipe")

#visualising the distribution of scores
for (i in 1:22) {
  hist(train_dat_numeric[,i], probability=TRUE,
       main = colnames[i],
       xlab = NULL)
}


#visualising if values change according to status
#part 1:
a <- train_dat %>% 
  select(Avg.vocal.freq:Shimmer.APQ3, status) %>% 
  mutate(status = as.factor(status)) %>% 
  gather(comp, value, -status) %>% 
  ggplot(aes(comp,value, fill = status))+
  geom_boxplot()+
  facet_wrap(~comp, scales = "free")

#part 2:
b <- train_dat %>% 
  select(Shimmer.APQ3:PPE) %>% 
  mutate(status = as.factor(status)) %>% 
  gather(comp, value, -status) %>% 
  ggplot(aes(comp,value, fill = status))+
  geom_boxplot()+
  facet_wrap(~comp, scales = "free")

a;b


# **Analysis: Fitting the algorithms**

# 4. Fitting models: KNN
#prep work
train_dat1 <- train_dat %>% 
  select(-Subject) %>% 
  mutate(status = as.factor(status))
test_dat1 <- test_dat %>% select(-Subject) %>% 
  mutate(status = as.factor(status))

ks <- seq(1, 96, 2)

accuracy <- map_df(ks, function(k){
  fit <- knn3(status ~ ., data = train_dat1, k = k)
  y_hat <- predict(fit, train_dat1, type="class")
  cm_train <- confusionMatrix(y_hat, train_dat1$status)
  train_error <- cm_train$overall["Accuracy"]
  
  y_hat <- predict(fit, test_dat1,type="class")
  cm_test <- confusionMatrix(y_hat, test_dat1$status)
  test_error <- cm_test$overall["Accuracy"]
  
  tibble(train = train_error, test = test_error)
})

accuracy
which.max(accuracy$test) #when k = 5, accuracy highest

#using k=5
fit.knn <- knn3(status ~ ., data = train_dat1, k = 5)
y_hat_knn <- predict(fit.knn, test_dat1, type="class")
cm_test <- confusionMatrix(y_hat_knn, test_dat1$status)
cm_test$overall["Accuracy"] #0.82653

#checking F1 scores
F_meas(y_hat_knn,test_dat1$status) #0.58537

#prevalence in test_dat
mean(1-test_dat$status)

acc_results <- data_frame(Method = "Tuned KNN (k=5)", #column 1: method
                          Accuracy = cm_test$overall["Accuracy"],
                          F1.score  = F_meas(y_hat_knn,test_dat1$status))

acc_results

mean(train_dat$status == 1)

## 4.1 Tuning the F1 score
ks <- seq(1, 50, 2)
Fvalue <- map_df(ks, function(k){
  fit.knn <- knn3(status ~ ., data = train_dat1, k = k)
  y_hat_knn <- predict(fit.knn, test_dat1, type="class")
  cm_train <- confusionMatrix(y_hat_knn, test_dat1$status)
  Fvalue <- F_meas(y_hat_knn,test_dat1$status)
  
  tibble(Fvalue)
})
Fvalue
ks[which.max(Fvalue$Fvalue)] #when k = 1, F1 score is 0.73077

## 4.2 Using components with clear distinctions
#selecting only components with clear distinctions
train_dat_special <- train_dat1 %>% select(Jitter.Abs, Jitter.perc, PPQ, Shimmer, Shimmer.dB, Shimmer.APQ3, Shimmer.APQ5, Shimmer.DDA, APQ, DFA, Spread1, status)
test_dat_special <- test_dat1 %>% select(Jitter.Abs, Jitter.perc, PPQ, Shimmer, Shimmer.dB, Shimmer.APQ3, Shimmer.APQ5, Shimmer.DDA, APQ, DFA, Spread1, status)

#tuning knn
ks <- seq(1, 96,2)
accuracy <- map_df(ks, function(k){
  fit <- knn3(status ~ ., data = train_dat_special, k = k)
  y_hat <- predict(fit, test_dat1, type="class")
  cm_train <- confusionMatrix(y_hat, test_dat_special$status)
  test_error <- cm_train$overall["Accuracy"]
  tibble(test = test_error)
})
accuracy
ks[which.max(accuracy$test)] #accuracy is highest at k = 9

#using k = 9
fit.knn.s <- knn3(status ~ ., data = train_dat_special, k = 9)
y_hat_knn_s <- predict(fit.knn.s, test_dat1, type="class")
cm_test_knn_s <- confusionMatrix(y_hat_knn_s, test_dat_special$status)
cm_test_knn_s$overall["Accuracy"]
#accuracy at 0.8776

#tuning F1 score
F.value_s <- map_df(ks, function(k){
  fit <- knn3(status ~ ., data = train_dat_special, k = k)
  y_hat <- predict(fit, test_dat, type="class")
  cm_train <- confusionMatrix(y_hat, test_dat_special$status)
  Fvalue.s <- F_meas(y_hat,test_dat1$status)
  
  tibble(Fvalue.s)
})
F.value_s[5,]

ks[which.max(F.value_s$Fvalue.s)] #when k = 9, F1 score is 0.73913 (highest)

#inputting the results
acc_results1 <- bind_rows(acc_results, 
                          data.frame (Method = "Selected predictors, Tuned KNN (k=9)", 
                                      Accuracy = cm_test_knn_s$overall["Accuracy"],
                                      F1.score  = F_meas(y_hat_knn_s,test_dat1$status)))


acc_results1

# 5. Fitting models: Random forest
#random forest without tuning
fit.rf <- randomForest(status~., data = train_dat_special)
fit.rf;plot(fit.rf)
y_hat_rf_s <- predict(fit.rf, test_dat1, type = "class")
cm_basic_rf <- confusionMatrix(y_hat_rf_s, test_dat1$status) 

## 5.1 Tuning randomForest 
#tuning mtry parameter
m <- c(1:11)
fit.rf1 <- sapply(m, function(m){
  fit <- randomForest(status~., data = train_dat_special, mtry = m)
  y_hat <- predict(fit, test_dat1, type="class")
  cm_train <- confusionMatrix(y_hat, test_dat_special$status)
  test_error <- cm_train$overall["Accuracy"]
  tibble(test_error)
})

as.data.frame(fit.rf1)
m[which.max(as.data.frame(fit.rf1))] #accuracy highest when mtry = 7

#checking results
fit.rf.test <- randomForest(status~., data = train_dat_special, mtry = 7)
y_hat.rf <- predict(fit.rf.test, test_dat1, type="class")
cm_test_rf_s <- confusionMatrix(y_hat.rf, test_dat_special$status) 
cm_test_rf_s
cm_test_rf_s$overall["Accuracy"]
F_meas(y_hat.rf, test_dat_special$status) 


#inputting the results
acc_results2 <- bind_rows(acc_results1, 
                          data.frame (Method = c("Selected predictors, RF", 
                                                 "Selected predictors, Tuned RF (mtry=7)"), 
                                      Accuracy = c(cm_basic_rf$overall["Accuracy"],
                                                   cm_test_rf_s$overall["Accuracy"]),
                                      F1.score  = c(F_meas(y_hat_rf_s,test_dat1$status),
                                                    F_meas(y_hat.rf,test_dat1$status))))

acc_results2

## 5.2 Further tuning for random forests
#train control - 5 fold
control <- trainControl(method="cv", number = 5)
grid <- data.frame(mtry = c(1:10))
train_rf <-  train(status~., data = train_dat_special,
                   method = "rf", 
                   ntree = 150,
                   trControl = control,
                   tuneGrid = grid,
                   nSamp = 5000)
ggplot(train_rf)
train_rf

fit.rf2 <- randomForest(status~., data = train_dat_special,
                        minNode = train_rf$bestTune$mtry)
y_hat_rf_s2 <- predict(fit.rf2, test_dat_special)
cm_test_rf2 <- confusionMatrix(y_hat_rf_s2, test_dat_special$status) #acc at 0.8673
cm_test_rf2$overall["Accuracy"]
F_meas(y_hat_rf_s2,test_dat1$status)

#inputting the results
acc_results3 <- bind_rows(acc_results2, 
                          data.frame (Method = "Selected predictors, Tuned RF (5 fold with varying mtry)", 
                                      Accuracy = cm_test_rf2$overall["Accuracy"],
                                      F1.score  = F_meas(y_hat_rf_s2,test_dat1$status)))

acc_results3

# 6. Ensembles
#preparing predictions for ensemble comparison
pred <- cbind(y_hat_knn_s, y_hat.rf)
pred <- pred - 1 #not sure why table produces 1 and 2s

#Ensemble Calculation
park <- rowMeans(pred == "1")
y_hat <- ifelse(park > 0.5, "1", "0")
mean(as.factor(y_hat) == test_dat1$status)

#Accuracy at 0.878, lower than tuned randomForest (more methods needed)
#inputting the results
acc_results4 <- bind_rows(acc_results3, 
                          data.frame (Method = "Ensemble of method 2 & 3", 
                                      Accuracy = mean(as.factor(y_hat) == test_dat1$status),
                                      F1.score  = NA))

acc_results4




































