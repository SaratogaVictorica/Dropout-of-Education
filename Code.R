library(readr)
Data <- read.csv("D:/Program Files/Documents/R/ML/Final Project/dataset.csv")
Data.Omit = na.omit(Data)
#View(Data.Omit)

#Dropout as 1, Enrolled as 2, Graduate as 3
Data.Omit$Target = factor(Data.Omit$Target)
Data.Omit$Target = as.integer(Data.Omit$Target)
#View(Data.Omit)

Data.N = scale(Data.Omit)
Data.N = as.data.frame(Data.N)
#View(Data)

library(caret)
ctrl <- trainControl(method = "cv", number = 10)

set.seed(2023)
Simple.Linear.Model <- train(Target~ .,  data = Data.N,  method = "lm",  trControl = ctrl)
print(Simple.Linear.Model)

set.seed(2023)
LASSO <- train(Target~ .,  data = Data.N,  method = "lasso",  trControl = ctrl)
print(LASSO)

set.seed(2023)
RIDGE <- train(Target~ .,  data = Data.N,  method = "ridge",  trControl = ctrl)
print(RIDGE)

set.seed(2023)
NET <- train(Target~ .,  data = Data.N,  method = "glmnet",  trControl = ctrl)
print(NET)

min(
Simple.Linear.Model$results$RMSE,LASSO$results$RMSE,RIDGE$results$RMSE,NET$results$RMSE)

library(glmnet)

plot(NET)

#Recording the best Tune via code, rather than directly typing.
#Reducing errors.
Optimal.Alpha = NET$bestTune[,1]
Optimal.Lambda = NET$bestTune[,2]

X = as.matrix(Data.N[,-which(names(Data.N) == "Target")])
Y = as.numeric(Data.N$Target)

Elastic.Net.Model <- glmnet(x=X, y=Y, alpha = Optimal.Alpha, lambda = Optimal.Lambda, standardize = TRUE)

Model.Coef = as.data.frame(coef(Elastic.Net.Model, s=Optimal.Lambda, exact=TRUE)[-1,])
colnames(Model.Coef) = "Coefficient"
Model.Coef

GLM.Object = glm(Target ~., data=Data.N, x=TRUE, y=TRUE)
Stepwise.Model <- step(GLM.Object, direction = "both", trace = 0)

GLM.Coef = as.data.frame(Stepwise.Model$coefficients)
colnames(GLM.Coef) = "Coefficient"
GLM.Coef

GLM.Object = glm(Target ~., data=Data.Omit, x=TRUE, y=TRUE)
Stepwise.Model <- step(GLM.Object, direction = "both", trace = 0)

GLM.Coef = as.data.frame(Stepwise.Model$coefficients)
colnames(GLM.Coef) = "Coefficient"
GLM.Coef
