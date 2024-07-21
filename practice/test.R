#!/usr/bin/env Rscript

Sys.setenv(TZ='GMT')

# dataset[rows, cols]

library(caret)

# attach the iris dataset to the environment
data(iris)

dataset <- iris

# create a list of 80% of the rows in the original dataset we can use for training
validation_index <- createDataPartition(dataset$Species, p=0.80, list=FALSE)
# select 20% of the data for validation
validation <- dataset[-validation_index,]
# use the remaining 80% of data to training and testing the models
dataset <- dataset[validation_index,]





# split input and output
# y vai ser a coluna da especie
# x sao todos os outros atributos

x <- dataset[,1:4]
y <- dataset[,5]

# pdf("boxplot.pdf")

# # boxplot for each attribute on one image
# par(mfrow=c(1,4))

# for(i in 1:4) {
#   boxplot(x[,i], main=names(iris)[i])
# }

# dev.off()

pdf("plot.pdf")
# scatterplot matrix
featurePlot(x=x, y=y, plot="ellipse")
dev.off()

pdf("plot2.pdf")
# scatterplot matrix
featurePlot(x=x, y=y, plot="box")
dev.off()


pdf("plot3.pdf")
# scatterplot matrix
scales <- list(x=list(relation="free"), y=list(relation="free"))
featurePlot(x=x, y=y, plot="density", scales=scales)
dev.off()

# Run algorithms using 10-fold cross validation
control <- trainControl(method="cv", number=10)
metric <- "Accuracy"

# a) linear algorithms
set.seed(7)
fit.lda <- train(Species~., data=dataset, method="lda", metric=metric, trControl=control)
# b) nonlinear algorithms
# CART
set.seed(7)
fit.cart <- train(Species~., data=dataset, method="rpart", metric=metric, trControl=control)
# kNN
set.seed(7)
fit.knn <- train(Species~., data=dataset, method="knn", metric=metric, trControl=control)
# c) advanced algorithms
# SVM
set.seed(7)
fit.svm <- train(Species~., data=dataset, method="svmRadial", metric=metric, trControl=control)
# Random Forest
set.seed(7)
fit.rf <- train(Species~., data=dataset, method="rf", metric=metric, trControl=control)


results <- resamples(list(lda=fit.lda, cart=fit.cart, knn=fit.knn, svm=fit.svm, rf=fit.rf))
summary(results)

pdf("plot4.pdf")
# como o modelo lda neste caso foi o melhor, ficou em primeiro 
dotplot(results)
dev.off()


#print(fit.lda)

# estimate skill of LDA on the validation dataset
predictions <- predict(fit.lda, validation)
confusionMatrix(predictions, validation$Species)