library(caret)
library(doParallel)
library(oce)
library(signal)
registerDoParallel(6)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

filePath <- "./wear_data/raw_data_wear_%s.csv"

# Because the sample with the most accelerometer features has 424 accel. values, read 500 accel values
xAccel <- read.table(sprintf(filePath, "x"), sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:500, sep='')))
yAccel <- read.table(sprintf(filePath, "y"), sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:500, sep='')))
zAccel <- read.table(sprintf(filePath, "z"), sep=',', fill = T, col.names = c('gesture', 'person', 'sample', paste('acc', 1:500, sep='')))

interpolateAccelData <- function(accelData, numberOfValues){
  stepwidth <- 1/numberOfValues
  
  newAccelData <- accelData[,1:3]
  
  newAccelData[,4:(numberOfValues+3)] <- t(apply(accelData[,-(1:3)], 1, function(row){
    rowClean <- row[!is.na(row)]
    approx(x = seq(0,1,1/(length(rowClean)-1)), y = rowClean, xout = seq(0,1-stepwidth,stepwidth))$y
  }))

  newAccelData
}

xAccel <- interpolateAccelData(xAccel, numberOfValues = 30)
yAccel <- interpolateAccelData(yAccel, numberOfValues = 30)
zAccel <- interpolateAccelData(zAccel, numberOfValues = 30)

# combine x, y and z acceleration into one dataframe
nOtherCols <- ncol(xAccel[, 1:3])
nAccelCols <- ncol(xAccel[, -(1:3)])
xyzAccel <- xAccel[,1:3]
xyzAccel[(0*nAccelCols+nOtherCols+1):(1*nAccelCols+nOtherCols)] <- xAccel[, -(1:3)]
xyzAccel[(1*nAccelCols+nOtherCols+1):(2*nAccelCols+nOtherCols)] <- yAccel[, -(1:3)]
xyzAccel[(2*nAccelCols+nOtherCols+1):(3*nAccelCols+nOtherCols)] <- zAccel[, -(1:3)]

rm(xAccel, yAccel, zAccel, nOtherCols, nAccelCols, filePath, interpolateAccelData) # clean environment

# save/load for later use
saveRDS(object = xyzAccel, file = 'xyzAccel_raw_interpolated_data_30.RData')
xyzAccel <- readRDS('xyzAccel_raw_interpolated_data_30.RData')

plot(as.numeric(xyzAccel[1,-(1:3)]), type='l', ylim=c(-6,6), ylab="acceleration [G]", xlab="featureNr - concat(X | Y | Z)", main="acceleration values of 3 samples", sub="(30 feature values per axis)")
lines(as.numeric(xyzAccel[2,-(1:3)]), col=2)
lines(as.numeric(xyzAccel[3,-(1:3)]), col=3)
legend(75, 6, legend=c(as.character(xyzAccel[1,1]), as.character(xyzAccel[2,1]), as.character(xyzAccel[3,1])), col=c(1, 2, 3),  lty=1)

# Lowpass Filter
#----------------
# apply lowpass filter on each axis
# not needed because we only used a very low numbers of values for the interpolation
#
# filterLength <- 101 #adjust to length
# xyzAccelFiltered = xyzAccel[,1:3]
# xyzAccelFiltered[,4:503] <- t(apply(xyzAccel[,4:503], 1, function(v){
#   lowpass(v, n=filterLength)
# }))
# xyzAccelFiltered[,504:1003] <- t(apply(xyzAccel[,504:1003], 1, function(v){
#   lowpass(v, n=filterLength)
# }))
# xyzAccelFiltered[,1004:1503] <- t(apply(xyzAccel[,1004:1503], 1, function(v){
#   lowpass(v, n=filterLength)
# }))


# balance check
table(xyzAccel$gesture)
hist(as.numeric(xyzAccel$gesture)) # balanced with 270 each

# concatinated gesture plot
matplot(t(subset(xyzAccel, gesture == "up")[,-(1:3)]), type="l", col=rgb(0, 0, 0, 0.1), ylab="acceleration [G]", xlab="featureNr - concat(X | Y | Z)", main="up")
matplot(t(subset(xyzAccel, gesture == "down")[,-(1:3)]), type="l", col=rgb(0, 0, 0, 0.1), ylab="acceleration [G]", xlab="featureNr - concat(X | Y | Z)", main="down")
matplot(t(subset(xyzAccel, gesture == "left")[,-(1:3)]), type="l", col=rgb(0, 0, 0, 0.1), ylab="acceleration [G]", xlab="featureNr - concat(X | Y | Z)", main="left")


# calculate mean of all 'up' and 'down' movements and plot it
# this should give an idea of how the average up/down movement looks like
xyzAccel_means_up <- colMeans(subset(xyzAccel, gesture == "up")[,-(1:3)])
xyzAccel_means_down <- colMeans(subset(xyzAccel, gesture == "down")[,-(1:3)])
xyzAccel_means_left <- colMeans(subset(xyzAccel, gesture == "left")[,-(1:3)])
plot(xyzAccel_means_up, type='l', col=1, ylim=c(-4,7), ylab="mean acceleration [G]", xlab="featureNr - concat(X | Y | Z)", main="comparison of mean acceleration of 3 gestures")
lines(xyzAccel_means_down, col=2)
lines(xyzAccel_means_left, col=3)
legend(-1, -1.7, legend=c("up", "down", "left"), col=c(1, 2, 3),  lty=1)


set.seed(12345) # make it reproducible
# use a training set with ONLY 30% of the data to speed up training, INCREASE FOR FINAL TRAINING!
indexes_train <- createDataPartition(xyzAccel$gesture, p = 0.7, list = F) 
indexes_test <- (1:nrow(xyzAccel))[-indexes_train]

training <- xyzAccel[indexes_train,]
testing <- xyzAccel[indexes_test,]

trControl <- trainControl(method = 'repeatedcv', 
                          number = 10, #USE 10 FOR FINAL TRAINING!
                          repeats = 20, #USE 20 FOR FINAL TRAINING!
                          returnData = F, 
                          classProbs = T, 
                          returnResamp = 'final', 
                          allowParallel = T,
                          preProcOptions = list(thresh = 0.99))

models <- list()

#------------------------
# Train KNN model
#------------------------
trainKnn <- function(data){
  train(x = data[,-(1:3)],
        y = data$gesture,
        preProcess = NULL,
        method = 'knn', 
        tuneGrid = expand.grid(k=1:5),
        metric = 'Kappa',
        trControl = trControl)
}

models$modelKnn <- trainKnn(training)

models$modelKnn
plot(models$modelKnn)


#------------------------
# Train KNN model with PCA
#------------------------
trainKnnPca <- function(data){
  train(x = data[,-(1:3)],
        y = data$gesture,
        preProcess = c('pca'), 
        method = 'knn', 
        tuneGrid = expand.grid(k=1:5),
        metric = 'Kappa',
        trControl = trControl)
}

models$modelKnnPca <- trainKnnPca(training)

models$modelKnnPca$preProcess
models$modelKnnPca
plot(models$modelKnnPca)


#------------------------
# Train LDA model
#------------------------
trainLda <- function(data){
  train(x = data[,-(1:3)],
        y = data$gesture,
        preProcess = NULL, 
        method = 'lda', 
        tuneGrid = NULL,
        metric = 'Kappa',
        trControl = trControl)
}

models$modelLda <- trainLda(training)

models$modelLda$preProcess
models$modelLda


#------------------------
# Train LDA model with PCA
#------------------------
trainLdaPca <- function(data){
  train(x = data[,-(1:3)],
        y = data$gesture,
        preProcess = c('pca'), 
        method = 'lda', 
        tuneGrid = NULL,
        metric = 'Kappa',
        trControl = trControl)
}

models$modelLdaPca <- trainLdaPca(training)

models$modelLdaPca$preProcess
models$modelLdaPca


#------------------------
# Train Random Forest model
#------------------------
trainRf <- function(data){
  train(x = data[,-(1:3)],
        y = data$gesture,
        preProcess = NULL, 
        method = 'rf', 
        tuneGrid = expand.grid(mtry = 3**(0:5)),
        metric = 'Kappa',
        trControl = trControl)
}

models$modelRf <- trainRf(training)

models$modelRf
plot(models$modelRf)


#------------------------
# Train Random Forest model with PCA
#------------------------
trainRfPca <- function(data){
  train(x = data[,-(1:3)],
        y = data$gesture,
        preProcess = c('pca'), 
        method = 'rf', 
        tuneGrid = expand.grid(mtry = 3**(0:5)),
        metric = 'Kappa',
        trControl = trControl)
}

models$modelRfPca <- trainRfPca(training)

models$modelRfPca$preProcess
models$modelRfPca
plot(models$modelRfPca)


saveRDS(object = models, file = 'all_models.RData')
models <- readRDS('all_models.RData')

# Compare models
results <- resamples(models)
summary(results)
bwplot(results)


# details of KNN with PCA
#-------------------------

#confusion matrix
cvConfMatrix <- confusionMatrix(models$modelKnnPca)
cvConfMatrix
levelplot(sweep(x = cvConfMatrix$table, STATS = colSums(cvConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))

#test with test data set
testPredicted <- predict(models$modelKnnPca, newdata = testing[,-(1:3)])
testConfMatrix <- confusionMatrix(data = testPredicted, reference = testing$gesture)
testConfMatrix # Accuracy : 0.9738
levelplot(sweep(x = testConfMatrix$table, STATS = colSums(testConfMatrix$table), MARGIN = 2, FUN = '/'), col.regions=gray(100:0/100))
