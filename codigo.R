#1. PREPROCESAMIENTO DE LOS DATOS.
library(tidyverse)
library(caret)
library(gridExtra)
library(readxl)
setwd("C:/Users/adria/Desktop/PROYECTOS/breastcancer")
data <- read_excel("clean_data.xlsx")
head(data)

#Comprobacion de valores nulos.
variables <- colnames(data)
sapply(variables,function(nombre_variable){
  any(is.na(nombre_variable))
})

#Corrección de la estructura.
str(data)
data <- mutate_all(data[3:30], as.numeric) %>%
  mutate(diagnosis = as.factor(data$diagnosis))
x <- data[,1:28] %>% scale()
y <- data$diagnosis
str(data)

#Conjuntos de testeo y entrenamiento.
set.seed(1990, sample.kind = "Rounding") 
ind <- createDataPartition(y, times = 1, p = 0.2, list = FALSE) #20% a datos de testeo
test_x <- x[ind,]
test_y <- y[ind]
train_x <- x[-ind,]
train_y <- y[-ind]
tibble(dim(train_x),dim(test_x))
c(length(train_y),length(test_y))
mean(test_y == "B")
mean(train_y == "B")

#2. ENTRENAMIENTO DE LOS ALGORITMOS Y GENERACIÓN DE LOS PREDICTORES.
#KNN
set.seed(1998, sample.kind = "Rounding") 
knn <- train(train_x,train_y,method = "knn",
             tuneGrid = data.frame(k = c(3:21)))
knn$bestTune
ggplot(knn, highlight = TRUE)
knn_preds <- predict(knn,test_x) %>% as.factor()

#RANDOM FOREST
set.seed(1998, sample.kind = "Rounding")
rf <- train(train_x,train_y,method = "rf",
            tuneGrid = data.frame(mtry = c(3,5,7,9,11,13)))
rf$bestTune
ggplot(rf, highlight = TRUE)
rf_preds <- predict(rf,test_x) %>% as.factor()

#K-MEDIAS
k <- kmeans(train_x, centers = 2)
predict_kmeans <- function(x, k) {
  centers <- k$centers    
  distances <- sapply(1:nrow(x), function(i){
    apply(centers, 1, function(y) dist(rbind(x[i,], y)))
  })
  max.col(-t(distances))
}
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M") %>% as.factor()

#REGRESIÓN LOGISTICA
set.seed(1998, sample.kind = "Rounding")
lr <- train(train_x,train_y,method = "glm")
lr_preds <- predict(lr,test_x) %>% as.factor()

#ENSEMBLE
ensembling_pred <- data.frame(kmeans = knn_preds,
                              lr = lr_preds,
                              knn = knn_preds,
                              rf = rf_preds)
ensembling_pred <- ensembling_pred %>%
  mutate(ensemble = ifelse(rowMeans(ensembling_pred == "B") > 0.5, "B","M") %>% as.factor())
head(ensembling_pred)
ensemble_preds <- ensembling_pred$ensemble

#3. EVALUACIÓN DE LOS ALGORITMOS
data.frame(Algoritmos = c("Kmeans","Knn","Random Forest","Logistic Regression","Ensemble"),
           Precision_general = c(confusionMatrix(kmeans_preds,test_y)$overall["Accuracy"],
                                 confusionMatrix(knn_preds,test_y)$overall["Accuracy"],
                                 confusionMatrix(rf_preds,test_y)$overall["Accuracy"],
                                 confusionMatrix(lr_preds,test_y)$overall["Accuracy"],
                                 confusionMatrix(ensemble_preds,test_y)$overall["Accuracy"]),
           Especificidad = c(specificity(kmeans_preds,test_y),
                             specificity(knn_preds,test_y),
                             specificity(rf_preds,test_y),
                             specificity(lr_preds,test_y),
                             specificity(ensemble_preds,test_y))) %>%
  arrange(desc(Especificidad))






