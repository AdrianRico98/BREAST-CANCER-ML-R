CÁNCER DE MAMA: MACHINE LEARNING
================
Proyecto creado por Adrián Rico -
17/10/2021

## OBJETIVO DEL PROYECTO.

En este proyecto selecciono un conjunto de datos del “UCI machine
learning repository” denominado [“Breast Cancer Wisconsin
(Diagnostic)”](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)
en el que se contienen 569 observaciones de distintos tumores mamarios y
32 variables que podemos dividir en dos bloques:

-   **Primer bloque**: contiene una variable que identifica cada
    observacion o tumor (ID) y otra que determina si es maligno (M) o
    benigno (B), denominada “diagnosis”.

-   **Segundo bloque**: se recogen 10 caracteristicas relativas a las
    observaciones calculadas a partir de una imagen digitalizada de un
    aspirado con aguja fina (FNA) de la masa mamaria. Describen las
    caracteristicas de los núcleos celulares presentes en la imagen
    (radius, texture, area, perimeter,
    smoothness,compactness,concavity,concave points, symmetry, fractal
    dimension). Para cada caracteristica se computa su media, su error
    estándar y el peor valor (o el mayor). De este modo, el campo 3 es
    la media del radio, el campo 13 es el error estándar del radio y el
    campo 23 es el peor radio.

El objetivo del proyecto es aplicar algoritmos de machine learning que
sean capaces de tener una **gran performance a la hora de clasificar a
los tumores** de un set de testeo como benignos o malignos en base a las
variables del segundo bloque.

Dividiré el proyecto en tres partes:

-   **Parte 1**: carga de librerias y datos y preprocesamiento de los
    mismos.

-   **Parte 2**: entrenamiento de los distintos algoritmos que vamos a
    utilizar.

-   **Parte 3**: testeo de la performance de los algoritmos, selección y
    valoración final.

## PARTE 1: PREPROCESAMIENTO DE LOS DATOS.

Primeramente cargamos las librerias que vamos a utilizar y los datos.

``` r
library(tidyverse)
library(caret)
library(gridExtra)
library(readxl)
setwd("C:/Users/adria/Desktop/PROYECTOS/breastcancer")
data <- read_excel("clean_data.xlsx")
head(data)
```

    ## # A tibble: 6 x 32
    ##         id diagnosis radius_mean texture_mean perimeter_mean area_mean
    ##      <dbl> <chr>     <chr>       <chr>        <chr>          <chr>    
    ## 1   842302 M         17.99       10.38        122.8          1001     
    ## 2   842517 M         20.57       17.77        132.9          1326     
    ## 3 84300903 M         19.69       21.25        130            1203     
    ## 4 84348301 M         11.42       20.38        77.58          386.1    
    ## 5 84358402 M         20.29       14.34        135.1          1297     
    ## 6   843786 M         12.45       15.7         82.57          477.1    
    ## # ... with 26 more variables: smoothness_mean <chr>, compactness_mean <chr>,
    ## #   concavity_mean <chr>, concave points_mean <chr>, symmetry_mean <chr>,
    ## #   fractal_dimension_mean <chr>, radius_se <chr>, texture_se <chr>,
    ## #   perimeter_se <chr>, area_se <chr>, smoothness_se <chr>,
    ## #   compactness_se <chr>, concavity_se <chr>, concave points_se <chr>,
    ## #   symmetry_se <chr>, fractal_dimension_se <chr>, radius_worst <chr>,
    ## #   texture_worst <chr>, perimeter_worst <chr>, area_worst <chr>,
    ## #   smoothness_worst <chr>, compactness_worst <chr>, concavity_worst <chr>,
    ## #   concave points_worst <chr>, symmetry_worst <chr>,
    ## #   fractal_dimension_worst <chr>

### 1.1 Estructura y limpieza de los datos.

Comprobamos que no hay valores nulos en ninguna de las variables.

``` r
variables <- colnames(data)
sapply(variables,function(nombre_variable){
  any(is.na(nombre_variable))
})
```

    ##                      id               diagnosis             radius_mean 
    ##                   FALSE                   FALSE                   FALSE 
    ##            texture_mean          perimeter_mean               area_mean 
    ##                   FALSE                   FALSE                   FALSE 
    ##         smoothness_mean        compactness_mean          concavity_mean 
    ##                   FALSE                   FALSE                   FALSE 
    ##     concave points_mean           symmetry_mean  fractal_dimension_mean 
    ##                   FALSE                   FALSE                   FALSE 
    ##               radius_se              texture_se            perimeter_se 
    ##                   FALSE                   FALSE                   FALSE 
    ##                 area_se           smoothness_se          compactness_se 
    ##                   FALSE                   FALSE                   FALSE 
    ##            concavity_se       concave points_se             symmetry_se 
    ##                   FALSE                   FALSE                   FALSE 
    ##    fractal_dimension_se            radius_worst           texture_worst 
    ##                   FALSE                   FALSE                   FALSE 
    ##         perimeter_worst              area_worst        smoothness_worst 
    ##                   FALSE                   FALSE                   FALSE 
    ##       compactness_worst         concavity_worst    concave points_worst 
    ##                   FALSE                   FALSE                   FALSE 
    ##          symmetry_worst fractal_dimension_worst 
    ##                   FALSE                   FALSE

Observamos la estructura de los datos importados.

``` r
str(data)
```

    ## tibble [569 x 32] (S3: tbl_df/tbl/data.frame)
    ##  $ id                     : num [1:569] 842302 842517 84300903 84348301 84358402 ...
    ##  $ diagnosis              : chr [1:569] "M" "M" "M" "M" ...
    ##  $ radius_mean            : chr [1:569] "17.99" "20.57" "19.69" "11.42" ...
    ##  $ texture_mean           : chr [1:569] "10.38" "17.77" "21.25" "20.38" ...
    ##  $ perimeter_mean         : chr [1:569] "122.8" "132.9" "130" "77.58" ...
    ##  $ area_mean              : chr [1:569] "1001" "1326" "1203" "386.1" ...
    ##  $ smoothness_mean        : chr [1:569] "0.1184" "0.08474" "0.1096" "0.1425" ...
    ##  $ compactness_mean       : chr [1:569] "0.2776" "0.07864" "0.1599" "0.2839" ...
    ##  $ concavity_mean         : chr [1:569] "0.3001" "0.0869" "0.1974" "0.2414" ...
    ##  $ concave points_mean    : chr [1:569] "0.1471" "0.07017" "0.1279" "0.1052" ...
    ##  $ symmetry_mean          : chr [1:569] "0.2419" "0.1812" "0.2069" "0.2597" ...
    ##  $ fractal_dimension_mean : chr [1:569] "0.07871" "0.05667" "0.05999" "0.09744" ...
    ##  $ radius_se              : chr [1:569] "1095" "0.5435" "0.7456" "0.4956" ...
    ##  $ texture_se             : chr [1:569] "0.9053" "0.7339" "0.7869" "1156" ...
    ##  $ perimeter_se           : chr [1:569] "8589" "3398" "4585" "3445" ...
    ##  $ area_se                : chr [1:569] "153.4" "74.08" "94.03" "27.23" ...
    ##  $ smoothness_se          : chr [1:569] "0.006399" "0.005225" "0.00615" "0.00911" ...
    ##  $ compactness_se         : chr [1:569] "0.04904" "0.01308" "0.04006" "0.07458" ...
    ##  $ concavity_se           : chr [1:569] "0.05373" "0.0186" "0.03832" "0.05661" ...
    ##  $ concave points_se      : chr [1:569] "0.01587" "0.0134" "0.02058" "0.01867" ...
    ##  $ symmetry_se            : chr [1:569] "0.03003" "0.01389" "0.0225" "0.05963" ...
    ##  $ fractal_dimension_se   : chr [1:569] "0.006193" "0.003532" "0.004571" "0.009208" ...
    ##  $ radius_worst           : chr [1:569] "25.38" "24.99" "23.57" "14.91" ...
    ##  $ texture_worst          : chr [1:569] "17.33" "23.41" "25.53" "26.5" ...
    ##  $ perimeter_worst        : chr [1:569] "184.6" "158.8" "152.5" "98.87" ...
    ##  $ area_worst             : chr [1:569] "2019" "1956" "1709" "567.7" ...
    ##  $ smoothness_worst       : chr [1:569] "0.1622" "0.1238" "0.1444" "0.2098" ...
    ##  $ compactness_worst      : chr [1:569] "0.6656" "0.1866" "0.4245" "0.8663" ...
    ##  $ concavity_worst        : chr [1:569] "0.7119" "0.2416" "0.4504" "0.6869" ...
    ##  $ concave points_worst   : chr [1:569] "0.2654" "0.186" "0.243" "0.2575" ...
    ##  $ symmetry_worst         : chr [1:569] "0.4601" "0.275" "0.3613" "0.6638" ...
    ##  $ fractal_dimension_worst: chr [1:569] "0.1189" "0.08902" "0.08758" "0.173" ...

Entenedemos que:

-   Todas las variables, salvo id, estan codificadas como caracteres
    (chr). Necesitamos que el output (diagnosis) sea un factor. El
    resto, es decir, los regresores, deben ser numéricos.

-   No vamos a necesitar el id.

Construimos el nuevo conjunto de datos que tenga en cuenta los dos
puntos anteriores. Después, separamos el output (vector “y”) y el
conjunto de regresores (matriz “x”). Al tener que computar
posteriormente algoritmos que tienen en cuenta las distancias,
tipificamos los regresores.

``` r
data <- mutate_all(data[3:30], as.numeric) %>%
  mutate(diagnosis = as.factor(data$diagnosis))
x <- data[,1:28] %>% scale()
y <- data$diagnosis
str(data)
```

    ## tibble [569 x 29] (S3: tbl_df/tbl/data.frame)
    ##  $ radius_mean           : num [1:569] 18 20.6 19.7 11.4 20.3 ...
    ##  $ texture_mean          : num [1:569] 10.4 17.8 21.2 20.4 14.3 ...
    ##  $ perimeter_mean        : num [1:569] 122.8 132.9 130 77.6 135.1 ...
    ##  $ area_mean             : num [1:569] 1001 1326 1203 386 1297 ...
    ##  $ smoothness_mean       : num [1:569] 0.1184 0.0847 0.1096 0.1425 0.1003 ...
    ##  $ compactness_mean      : num [1:569] 0.2776 0.0786 0.1599 0.2839 0.1328 ...
    ##  $ concavity_mean        : num [1:569] 0.3001 0.0869 0.1974 0.2414 0.198 ...
    ##  $ concave points_mean   : num [1:569] 0.1471 0.0702 0.1279 0.1052 0.1043 ...
    ##  $ symmetry_mean         : num [1:569] 0.242 0.181 0.207 0.26 0.181 ...
    ##  $ fractal_dimension_mean: num [1:569] 0.0787 0.0567 0.06 0.0974 0.0588 ...
    ##  $ radius_se             : num [1:569] 1095 0.543 0.746 0.496 0.757 ...
    ##  $ texture_se            : num [1:569] 0.905 0.734 0.787 1156 0.781 ...
    ##  $ perimeter_se          : num [1:569] 8589 3398 4585 3445 5438 ...
    ##  $ area_se               : num [1:569] 153.4 74.1 94 27.2 94.4 ...
    ##  $ smoothness_se         : num [1:569] 0.0064 0.00522 0.00615 0.00911 0.01149 ...
    ##  $ compactness_se        : num [1:569] 0.049 0.0131 0.0401 0.0746 0.0246 ...
    ##  $ concavity_se          : num [1:569] 0.0537 0.0186 0.0383 0.0566 0.0569 ...
    ##  $ concave points_se     : num [1:569] 0.0159 0.0134 0.0206 0.0187 0.0188 ...
    ##  $ symmetry_se           : num [1:569] 0.03 0.0139 0.0225 0.0596 0.0176 ...
    ##  $ fractal_dimension_se  : num [1:569] 0.00619 0.00353 0.00457 0.00921 0.00511 ...
    ##  $ radius_worst          : num [1:569] 25.4 25 23.6 14.9 22.5 ...
    ##  $ texture_worst         : num [1:569] 17.3 23.4 25.5 26.5 16.7 ...
    ##  $ perimeter_worst       : num [1:569] 184.6 158.8 152.5 98.9 152.2 ...
    ##  $ area_worst            : num [1:569] 2019 1956 1709 568 1575 ...
    ##  $ smoothness_worst      : num [1:569] 0.162 0.124 0.144 0.21 0.137 ...
    ##  $ compactness_worst     : num [1:569] 0.666 0.187 0.424 0.866 0.205 ...
    ##  $ concavity_worst       : num [1:569] 0.712 0.242 0.45 0.687 0.4 ...
    ##  $ concave points_worst  : num [1:569] 0.265 0.186 0.243 0.258 0.163 ...
    ##  $ diagnosis             : Factor w/ 2 levels "B","M": 2 2 2 2 2 2 2 2 2 2 ...

Pese a que hay una correlación importante entre ciertos regresores
(ejecutar -&gt; cor(x)), no vamos a entrenar algoritmos que requieran de
un gran tiempo de procesamiento y estamos interesados en observar
brevemente las variables de forma individual, por lo que descartamos un
resumen factorial.

### 1.2 Generamos las particiones de entrenamiento y de testeo.

``` r
set.seed(1990, sample.kind = "Rounding") 
ind <- createDataPartition(y, times = 1, p = 0.2, list = FALSE) #20% a datos de testeo
test_x <- x[ind,]
test_y <- y[ind]
train_x <- x[-ind,]
train_y <- y[-ind]
tibble(dim(train_x),dim(test_x))
```

    ## # A tibble: 2 x 2
    ##   `dim(train_x)` `dim(test_x)`
    ##            <int>         <int>
    ## 1            454           115
    ## 2             28            28

``` r
c(length(train_y),length(test_y))
```

    ## [1] 454 115

Vemos que existe una prevalencia o reparto proporcional de tumores
benignos y malignos similar en el conjunto de testeo y en el de
entrenamiento.

``` r
mean(test_y == "B")
```

    ## [1] 0.626087

``` r
mean(train_y == "B")
```

    ## [1] 0.6277533

## PARTE 2: ENTRENAMIENTO DE LOS ALGORITMOS.

Entrenaremos y generaremos las predicciones de cuatro algoritmos:

-   **K Vecinos más cercanos**.

-   **Random Forest**.

-   **K Medias**.

-   **Regresión Logística**.

Un “quinto algoritmo” será un **modelo ensamblado** generado a partir de
los anteriores. No se eligen algoritmos generativos (como LDA O QDA)
dado del grán número de variables independientes o regresores y sus
suposiciones de partida.

### 2.1 K Vecinos más cercanos.

En este algoritmo, para cada punto en un espacio n dimensional (siendo n
el número de regresores), se seleccionan los k puntos más cercanos y por
la “regla de la mayoría” se decide la clase de pertenencia del punto
original. Esto se hace para todos los puntos del conjunto.

El único parámetro modificable del algoritmo es “k” y vamos a elegirlo
mediante una validación cruzada bootstrap (por defecto en caret) para
k\`s de 3 a 20 (por defecto caret evaluaria solo k= 5,7 y 9).

``` r
set.seed(1998, sample.kind = "Rounding") 
#entrenamos el algoritmo
knn <- train(train_x,train_y,method = "knn",
             tuneGrid = data.frame(k = c(3:21)))
#vemos el k que maximiza la precisión general en la validación cruzada.
knn$bestTune
```

    ##     k
    ## 11 13

``` r
ggplot(knn, highlight = TRUE)
```

![validacion_knn](https://raw.githubusercontent.com/AdrianRico98/BREAST-CANCER-ML-R/master/imagenes/unnamed-chunk-7-1)<!-- -->
Vemos que el k que maximiza la precisión en base a la validación cruzada
es 13. Podemos generar las predicciones y guardarlas en un vector para
evaluarlas en la parte 3.

``` r
knn_preds <- predict(knn,test_x) %>% as.factor()
```

### 2.2 Random Forest.

En este algoritmo, se generan múltiples arboles de decision en base al
mismo conjunto de datos, introduciendo aleatoriedad con el muestreo
bootstrap y la selección aleatoria de regresores. Se evalua la
pertenencia de un punto a una clase en base a la regla de la mayoría
aplicada sobre los resultados de los múltiples árboles.

El único parámetro modificable de este algoritmo es “mtry”, referido al
número de regresores aleatoriamente escogidos. Vamos a elegirlo mediante
una validación cruzada bootstrap para mtry\`s = 3,5,7,9,11,13.

``` r
set.seed(1998, sample.kind = "Rounding")
#entrenamos el algoritmo
rf <- train(train_x,train_y,method = "rf",
             tuneGrid = data.frame(mtry = c(3,5,7,9,11,13)))
#vemos el k que maximiza la precisión general en la validación cruzada.
rf$bestTune
```

    ##   mtry
    ## 1    3

``` r
ggplot(rf, highlight = TRUE)
```

![validacion_rf](https://raw.githubusercontent.com/AdrianRico98/BREAST-CANCER-ML-R/master/imagenes/unnamed-chunk-9-1)<!-- -->

Vemos que el mtry que maximiza la precisión en base a la validacion
cruzada es 3. Generamos el vector de predicciones.

``` r
rf_preds <- predict(rf,test_x) %>% as.factor()
```

### 2.3 K-medias.

En este algoritmo no supervisado, se generan clusters o grupos en base a
la similitud de las observaciones (menor distancia). En este caso, el
numero de grupos “k” es dos, ya que son las categorias que tiene el
output (benigno y maligno).

``` r
#entrenamos el algoritmo
k <- kmeans(train_x, centers = 2)
```

Para calculas predicciones sobre los datos de testeo, creamos una
función (predict\_kmeans). Esta función recibe la matriz de regresores
de los datos de testeo y el modelo generado (k) y calcula las distancias
de las observaciones a los centroides de los clusters generados.

``` r
#creamos la funcion
predict_kmeans <- function(x, k) {
    centers <- k$centers    
    distances <- sapply(1:nrow(x), function(i){
                        apply(centers, 1, function(y) dist(rbind(x[i,], y)))
                 })
  max.col(-t(distances))
}
#generamos las predicciones.
kmeans_preds <- ifelse(predict_kmeans(test_x, k) == 1, "B", "M") %>% as.factor()
```

### 2.4 Regresión Logística.

En este algoritmo, se generan las probabilidades predecidas de que una
observación pertenezca o no a una categoria del output basándose en la
transformación logistica de la regresión lineal. Para más detalles sobre
el algoritmo consulte “Statquest: logistic regression” en YouTube.

``` r
set.seed(1998, sample.kind = "Rounding")
#entrenamos el algoritmo
lr <- train(train_x,train_y,method = "glm")
```

Generamos las predicciones.

``` r
#generamos las predicciones.
lr_preds <- predict(lr,test_x) %>% as.factor()
```

### 2.5 Modelo ensamblado.

El modelo ensamblado predicirá, para cada observacion, el output que
mayoritariamente haya sido predicho por los algoritmos anteriores (para
todas las observaciones existe una predicción mayoritaria, no hay
empates). Primeramente, generamos un data frame con las predicciones de
los cuatro algoritmos. La quinta columna del data frame serán las
predicciones del modelo ensamblado.

``` r
#generamos el data frame
ensembling_pred <- data.frame(kmeans = knn_preds,
                              lr = lr_preds,
                              knn = knn_preds,
                              rf = rf_preds)
#añadimos una columna que tenga la prediccion del modelo ensamblado
ensembling_pred <- ensembling_pred %>%
  mutate(ensemble = ifelse(rowMeans(ensembling_pred == "B") > 0.5, "B","M") %>% as.factor())
head(ensembling_pred)
```

    ##   kmeans lr knn rf ensemble
    ## 1      M  M   M  M        M
    ## 2      M  M   M  M        M
    ## 3      B  B   B  B        B
    ## 4      M  M   M  M        M
    ## 5      M  M   M  M        M
    ## 6      M  M   M  M        M

``` r
#extraemos la prediccion
ensemble_preds <- ensembling_pred$ensemble
```

## PARTE 3: EVALUACIÓN DE LOS ALGORITMOS Y VALORACIÓN FINAL.

En esta parte elegimos el método de clasificación más eficaz. Para ello,
debemos considerar dos cuestiones:

-   Necesitamos un algoritmo que tenga un **gran porcentaje de precisión
    general** a la hora de clasificar los tumores. Es decir, nos
    interesa que el algoritmo prediga bien los casos benignos y
    malignos.

-   En particular, nos interesa que la **especificidad sea muy alta**,
    esto es, el porcentaje de tumores malignos clasificados por el
    algoritmo como malignos.

``` r
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
```

    ##            Algoritmos Precision_general Especificidad
    ## 1            Ensemble         0.9652174     0.9302326
    ## 2                 Knn         0.9652174     0.9069767
    ## 3 Logistic Regression         0.9304348     0.8837209
    ## 4       Random Forest         0.9130435     0.8604651
    ## 5              Kmeans         0.8956522     0.7906977

El modelo ensamblado es el que consigue una mayor especificidad y
precisión general, esto último junto al knn.

**Como conclusión** se ha logrado una precisión general del 96.5% y una
especificidad del 93% y podemos estar seguros de que el modelo
ensamblado es un gran modelo a la hora de predecir la naturaleza de los
tumores.

*NOTA ACLARATORIA*: El objetivo es predecir correctamente, no explorar
por qué un tumor con las caracteristicas x es maligno. Por supuesto,
también podría observar la importancia de las variables en cada
algoritmo con el calculo de oddratios, anovas sobre el cluster de
pertenencia, importancia de variables en el random forest, etc. Sin
embargo, considero que esto se deberia de hacer con un equipo médico que
permita analizar estas cuestiones de forma medianamente lógica.
