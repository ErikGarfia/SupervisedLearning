# REGRESION

En los problemas de regresion, los valores de objetivo son continuamente variables como el PIB de un pais o el precio de una casa

~~~
import pandas as pd

# importamos los datos de un archivo csv con pandas
boston = pd.read_csv('boston.csv)

#imprimimos los primeros 5 valores
print(boston.head())
~~~
<p align="center">
  <img src="SegundoCapitulo/1.png" width="350" title="hover text">
</p>

en donde:
* CRIM: tasa de criminalidad per capita
* NX: concentracion de oxidos nitricos
* RM: numero de habitaciones promedio por vivienda
* MEDV: valor medio de las viviendas ocupadas por sus propietarios en miles de dolares

para poder manejar la informacion del cvs, dividimos nuestro dataframe: 
~~~
X = boston.drop('MEDV', axis=1).values
y = boston['MEDV'].values
~~~
Para empezar, predeciremos el valor de una sola caracteristica que en este caso es X, el numero promedio de habitaciones en un bloque, para esto, dividimos el numero de habitaciones de la columna del dataFrameX, que es la quinta columna en las salas X

~~~
X_room = X[:,5]
type(X_rooms), type(y)
~~~
Dando como resultado que ambas son matrices Numpy, sin embargo, no tienen el formato que queremos, para hacerlo, aplicammos el metodo reshape para mantener la primera version 
~~~
y = y.reshape(-1,1)
X_rooms =    X_rooms.reshape(-1,1)
~~~
Ahora graficamos
~~~
plt.scatter(X_room, y)
plt.ylabel('Value of house /1000($)')
plt.xlabel('Number of rooms')
plt.show()
~~~
teniendo como resultado 
<p align="center">
  <img src="SegundoCapitulo/2.png" width="350" title="hover text">
</p>

**Ahora procedemos a hacer nuestro modelo en una regresion de datos**. En el proximo ejemplo utilizaremos un modelo de regresion lineal

~~~
import numpy as np
from sklearn.linea_model import LinearRegression

reg = LinearRegression()

#ajustamos la regresion a los datos
reg.fit(X_rooms, y)

#verificamos las predicciones dentro del rango de los datos
prediction_shape = np.linspace(min(X_rooms),max(X_rooms)).reshape(-1,1)

#y graficamosss
plt.scatter(X_rooms, y, color= 'blue')
plt.plot(prediction_space, reg.predict(prediction_space),color = 'black', linewidth=3)
plt.show()
~~~
obteniendo como respuesta
<p align="center">
  <img src="SegundoCapitulo/3.png" width="350" title="hover text">
</p>

### EJEMPLO
Importing data for supervised learning

In this chapter, you will work with Gapminder data that we have consolidated into one CSV file available in the workspace as 'gapminder.csv'. Specifically, your goal will be to use this data to predict the life expectancy in a given country based on features such as the country's GDP, fertility rate, and population. As in Chapter 1, the dataset has been preprocessed.

Since the target variable here is quantitative, this is a regression problem. To begin, you will fit a linear regression with just one feature: 'fertility', which is the average number of children a woman in a given country gives birth to. In later exercises, you will use all the features to build regression models.

Before that, however, you need to import the data and get it into the form needed by scikit-learn. This involves creating feature and target variable arrays. Furthermore, since you are going to use only one feature to begin with, you need to do some reshaping using NumPy's .reshape() method. Don't worry too much about this reshaping right now, but it is something you will have to do occasionally when working with scikit-learn so it is useful to practice.

* Import numpy and pandas as their standard aliases.
* Read the file 'gapminder.csv' into a DataFrame df using the read_csv() function.
*    Create array X for the 'fertility' feature and array y for the 'life' target variable.
*    Reshape the arrays by using the .reshape() method and passing in -1 and 1.
~~~
# Import numpy and pandas
import numpy as np
import pandas as pd

# Read the CSV file into a DataFrame: df
df = pd.read_csv('gapminder.csv')

# Create arrays for features and target variable
y = df['life'].values
X = df['fertility'].values

# Print the dimensions of X and y before reshaping
print("Dimensions of y before reshaping: {}".format(y.shape))
print("Dimensions of X before reshaping: {}".format(X.shape))

# Reshape X and y
y = y.reshape(-1,1)
X = X.reshape(-1,1)

# Print the dimensions of X and y after reshaping
print("Dimensions of y after reshaping: {}".format(y.shape))
print("Dimensions of X after reshaping: {}".format(X.shape))

~~~

Vamos a ver cómo funciona la regresion lineal, para poder emplear la regresion linesl, debemos de tomar el modelo de **y=ax+b** en donde **y = objetivo**, **x = caracteristica unica**, **a,b = parameters of model**, de este modelo una de las principales pregunta es ¿como elegimos a y b?
Un metodo comun es definir una fucnion de error para cual¿quier linea y lueog elegir la linea que minimiza la funcion de error, tal funcion de error se denomina **perdida** o funcion de costo. El residual de cada funcion de errores la linea vertical que se forma entre cada uno de los puntos y la linea de error.
<p align="center">
  <img src="SegundoCapitulo/4.png" width="350" title="hover text">
</p>

y para poder sacar ese error, sumamos toda la distancia entre cada punto y la linea de error y luego la elevamos al cuadrado. lo que obtenemos es comunmente llamado minimos cuadrados ordinarios u OLS.

Para realizar esta regresion lineal en un caso de scikit-learn, por ejemplo teninedo dos caracteristicas y un objetivo 
~~~
y = a1x1 + a2x2 + b
~~~
entonces para ajustar el modelo de regresion linea necesitamos especificar tres variables **a1 a2 b** y así, si tenemos mas variables aNxN entonces vamos a tener a1 a2 ... aN b

La API de scikit funciona:
* Pasandoles dos arreglos: caracteristicas y objetivos

### EJEMPLO
<p align="center">
  <img src="SegundoCapitulo/5.png" width="360" title="hover text">
</p>

### EJEMPLO
Fit & predict for regression

Now, you will fit a linear regression and predict life expectancy using just one feature. You saw Andy do this earlier using the 'RM' feature of the Boston housing dataset. In this exercise, you will use the 'fertility' feature of the Gapminder dataset. Since the goal is to predict life expectancy, the target variable here is 'life'. The array for the target variable has been pre-loaded as y and the array for 'fertility' has been pre-loaded as X_fertility.

A scatter plot with 'fertility' on the x-axis and 'life' on the y-axis has been generated. As you can see, there is a strongly negative correlation, so a linear regression should be able to capture this trend. Your job is to fit a linear regression and then predict the life expectancy, overlaying these predicted values on the plot to generate a regression line. You will also compute and print the R2
score using sckit-learn's .score() method.

* Import LinearRegression from sklearn.linear_model.
* Create a LinearRegression regressor called reg.
* Set up the prediction space to range from the minimum to the maximum of X_fertility. This has been done for you.
* Fit the regressor to the data (X_fertility and y) and compute its predictions using the .predict() method and the prediction_space array.
* Compute and print the R2
score using the .score() method.
* Overlay the plot with your linear regression line. This has been done for you, so hit 'Submit Answer' to see the result!
~~~
# Import LinearRegression
from sklearn.linear_model import LinearRegression

# Create the regressor: reg
reg = LinearRegression()

# Create the prediction space
prediction_space = np.linspace(min(X_fertility), max(X_fertility)).reshape(-1,1)

# Fit the model to the data
reg.fit(X_fertility, y)

# Compute predictions over the prediction space: y_pred
y_pred = reg.predict(prediction_space)

# Print R^2 
print(reg.score(X_fertility, y))

# Plot regression line
plt.plot(prediction_space, y_pred, color='black', linewidth=3)
plt.show()

~~~
<p align="center">
  <img src="SegundoCapitulo/6.png" width="360" title="hover text">
</p>

### EJEMPLO
Train/test split for regression

As you learned in Chapter 1, train and test sets are vital to ensure that your supervised learning model is able to generalize well to new data. This was true for classification models, and is equally true for linear regression models.

In this exercise, you will split the Gapminder dataset into training and testing sets, and then fit and predict a linear regression over all features. In addition to computing the R2
score, you will also compute the Root Mean Squared Error (RMSE), which is another commonly used metric to evaluate regression models. The feature array X and target variable array y have been pre-loaded for you from the DataFrame df.
* Import LinearRegression from sklearn.linear_model, mean_squared_error from sklearn.metrics, and train_test_split from sklearn.model_selection.
* Using X and y, create training and test sets such that 30% is used for testing and 70% for training. Use a random state of 42.
* Create a linear regression regressor called reg_all, fit it to the training set, and evaluate it on the test set.
* Compute and print the R2
score using the .score() method on the test set.
* Compute and print the RMSE. To do this, first compute the Mean Squared Error using the mean_squared_error() function with the arguments y_test and y_pred, and then take its square root using np.sqrt().
~~~
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3 , random_state=42)

# Create the regressor: reg_all
reg_all = LinearRegression()

# Fit the regressor to the training data
reg_all.fit(X_train, y_train)

# Predict on the test data: y_pred
y_pred = reg_all.predict(X_test)

# Compute and print R^2 and RMSE
print("R^2: {}".format(reg_all.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

~~~
En el caso anterior podemos detectar una pequeña trampa, si estamos calculando R² en el conjunto de prueba, entonces la R² que obtenemos de resultado depende de la forma en la que divide los datos. Los puntos de datos en el conjunto de prueba pueden tener algunas peculiaridades que significan que la R² calculado no es representativo de la capacidad del modelo  para generalizar a datos visibles. Para evitar el problema anterior recurrimos a emplear una tecnica llamada **validacion cruzada**

## Validacion cruzada- Cross validation basics

* Comenzamos diviendo el conjunto de datos en cinco grupos
<p align="center">
  <img src="SegundoCapitulo/7.png" width="360" title="hover text">
</p>

* Luego, el primer despliegue lo mostramos como el conjunto de prueba y los cuatro restantes los mostramos como el conjunto de entreno
<p align="center">
  <img src="SegundoCapitulo/8.png" width="360" title="hover text">
</p>

* Y calculamos la metrica de este conjunto
<p align="center">
  <img src="SegundoCapitulo/9.png" width="360" title="hover text">
</p>

* Y de manera continua, realizamos los pasos anteriores con cada uno de los cinco grupos
<p align="center">
  <img src="SegundoCapitulo/10.png" width="360" title="hover text">
</p>

Como resultado de lo anterior, obtenemos 5 valores de R² a partir de los cuales podemos calcular estadisticas de interes, como la media, la mediana, intervalos de confianza del 95%. El dividir enn grupos el conjunto de datos es llamado **k-fold CV** ya que el conjunto de datos se puede dividir en k grupos, sin embargo, este no es el proceso más adecuado ya que al utilizar mas divisiones de k, el proceso es más costoso desde el punto de vista computacional. Lo que realiza este metodo es evitar el problema que su metrica dependa de la division de prueba del tren.

### EJEMPLO
<p align="center">
  <img src="SegundoCapitulo/11.png" width="390" title="hover text">
</p>

### EJEMPLO
5-fold cross-validation

Cross-validation is a vital step in evaluating a model. It maximizes the amount of data that is used to train the model, as during the course of training, the model is not only trained, but also tested on all of the available data.

In this exercise, you will practice 5-fold cross validation on the Gapminder data. By default, scikit-learn's cross_val_score() function uses R2

as the metric of choice for regression. Since you are performing 5-fold cross-validation, the function will return 5 scores. Your job is to compute these 5 scores and then take their average.

The DataFrame has been loaded as df and split into the feature/target variable arrays X and y. The modules pandas and numpy have been imported as pd and np, respectively.


* Import LinearRegression from sklearn.linear_model and cross_val_score from sklearn.model_selection.
* Create a linear regression regressor called reg.
* Use the cross_val_score() function to perform 5-fold cross-validation on X and y.
* Compute and print the average cross-validation score. You can use NumPy's mean() function to compute the average.
~~~
# Import the necessary modules
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Create a linear regression object: reg
reg = LinearRegression()

# Compute 5-fold cross-validation scores: cv_scores
cv_scores = cross_val_score(reg, X, y, cv=5)

# Print the 5-fold cross-validation scores
print(cv_scores)

print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

~~~

### EJEMPLO
K-Fold CV comparison

Cross validation is essential but do not forget that the more folds you use, the more computationally expensive cross-validation becomes. In this exercise, you will explore this for yourself. Your job is to perform 3-fold cross-validation and then 10-fold cross-validation on the Gapminder dataset.

In the IPython Shell, you can use %timeit to see how long each 3-fold CV takes compared to 10-fold CV by executing the following cv=3 and cv=10:

%timeit cross_val_score(reg, X, y, cv = ____)

pandas and numpy are available in the workspace as pd and np. The DataFrame has been loaded as df and the feature/target variable arrays X and y have been created.


* Import LinearRegression from sklearn.linear_model and cross_val_score from sklearn.model_selection.
* Create a linear regression regressor called reg.
* Perform 3-fold CV and then 10-fold CV. Compare the resulting mean scores.
~~~
# Import necessary modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

# Create a linear regression object: reg
reg = LinearRegression()

# Perform 3-fold CV
cvscores_3 = cross_val_score(reg, X, y, cv=3)
print(np.mean(cvscores_3))

# Perform 10-fold CV
cvscores_10 = cross_val_score(reg, X, y, cv=10)
print(np.mean(cvscores_10))

~~~

Recordemos que lo que hace una regresion lineal es minimizar una funcion de perdidad para elegir un coeficiente ai para cada variable caracteristica. Lo que hay que aplicar en una regresion lineal es el penalizar cada que se encuentre un coeficiente muy grande, a esto se le conoce como **regularizacion**.

***
El primer tipo de regresion regularizada se llama **regresion de cresta** en el que la funcion de perdida estandar es:  
<p align="center">
  <img src="SegundoCapitulo/12.png" width="390" title="hover text">
</p>

En el que, al minimizar la funcion de perdida para que se ajuste a nuestros datos, los modelos son penalizados por coeficientes de gran magnitud, tanto negativos como positivos.

En la imagen anterior, alpha es un parametro que se dee de elegir para ajustar y predecir, puede considerarse como un parametro que controla la complejidad del modelo. Tener un alpha demasiado grande significa que los coeficientes grandes se penalizan significativamente, lo que puede conducir a un modelo que es demasiado simple y termina ajustando los datos de manera insuficiente

### EJEMPLO
<p align="center">
  <img src="SegundoCapitulo/13.png" width="390" title="hover text">
</p>
En el ejemplo, establecemos un alfa de 0.1 y el argumento normalizar lo establecemos en true para asegurar que todas nuestras variables estan en la misma escala

***
Otro tipo de regresion regularizada es llamada **regresion de lazo** en el que la funcion de perdida 
<p align="center">
  <img src="SegundoCapitulo/14.png" width="360" title="hover text">
</p>

Uno de los aspectos interesantes de la regresion de lazo es que se puede usar  para seleccionar caracteristicas importantes de un conjunnto de datos, esto se debe a que tiende a reducir los coeficientes de las caracteristicas menos importantes para que sean exactamente cero, las caracteristicas cuyos coeficientes no son cero son seleccionadas por el algoritmo

### EJEMPLO
<p align="center">
  <img src="SegundoCapitulo/15.png" width="360" title="hover text">
</p>

Y obtenemos la siguiente figura que muestra que el predictor más immportannte para nuestra variable objetivo (el precio de vivienda) es el numero de habitaciones
<p align="center">
  <img src="SegundoCapitulo/16.png" width="360" title="hover text">
</p>

### EJEMPLO
Regularization I: Lasso

In the video, you saw how Lasso selected out the 'RM' feature as being the most important for predicting Boston house prices, while shrinking the coefficients of certain other features to 0. Its ability to perform feature selection in this way becomes even more useful when you are dealing with data involving thousands of features.

In this exercise, you will fit a lasso regression to the Gapminder data you have been working with and plot the coefficients. Just as with the Boston data, you will find that the coefficients of some features are shrunk to 0, with only the most important ones remaining.

The feature and target variable arrays have been pre-loaded as X and y.


* Import Lasso from sklearn.linear_model.
* Instantiate a Lasso regressor with an alpha of 0.4 and specify normalize=True.
* Fit the regressor to the data and compute the coefficients using the coef_ attribute.
* Plot the coefficients on the y-axis and column names on the x-axis. This has been done for you, so hit 'Submit Answer' to view the plot!

~~~
# Import Lasso
from sklearn.linear_model import Lasso

# Instantiate a lasso regressor: lasso
lasso = Lasso(alpha=0.4, normalize=True)

# Fit the regressor to the data
lasso_coef=lasso.fit(X,y).coef_

# Compute and print the coefficients
lasso_coef = lasso.coef_
print(lasso_coef)

# Plot the coefficients
plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns.values, rotation=60)
plt.margins(0.02)
plt.show()

~~~
<p align="center">
  <img src="SegundoCapitulo/17.png" width="360" title="hover text">
</p>

### EJEMPLO
Regularization II: Ridge

Lasso is great for feature selection, but when building regression models, Ridge regression should be your first choice.

Recall that lasso performs regularization by adding to the loss function a penalty term of the absolute value of each coefficient multiplied by some alpha. This is also known as L1
regularization because the regularization term is the L1

norm of the coefficients. This is not the only way to regularize, however.

If instead you took the sum of the squared values of the coefficients multiplied by some alpha - like in Ridge regression - you would be computing the L2
norm. In this exercise, you will practice fitting ridge regression models over a range of different alphas, and plot cross-validated R2 scores for each, using this function that we have defined for you, which plots the R2 score as well as standard error for each alpha:
~~~
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()
~~~
Don't worry about the specifics of the above function works. The motivation behind this exercise is for you to see how the R2 score varies with different alphas, and to understand the importance of selecting the right value for alpha. You'll learn how to tune alpha in the next chapter.

* Instantiate a Ridge regressor and specify normalize=True.
* Inside the for loop:
  *    Specify the alpha value for the regressor to use.
  *   Perform 10-fold cross-validation on the regressor with the specified alpha. The data is available in the arrays X and y.
  * Append the average and the standard deviation of the computed cross-validated scores. NumPy has been pre-imported for you as np.
* Use the display_plot() function to visualize the scores and standard deviations.

~~~
# Import necessary modules
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

~~~
<p align="center">
  <img src="SegundoCapitulo/18.png" width="360" title="hover text">
</p>