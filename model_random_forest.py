import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from tools import openData, rmsle


def randomForest():
    dataset = pd.read_csv("data/day.csv")
    cnt = openData()[:147]['cnt']
    print("cnt", cnt)
    X = dataset.iloc[:, 2:13].values  # Independent variables. Removed the instant and the dteday columns
    print("X", X)
    y = dataset.iloc[:,15].values  # Dependent variable. The data is a range of numbers. Hence it is a regression problem
    print("y", y)
    # Categorical data. We have to OneHotEncode them
    ohe = OneHotEncoder(categories=range(0, 7))  # These are the columns containing categorical data
    X = ohe.fit_transform(X).toarray()

    # Splitting the data set into training and test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Random Forest Regression model
    regressor = RandomForestRegressor(n_estimators=250,
                                      max_features='sqrt')  # Chose the max_features value from the grid serach method
    regressor.fit(X_train, y_train)

    # Predicting the values
    y_pred = regressor.predict(
        X_test)  # 1-D Vector containing the predicted values of the dependent variable. That is total count of bike rented.
    plt.title("Prédiction sur les 147 1er jours par RF")
    plt.plot(y_pred)
    plt.show()
    plt.title("Données actuelles sur les 147 1ers jours")
    plt.plot(openData()[:147]['cnt'], 'r')
    plt.show()

    # Evaluating the model using k-fold cross validation technique
    accuracy = cross_val_score(estimator=regressor, X=X_train, y=y_train, cv=10)
    print("MSLE moyen = ", accuracy.mean())  # score moyen
    print("intervalle de confiance = ", accuracy.std())  # intervalle de confiance
    print("RMSLE = ", rmsle(y_pred, cnt))


"""from sklearn.model_selection import GridSearchCV
parameters = [ {'n_estimators' :[200,220,250,300,325,350], 'max_features' : ['auto','sqrt' , 'log2'], 'min_samples_split' : [2,3,4,5,6] },
                ]
grid_search = GridSearchCV(estimator = regressor, param_grid = parameters)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
"""
