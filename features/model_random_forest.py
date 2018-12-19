import pandas as pd
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from features.tools import rmsle, openData


def randomForest():

    cnt_actual = openData()[:]['cnt']
    dataset = pd.read_csv("data/day.csv")
    X = dataset.iloc[:, 2:13].values #Independent variables. Removed the instant and the dteday columns
    y = dataset.iloc[:, 15].values
    ohe = OneHotEncoder(categorical_features=range(0, 7)) #These are the columns containing categorical data
    X = ohe.fit_transform(X).toarray()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size = 0.5)
    regressor = RandomForestRegressor(n_estimators=250, max_features='sqrt')
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    pyplot.title("Prédiction par random forest")
    pyplot.plot(y_pred)
    pyplot.show()
    accuracy = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv =10)
    print("moyenne des écarts types = ", accuracy.mean())
    print("erreur standard = ", accuracy.std())
    print("rmsle = ",rmsle(y_pred, cnt_actual[:len(y_pred)]))
    print(len(y_pred))
    return y_pred