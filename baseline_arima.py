import pandas
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from type_random_forest import OpenData


def arima():
    variables = OpenData()
    count = variables['cnt']
    lnprice = np.log(count)

    plt.plot(lnprice)
    plt.show()
    acf_1 = acf(lnprice)[1:360]
    plt.plot(acf_1)
    plt.show()
    test_df = pandas.DataFrame([acf_1]).T
    test_df.columns = ['Autocorrelation par pandas']
    test_df.index += 1
    test_df.plot(kind='bar')
    pacf_1 = pacf(lnprice)[1:360]
    plt.plot(pacf_1)
    plt.show()
    test_df = pandas.DataFrame([pacf_1]).T
    test_df.columns = ['Autocorrelation partiel par pandas']
    test_df.index += 1
    test_df.plot(kind='bar')
    result = ts.adfuller(lnprice, 1)

    lnprice_diff = lnprice - lnprice.shift()
    diff = lnprice_diff.dropna()
    acf_1_diff = acf(diff)[1:360]
    test_df = pandas.DataFrame([acf_1_diff]).T
    test_df.columns = ['Autocorrelation des différences premières']
    test_df.index += 1
    test_df.plot(kind='bar')
    pacf_1_diff = pacf(diff)[1:360]
    plt.plot(pacf_1_diff)
    plt.show()

    price_matrix = lnprice.as_matrix()
    model = ARIMA(price_matrix, order=(0, 1, 0))
    model_fit = model.fit(disp=0)
    print(model_fit.summary())

    predictions = model_fit.predict(1, 731, typ='levels')
    print(predictions)
    predictionsadjusted = np.exp(predictions)
    return predictionsadjusted,count


def rmsle(predict_cnt, actual_cnt):
    for i in range(1,731):
        somme = (np.log(predict_cnt[i] + 1) - np.log(actual_cnt[i] + 1)) ** 2
    print("rsmle")
    rmsle = np.sqrt((1 / 731) * somme)
    print(rmsle)

