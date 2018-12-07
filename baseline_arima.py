import pandas
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import math

#  La bibliothèque d'importance majeure dans ce cas est statsmodels,
#  puisque nous utilisons cette bibliothèque
#  pour calculer les statistiques ACF et PACF, et aussi pour formuler le modèle ARIMA.
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.arima_model import ARIMA
from tools import openData


def arima(data):
    """
    Le modèle de prédiction ARIMA n'est pas le plus pertinant mais est indispendsable
    pour évaluer l'efficacité des autres modèles.

    :return: prédictions du jour 1 au jour 731  du
    nombre de vélos loués d'après le modèle ARIMA.
    count : valeurs réelles / actuelles du nombre de vélos loués.
    """
    variables = data
    count = variables['cnt']

    #  Dans un premier temps, pour effectuer une analyse de séries chronologiques,
    #  nous devons exprimer notre ensemble de données en termes de logarithmes.
    #  Si nos données sont exprimées uniquement en termes de compte,
    #  cela ne permet pas une capitalisation continue des rendements dans le temps et donnera des résultats trompeurs.
    lncount = np.log(count)

    plt.title("Log(count)")
    plt.plot(lncount)
    # on affiche le compte de vélos loués (exprimé en termes de logarithmes) en fonction du jour
    plt.show()

    acf_1 = acf(lncount)[1:731]
    plt.title("ACF")
    plt.plot(acf_1)
    plt.show()

    test_df = pandas.DataFrame([acf_1]).T
    test_df.columns = ['Autocorrelation']
    test_df.index += 1
    test_df.plot(kind='bar')
    plt.title("PACF")
    pacf_1 = pacf(lncount)[1:731]
    plt.plot(pacf_1)
    plt.show()

    test_df = pandas.DataFrame([pacf_1]).T
    test_df.columns = ['Autocorrelation partielle']
    test_df.index += 1
    test_df.plot(kind='bar')
    result = ts.adfuller(lncount, 1) # on utilise le test de Dickey-Fuller pour déterminer si les données sont stationnaires.
    print(result)
    lncount_diff = lncount - lncount.shift()
    diff = lncount_diff.dropna()
    acf_1_diff = acf(diff)[1:731]
    test_df = pandas.DataFrame([acf_1_diff]).T
    test_df.columns = ['Autocorrelation des différences premières']
    test_df.index += 1
    test_df.plot(kind='bar')
    pacf_1_diff = pacf(diff)[1:20]
    plt.title("PACF DIFF")
    plt.plot(pacf_1_diff)
    plt.show()

    count_matrix = lncount.as_matrix()
    model = ARIMA(count_matrix, order=(0, 1, 0))
    model_fit = model.fit(disp=0)

    predictions = model_fit.predict(1, 731, typ='levels')
    predictionsadjusted = np.exp(predictions)

    return predictionsadjusted, count

