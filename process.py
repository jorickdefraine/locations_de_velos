'''
Fichier principal
'''

import pandas as pd
from baseline_arima import arima, rmsle
from matrice_de_correlation import affiche_corr


def openData():
    df = pd.read_csv('data/day.csv')
    df.index = df['dteday']
    return df


affiche_corr(openData())
arima()
rmsle(arima()[0], arima()[1])
