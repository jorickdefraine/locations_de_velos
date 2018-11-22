'''
si vous avez bien pull, ce message s'affiche.
'''

from matrice_de_correlation import affiche_corr
from baseline_arima import arima, rmsle
import pandas as pd


def OpenData():
    df = pd.read_csv('data/day.csv')
    return df


affiche_corr(OpenData())
arima()
rmsle(arima()[0], arima()[1])
