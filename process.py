'''
Fichier principal
'''

import pandas as pd
from baseline_arima import arima, rmsle
from matrice_de_correlation import affiche_corr
from tools import openData

"""
affiche_corr(openData())
arima()
rmsle(arima()[0], arima()[1])
"""