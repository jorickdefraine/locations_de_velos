'''
Fichier principal
'''

from baseline_arima import arima
from matrice_de_correlation import affiche_corr
from tools import openData, rmsle

"""
affiche_corr(openData())
arima()
rmsle(arima()[0], arima()[1])
"""