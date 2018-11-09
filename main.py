'''
si vous avez bien pull, ce message s'affiche.
'''

from type_random_forest import OpenData, affiche_corr
from baseline_arima import arima
print(OpenData())

affiche_corr(OpenData())
arima()
