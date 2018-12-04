"""
Ce fichier continent les différents tests de fonctions.
Ne pas oublier d'importer chaque fonction et fichier utilisé (une fois suffit).
"""

from tools import rmsle, openData


predict = openData()['cnt']
actual = openData()['cnt']

assert rmsle(predict, actual) == 0  # 0.0

