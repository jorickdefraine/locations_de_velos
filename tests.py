"""
Ce fichier continent les différents tests de fonctions.
Ne pas oublier d'importer chaque fonction et fichier utilisé (une fois suffit).
"""

from tools import rmsle, openData

predict = openData()['cnt']
actual = openData()['cnt']

l = list(range(10))
l2 = list(range(1, 11))
assert rmsle(predict, actual) == 0  # 0.0
assert rmsle(l, l2) == 0.29770806408565864
