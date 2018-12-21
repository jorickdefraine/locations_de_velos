"""
Ce fichier continent les différents tests de fonctions.
Ne pas oublier d'importer chaque fonction et fichier utilisé (une fois suffit).
"""

from tools import rmsle, openData
import numpy as np

l = list(range(10))
l2 = list(range(1, 11))


def test():
    test_RMSLE1 = np.testing.assert_almost_equal(rmsle(openData()['cnt'], openData()['cnt']), 0.000, decimal=3)
    test_RMSLE2 = np.testing.assert_almost_equal(rmsle(l, l2), 0.2977, decimal=3)