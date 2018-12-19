"""
Ce fichier continent les différents tests de fonctions.
Ne pas oublier d'importer chaque fonction et fichier utilisé (une fois suffit).
"""

from features.tools import rmsle, openData
import numpy as np

l = list(range(10))
l2 = list(range(1, 11))

np.testing.assert_almost_equal(rmsle(openData()['cnt'], openData()['cnt']), 0.000, decimal=3)
np.testing.assert_almost_equal(rmsle(l, l2), 0.2977, decimal=3)