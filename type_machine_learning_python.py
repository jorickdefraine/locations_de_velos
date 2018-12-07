# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:12:45 2018

@author: charpak4.21
"""
#from type_random_forest import OpenData
from tools import openData

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

#Pour nos données
dataset = openData()

#Walk Forward Validation
X = dataset.values
n_train = 500
n_records = len(X)
for i in range(n_train, n_records):
    train, test = X[0:i], X[i:i+1]
    print('train=%d, test=%d' % (len(train), len(test)))

plt.show()