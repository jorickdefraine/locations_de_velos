# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:12:45 2018

@author: charpak4.21
"""

from programme.tools import openData

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

# Pour nos données
def pourNosDonnees():
    dataset = openData()
    print(dataset.dtypes)

    print(dataset)
    print(dataset.columns)

    #  3.1 Dimensions of Dataset
    print(dataset.shape)

    # 3.2 Peek at the Data
    print(dataset.head(20))

    # 3.3 Statistical Summary
    print(dataset.describe())

    # 3.4 Class Distribution
    print(dataset.groupby('holiday').size())
    print(dataset.columns)
    print(dataset.groupby('season').size())
    print(dataset.groupby('weathersit').size())

    # 4. Data Visualization
    # 4.1 Univariate Plots
    dataset.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False)
    plt.show()

    dataset.hist()
    plt.show()

    # 4.2 Multivariate Plots
    scatter_matrix(dataset)
    plt.show()
