# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:13:00 2018

@author: Charpak 14
"""
from type_random_forest import OpenData

import numpy
import pandas
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#################
dataset = OpenData() #je donne un nom de variable à dataframe
print(dataset.dtypes) #affiche le type des variables dans les données

"""
#permet de dessiner les graphes intitulés avec les variables
for col in dataset.columns:
    plt.plot(dataset.index, dataset[col])
    plt.title(col)
    plt.figure()
plt.show()
"""
#permet de transformer les dates en datetime
dataset['dteday'] = pandas.to_datetime(dataset['dteday'])

#index affiche correctement que les dates sont datetime
dataset.index = dataset['dteday']
print(dataset.dtypes)

#day permet de récupérer le jour du mois [0,31]
t1 = dataset['dteday'][31].day
print(t1)

#weekday permet de récupérer le jour de la semaine [0,6]
t2 = dataset['dteday'][6].weekday()
print(t2)