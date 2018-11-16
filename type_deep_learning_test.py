# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:13:00 2018

@author: Charpak 14
"""
import numpy
import pandas
import matplotlib.pyplot as plt
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#################
dataset = pandas.read_csv(r"data\day.csv") #je donne un nom de variable à dataframe

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