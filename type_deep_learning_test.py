# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:13:00 2018

@author: Charpak 14
"""

import pandas
import matplotlib.pyplot as plt
dataset = pandas.read_csv(r"data\day.csv", usecols=[1], engine='python', skipfooter=3)
plt.plot(dataset)
plt.show()

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error