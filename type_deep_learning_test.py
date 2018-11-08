# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:13:00 2018

@author: Charpak 14
"""
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error