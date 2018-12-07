# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:28:55 2018

@author: Villebon
"""
from keras.models import Sequential
from keras.layers import Dense
import numpy

seed = 7
numpy.random.seed(seed)

dataset = numpy.loadtxt("data.csv", delimiter=",")

X = dataset[:,0:2]
Y = dataset[:,2]

model = Sequential()
model.add(Dense(12, input_dim=2, init='uniform', activation='relu'))
model.add(Dense(2, init='uniform', activation='relu'))
model.add(Dense(1, input_dim=2, init='uniform', activation='linear'))


model.compile(loss='mean_squared_error', opt imizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)


predictions = model.predict(X)
rounded = [round(x[0]) for x in predictions]
print(rounded)
history = model.fit(X, Y, epochs=150, batch_size=10,  verbose=2)
print(history.history)