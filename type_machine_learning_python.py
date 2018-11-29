# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:12:45 2018

@author: charpak4.21
"""

from tools import openData

import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#Pour nos données
dataset = openData()
print(dataset.dtypes)

print(dataset)
print(dataset.columns)

#3.1 Dimensions of Dataset
print(dataset.shape)

#3.2 Peek at the Data
print(dataset.head(20))

#3.3 Statistical Summary
print(dataset.describe())

#3.4 Class Distribution
print(dataset.groupby('holiday').size())
print(dataset.columns)
print(dataset.groupby('season').size())
print(dataset.groupby('weathersit').size())

#4. Data Visualization
#4.1 Univariate Plots
dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()

dataset.hist()
plt.show()

#4.2 Multivariate Plots
scatter_matrix(dataset)
plt.show()

#5. Evaluate Some Algorithms
#5.1 Create a Validations Dataset
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

#5.2 Test Harness
seed = 7
scoring = 'accuracy'

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

