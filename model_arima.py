# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:12:45 2018

@author: charpak4.21
"""

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

#url = "\Users\charpak4.21\locations_de_velos"
#names = ['1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15','16']
#dataset = pandas.read_csv(url, names=names)
#
#print(dataset.shape)
#print(dataset.head(20))
#print(dataset.describe())
