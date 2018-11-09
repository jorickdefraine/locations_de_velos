import pandas as pd
from pandas import datetime
import numpy as np
import matplotlib.pyplot as plt
import data
import seaborn as sns
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import datasets

def OpenData():
    df = pd.read_csv('data/day.csv')
    return df


def affiche_corr(df):
    '''
    fonction qui affiche une matrice de correlation  pour chaque pair de colonne de données.

    Paramètres:
        DataFrames: pandas DataFrame
        size: taille verticale et horizontale de la matrice de correlation'''
    correlation = df.corr(method='pearson')
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
    plt.title("Matrice de corrélation pour la location de vélo")
    plt.show()