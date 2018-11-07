import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data

def OpenData():
    df = pd.read_csv('data/day.csv')
    return df.head()
def affiche_corr(DataFrame,size):
    '''
    fonction qui affiche une matrice de correlation  pour chaque pair de colonne des données.

    Paramètres:
        DataFrames: pandas DataFrame
        size: taille verticale et horizontale de la matrice de correlation'''
    correlation = DataFrame.corr(method='pearson')
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(correlation)
    plt.xticks(range(len(correlation.columns)), correlation.columns)
    plt.yticks(range(len(correlation.columns)), correlation.columns)
    plt.show()