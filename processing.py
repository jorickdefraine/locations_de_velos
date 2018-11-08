import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

def OpenData():
    df = pd.read_csv('data/day.csv')
    return df
def firstLine(ligne):
    df = pd.read_csv('data/day.csv')
    return df.head(ligne)

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
    plt.title("Matrice de corrélation pour la location de vélo")
    plt.show()

def affiche_hm(DataFrame):
    '''
    fonction qui affiche la heat map pour chaque pair de colonne des données.

    Paramètres:
        DataFrames: pandas DataFrame
        size: taille verticale et horizontale de la matrice de correlation'''
    correlation = DataFrame.corr(method='pearson')
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
    plt.title("Matrice de corrélation pour la location de vélo")
    plt.show()

def feature_imp(X,y):
    # Build a classification task using 3 informative features
    #X, y = make_classification(n_samples=1000,
    #                           n_features=10,
    #                           n_informative=3,
    #                           n_redundant=0,
    #                           n_repeated=0,
    #                           n_classes=2,
    #                           random_state=0,
    #                           shuffle=False)

    # Build a forest and compute the feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]

    # Print the feature ranking
    print("Feature ranking:")

    for f in range(X.shape[1]):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(X.shape[1]), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(X.shape[1]), indices)
    plt.xlim([-1, X.shape[1]])
    plt.show()