import matplotlib.pyplot as plt
import seaborn as sns
from tools import openData

def affiche_corr():
    '''
    fonction qui affiche une matrice de correlation  pour chaque pair de colonne de données.

    Paramètres:
        DataFrames: pandas DataFrame
        size: taille verticale et horizontale de la matrice de correlation
    '''

    correlation = openData().corr(method='pearson')
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns)
    plt.title("Matrice de corrélation pour la location de vélo")
    plt.show()