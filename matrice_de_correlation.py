import matplotlib.pyplot as plt
import seaborn as sns



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