import pandas as pd
import numpy as np
from matplotlib import pyplot


def openData():
    """
    panda permet de lire les données d'après un fichier .csv
    :return: données sous forme de tableau en indexant la colonne du temps (en jour).
    """
    df = pd.read_csv('data/day.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.index = df['dteday']
    return df


def splitData():
    """
    Sépare la liste des données en deux parties (60/40) pour faciliter la cross-validation.
    :return: en premier la liste des données d'entrainement pour la cross-validation,
    en second la liste des données de tests pour la cross-validation.
    """
    split = openData().values
    train_size = int(len(split) * 0.70)
    train, test = split[0:train_size], split[train_size:len(split)]
    print('Nb de données totales: ', (len(split)))
    print('Nb données d entrainement: ', (len(train)))
    print('Nb données de tests:', (len(test)))
    pyplot.title('Séparation des données en "train" (bleu) et "test" (rouge)')
    pyplot.plot([x for x in range(train_size)], train, 'b')
    pyplot.plot([x for x in range(train_size, len(split))], test, 'r')
    pyplot.axis([0, 750, 0, 9000])
    pyplot.show()
    return train, test


def rmsle(predict_cnt, actual_cnt):
    """
    Root Mean Squared Logarithmic Error
    :param predict_cnt: prédiction du nombre de vélos loués modèle
    :param actual_cnt:  compte réel du nombr de vélos loués
    :return: le score d'un modèle grâce au RMSLE  (compris entre 0 et 1).
    """
    #  print(predict_cnt)
    #  print("pause")
    #  print(actual_cnt)
    somme = 0
    for i in range(len(actual_cnt)):
        somme += (np.log(predict_cnt[i] + 1) - np.log(actual_cnt[i] + 1)) ** 2
    score = np.sqrt((1 / len(actual_cnt)) * somme)
    return score
