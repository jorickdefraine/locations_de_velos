import pandas as pd
import numpy as np


def openData():
    """
    panda permet de lire les données d'après un fichier .csv
    :return: données sous forme de tableau en indexant la colonne du temps (en jour).
    """
    df = pd.read_csv('data/day.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.index = df['dteday']

    return df


def rmsle(predict_cnt, actual_cnt):
    """
    :param predict_cnt: prédiction du nombre de vélos loués modèle
    :param actual_cnt:  compte réel du nombr de vélos loués
    :return: le score d'un modèle entre 0 et 1.
    """
    print(predict_cnt)
    print("pause")
    print(actual_cnt)
    for i in range(1, 731):
        somme = (np.log(predict_cnt[i] + 1) - np.log(actual_cnt[i] + 1)) ** 2
    print("rsmle")
    rmsle = np.sqrt((1 / 731) * somme)
    print(rmsle)