from tools import openData
import numpy as np
import matplotlib.pyplot as plt


def countLessOne():
    """
    prédiction la plus basique.
    prediction(t) = cnt(t-1)
    :return: prediction du jour 2 au jour 731.
    """
    variables = openData()
    count = variables[:]["cnt"]
    prediction = []
    for i in range(1, len(count)):
        prediction.append(count[:][i - 1])
    prediction = np.reshape(prediction,len(prediction))
    return prediction, count
