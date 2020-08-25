from src.tools import openData


def countLessOne():
    """
    prédiction la plus basique.
    prediction(t) = cnt(t-1)

    :return: prediction sur les 6er mois puis 6er mois +1j ...
    count2 : valeurs réelles / actuelles du nombre de vélos loués.
    """
    variables = openData()
    count = variables[:]["cnt"]  # les valeurs actuelles (dans data)
    prediction = []
    count2 = [x for i, x in enumerate(count) if i != 1]  # enlève la première ligne de count
    for i in range(1, len(count)):
        prediction.append(count[:][i - 1])
    return prediction, count2
