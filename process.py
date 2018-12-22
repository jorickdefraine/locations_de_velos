'''
Fichier principal
Importez votre fichier comportant les fonctions que vous souhaitez exécuter,
Puis tapez la commande à la suite des autres (vous pouvez commenter les autres pendant vos tests).
Si une fonction est déjà créee et importée ici, pas besoin de le faire dans votre fichier.
(exemple : openData et rmsle)
'''

from model_random_forest import randomForest
import matplotlib.pyplot as plt
from baseline_arima import arima
from tools import openData
#affiche_corr()

print(randomForest())
#arima(openData())
#plt.plot(arima(openData())[0])
#plt.show
#print(rmsle(arima()[0], arima()[1]))
#print(rmsle(countLessOne()[0], countLessOne()[1]))


