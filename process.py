'''
Fichier principal
Importez votre fichier comportant les fonctions que vous souhaitez exécuter,
Puis tapez la commande à la suite des autres (vous pouvez commenter les autres pendant vos tests).
Si une fonction est déjà créee et importée ici, pas besoin de le faire dans votre fichier.
(exemple : openData et rmsle)
'''

from baseline_arima import arima
from matrice_de_correlation import affiche_corr
from tools import openData, rmsle
import tests
from model_count_less_one import countLessOne

#affiche_corr()


arima(openData())
#print(rmsle(arima()[0], arima()[1]))
#print(rmsle(countLessOne()[0], countLessOne()[1]))


