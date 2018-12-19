'''
Fichier principal
Importez votre fichier comportant les fonctions que vous souhaitez exécuter,
Puis tapez la commande à la suite des autres (vous pouvez commenter les autres pendant vos tests).
Si une fonction est déjà créee et importée ici, pas besoin de le faire dans votre fichier.
(exemple : openData et rmsle)
'''

from features.tools import walkForwardValidation
from features.model_random_forest import randomForest
from features.matrice_de_correlation import affiche_corr
#affiche_corr()


#arima(openData())
#print(randomForest())
affiche_corr()
#print(rmsle(arima()[0], arima()[1]))
#print(rmsle(countLessOne()[0], countLessOne()[1]))


