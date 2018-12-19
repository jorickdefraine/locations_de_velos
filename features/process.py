'''
Fichier principal
Importez votre fichier comportant les fonctions que vous souhaitez exécuter,
Puis tapez la commande à la suite des autres (vous pouvez commenter les autres pendant vos tests).
Si une fonction est déjà créee et importée ici, pas besoin de le faire dans votre fichier.
(exemple : openData et rmsle)
'''

from features.tools import walkForwardValidation
from features.model_random_forest import randomForest
#affiche_corr()


#arima(openData())
print(randomForest())

#print(rmsle(arima()[0], arima()[1]))
#print(rmsle(countLessOne()[0], countLessOne()[1]))


