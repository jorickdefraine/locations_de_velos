'''
Fichier principal
Importez votre fichier comportant les fonctions que vous souhaitez exécuter,
Puis tapez la commande à la suite des autres (vous pouvez commenter les autres pendant vos tests).
Si une fonction est déjà créee et importée ici, pas besoin de le faire dans votre fichier.
(exemple : openData et rmsle)
'''

from tools import openData
import matplotlib.pyplot as plt
#affiche_corr()

plt.plot(openData()['cnt'])
plt.show()
#arima(openData())

#print(rmsle(arima()[0], arima()[1]))
#print(rmsle(countLessOne()[0], countLessOne()[1]))


