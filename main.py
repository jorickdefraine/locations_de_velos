from model_random_forest import randomForest
import matplotlib.pyplot as plt
from baseline_arima import arima
from tools import openData, rmsle
from model_count_less_one import countLessOne
from matrice_de_correlation import affiche_corr

if __name__ == '__main__':
    #affiche_corr()

    #print(randomForest())
    arima(openData())
    #plt.plot(arima(openData())[0])
    #plt.show
    #print(rmsle(arima()[0], arima()[1]))
    #print(rmsle(countLessOne()[0], countLessOne()[1]))


