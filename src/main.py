from baseline_arima import arima
from tools import openData

if __name__ == '__main__':
    #affiche_corr()

    #print(randomForest())
    arima(openData())
    #plt.plot(arima(openData())[0])
    #plt.show
    #print(rmsle(arima()[0], arima()[1]))
    #print(rmsle(countLessOne()[0], countLessOne()[1]))


