from baseline_arima import rmsle
from process import openData


def countLessOne():
    variables = openData()
    count = variables['cnt']
