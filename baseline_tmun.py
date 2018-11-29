from baseline_arima import rmsle
from process import openData


def tmnun():
    variables = openData()
    count = variables['cnt']
