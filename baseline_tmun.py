from baseline_arima import rmsle
from main import open_data


def tmnun():
    variables = open_data()
    count = variables['cnt']
