from baseline_arima import rmsle
from process import openData
import matplotlib.pyplot as plt

def countLessOne():
    variables = openData()
    count = variables['cnt']

