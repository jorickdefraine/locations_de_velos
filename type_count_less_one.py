from process import openData
import matplotlib.pyplot as plt

def countLessOne():
    variables = openData()
    count = variables['cnt']
    predict = count-1
    plt.scatter