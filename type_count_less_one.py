from process import openData
import matplotlib.pyplot as plt

def countLessOne():
    variables = openData()
    count = variables['cnt']
    return count