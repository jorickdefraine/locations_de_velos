import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def OpenData():
    df = pd.read_csv('day.csv')
    return df.head()
