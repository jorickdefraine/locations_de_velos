import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data

def OpenData():
    df = pd.read_csv('data/day.csv')
    return df.head()
