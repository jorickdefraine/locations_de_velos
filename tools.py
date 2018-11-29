import pandas as pd


def openData():
    df = pd.read_csv('data/day.csv')
    df.index = df['dteday']
    return df
