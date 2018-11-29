import pandas as pd


def openData():
    df = pd.read_csv('data/day.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.index = df['dteday']

    return df
