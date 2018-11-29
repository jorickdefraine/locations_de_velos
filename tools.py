import pandas as pd
import numpy as np


def openData():
    df = pd.read_csv('data/day.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df.index = df['dteday']

    return df


def rmsle(predict_cnt, actual_cnt):
    print(predict_cnt)
    print("pause")
    print(actual_cnt)
    for i in range(1, 731):
        somme = (np.log(predict_cnt[i] + 1) - np.log(actual_cnt[i] + 1)) ** 2
    print("rsmle")
    rmsle = np.sqrt((1 / 731) * somme)
    print(rmsle)