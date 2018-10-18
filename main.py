'''
si vous avez bien pull, ce message s'affiche.
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('day.csv')

print(df.head())