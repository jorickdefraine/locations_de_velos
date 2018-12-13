# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 09:28:55 2018

@author: Villebon
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)