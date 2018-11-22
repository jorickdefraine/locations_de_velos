# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 08:37:53 2018

@author: hp
"""

from ensae_teaching_cs.data import google_trends
t = google_trends("macron") # export CSV, donc pas à jour
import pandas
df = pandas.read_csv(t, sep=",")
df.tail()
df.plot(x="Week", y="macron", figsize=(14,4))