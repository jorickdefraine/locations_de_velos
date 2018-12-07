# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:45:24 2018

@author: Villebon
"""

# Import pandas 
import pandas as pd

# Read in white wine data 
white = pd.read_csv("winequality-white.csv", sep=';')

# Read in red wine data 
red = pd.read_csv("winequality-red.csv", sep=';')

# Print info on white wine
print(white.info())

# Print info on red wine
print(red.info())

# First rows of `red` 
red.head()

# Last rows of `white`
white.tail()

# Take a sample of 5 rows of `red`
print(red.sample(5))

# Describe `white`
white.describe()

# Double check for null values in `red`
pd.isnull(red)


""" One variable that you could find interesting at first sight is alcohol.
 It’s probably one of the first things that catches your attention when
 you’re inspecting a wine data set. You can visualize the distributions
 with any data visualization library, but in this case, the tutorial 
 makes use of matplotlib to quickly plot the distributions:
"""
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2)

ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")

fig.subplots_adjust(left=0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)
ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")
#ax[0].legend(loc='best')
#ax[1].legend(loc='best')
fig.suptitle("Distribution of Alcohol in % Vol")

plt.show()


"""Note that you can double check this if you use the histogram() 
function from the numpy package to compute the histogram of the white
 and red data, just like this"""
import numpy as np
print(np.histogram(red.alcohol, bins=[7,8,9,10,11,12,13,14,15]))
print(np.histogram(white.alcohol, bins=[7,8,9,10,11,12,13,14,15]))




"""Next, one thing that interests me is the relation
 between the sulphates and the quality of the wine.
 As you have read above, sulphates can cause people 
 to have headaches and I’m wondering if this infuences
 the quality of the wine. What’s more, I often hear that
 women especially don’t want to drink wine exactly because 
 it causes headaches. Maybe this affects the ratings for the red wine?
Let’s take a look."""

import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(red['quality'], red["sulphates"], color="red")
ax[1].scatter(white['quality'], white['sulphates'], color="white", edgecolors="black", lw=0.5)

ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlabel("Quality")
ax[1].set_xlabel("Quality")
ax[0].set_ylabel("Sulphates")
ax[1].set_ylabel("Sulphates")
ax[0].set_xlim([0,10])
ax[1].set_xlim([0,10])
ax[0].set_ylim([0,2.5])
ax[1].set_ylim([0,2.5])
fig.subplots_adjust(wspace=0.5)
fig.suptitle("Wine Quality by Amount of Sulphates")

plt.show()



""" Let’s put the data to the test and make 
a scatter plot that plots the alcohol versus
 the volatile acidity. The data points should 
 be colored according to their rating or quality label:"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(570)

redlabels = np.unique(red['quality'])
whitelabels = np.unique(white['quality'])

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
redcolors = np.random.rand(6,4)
whitecolors = np.append(redcolors, np.random.rand(1,4), axis=0)

for i in range(len(redcolors)):
    redy = red['alcohol'][red.quality == redlabels[i]]
    redx = red['volatile acidity'][red.quality == redlabels[i]]
    ax[0].scatter(redx, redy, c=redcolors[i])
for i in range(len(whitecolors)):
    whitey = white['alcohol'][white.quality == whitelabels[i]]
    whitex = white['volatile acidity'][white.quality == whitelabels[i]]
    ax[1].scatter(whitex, whitey, c=whitecolors[i])
    
ax[0].set_title("Red Wine")
ax[1].set_title("White Wine")
ax[0].set_xlim([0,1.7])
ax[1].set_xlim([0,1.7])
ax[0].set_ylim([5,15.5])
ax[1].set_ylim([5,15.5])
ax[0].set_xlabel("Volatile Acidity")
ax[0].set_ylabel("Alcohol")
ax[1].set_xlabel("Volatile Acidity")
ax[1].set_ylabel("Alcohol") 
#ax[0].legend(redlabels, loc='best', bbox_to_anchor=(1.3, 1))
ax[1].legend(whitelabels, loc='best', bbox_to_anchor=(1.3, 1))
#fig.suptitle("Alcohol - Volatile Acidity")
fig.subplots_adjust(top=0.85, wspace=0.7)

plt.show()

""" Now that you have explored your data, 
it’s time to act upon the insights that you have gained! 
Let’s preprocess the data so that you can start building 
your own neural network!"""

# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)

"""Since it can be somewhat difficult to interpret graphs,
 it’s also a good idea to plot a correlation matrix.
 This will give insights more quickly about which variables correlate:"""
import seaborn as sns
corr = wines.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
sns.plt.show()

"""For now, import the train_test_split from sklearn.model_selection 
and assign the data and the target labels to the variables X and y.
 You’ll see that you need to flatten the array of target labels 
 in order to be totally ready to use the X and y variables as
 input for the train_test_split() function. Off to work, get 
 started in the DataCamp Light chunk below! """
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data 
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array
y= np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

