import inline as inline
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import rcParams
import numpy as np
from numpy.random import seed
import random

#set the size of the plot
rcParams["figure.figsize"] = 10, 10

def createData():
    df = pd.DataFrame() # create an empty dataframe

    randVals = [] #empty list for the random values
    for i in range(999):
        x = random.randint(0, 150) #create a random number
        randVals.append(x) #append it to the list 1000 times

    df['Weight'] = randVals #assign the list to the dataframe


    height = [] #empty list for the tag values
    classifier = [] #empty list for the classification
    for row in df['Weight']: #assigning binary classifiers to the pandas dataframe
        if row > 96:
            height.append(random.randint(0, 50)) #make a random small height
            classifier.append("Red") #make a classification
        else:
            height.append(random.randint(51, 100)) #make a random big height
            classifier.append("Black") #make a classification

    df['Height'] = height #assign the list to the dataframe
    df['Classification'] =classifier #assign the answers to the dataframe
    print(df)

    return df

def plotVals():
    df = createData()
    x = df['Weight'].values
    y = df['Height'].values
    colors = ('orange', 'black')
    groups = ("red", "black")

    #Create a plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, facecolor='w')

    for color, group in zip(colors, groups):
        ax.scatter(x, y, alpha=0.5, c=color, edgecolors='none', s=25, label=group)

    plt.title('Binary Classification')
    plt.show()

plotVals()


