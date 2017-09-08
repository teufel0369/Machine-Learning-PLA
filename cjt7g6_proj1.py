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
        x = random.randint(0, 500) #create a random number
        randVals.append(x) #append it to the list 1000 times

    df['randVals'] = randVals #assign the list to the dataframe

    tagVals = [] #empty list for the tag values
    for row in df['randVals']: #assigning binary classifiers to the pandas dataframe
        if row > 327:
            tagVals.append('Red') #assign red to the list element if greater than 327
        else:
            tagVals.append('Black') #otherwise assign black

    df['tagVals'] = tagVals #assign the list to the dataframe

    return df

def plotVals():
    df = createData()
    plt.plot(df['randVals'])
    plt.show()

plotVals()


