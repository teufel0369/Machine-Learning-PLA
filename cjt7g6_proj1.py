import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

def createData():
    '''create some empty list variables to put randomized data into'''
    x = []
    y = []
    d = []

    '''create some random values for x and y. if the y value > 0, tag it with a 1, otherwise tag it with a -1'''
    for i in range(100):
        x1 = random.uniform(-5.0, 5.0)
        y1 = random.uniform(-5.0, 5.0)
        x.append(x1)
        y.append(y1)

        if y[i] > 0:
            d1 = 1
        else:
            d1 = -1

        d.append(d1)

    '''create an empty dataframe and stick all the lists in there'''
    df = pd.DataFrame()
    df['X'] = x
    df['Y'] = y
    df['Class'] = d

    return df

'''This function will plot the linearly separable data'''
def plotVals(df):
    x = df['X'].values
    y = df['Y'].values

    '''create a plot and two subplots for -1 and 1'''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)

    '''iterate over the x,y values and tag and plot them appropriately'''
    for index in range(len(x)):
        if y[index] > 0:
            ax1.scatter(x[index], y[index], alpha=0.5, c='red', edgecolors='none', s=25, label='red')
        else:
            ax2.scatter(x[index], y[index], alpha=0.5, c='black', edgecolors='none', s=25, label='black')
    plt.title("Perceptron Learning Algorithm Randomized Data")
    plt.show()


'''This function will update the weights. 3 weights; 2 for the points themselves and 1 for bias'''
def weightsUpdate(weights, constantC, constantK, classificationd, x, y):
    weights[0] = weights[0] + constantC * classificationd * constantK # w0 = w0 + cdk
    weights[1] = weights[1] + constantC * classificationd * x #w1 = w1 + cdx
    weights[2] = weights[2] + constantC * classificationd * y #w2 = w2 + cdy

    return weights

'''this function will train the weights to make a correct discriminant output'''
def trainModel(df, weights, constantC, constantK, maxIter):
    #grab the values in a list
    x = df['X'].values
    y = df['Y'].values
    d = df['Class'].values

    #define some variables to keep track
    numTurns = 0
    numErrors = 0

    while numTurns < maxIter:
        errorRate = 0
        numErrors = 0


        for i in range(len(x)):
            ''' calculate the discriminant D = w0 + w1*xi + w2*yi '''
            discriminant = weights[0] + (weights[1] * x[i]) + (weights[2] * y[i])

            '''if the discriminant is not correct when compared to the correct output'''
            if discriminant != d[i]:
                '''update the weights'''
                weights = weightsUpdate(weights, constantC, constantK, d[i], x[i], y[i])
                numErrors += 1
                print(discriminant)
                print(d[i])

        numTurns += 1 #increase number of turns by 1 iteration
        print("Number of errors: " + str(numErrors))
        errorRate = numErrors / len(x) * 100
        print("Error rate: " + str(errorRate) + "%")
        print("Weights: " + str(weights)) #debugging


def main():
    weights = [0.0, 0.0, 0.0]
    df = createData()
    #plotVals(df) #just to show that the data is linearly separable
    trainModel(df, weights, 0.00000, 0.00000, 1000000)

main()
