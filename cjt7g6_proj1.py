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
    for i in range(1000):
        x1 = random.uniform(-10.0, 10.0)
        y1 = random.uniform(-10.0, 10.0)
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
def trainModel(df, weights, constantC, constantK, maxIter, threshHold):
    #grab the values in a list
    x = df['X'].values
    y = df['Y'].values
    d = df['Class'].values

    #define some variables to keep track
    numTurns = 0

    while numTurns < maxIter:
        errorRate = 0
        falsePosNeg = 0
        truePosNeg = 0

        '''assign som threshhold values. must accomodate for slight variance.'''
        posThreshHoldCeiling = 1 + threshHold
        posThreshHoldFloor = 1 - threshHold
        negThreshHoldFloor = -1 - threshHold
        negThreshHoldCeiling = -1 + threshHold

        for i in range(len(x)):
            ''' calculate the discriminant D = w0 + w1*xi + w2*yi '''
            discriminant = weights[0] + (weights[1] * x[i]) + (weights[2] * y[i])

            '''if the discriminant is not correct when compared to the correct output'''
            if ((discriminant >= posThreshHoldFloor and discriminant <= posThreshHoldCeiling) or
                    (discriminant >= negThreshHoldFloor and discriminant <= negThreshHoldCeiling)):
                truePosNeg += 1
                #weights = weightsUpdate(weights, constantC, constantK, d[i], x[i], y[i])
            else:
                '''update the weights'''
                weights = weightsUpdate(weights, constantC, constantK, d[i], x[i], y[i])
                falsePosNeg += 1


        numTurns += 1 #increase number of turns by 1 iteration
        print("Number of False Positive/Negative: " + str(falsePosNeg))
        print("Number of True Positive/Negative: " + str(truePosNeg))
        errorRate = falsePosNeg / len(x) * 100
        print("Error rate: " + str(errorRate) + "%")


        '''add stop conditions'''
        if (errorRate < 25):
            break
        else:
            continue

def main():
    weights = [0.1, 0.1, 0.1]
    df = createData()
    #plotVals(df) #just to show that the data is linearly separable
    trainModel(df, weights, 0.0000001, 0.0000001, 2000000, 0.30)

main()
