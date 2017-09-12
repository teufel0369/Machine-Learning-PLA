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
        x1 = random.uniform(-10, 10)
        y1 = random.uniform(-10, 10)
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

'''this function will test whether the value is positive'''
def isPos(val):
    if val > 0:
        return True
    else:
        return False

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

'''NOTE: Write in loop controls for training the model
1) if the errorRate does not continually improve, spit out the model weights
2) if the errorRate reaches 0, spit out the model weights
3) TBD'''

'''this function will train the weights to make a correct discriminant output'''
def trainModel(df, weights, constantC, constantK, maxIter):
    #grab the values in a list
    x = df['X'].values
    y = df['Y'].values
    d = df['Class'].values
    globalErrorRate = 0

    #define some variables to keep track
    numTurns = 0
    numWeightUpdates = 0

    while numTurns < maxIter:
        localErrorRate = 0
        successRate = 0
        falsePosNeg = 0
        truePosNeg = 0

        for i in range(len(x)):
            #calculate the discriminant
            discriminant = weights[0] + (weights[1] * x[i]) + (weights[2] * y[i])

            if(isPos(discriminant) and d[i] == 1):
                truePosNeg += 1

            elif(isPos(discriminant) == False and d[i] == -1):
                truePosNeg += 1

            else:
                falsePosNeg += 1
                weights = weightsUpdate(weights, constantC, constantK, d[i], x[i], y[i])

        numTurns += 1 #increase number of turns by 1 iteration
        print("Number of False Positive/Negative: " + str(falsePosNeg))
        print("Number of True Positive/Negative: " + str(truePosNeg))
        localErrorRate = falsePosNeg / len(x) * 100
        successRate = truePosNeg / len(x) * 100
        print("Error rate: " + str(localErrorRate) + "%")
        print("Success rate: " + str(successRate) + "%")


        if successRate == 100:
            print("\n\nTrained Weight Values: " + str(weights))
            print("Number of iterations: " + str(numTurns))
            break

        else:
            continue

    return weights

'''this function will test the accuracy of the model with one pass through the dataset'''
def testModel(weights):
    testDF = createData() #create data to test the model with

    #grab the values from the test data
    x = testDF['X'].values
    y = testDF['Y'].values
    d = testDF['Class'].values

    #create an empty list of predicted values
    D = []
    numErrors = 0
    numCorrect = 0

    for i in range(len(x)):
        discriminant = weights[0] + (weights[1] * x[i]) + (weights[2] * y[i]) #make the prediction

        D.append(discriminant) #append the value to the list

    '''iterate through the list and let's see how we did'''
    for x in range(len(d)):
        if(isPos(D[i]) and d[i] == 1):
            numCorrect += 1
        elif(isPos(D[i]) == False and d[i] == -1):
            numCorrect += 1
        else:
            numErrors += 1

    print("\nNumber of errors on test data: " + str(numErrors))
    print("Error Rate: " + str(numErrors / len(d)))

    resultsDF = pd.DataFrame()
    resultsDF['Predicted Output'] = D
    resultsDF['Expected Output'] = d
    print(resultsDF)

def main():
    weights = [0.00000, 0.00000, 0.00000]
    df = createData()
    #plotVals(df) #just to show that the data is linearly separable
    weights = trainModel(df, weights, 0.01, 1, 1000000)
    testModel(weights)

main()
