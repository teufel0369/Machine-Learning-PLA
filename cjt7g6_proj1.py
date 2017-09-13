import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import os

'''Global file I/O'''
modelLog = open('modelLog.txt', 'w')

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

    for count in range(50):
        randChoice = random.randint(0, 1)
        x1 = random.uniform(-10, 5)
        y1 = random.uniform(-10, 5)
        x.append(x1)
        y.append(y1)

        if(randChoice == 1):
            d1 = 1
        else:
            d1 = -1

        d.append(d1)



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
def plotVals(df,):
    x = df['X'].values
    y = df['Y'].values
    d = df['Class'].values

    '''create a plot and two subplots for -1 and 1'''
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = fig.add_subplot(111)

    '''iterate over the x,y values and tag and plot them appropriately'''
    for index in range(len(x)):
        if d[index] == -1:
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
    globalErrorRate = 0

    #define some variables to keep track
    numTurns = 0
    numWeightUpdates = 0
    runningSuccessRate = []

    while numTurns < maxIter:
        localErrorRate = 0
        successRate = 0
        falsePosNeg = 0
        truePosNeg = 0

        for i in range(len(x)):

            #calculate the discriminant
            discriminant = weights[0] + (weights[1] * x[i]) + (weights[2] * y[i])

            if(isPos(discriminant) and d[i] == 1): #if sign(D) == d
                truePosNeg += 1 #add one to the successes

            elif(isPos(discriminant) == False and d[i] == -1):
                truePosNeg += 1

            else: #if sign(D) != d
                falsePosNeg += 1 #add one to the errors
                weights = weightsUpdate(weights, constantC, constantK, d[i], x[i], y[i]) #update the weights
                numWeightUpdates += 1

        '''take some stats about the iteration and print some updates'''
        numTurns += 1 #increase number of turns by 1 iteration
        print("\n\nNumber of False Positive/Negative: " + str(falsePosNeg))
        modelLog.write("\nNumber of False Positive/Negative: " + str(falsePosNeg))
        print("Number of True Positive/Negative: " + str(truePosNeg))
        modelLog.write("\nNumber of True Positive/Negative: " + str(truePosNeg))
        localErrorRate = falsePosNeg / len(x) * 100
        successRate = truePosNeg / len(x) * 100
        print("Error rate: " + str(localErrorRate) + "%")
        modelLog.write("\nError rate: " + str(localErrorRate) + "%")
        print("Success rate: " + str(successRate) + "%")
        modelLog.write("\nSuccess rate: " + str(successRate) + "%" + "\n")
        print("Number of iterations: " + str(numTurns))
        runningSuccessRate.append(successRate)

        '''***find a way to make it stop here using a running success rate not improving of the course of
        x amount of iterations****'''

        '''if the success rate reaches 100%, print the stats about the weights and number
        of turns then break out of the loop. Otherwise, continue until 100% success rate is reached.'''
        if successRate == 100:
            print("\nLine equation: " + lineEquation(weights))
            modelLog.write("\nLine equation: " + lineEquation(weights))
            print("Trained Weight Values: " + str(weights))
            modelLog.write("\nTrained Weight Values: " + str(weights))
            print("Number of weight updates: " + str(numWeightUpdates))
            modelLog.write("\nNumber of weight updates: " + str(numWeightUpdates))
            print("Number of iterations: " + str(numTurns))
            modelLog.write("\nNumber of iterations: " + str(numTurns) + "\n")
            break
        else:
            continue

    return weights #return the trained weights

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
        if(discriminant >= 0):
            D.append(1)
        else:
            D.append(-1)

    resultsDF = pd.DataFrame()
    resultsDF['Predicted Output'] = D
    resultsDF['Expected Output'] = d

    D1 = resultsDF['Predicted Output'].values
    d1 = resultsDF['Expected Output'].values

    for i in range(len(d1)):
        if d1[i] != D[i]:
            numErrors += 1

    errorRate = numErrors / 1050 * 100
    print("\nNumber of errors on test data: " + str(numErrors))
    modelLog.write("\nNumber of errors on test data: " + str(numErrors) + "\n")
    print("Error Rate: " + str(errorRate) + "\n")
    modelLog.write("Error Rate: " + str(errorRate) + "\n")
    print(resultsDF)
    modelLog.write("\n" + str(resultsDF))

'''create function to give equation of a line --->  y = mx + b '''
def lineEquation(weights):
    '''Ax + By + c = 0 ---> m = -A/B ---> y = -A/B - c'''
    c = weights[0]
    A = weights[1]
    B = weights[2]

    #create the string for equation of the line
    string = "y = " + str(A * -1) + " / " + str(B) + " + " + str(c * -1)

    return string

def main():
    weights = [0.00000, 0.00000, 0.00000]
    df = createData()
    plotVals(df) #just to show that the data is linearly separable
    weights = trainModel(df, weights, 0.01, 1, 10000)
    testModel(weights)
    modelLog.close()

main()
