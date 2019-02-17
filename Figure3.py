import numpy as np
import random
import math

random.seed(1)


def randomWalk():
    walk = []
    S = []
    leftTerm = 0 #A
    center = 3 #D
    rightTerm = 6 #G
    walk.append(center)
    S.append(0)
    currentState = center
    while currentState != leftTerm and currentState != rightTerm:
        if random.random() < 0.5:
            newState = currentState - 1
            walk.append(newState)
            S.append(0)
            currentState = newState
        else:
            newState = currentState + 1
            walk.append(newState)
            if newState == rightTerm:
                S.append(1)
            else:
                S.append(0)
            currentState = newState
    return walk, S

trainingSetArray = []
for i in range(0,100):
    walkArray = []
    scoreArray = []
    for i in range(0,10):
        walk1, S1 = randomWalk()
        walkArray.append(walk1)
        scoreArray.append(S1)
    trainingSetArray.append([walkArray, scoreArray])

# print(walkArray)
# print(scoreArray)
# print(trainingSetArray)

def loopThroughSequence(walkArray, scoreArray, previousValueFunction, alpha, gamma, lambdaVar):
    eligibilityMatrix = []
    currentValueFunction = previousValueFunction
    for sequence in range(len(walkArray)):
        eligibility = np.array([0., 0., 0., 0., 0., 0., 0.])
        for currentStateNumber in range(1, len(walkArray[sequence])):
            currentState = walkArray[sequence][currentStateNumber]
            currentScore = scoreArray[sequence][currentStateNumber]
            previousState = walkArray[sequence][currentStateNumber - 1]
            eligibility[previousState] = 1
            reward = currentScore
            valueFunctionUpdatePortion = alpha * (
                        reward + gamma * previousValueFunction[currentState] - previousValueFunction[previousState])
            currentValueFunction = currentValueFunction + valueFunctionUpdatePortion * eligibility
            eligibility = gamma * lambdaVar * eligibility
        eligibilityMatrix.append(eligibility)
    return currentValueFunction, eligibilityMatrix

def LoopUntilConverged(walkArray, scoreArray, previousValueFunction, alpha, gamma, epsilon, lambdaVar):
    error = epsilon + 1
    counter = 0
    while error >= epsilon:
        counter += 1  # Count iterations for this training set
        # currentValueFunction, eligibilityMatrix = loopThroughSequence(walkArray, scoreArray, previousValueFunction,
        #                                                               alpha, gamma, lambdaVar)
        eligibilityMatrix = []
        currentValueFunction = previousValueFunction
        for sequence in range(len(walkArray)):
            eligibility = np.array([0., 0., 0., 0., 0., 0., 0.])
            for currentStateNumber in range(1, len(walkArray[sequence])):
                currentState = walkArray[sequence][currentStateNumber]
                currentScore = scoreArray[sequence][currentStateNumber]
                previousState = walkArray[sequence][currentStateNumber - 1]
                eligibility[previousState] = 1
                reward = currentScore
                valueFunctionUpdatePortion = alpha * (
                        reward + gamma * previousValueFunction[currentState] - previousValueFunction[previousState])
                currentValueFunction = currentValueFunction + valueFunctionUpdatePortion * eligibility
                eligibility = gamma * lambdaVar * eligibility
            eligibilityMatrix.append(eligibility)


        error = sum(abs(currentValueFunction - previousValueFunction))  # Calculate Error
        previousValueFunction = currentValueFunction
    return currentValueFunction, eligibilityMatrix, counter



# basic implementation
alpha = 0.01
gamma = 1
epsilon = 0.01
correctValueFunction = np.array([0., 1./6, 2./6, 3./6, 4./6, 5./6, 0.])

# lambdaArray = [0, .1, .3, .5, .7, .9, 1]
lambdaArray = [1]
# avgConvergenceSteps = []
for lambdaVar in lambdaArray:
    counters = [] # Count the iterations to Converge
    allValueFunctions = [] # Store the value function for each training set
    for trainingSet in trainingSetArray:
        walkArray = trainingSet[0]
        scoreArray = trainingSet[1]
        previousValueFunction = np.array([0., 0., 0., 0., 0., 0., 0.])
        currentValueFunction, eligibilityMatrix, counter = LoopUntilConverged(walkArray, scoreArray,
                                                                              previousValueFunction, alpha, gamma,
                                                                              epsilon, lambdaVar)

        # print("Value Function: " + str(currentValueFunction) + " Error: " + str(error))
        allValueFunctions.append(currentValueFunction)
        counters.append(counter)
        # print("Count of iterations to convergence: " + str(counter))
        # for i in range(len(eligibilityMatrix)):
        #     print(eligibilityMatrix[i])
    avgIterationsToConverge = np.average(counters)
    # avgConvergenceSteps.append(avgIterationsToConverge)
    avgValueFunction = np.average(allValueFunctions, axis=0)
    print("Lambda: " + str(lambdaVar) + " Avg Steps to Convege: " + str(avgIterationsToConverge))
    print("Average Value Function: " + str(avgValueFunction))
    valFuncError = correctValueFunction - avgValueFunction
    print("Average Error Function: " + str(valFuncError))
    squareError = np.square(valFuncError)
    rootMeanSquaredError = math.sqrt(np.mean(squareError)*7/5) # this takes the mean of the array of length 7, but only 5 of the values are meaningful
    print("Root Mean Squared Error: " + str(rootMeanSquaredError))

