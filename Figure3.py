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

def LoopUntilConverged(walkArray, alpha, epsilon, lambdaVar):
    error = epsilon + 1
    counter = 0
    weight = np.array([0, .5, .5, .5, .5, .5, 1])
    # weight = np.array([0., 0., 0., 0., 0., 0., 1])
    while error >= epsilon:
        counter += 1  # Count iterations for this training set
        delta_w = np.array([0., 0., 0., 0., 0., 0., 0.])
        for sequence in range(len(walkArray)):
            for t in range(0, len(walkArray[sequence])-1):
                t_1_State = walkArray[sequence][t+1]
                t_state = walkArray[sequence][t]
                x_t = np.array([0., 0., 0., 0., 0., 0., 0.])
                x_t_1 = np.array([0., 0., 0., 0., 0., 0., 0.])
                x_t[t_state] = 1
                x_t_1[t_1_State] = 1
                P_t = np.matmul(np.transpose(weight), x_t)
                P_t_1 = np.matmul(np.transpose(weight), x_t_1)
                gradientP = np.array([0., 0., 0., 0., 0., 0., 0.])
                lambdaGradientP = gradientP
                for k in range(0, t+1):
                    k_state = walkArray[sequence][k]
                    gradientP[k_state] = 1
                    lambdaGradientP[k_state] = (lambdaVar ** (t-k)) * gradientP[k_state]
                    pass
                current_delta_w = alpha * (P_t_1 - P_t) * lambdaGradientP
            delta_w += current_delta_w
            pass

        error = sum(abs(delta_w))  # Calculate Error
        weight = weight + delta_w
    return weight, counter



# basic implementation
alpha = 0.01
gamma = 1
epsilon = 0.01
correctValueFunction = np.array([0., 1./6, 2./6, 3./6, 4./6, 5./6, 1.])

lambdaArray = [0, .1, .3, .5, .7, .9, 1]
# lambdaArray = [1]
# avgConvergenceSteps = []
for lambdaVar in lambdaArray:
    counters = [] # Count the iterations to Converge
    allWeightsArray = [] # Store the weights for each training set
    allErrorsArray = []  # Store the errors for each training set
    allRMSEArray = []  # Store the RMSE for each training set
    for trainingSet in trainingSetArray:
        walkArray = trainingSet[0]
        scoreArray = trainingSet[1]
        previousValueFunction = np.array([0., 0., 0., 0., 0., 0., 0.])
        # currentValueFunction, eligibilityMatrix, counter = LoopUntilConverged(walkArray, scoreArray,
        #                                                                       previousValueFunction, alpha, gamma,
        #                                                                       epsilon, lambdaVar)

        weight, counter = LoopUntilConverged(walkArray, alpha, epsilon, lambdaVar)

        weightError = correctValueFunction - weight
        weightError = weightError[1:-1]
        squareError = np.square(weightError)
        meanSquareError = np.mean(squareError)
        rootMeanSquaredError = np.sqrt(meanSquareError)

        allWeightsArray.append(weight)
        allErrorsArray.append(weightError)
        allRMSEArray.append(rootMeanSquaredError)

        counters.append(counter)
    avgIterationsToConverge = np.average(counters)
    avgWeight = np.average(allWeightsArray, axis=0)
    avgError = np.average(allErrorsArray, axis=0)
    avgRMSE = np.average(allRMSEArray, axis=0)
    print("Lambda: " + str(lambdaVar) + " Avg Steps to Convege: " + str(avgIterationsToConverge))
    print("Average Weight: " + str(avgWeight))
    print("Average Error: " + str(avgError))
    print("Root Mean Squared Error: " + str(avgRMSE))
    print("")

