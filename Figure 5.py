import numpy as np
import random
from matplotlib import pyplot as plt
from RandomWalk import randomWalk
from Experiment2 import ExperimentTwo

random.seed(1)

trainingSetArray = []
for i in range(0,100):
    walkArray = []
    scoreArray = []
    for i in range(0,10):
        walk1, S1 = randomWalk()
        walkArray.append(walk1)
        scoreArray.append(S1)
    trainingSetArray.append([walkArray, scoreArray])

# basic implementation
alphaArray = [0.001, 0.005, 0.01, 0.05, 0.1, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]
correctValueFunction = np.array([0., 1./6, 2./6, 3./6, 4./6, 5./6, 1.])

lambdaArray = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# lambdaArray = [0.5]
avgRMSEArray = []
for lambdaVar in lambdaArray:
    allRMSEArray = []
    avgRMSEArrayAlpha = []
    for alpha in alphaArray:
        allWeightsArray = [] # Store the weights for each training set
        allErrorsArray = []  # Store the errors for each training set
        alphaRMSEArray = []  # Store the RMSE for each training set
        for trainingSet in trainingSetArray:
            walkArray = trainingSet[0]
            scoreArray = trainingSet[1]

            weight = ExperimentTwo(walkArray, alpha, lambdaVar)

            weightError = correctValueFunction - weight
            weightError = weightError[1:-1]
            squareError = np.square(weightError)
            meanSquareError = np.mean(squareError)
            rootMeanSquaredError = np.sqrt(meanSquareError)

            allWeightsArray.append(weight)
            allErrorsArray.append(weightError)
            alphaRMSEArray.append(rootMeanSquaredError)

        avgWeight = np.average(allWeightsArray, axis=0)
        avgError = np.average(allErrorsArray, axis=0)
        avgRMSE = np.average(alphaRMSEArray, axis=0)
        avgRMSEArrayAlpha.append(avgRMSE)
        print("Lambda: " + str(lambdaVar) + " Alpha: " + str(alpha) + " Root Mean Squared Error: " + str(avgRMSE))
    bestRMSE = min(avgRMSEArrayAlpha)
    avgRMSEArray.append(bestRMSE)

plt.plot(lambdaArray, avgRMSEArray, marker='o')
plt.title('Figure 5')
plt.xlabel("Lambda")
plt.ylabel("RMSE using best alpha")
plt.show()
