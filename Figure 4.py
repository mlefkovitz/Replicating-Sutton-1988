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
alphaArray = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
correctValueFunction = np.array([0., 1./6, 2./6, 3./6, 4./6, 5./6, 1.])

lambdaArray = [0.0, .3, .8, 1.0]
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
    avgRMSEArray.append(avgRMSEArrayAlpha)

labels = ["Lambda = " + str(lambdaArray[0]),
            "Lambda = " + str(lambdaArray[1]),
            "Lambda = " + str(lambdaArray[2]),
            "Lambda = " + str(lambdaArray[3])]

for i,l in zip(avgRMSEArray,labels):
    plt.plot(alphaArray, i, label=l, marker='o')
    plt.legend(labels)
plt.title('Figure 4')
plt.xlabel("alpha")
plt.ylabel("RMSE")
plt.ylim(0, 0.65)
plt.savefig('Figure4.png', bbox_inches='tight')
plt.show()