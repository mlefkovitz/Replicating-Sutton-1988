import numpy as np
import random
from matplotlib import pyplot as plt
from RandomWalk import randomWalk

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

def ExperimentTwo(walkArray, alpha, lambdaVar):
    weight = np.array([0, .5, .5, .5, .5, .5, 1])
    # weight = np.array([0, 0., 0., 0., 0., 0., 1])
    for sequence in range(len(walkArray)):
        delta_w = np.array([0., 0., 0., 0., 0., 0., 0.])  # Fresh delta_w each iteration
        for t in range(0, len(walkArray[sequence])-1):
            # identify the states
            t_1_State = walkArray[sequence][t+1]
            t_state = walkArray[sequence][t]
            # initialize the x arrays
            x_t = np.array([0., 0., 0., 0., 0., 0., 0.])
            x_t_1 = np.array([0., 0., 0., 0., 0., 0., 0.])
            x_t[t_state] = 1
            x_t_1[t_1_State] = 1
            # Calculate Pt and PT+1
            P_t = np.matmul(np.transpose(weight), x_t)
            P_t_1 = np.matmul(np.transpose(weight), x_t_1)
            # initialize the gradient variables
            gradientP = np.array([0., 0., 0., 0., 0., 0., 0.])
            lambdaGradientP = np.array([0., 0., 0., 0., 0., 0., 0.])
            for k in range(0, t+1): #
                k_state = walkArray[sequence][k]
                gradientP[k_state] = 1
                lambdaGradientP[k_state] = (lambdaVar ** (t-k)) * gradientP[k_state]
            delta_w += alpha * (P_t_1 - P_t) * lambdaGradientP
        weight = weight + delta_w  # after this training set has been evaluated, add delta_w to the weight
    return weight

# basic implementation
alphaArray = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
epsilon = 0.01
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
plt.show()