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

def LoopUntilConverged(walkArray, alpha, epsilon, lambdaVar):
    error = epsilon + 1
    counter = 0
    weight = np.array([0, .5, .5, .5, .5, .5, 1])
    # weight = np.array([0, 0., 0., 0., 0., 0., 1])
    while error >= epsilon:
        counter += 1  # Count iterations for this training set
        delta_w = np.array([0., 0., 0., 0., 0., 0., 0.]) # Fresh delta_w each iteration
        for sequence in range(len(walkArray)):
            current_delta_w = np.array([0., 0., 0., 0., 0., 0., 0.])
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
                current_delta_w += alpha * (P_t_1 - P_t) * lambdaGradientP
            delta_w += current_delta_w # At the end of each sequence add each delta_w to a big delta_w variable
        error = sum(abs(delta_w))  # Calculate Error
        weight = weight + delta_w # after all of the sequences in this training set have been evaluated, add delta_w to the weight
    return weight, counter

# basic implementation
alpha = 0.01
epsilon = 0.01
correctValueFunction = np.array([0., 1./6, 2./6, 3./6, 4./6, 5./6, 1.])

lambdaArray = [0, .1, .3, .5, .7, .9, 1]
# lambdaArray = [0.5]
avgRMSEArray = []
for lambdaVar in lambdaArray:
    counters = [] # Count the iterations to Converge
    allWeightsArray = [] # Store the weights for each training set
    allErrorsArray = []  # Store the errors for each training set
    allRMSEArray = []  # Store the RMSE for each training set
    for trainingSet in trainingSetArray:
        walkArray = trainingSet[0]
        scoreArray = trainingSet[1]

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
    avgRMSEArray.append(avgRMSE)
    print("Lambda: " + str(lambdaVar) + " Avg Steps to Convege: " + str(avgIterationsToConverge))
    print("Average Weight: " + str(avgWeight))
    print("Average Error: " + str(avgError))
    print("Root Mean Squared Error: " + str(avgRMSE))
    print("")

plt.plot(lambdaArray, avgRMSEArray, marker='o')
plt.title('Figure 3')
plt.xlabel("Lambda")
plt.ylabel("RMSE")
plt.savefig('Figure3.png', bbox_inches='tight')
plt.show()

