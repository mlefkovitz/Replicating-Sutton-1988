import numpy as np

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