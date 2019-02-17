import random

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