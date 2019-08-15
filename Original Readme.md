# 7642Spring2019gth836x

## Project 1
Python Files:
- RandomWalk.py
- Figure3.py
- Experiment2.py
- Figure4.py
- Figure5.py

### RandomWalk.py
- Contains the randomWalk function.
- When called, this function returns a single random walk.

### Figure3.py
- Implements numpy, random, and matplotlib. Also implements the randomWalk function above.
- Seeded with random.seed(1)
- Initially creates 100 training sets of 10 randomWalk sequences.
-Loops through each training set with a set of lambdas to compute RMSE for each lambda, repeating until convergence (Experiment 1).
- Plots RMSE vs Lambda and outputs to file (Sutton Figure 3).

### Experiment2.py
- Contains the ExperimentTwo function.
- When called, this function returns the weight calculated by the TD equation for the given randomWalk sequence, lambda, and alpha.

### Figure4.py
- Implements numpy, random, and matplotlib. Also implements the randomWalk and ExperimentTwo functions above.
- Seeded with random.seed(1)
- Initially creates 100 training sets of 10 randomWalk sequences.
- Loops through each training set with a set of lambdas and alphas to compute RMSE for each lambda-alpha combination (Experiment 2).
- Plots RMSE vs alpha for each lambda and outputs to file (Sutton Figure 4).

### Figure5.py
- Implements numpy, random, and matplotlib. Also implements the randomWalk and ExperimentTwo functions above.
- Seeded with random.seed(1)
- Initially creates 100 training sets of 10 randomWalk sequences.
- Loops through each training set with a set of lambdas and alphas to compute RMSE for each lambda-alpha combination (Experiment 2).
- Identifies best RMSE for each lambda.
- Plots RMSE vs lambda and outputs to file (Sutton Figure 5).