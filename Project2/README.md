# 7642Spring2019gth836x

## Project 2
Python Files:
- Lunar Lander Heuristic.py
- Lunar Lander PG.py
- Lunar Lander PG Hyperparameters.py

### Lunar Lander Heuristic.py
- Uses the heuristic laid out in OpenAI's documentation
- Solves the problem in a single episode (no learning needed)

### Lunar Lander PG.py
- Implements numpy, gym, math, copy, and matplotlib.
- Seeded with seed = 100 (I like this number better than 1 or 0)
- Uses Policy Gradient algorithm originally from Karpathy's blog to solve the Lunar Lander problem
- Saves the best model before the maximum episodes and runs 100 trials against that model
- Plots all of the training episodes vs reward
- Plots all of the trials vs reward

### Lunar Lander PG Hyperparameters.py
- Same as above, but as a function that can be called with hyperparameters as inputs (and outputs to appropriately named files)
