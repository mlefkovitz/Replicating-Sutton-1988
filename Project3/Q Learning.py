import numpy as np
from soccerpdvelez import Player, World
from testbenchpdvelez import create_state_comb, print_status
import random
import math
from matplotlib import pyplot as plt

# state space = total possible spaces for each player with each combination of ball positions
# state space = grid size * (grid size - 1) * 2
rows = 2
cols = 4
num_states = rows * cols
state_size = num_states * (num_states - 1) * 2

# action_space = 5 actions for each player = 5*5
print("actions: [N: 0, S: 1, E: 2, W: 3, Stay: 4] \n")
num_actions = 5
action_size = num_actions * num_actions

qtable = np.zeros((state_size, action_size))

total_episodes = 1000000        # Total episodes

learning_rate = 0.7           # Learning rate
gamma = 0.9                   # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

world = World()
world.set_world_size(x=cols, y=rows)
world.set_goals(100, 0, 'A')
world.set_goals(100, 3, 'B')
world.set_commentator_on()

current_episode = 0
total_states = create_state_comb(range(num_states), range(num_states))

error_array = []

while current_episode < total_episodes:
    goal = False
    player_a = Player(x=2, y=0, has_ball=False, p_id='A')
    player_b = Player(x=1, y=0, has_ball=True, p_id='B')

    world.place_player(player_a, player_id='A')
    world.place_player(player_b, player_id='B')

    world.plot_grid()

    stateName = world.map_player_state()
    state = int(total_states[stateName])
    while goal == False and current_episode < total_episodes:
        print("Current Episode = " + str(current_episode) + "\n")
        current_episode += 1
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action_id = np.argmax(qtable[state, :])
            action_a = math.floor(action_id / 5)
            action_b = action_id % 5
        # Else doing a random choice --> exploration
        else:
            action_a = np.random.randint(5)
            action_b = np.random.randint(5)
            action_id = action_a * 5 + action_b

        actions = {'A': action_a, 'B': action_b}
        new_state, rewards, goal = world.move(actions)
        reward_a = rewards['B']
        reward_b = rewards['A']
        reward = reward_a
        world.plot_grid()
        print_status(goal, new_state, rewards, total_states)

        new_state_num = int(total_states[new_state])

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        current_err = learning_rate * (reward + gamma * np.max(qtable[new_state_num, :]) - qtable[state, action_id])
        qtable[state, action_id] = qtable[state, action_id] + current_err

        # Save errors to array for easy graphing
        error_array.append(abs(current_err)/100)

        # Our new state is state
        state = new_state_num

        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * current_episode)



plt.plot(error_array, marker=None)
plt.title('Figure 3d. Q-Learner')
plt.xlabel("Simulation Iteration")
plt.ylabel("Q-Value Difference")
plt.savefig('Figure3d Q-Learning.png', bbox_inches='tight')
plt.show()