import gym
env = gym.make('CartPole-v0')
for i_episode in range(1):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            # print("Episode finished after {} timesteps".format(t+1))
            break




env = gym.make("Taxi-v2")
env.render()
action_size = env.action_space.n
print("Action size ", action_size)

state_size = env.observation_space.n
print("State size ", state_size)

import numpy as np
qtable = np.zeros((state_size, action_size))
print(qtable)


total_episodes = 2500000        # Total episodes
total_test_episodes = 100     # Total test episodes
max_steps = 200                # Max steps per episode

learning_rate = 0.7           # Learning rate
gamma = 0.900                 # Discounting rate

# Exploration parameters
epsilon = 1.0                 # Exploration rate
max_epsilon = 1.0             # Exploration probability at start
min_epsilon = 0.01            # Minimum exploration probability
decay_rate = 0.01             # Exponential decay rate for exploration prob

state = env.reset()

import random
# 2 For life or until learning is stopped
for episode in range(total_episodes):
    # Reset the environment
    state = env.reset()
    step = 0
    done = False

    for step in range(max_steps):
        # 3. Choose an action a in the current world state (s)
        ## First we randomize a number
        exp_exp_tradeoff = random.uniform(0, 1)

        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
        if exp_exp_tradeoff > epsilon:
            action = np.argmax(qtable[state, :])

        # Else doing a random choice --> exploration
        else:
            action = env.action_space.sample()

        # Take the action (a) and observe the outcome state(s') and reward (r)
        new_state, reward, done, info = env.step(action)

        # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma *
                                                                         np.max(qtable[new_state, :]) - qtable[
                                                                             state, action])

        # Our new state is state
        state = new_state

        # If done : finish episode
        if done == True:
            break

    # Reduce epsilon (because we need less and less exploration)
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)

    if episode % 1000 == 0:
        print(f"Episode: {episode}")

env.reset()
rewards = []

# for episode in range(total_test_episodes):
#     state = env.reset()
#     step = 0
#     done = False
#     total_rewards = 0
#     # print("****************************************************")
#     # print("EPISODE ", episode)
#
#     for step in range(max_steps):
#         # UNCOMMENT IT IF YOU WANT TO SEE OUR AGENT PLAYING
#         # env.render()
#         # Take the action (index) that have the maximum expected future reward given that state
#         action = np.argmax(qtable[state, :])
#
#         new_state, reward, done, info = env.step(action)
#
#         total_rewards += reward
#
#         if done:
#             rewards.append(total_rewards)
#             # print ("Score", total_rewards)
#             break
#         state = new_state
# env.close()
# print("Score over time: " + str(sum(rewards) / total_test_episodes))
# print(qtable)


Q = qtable

print(Q[0])
print("")
print("328: ")
print(Q[328])
print("")
print("462,4 -11.374402515: " + str(Q[462][4]))
print(Q[462])
print("")
print("398,3 = 4.348907: " + str(Q[398][3]))
print(Q[398])
print("")
print("253,0 = -0.5856821173: " + str(Q[253][0]))
print(Q[253])
print("")
print("377,1 = 9.683: " + str(Q[377][1]))
print(Q[377])
print("")
print("83,5 = -12.8232660372: " + str(Q[83][5]))
print(Q[83])
print("")
print("392,2 = " + str(Q[392][2]))
print("")
print("214,3 = " + str(Q[214][3]))
print("")
print("368,1 = " + str(Q[368][1]))
print("")
print("274,5 = " + str(Q[274][5]))
print("")
print("469,4 = " + str(Q[469][4]))
print("")
print("84,5 = " + str(Q[84][5]))
print("")
print("72,2 = " + str(Q[72][2]))
print("")
print("408,3 = " + str(Q[408][3]))
print("")
print("74,1 = " + str(Q[74][1]))
print("")
print("141,0 = " + str(Q[141][0]))




