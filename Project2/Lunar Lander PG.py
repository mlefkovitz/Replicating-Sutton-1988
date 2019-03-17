""" Trains an agent with (stochastic) Policy Gradients on Pong. Uses OpenAI Gym. """
import numpy as np
import gym
import matplotlib.pyplot as plt
import math
import copy
# model initialization
I = 8 # input dimensionality: 8 dimensional state space
O = 4 # output dimensionality: 4 dimensional action space

# hyperparameters
H = 10 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad^2
# resume = False # resume from previous checkpoint?
render = False
# render = True
seed = 100
# seed = 1
deterministic_action = False
# deterministic_action = True
max_episodes = 4000

np.random.seed(seed)

model = {}
model['W1'] = np.random.randn(H, I) / np.sqrt(I) # "Xavier" initialization
model['W2'] = np.random.randn(H,O) / np.sqrt(O) # "Xavier" initialization
grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
models = [] # store recent models
models.append(copy.deepcopy(model))
best_model = { k : np.zeros_like(v) for k,v in model.items() } # store best model

def sigmoid(x):
  return 1.0 / (1.0 + np.exp(-x)) # sigmoid "squashing" function to interval [0,1]

def discount_rewards(r):
  """ take 1D float array of rewards and compute discounted reward """
  discounted_r = np.zeros_like(r)
  running_add = 0
  for t in reversed(range(0, r.size)):
    running_add = running_add * gamma + r[t]
    discounted_r[t] = running_add
  return discounted_r

def softmax(A):
    expA = np.exp(A)
    softmax = expA / np.sum(expA)
    return softmax

def policy_forward(x):
  Z = np.dot(model['W1'], x)
  A = sigmoid(Z)
  Z2 = np.dot(A, model['W2'])
  Y = softmax(Z2) # probability matrix
  return Y, A # return probability array and hidden state

def policy_backward(Y, T, eph, epdp, epx):
  """ backward pass. (eph is array of intermediate hidden states) """
  delta2 = Y - T
  delta1 = (delta2).dot(model['W2'].T) * eph * (1 - eph)

  dW2 = eph.T.dot(delta2)

  dW1 = epx.T.dot(delta1)

  dW2 = np.dot(eph.T, epdp)
  dh = np.dot(epdp, model['W2'].T)
  dW1 = np.dot(dh.T, epx)
  return {'W1':dW1, 'W2':dW2}

def log_loss(actions, probabilities):
    n = np.shape(probabilities)[0]
    # logprobabilities = np.log(probabilities)
    # multiplied = actions * logprobabilities
    # summed = np.sum(multiplied)
    # loss = (-1.0/n) * summed
    loss = np.sum(-actions * np.log(probabilities))/n # simplified expression
    return loss

env = gym.make('LunarLander-v2')
env.seed(seed)
print('State shape: ', env.observation_space.shape)
print('Number of actions: ', env.action_space.n)

observation = env.reset()
prev_x = None # used in computing the difference frame
actions,avecs,xs,hs,ps,dps,drs = [],[],[],[],[],[],[]
reward_array = []
running_reward = None
running_reward_array = []
reward_sum = 0
episode_number = 0
step = 0

#Training Zone

while True:
    if render: env.render()

    x = observation

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    if abs(1 - sum(aprob)) > .01:
        print("probabilities out of range")
        break

    if deterministic_action: # deterministic (exploit)
        action = np.argmax(aprob)
    else: # stochastic (explore)
        randomnum = np.random.uniform()
        if randomnum < aprob[0]: action = 0
        elif randomnum < aprob[0] + aprob[1]: action = 1
        elif randomnum < aprob[0] + aprob[1] + aprob[2]: action = 2
        else: action = 3

    # record various intermediates (needed later for backprop)
    xs.append(x) # observation
    hs.append(h) # hidden state
    actions.append(action)
    # y = 1 if action == 2 else 0 # a "fake label"
    y = np.array([0,0,0,0])
    y[action] = 1
    avecs.append(y)
    ps.append(aprob)
    # dps.append(aprob)
    actionprob = y * aprob
    dps.append(y - aprob) # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)
    # dps.append(-actionprob)  # grad that encourages the action that was taken to be taken (see http://cs231n.github.io/neural-networks-2/#losses if confused)


    step += 1

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)


    if step > 999:
        done = True
        reward += -100

    reward_sum += reward

    # print("step {} x {} y {} vx {} vy {} theta {} vtheta {} left-leg {} right-leg {} reward {} action {}".format(
    #     step, observation[0], observation[1], observation[2], observation[3], observation[4], observation[5],
    #     observation[6], observation[7], reward, action))


    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdp = np.vstack(dps)
        epr = np.vstack(drs)
        epactions = np.vstack(actions)
        epavecs = np.vstack(avecs)
        epprobs = np.vstack(ps)
        actions, avecs, xs, hs, ps, dps, drs = [], [], [], [], [], [], [] # reset array memory

        logloss = log_loss(epavecs, epprobs)

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(epprobs, epavecs, eph, epdp, epx)
        for k in model: grad_buffer[k] += grad[k] # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k,v in model.items():
                g = grad_buffer[k] # gradient
                # rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g**2
                # model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                model[k] += learning_rate * g
                # print(learning_rate * g)
                grad_buffer[k] = np.zeros_like(v) # reset batch gradient buffer
            current_model = copy.deepcopy(model)
            models.append(current_model)

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        reward_array.append(reward_sum)
        hundred_running_reward = np.mean(reward_array[-100:])
        running_reward_array.append(hundred_running_reward)

        if np.max(running_reward_array) == hundred_running_reward:
            best_ep = 0
            model_iterator = 0
            if episode_number > 100:
                best_ep = episode_number - 100
                model_iterator = math.floor(best_ep / batch_size)
                model_iterator += math.floor(batch_size/2)
            best_model = copy.deepcopy(models[model_iterator])
            # print('max reward', hundred_running_reward, best_ep, model_iterator)

        max_reward_so_far = np.amax(reward_array)
        max_reward_past_100 = np.amax(reward_array[-100:])

        print('episode ' + str(episode_number) + ' took ' + str(step) + ' steps. reward total: %0.2f. all time max: %0.2f, recent max: %0.2f, running mean: %0.2f, recent mean: %0.2f, loss: %0.4f' % (reward_sum, max_reward_so_far, max_reward_past_100, running_reward, hundred_running_reward, logloss))
        # if episode_number % 100 == 0: pickle.dump(model, open('save.p', 'wb'))

        # Break Criteria
        if np.mean(reward_array[-100:]) >= 200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode_number - 100, np.mean(reward_array[-100:])))
            break

        if episode_number >= max_episodes:
            break

        reward_sum = 0
        observation = env.reset() # reset env
        step = 0


# Trials with Trained Algorithm
observation = env.reset()
trained_reward_array = []
running_reward = 0
reward_sum = 0
episode_number = 0
step = 0

model = copy.deepcopy(best_model)

while True:
    if render: env.render()

    x = observation

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    if abs(1 - sum(aprob)) > .01:
        print("probabilities out of range")
        break

    if deterministic_action: # deterministic (exploit)
        action = np.argmax(aprob)
    else: # stochastic (explore)
        randomnum = np.random.uniform()
        if randomnum < aprob[0]: action = 0
        elif randomnum < aprob[0] + aprob[1]: action = 1
        elif randomnum < aprob[0] + aprob[1] + aprob[2]: action = 2
        else: action = 3

    step += 1

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)

    reward_sum += reward

    # print("step {} x {} y {} vx {} vy {} theta {} vtheta {} left-leg {} right-leg {} reward {} action {}".format(
    #     step, observation[0], observation[1], observation[2], observation[3], observation[4], observation[5],
    #     observation[6], observation[7], reward, action))

    drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

    if done: # an episode finished
        episode_number += 1
        trained_reward_array.append(reward_sum)
        mean_reward = np.mean(trained_reward_array)
        print('episode ' + str(episode_number) + ' took ' + str(step) +
              ' steps. reward: %0.2f. running mean: %0.2f' % (reward_sum, mean_reward))

        if episode_number >= 100:
            break

        reward_sum = 0
        observation = env.reset()  # reset env
        step = 0


print('best model found at episode ', best_ep, ' average score ', np.max(running_reward_array))
print('100 trial episodes had a reward of ', mean_reward)

# plot the scores
fig = plt.figure()
plt.plot(np.arange(len(reward_array)), reward_array,linestyle="",marker="o")
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.title('Score per Training Episode')
plt.savefig('Score per Training Episode PG.png', bbox_inches='tight')
plt.show()

# plot the scores
fig = plt.figure()
plt.plot(np.arange(len(trained_reward_array)), trained_reward_array,linestyle="",marker="o")
plt.ylabel('Score')
plt.xlabel('Trial #')
plt.title('Score per Trail (trained agent)')
plt.savefig('Score per Trail (trained agent) PG.png', bbox_inches='tight')
plt.show()

