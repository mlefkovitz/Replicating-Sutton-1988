import gym
import numpy as np

def heuristic(env, s):
    # Heuristic for:
    # 1. Testing.
    # 2. Demonstration rollout.
    angle_targ = s[0]*0.5 + s[2]*1.0         # angle should point towards center (s[0] is horizontal coordinate, s[2] hor speed)
    if angle_targ >  0.4: angle_targ =  0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4: angle_targ = -0.4
    hover_targ = 0.55*np.abs(s[0])           # target y should be proporional to horizontal offset

    # PID controller: s[4] angle, s[5] angularSpeed
    angle_todo = (angle_targ - s[4])*0.5 - (s[5])*1.0
    #print("angle_targ=%0.2f, angle_todo=%0.2f" % (angle_targ, angle_todo))

    # PID controller: s[1] vertical coordinate s[3] vertical speed
    hover_todo = (hover_targ - s[1])*0.5 - (s[3])*0.5
    #print("hover_targ=%0.2f, hover_todo=%0.2f" % (hover_targ, hover_todo))

    if s[6] or s[7]: # legs have contact
        angle_todo = 0
        hover_todo = -(s[3])*0.5  # override to reduce fall speed, that's all we need after contact

    # if env.continuous:
    #     a = np.array( [hover_todo*20 - 1, -angle_todo*20] )
    #     a = np.clip(a, -1, +1)
    # else:
    a = 0
    if hover_todo > np.abs(angle_todo) and hover_todo > 0.05: a = 2
    elif angle_todo < -0.05: a = 3
    elif angle_todo > +0.05: a = 1
    return a

def demo_heuristic_lander(env, seed=None, render=False):
    env.seed(seed)
    total_reward = 0
    steps = 0
    s = env.reset()
    while True:
        a = heuristic(env, s)
        s, r, done, info = env.step(a)
        print("step {} x {} y {} vx {} vy {} theta {} vtheta {} left-leg {} right-leg {} reward {} action {}".format(steps, s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], r, a))
        total_reward += r
        frames = []
        if render:
            still_open = env.render()
            # env.monitor.start('/tmp/cartpole-experiment-1', force=True)

            # env = wrappers.Monitor(env, 'your directory here', force=True)
            if still_open == False: break

        if steps % 20 == 0 or done:
            # print("observations:", " ".join(["{:+0.2f}".format(x) for x in s]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
        steps += 1
        if done: break
    return total_reward


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env = gym.wrappers.Monitor(env, "recording")
    demo_heuristic_lander(env, seed=0, render=True)