# a: Nick Knowles
# d: 8/21
# -A batched version of simple_rl.py w/ experience replay,
#  learn rate decay, and exploration rate decay.  
import math
import numpy as np
import random
import gym
from collections import deque
env = gym.make('CartPole-v0')

""" HYPERPARAMETERS """
mb_size = 128
discount_factor = 0.99 # static enviornment (cartpole)
use_buckets = True


Q_table = {}
replay_buffer = deque(maxlen=200)
num_actions = env.action_space.n
angle_bins = [-0.3, -0.15, 0.0, 0.15, 0.3] 
a_veloc_bins = [-0.87, 0, 0.87]
x_veloc_bins = [-0.5, 0, 0.5]
x_bins = [-3, 0, 3]

def add_state(S):
    """
    ::  -Adds a state to the table and initializes its Q values to 0
    """
    Q_table[S] = [0.0 for i in range(num_actions)]


def action(S):
    """
    ::  -A function to obtain the next action to take.  
    ::  -takes: current state
    ::  -returns: Either optimal action for state, or random
    ::            action with chance exploration_rate.
    """
    action_list = Q_table[S]
    if np.random.uniform(0, 1.0) >= exploration_rate:
	v = max(action_list)
	act = action_list.index(v)
    else:
        act = random.choice([0, 1])
    return int(act) 


def Q(batch):
    """
    ::  -updates: the Q values associated w/ each state in batch.
    ::  -takes: minibatch of experience tuples (MB x 5)
    """
    global discount_factor
    S, a, r, S_, T = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
    r = r.astype(float, copy=False)
    Qs = np.asarray([float(Q_table[state][int(action)]) for state,action in zip(S, a)])
    Q_s = np.asarray([float(max(Q_table[state])) for state in S_])
    
    #updated_Q = ((1.0-learn_rate) * Qs) + learn_rate * ((discount_factor * Q_s) + r) 
    updated_Q = Qs + (learn_rate * (r + discount_factor * Q_s - Qs))
    for i, action in enumerate(a):
        state, terminal, new_value, reward = S[i], T[i], updated_Q[i], r[i]
        #if terminal:
        #    Q_table[state][int(action)] = reward
        #else:
        Q_table[state][int(action)] = new_value


    """ TRAINING LOOP """
for i in range(1000):
    # note: discarded observation[0], it is just 
    #       the x position -scales poorly w/ Q table && 
    #       doesn't have a stong enough correlation 
    #       to success or failure in the env. 
    S = env.reset()[1:]
    S[0] = np.digitize(S[0], x_bins)
    S[-1] = np.digitize(S[-1], a_veloc_bins)
    S[-2] = np.digitize(S[-2], angle_bins)
    S[-3] = np.digitize(S[-3], x_veloc_bins)

    S = str(S)
    add_state(S)
    learn_rate = max(0.1, min(1.0 - math.log10((i + 1.0)/25.0), 0.5))	
    exploration_rate = max(0.01, min(1.0 - math.log10((i + 1.0)/25.0), 1.0))	
    for t in range(1000):
    	a = action(S) 
	S_, r, done, info = env.step(a)
        env.render()

        # fix the types & bin the angle 
        S_[0] = np.digitize(S_[0], x_bins)
        S_[-1] = np.digitize(S_[-1], x_veloc_bins)
        S_[-2] = np.digitize(S_[-2], angle_bins)
        S_[-3] = np.digitize(S_[-3], a_veloc_bins)
        S_ = str(S_)
        #print S_
        r = float(r)

        # add the experience tuple to memory, 
	replay_buffer.append([S, a, float(r), S_, done])

        # put next state into the table,
	if S_ not in Q_table.keys():
	    add_state(S_)

        # once there are enough things to batch over,
        if mb_size <= len(replay_buffer):
	    batch = np.asarray(random.sample(replay_buffer, mb_size))
            Q(batch)
        env.render()
	S = S_
        if done:
            env.render()
	    print('Ep' + str(i) + ': ' + str(t) + ' steps.')
            break

