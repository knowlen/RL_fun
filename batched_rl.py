# a: Nick Knowles
# d: 8/21
# Some basic openai gym use & Q learning for general reference. 
# -maybe play around w/ policy iteration here some day if time?

import numpy as np
import random
import gym
from collections import deque
env = gym.make('CartPole-v0')

""" HYPERPARAMETERS """
learn_rate = 0.1
mb_size = 64
exploration_rate = 0.99 # decrease over timestep
discount_factor = 0.001 # increase over timestep


Q_table = {}
replay_buffer = deque(maxlen=1000)
num_actions = env.action_space.n


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
    S, a, r, S_, T = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
    r = r.astype(float, copy=False)
    Qs = np.asarray([float(Q_table[state][int(action)]) for state,action in zip(S, a)])
    Q_s = np.asarray([float(max(Q_table[state])) for state in S_])
    
    updated_Q = ((1.0-learn_rate) * Qs) + learn_rate * ((discount_factor * Q_s) + r) 
    for i, action in enumerate(a):
        state, terminal, new_value, reward = S[i], T[i], updated_Q[i], r[i]
        if terminal == True:
            Q_table[state][int(action)] = reward
        else:
            Q_table[state][int(action)] = new_value


""" TRAINING LOOP """
for i in range(200):
    S = str(env.reset().round(2))
    add_state(S)
	
    for t in range(1000):
            
    	a = action(S) 
	S_, r, done, info = env.step(a)
        env.render()

        # fix the types, 
        S_ = str(S_)
        r = float(r)

        # add the experience tuple to memory, 
	replay_buffer.append([S, a, float(r), S_, done])

        # put next state into the table,
	if S_ not in Q_table.keys():
	    add_state(S_)

        # once there are enough things to batch over,
        if 3*mb_size <= len(replay_buffer):
	    batch = np.asarray(random.sample(replay_buffer, mb_size))
            Q(batch)

	S = S_
        if done:
	    print('Ep' + str(i) + ': ' + str(t) + ' steps.')
            exploration_rate *= 0.99
            break

