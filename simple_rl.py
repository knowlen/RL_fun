# a: Nick Knowles
# d: 8/21
# Some basic openai gym use & Q learning for general reference. 
# -maybe play around w/ policy iteration here some day if time?

import numpy as np
import random
import math
import gym
env = gym.make('CartPole-v0')
Q_table = {}
num_actions = env.action_space.n
discount_factor = 0.99

""" BINS for state space """
angle_bins = [-0.25, -0.15, 0.0, 0.15, 0.25]
a_veloc_bins = [-0.87, 0.87]
x_veloc_bins = [-0.5, 0.5]
x_bins = [0]


def add_state(S):
    global Q_table
    Q_table[S] = {}
    for i in range(num_actions):
	Q_table[S][i] = -1 
    

def policy(S):
    """ 
    Either return action with highest value
    given state S, or a random action.
    """
    global Q_table
    if S not in Q_table.keys():
	add_state(S) 
    action_list = Q_table[S].values()
    if np.random.uniform(0, 1.0) >= exploration_rate: 
	v = max(action_list)
	act = action_list.index(v)
    else:
        act = random.choice([0,1])
    return act 

def Q(S, a, r, S_):
    global Q_table
    old_Q = Q_table[S][a]
    new_Q = max(Q_table[S_].values()) 
    Q_table[S][a] += (learn_rate * (r + discount_factor * new_Q - old_Q))


def bin_state(S):
    S[0] = np.digitize(S[0], x_bins)
    S[1] = np.digitize(S[1], x_veloc_bins)
    S[2] = np.digitize(S[2], angle_bins)
    S[3] = np.digitize(S[3], a_veloc_bins)
    return S


learn_rate = 0.01
exploration_rate = 0.99

""" TRAINING LOOP  """
for episode in range(1, 2000):
    S = str(bin_state(env.reset()))
    for t in range(200):
        # Create new row for S if needed
	if S not in Q_table.keys():
	    add_state(S)
        
        # Select an action and execute it in the environment   
	a = policy(S)
	S_, r, done, info = env.step(a)
	S_ = str(bin_state(S_))
	
        # Create new row for S' if needed
        if S_ not in Q_table.keys():
	    add_state(S_)

        # Update the Q table
        Q(S, a, r, S_)
	S = S_
        if done:
            # (Dropped the pole...)
	    print('Ep' + str(episode) + ': ' + str(t) + ' steps.')
            break




#testing
    #Q_table[S][a] = r + 0.99*(max(Q_table[S_].values()) - Q_table[S][a])
    #Q_table[S][a] = (0.01 * Q_table[S][a]) + (0.99 * (r + 0.99 * max(Q_table[S_].values())))
 
 #learn_rate = max(0.1, min(1.0 - math.log10(episode/25.0), 0.5))
    #exploration_rate = max(0.01, min(1.0 - math.log10(episode/25.0), 1.0))

