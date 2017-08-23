# a: Nick Knowles
# d: 8/21
# Some basic openai gym use & Q learning for general reference. 
# -maybe play around w/ policy iteration here some day if time?

import numpy as np
import random
import gym
from collections import deque
env = gym.make('CartPole-v0')
learn_rate = 0.1
mb_size = 64
exploration_rate = 0.99 # decrease over timestep
discount_factor = 0.001 # increase over timestep
# hash string(state variables list) => action space
Q_table = {}

replay_buffer = deque(maxlen=1000)
#replay_buffer = []
num_actions = env.action_space.n

def add_state(S):
    global Q_table
    Q_table[S] = [-1.0 for i in range(num_actions)]

def action(S, env):
    global Q_table, exploration_rate
    if S not in Q_table.keys():
	add_state(S) 
    
    action_list = Q_table[S]
    if np.random.uniform(0, 1.0) >= exploration_rate:
	v = max(action_list)
	act = action_list.index(v)
    else:
        act = env.action_space.sample()
    return act 

#def Q(S, a, r, S_, done):
    #Q_table[S][a] = ((1-learn_rate) * Q_table[S][a]) + (learn_rate) * (r + discount_factor * max(Q_table[S_].values())))

def Q(batch):
   # TODO:
   #	-Batching
    global Q_table
    S, a, r, S_, T = batch[:,0], batch[:,1], batch[:,2], batch[:,3], batch[:,4]
    Qs = np.asarray([float(Q_table[state][int(action)]) for state,action in zip(S, a)])
    Q_s = np.asarray([float(max(Q_table[state])) for state in S_])
    updated_Q = ((1-learn_rate) * Qs) + (learn_rate * (r + (discount_factor * Q_s)))
    for i, action in enumerate(a):
        state, terminal, new_value, reward = S[i], T[i], updated_Q[i], r[i]
        if terminal:
            Q_table[state][action] = reward
        else:
            Q_table[state][action] = new_value

    #Q_table[S][a] = ((1-learn_rate) * Q_table[S][a]) + (learn_rate) * (r + discount_factor * max(Q_table[S_].values())))

for i in range(200):
    S = str(env.reset().round(2))
    for t in range(1000):
        #env.render()
         
	if S not in Q_table.keys():
	    add_state(S)
	
	a = action(S, env) 
	S_, r, done, info = env.step(a)
	S_ = str(S_.round(2))
        r = float(r)
	replay_buffer.append([S, a, r, S_, done])
	if S_ not in Q_table.keys():
	    add_state(S_)
	#Q(S, a, r, S_)
        if mb_size <= len(replay_buffer):
	    batch = np.asarray(random.sample(replay_buffer, mb_size))
            Q(batch)
       #else:
       #    batch = np.asarray(replay_buffer[-1]).reshape(1,5)
       #Q(batch) 
########for i in range(len(replay_buffer)/2):
########    replay = replay_buffer[np.random.randint(0,len(replay_buffer))]
########    rQ  = replay[0] 
########    ra  = replay[1]
########    rr  = replay[2]
########    rS_ = replay[3]
########    Q(rQ, ra, rr, rS_)
	S = S_
        if done:
	    print('Ep' + str(i) + ': ' + str(t) + ' steps.')
            exploration_rate *= 0.99
            break

