# a: Nick Knowles
# d: 8/21
# Some basic openai gym use & Q learning for general reference. 
# -maybe play around w/ policy iteration here some day if time?

import numpy as np
import gym
env = gym.make('CartPole-v0')

# hash string(state variables list) => action space
Q_table = {}

replay_buffer = []
num_actions = env.action_space.n

def add_state(S):
    global Q_table
    Q_table[S] = {}
    for i in range(num_actions):
	Q_table[S][i] = -1 #np.random.uniform(0, 1.0)
    

def action(S, env):
    global Q_table
    if S not in Q_table.keys():
	add_state(S) 
    
    action_list = Q_table[S].values()
    act = env.action_space.sample()
    if np.random.uniform(0, 1.0) >= 0.1 and len(replay_buffer) > 500: 
	v = max(action_list)
	act = action_list.index(v)
    return act 

def Q(S, a, r, S_):
   # TODO:
   #	-Take out args, just sample from replay randomly
    global Q_table
    #Q_table[S][a] = r + 0.99*(max(Q_table[S_].values()) - Q_table[S][a])
    Q_table[S][a] = (0.01 * Q_table[S][a]) + (0.99 * (r + 0.99 * max(Q_table[S_].values())))

for i in range(200):
    S = str(env.reset().round(2))
    for t in range(1000):
        env.render()
         
	if S not in Q_table.keys():
	    add_state(S)
	
	a = action(S, env) 
	S_, r, done, info = env.step(a)
	S_ = str(S_.round(2))
	replay_buffer.append([S, a, r, S_])	
	if S_ not in Q_table.keys():
	    add_state(S_)
	Q(S, a, r, S_)
	for i in range(len(replay_buffer)/2):
	    replay = replay_buffer[np.random.randint(0,len(replay_buffer))]
	    rQ  = replay[0] 
	    ra  = replay[1]
	    rr  = replay[2]
	    rS_ = replay[3]
	    Q(rQ, ra, rr, rS_)
	S = S_
        if done:
	    print('Ep' + str(i) + ': ' + str(t) + ' steps.')
            break

