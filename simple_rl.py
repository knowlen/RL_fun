# a: Nick Knowles
# d: 8/21
# Some basic openai gym use & Q learning for general reference. 
# -maybe play around w/ policy iteration here some day if time?

import numpy as np
from PIL import Image
import random
import math
import gym
from collections import deque
env = gym.make('CartPole-v0')
Q_table = {}
num_actions = env.action_space.n
discount_factor = 0.99999

""" BINS for state space """
angle_bins = [-0.25, -0.15, 0.0, 0.15, 0.25]
a_veloc_bins = [-0.87, 0.87]
x_veloc_bins = [-0.5, 0.5]
x_bins = [0]

frames =  deque(maxlen= 1000)
def make_gif(name):
    with open(name+'.gif', 'wb') as f:  # change the path if necessary
        im = Image.new('RGB', frames[0].size)
        im.save(f, save_all=True, append_images=frames)
 
ave_score = 0.0

scores = deque(maxlen=100)

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
    Q_table[S][a] += (learn_rate * (r + (discount_factor * new_Q) - old_Q))


def bin_state(S):
    S[0] = np.digitize(S[0], x_bins)
    S[1] = np.digitize(S[1], x_veloc_bins)
    S[2] = np.digitize(S[2], angle_bins)
    S[3] = np.digitize(S[3], a_veloc_bins)
    return S


learn_rate = 0.01
exploration_rate = 0.99
mb = 0
T_i = 5
T_e = 5
solved = 0

m_learn_rate = 0.009 #max(0.001, min(1.0 - math.log10(episode/25.0), 0.9))
l_learn_rate = 0.0005
""" TRAINING LOOP  """
for episode in range(1, 2000):
    S = str(bin_state(env.reset()))
    m_learn_rate = max(0.0001, min(1.0 - math.log10(episode/5.0), 0.09))
    #learn_rate = max(0.1, min(1.0 - math.log10(episode/25.0), 0.5))

    mb += 1
    m_exploration_rate = max(0.0001, min(1.0 - math.log10(episode/5.0), 1.00))
    #if episode > 55:
    #    exploration_rate = 0.0001
    exploration_rate = 0.001 + (0.5 * (m_exploration_rate - 0.001)) * (1 + math.cos((mb / T_e) * math.pi)) 

    learn_rate = l_learn_rate + (0.5 * (m_learn_rate - l_learn_rate)) * (1 + math.cos((mb / T_i) * math.pi)) 
    if mb == T_i:
            mb = 1


    for t in range(2000000):
        #learn_rate = 0.001 + (0.5 * (m_learn_rate- 0.001)) * (1 + math.cos((mb / T_i) * math.pi)) 
        
                # Create new row for S if needed

        frames.append(Image.fromarray(env.render(mode='rgb_array')))  # save each frames
	if S not in Q_table.keys():
	    add_state(S)
        
        # Select an action and execute it in the environment   
	a = policy(S)
	S_, r, done, info = env.step(a)
        S_ = str(bin_state(S_)[2:])
	
        # Create new row for S' if needed
        if S_ not in Q_table.keys():
	    add_state(S_)

        # Update the Q table
        Q(S, a, r, S_)
	S = S_
        if done:
            # (Dropped the pole...)
            print('Ep' + str(episode) + ': ' + str(t) + ' steps. Ave Score: ' + str(ave_score))
            if len(scores) >= 100:
                ave_score -= scores.popleft()/100
            if t > 100:
                make_gif('solving')
            if ave_score > 200:
                make_gif('solved')
                print('Solved!')
                exit() 
            scores.append(t)
            ave_score += (t / 100)

            break




#testing
    #Q_table[S][a] = r + 0.99*(max(Q_table[S_].values()) - Q_table[S][a])
    #Q_table[S][a] = (0.01 * Q_table[S][a]) + (0.99 * (r + 0.99 * max(Q_table[S_].values())))
 
 
 
