# a: Nick Knowles
# d: 8/21
#   (WIP) 
# -A hacky implementation of deep Q learning 
#  using a regression vector to approximate the Q
#  table w/ a deep neural net.
#   -missing significant components from the 
#    the literature, namely a target network, ddqn,
#    exploration decay, and loss clipping.
from PIL import Image
import math
import numpy as np
import random
import gym
from collections import deque
import tensorflow as tf
from keras.layers import Dense


env = gym.make('CartPole-v0')

""" HYPERPARAMETERS """
mb_size = 128
discount_factor = 0.99   # static enviornment -cartpole

Q_table = {}
replay_buffer = deque(maxlen=10000)
num_actions = env.action_space.n
state_dim = 4


def dnn(input_layer, num_units=50, num_layers=2):
    hidden_layer = Dense(num_units, activation='tanh', bias_initializer='random_uniform')(input_layer)
    for i in range(0, num_layers):
        hidden_layer = Dense(num_units, activation='tanh', 
                  kernel_initializer='random_uniform', 
                  bias_initializer='random_uniform')(hidden_layer)
    output_layer = Dense(num_actions, bias_initializer='random_uniform', 
                         activation='linear')(hidden_layer)
    return output_layer;

input_layer = tf.placeholder(tf.float32, shape=(None, state_dim))
labels = tf.placeholder(tf.float32, shape=(None, num_actions))
outputs = dnn(input_layer)
loss = tf.losses.mean_squared_error(labels, outputs)
train_step = tf.train.RMSPropOptimizer(learning_rate=0.025).minimize(loss)


"""         TRAINING LOOP       """
"""
:: -TODO: -MSE error clipping
::        -Decaying exploration
::        -Target network
::        -embeddings  
::        -clean up code
"""

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
streak = 0
ep = 1.0
exploration_rate = 0.99
learn_rate = 0.9
frames = []
with sess.as_default():
    while streak < 1:
        S = env.reset()
        learn_rate = max(0.1, min(1.0 - math.log10((float(ep))/25.0), 0.5))	

        env.render()
        exploration_rate *= max(0.1, exploration_rate*0.99) #max(0.01, min(1.0 - math.log10((ep)/25.0), 1.0))	
        for t in xrange(200):
            env.render()
            
            frames.append(Image.fromarray(env.render(mode='rgb_array')))  # save each frames
            
            if np.random.uniform(0, 1.0) >= exploration_rate:
                a = sess.run(outputs, feed_dict={input_layer:S.reshape(1,4)}).argmax()
            else:
                a = random.choice([0, 1])

            S_, r, done, info = env.step(a)
            replay_buffer.append([S, a, float(r), S_, done])
            S = S_

            env.render()
            if len(replay_buffer) > mb_size:
                batch = random.sample(replay_buffer, mb_size)
                batch_S = np.array([row[0] for row in batch])
                batch_next_S = np.array([row[3] for row in batch])
                batch_reward = np.array([row[2] for row in batch])
                actions =  np.array([row[1] for row in batch])

                Qvec = sess.run(outputs, feed_dict={input_layer:batch_S})
                Qnext = sess.run(outputs, feed_dict={input_layer:batch_next_S}).max(axis=1)
                Qnext = (Qnext * discount_factor) + r
                label_vec = Qvec
                for ind,a in enumerate(actions):
                    label_vec[ind][a] = Qnext[ind]
                 
                _ = sess.run(train_step, feed_dict={input_layer:batch_S, labels:label_vec})
            if done:
                env.render()
                print('Ep' + str(int(ep)) + ': ' + str(t) + ' steps.')
                if t >= 100:
                    make_gif()
                    streak += 1
                else:
                    streak = 0
                ep +=1.0
                break

def make_gif():
    with open('./test.gif', 'wb') as f:  # change the path if necessary
        im = Image.new('RGB', frames[0].size)
        im.save(f, save_all=True, append_images=frames)
    
 
 
