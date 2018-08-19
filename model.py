from __future__ import division
from keras.models import Sequential      # One layer after the other
from keras.layers import Dense, Flatten, Dropout, Conv2D  # Dense layers are fully connected layers, Flatten layers flatten out multidimensional inputs
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from collections import deque            # For storing moves 
import keras.backend as K
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import TrainEpisodeLogger, ModelIntervalCheckpoint
import argparse
import sys
sys.path.append('../../keras-rl')
from PIL import Image
import numpy as np
import gym                                # To train our network


import random     # For sampling batches from the observations

# For making it easier too train i downsize the maze too 84 by 84 reducing the amout of pixels also gives a short sequence for knowing speed and direction
INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

#This processor changes the picture and grayscales it and saves it in storage
class AtariProcessor(Processor):
    def process_observation(self, observation):
        assert observation.ndim == 3
        img = Image.fromarray(observation)
        img = img.resize(INPUT_SHAPE).convert('L')  # resize and convert to grayscale
        processed_observation = np.array(img)
        assert processed_observation.shape == INPUT_SHAPE
        return processed_observation.astype('uint8')  # saves storage in experience memory

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

#For commandline arguments like training and testing
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default='MsPacman-v0')
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

#Defines which game and the number of actions it can take in that game, in pacman there is nine for a joystick.
env = gym.make(args.env_name)
np.random.seed(231)
env.seed(123)
nb_actions = env.action_space.n
print("NUMBER OF ACTIONS: " + str(nb_actions))

nb_actions = env.action_space.n
#This is the CNN model that extract the information out of the games pixels, and outputs the 9 values for the 9 actions.
model = Sequential()
model.add(Conv2D(32, kernel_size=(8,8), strides=4,  input_shape=(WINDOW_LENGTH, INPUT_SHAPE[0], INPUT_SHAPE[1]), data_format='channels_first', activation='relu'))
model.add(Conv2D(64, kernel_size=(4,4), strides=2, data_format='channels_first', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), strides=1, data_format='channels_first', activation='relu'))
model.add(Flatten())       # Flatten input so as to have no problems with processing
model.add(Dense(512, init='uniform', activation='relu'))
model.add(Dense(256, init='uniform', activation='relu'))
model.add(Dense(128, init='uniform', activation='relu'))
model.add(Dense(nb_actions, init='uniform', activation='linear'))    # Same number of outputs as possible actions
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

memory = SequentialMemory(limit=1000000, window_length=4)
processor = AtariProcessor()
#The policy is based around how willing the agent is too do a random move and how much it care about future rewards
#Example would be going through a part of the maze with no cheese, but beacuse it took the pill so it can eat the ghosts, the reward will be bigger
policy = policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05, nb_steps=1250000)

#Keras-rl makes it easy too make this Q agent, by using the following function
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory,
               processor=processor, enable_double_dqn=False, enable_dueling_network=False, nb_steps_warmup=50000, gamma=.99, target_model_update=10000,
               train_interval=4, delta_clip=1.)

dqn.compile(Adam(lr=.00025), metrics=['mae'])

folder_path = 'C:/Users/Mads/Desktop/model_saves/Vanilla' #Write your own folder addres here

#This for training it is possible too adjust how many steps of training you wanna do, change the interval value this is what saves the weights, and change the steps how many training steps it should take
if args.mode == 'train':
    weights_filename = folder_path + 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = folder_path + 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = folder_path + 'dqn_' + args.env_name + '_REWARD_DATA.txt'
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=500000)]
    
    dqn.fit(env, callbacks=callbacks, nb_steps=1000000, verbose=1, nb_max_episode_steps=20000)
    
#Test phase is where you can see pacman play the game, be sure that the file name is the same as written at: weights_filename
#nb_episodes decides how many games it will play on it's own.
elif args.mode == 'test':
    weights_filename = folder_path + 'dqn_MsPacman-v0_weights_1000000.h5f'
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True, nb_max_start_steps=80)
