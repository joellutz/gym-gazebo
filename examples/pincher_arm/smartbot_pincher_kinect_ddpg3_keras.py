import numpy as np

import gym
from gym import wrappers
import gym_gazebo

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate, Convolution2D, MaxPooling2D
from keras.optimizers import Adam

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

# DOESN'T QUITE WORK (DOESN'T LEARN CORRECTLY, NOT FULLY WORKED OUT)

class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)


def shrink(data, rows, cols):
    return data.reshape(rows, data.shape[0]/rows, cols, data.shape[1]/cols).sum(axis=1).sum(axis=2)

ENV_NAME = 'GazeboSmartBotPincherKinect-v0'
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
env = wrappers.Monitor(env, '/tmp/{}'.format(ENV_NAME), force=True)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0] # 6

# Next, we build a very simple model.
# Recipe of deep reinforcement learning model
# actor = Sequential()
# # actor.add(Convolution2D(16, kernel_size=3, input_shape=(64, 64, 1), activation='relu'))
# actor.add(Convolution2D(16, kernel_size=3, input_shape=(env.observation_space.shape[0], env.observation_space.shape[1], 1), activation='relu'))
# actor.add(MaxPooling2D(pool_size=(3,3)))
# actor.add(Convolution2D(16, kernel_size=3, activation='relu'))
# actor.add(Flatten())
# actor.add(Dense(100, activation='relu'))
# actor.add(Dense(nb_actions))
# actor.add(Activation('tanh'))
# # if(reloadModel):
# #     self.model.load_weights('model.h5')
# print(actor.summary())

actor = Sequential()
actor.add(Flatten(input_shape=(1,) + env.observation_space.shape))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(64))
actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('tanh'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')

# observation_input = Input(shape=(64, 64, 1), name='observation_input')
# x = Convolution2D(16, kernel_size=3)(observation_input)
# x = Activation('relu')(x)
# x = MaxPooling2D(pool_size=(3,3))(x)
# x = Convolution2D(16, kernel_size=3)(x)
# x = Activation('relu')(x)
# x = Flatten()(x)
# x = Concatenate()([x, action_input])
# x = Dense(100)(x)
# x = Activation('relu')(x)
# x = Dense(1)(x)
# x = Activation('linear')(x)
# critic = Model(inputs=[action_input, observation_input], outputs=x)
# print(critic.summary())

observation_input = Input(shape=(1,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Dense(64)(flattened_observation)
x = Activation('relu')(x)
x = Concatenate()([x, action_input])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=100000, window_length=1)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.1)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic, critic_action_input=action_input,
                  memory=memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                  random_process=random_process, gamma=.99, target_model_update=1e-3,
                  processor=MujocoProcessor())
agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
agent.fit(env, nb_steps=1000000, visualize=False, verbose=1)

# After training is done, we save the final weights.
agent.save_weights('ddpg_keras_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)