#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy as np
import random
import time
import liveplot
import pandas
from deepQ import DeepQ
import os

# DOES NOT QUITE WORK (FAR TOO MANY ACTIONS, NOT FULLY WORKED OUT)

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (epoch%render_interval == 0) and (epoch != 0) and (epoch > render_skip):
        env.render()
    elif ((epoch-render_episodes)%render_interval == 0) and (epoch != 0) and (epoch > render_skip) and (render_episodes < epoch):
        env.render(close=True)
# render

if __name__ == '__main__':

    epochs = 1000
    steps = 100000
    stepCounter = 0
    explorationRate = 1
    minibatch_size = 16
    learnStart = 16
    saveModelFreq = 128
    learningRate = 0.00025
    discountFactor = 0.99
    memorySize = 1000000

    moveRealArm = False

    last100Scores = [0] * 100
    last100ScoresIndex = 0
    last100Filled = False

    print("initializing environment")
    if(moveRealArm):
        env = gym.make('GazeboSmartBotPincherKinectREAL_ARM-v0')    # id registered in gym-gazebo/gym_gazebo/__init__.py
    else:
        env = gym.make('GazeboSmartBotPincherKinect-v0')            # id registered in gym-gazebo/gym_gazebo/__init__.py

    # print(env.action_space) # Box(6,)
    # print(env.action_space.shape[0]) # 6
    # print(env.action_space.high) # [0.3       0.25      0.5       6.2831855 6.2831855 6.2831855]
    # print(env.action_space.low) # [ 0.04 -0.25  0.    0.    0.    0.  ]
    # print(env.observation_space) # Box(220, 300)
    # print(env.observation_space.shape) # (220, 300)
    # print(env.observation_space.shape[0]) # 220
    # print(env.observation_space.high) # [[255 255 255 ... 255 255 255] ... [255 255 255 ... 255 255 255]]
    # print(env.observation_space.low) # [[0 0 0 ... 0 0 0] ... [0 0 0 ... 0 0 0]]

    actions_upper = env.action_space.high
    actions_lower = env.action_space.low

    # we fix the orientation of roll & pitch to 0 and pi/2, discretize the position & yaw space
    # into 10 classes/bins each, results in 10^4 possible actions
    number_of_features = env.action_space.shape[0] - 2 # 4
    n_bins = 10

    print(pandas.cut([actions_lower[0], actions_upper[0]], bins=n_bins, retbins=True)) # 0.04 ... 0.3 in 10 bins
    #     ([(0.0397, 0.066], (0.274, 0.3]]
    # Categories (10, interval[float64]): [(0.0397, 0.066] < (0.066, 0.092] < (0.092, 0.118] <
    #                                      (0.118, 0.144] ... (0.196, 0.222] < (0.222, 0.248] <
    #                                      (0.248, 0.274] < (0.274, 0.3]], array([0.03974   , 0.066     , 0.092     , 0.118     , 0.144     ,
    #        0.17000001, 0.19600001, 0.22200001, 0.24800001, 0.27400001, 0.30000001]))
    print(pandas.cut([actions_lower[0], actions_upper[0]], bins=n_bins, retbins=True)[1])
    # x_position_bins =     [0.03974    0.066      0.092      0.118      0.144      0.17000001   0.19600001 0.22200001 0.24800001 0.27400001 0.30000001]
    # index of bin for x =  [0          1          2          3          4          5            6          7          8          9          10]
    x_position_bins = pandas.cut([actions_lower[0], actions_upper[0]], bins=n_bins, retbins=True)[1]
    y_position_bins = pandas.cut([actions_lower[1], actions_upper[1]], bins=n_bins, retbins=True)[1]
    z_position_bins = pandas.cut([actions_lower[2], actions_upper[2]], bins=n_bins, retbins=True)[1]
    yaw_angle_bins = pandas.cut([actions_lower[5], actions_upper[5]], bins=n_bins, retbins=True)[1]

    deepQ = DeepQ(env.observation_space.shape, n_bins**number_of_features, memorySize, discountFactor, learningRate, learnStart)
    deepQ.initNetworks()

    # number of reruns
    for epoch in xrange(epochs):
        print("resetting environment")
        observation = env.reset() # array([[255, 241, 241, ..., 237, 247, 243], ...,  [255, 255, 255, ..., 255, 248, 244]], dtype=uint8)
        print(explorationRate)
        # number of timesteps
        for t in xrange(steps):
            # env.render()
            # time.sleep(5)
            # predict the q-values for each action according to the current observation (=state)
            qValues = deepQ.getQValues(observation) # qValues = array([ 0.00283056, ..., -0.00947946], dtype=float32)

            # Pick an action based on the current state
            action = deepQ.selectAction(qValues, explorationRate) # 0 ... 9'999

            # from this action index, get the position & orientation to which the arm has to move
            binIdx = str(action) # e.g. "1473"
            binIdx = binIdx.zfill(4) # in case action is < 1'000, pad string with zeros on the left
            binIdx = [int(x) for x in binIdx]
            binIdx = binIdx[::-1] # reverse array
            # binIdx[0] indicates the index of the lower bound of the bin for the x-position (e.g. bin index 3).
            # To get the x-position, calculate the middle between the lower & upper bound of this bin (e.g. mean(bin3, bin4).
            x = (x_position_bins[binIdx[0]] + x_position_bins[binIdx[0] + 1])/2
            y = (y_position_bins[binIdx[1]] + y_position_bins[binIdx[1] + 1])/2
            z = (z_position_bins[binIdx[2]] + z_position_bins[binIdx[2] + 1])/2
            roll = 0
            pitch = np.pi/2
            yaw = (yaw_angle_bins[binIdx[3]] + yaw_angle_bins[binIdx[3] + 1])/2

            targetPose = np.array([x, y, z, roll, pitch, yaw])

            # Execute the action and get feedback
            print("calling env.step... "+  str(t))
            newObservation, reward, done, info = env.step(targetPose)
            # newObservation = array([[255, 241, 241, ..., 237, 247, 243], ...,  [255, 255, 255, ..., 255, 248, 244]]), reward = 0.0, done = False, info = {}

            deepQ.addMemory(observation, action, reward, newObservation, done)

            # first we have to gain some experience in order to have a mini-batch for the network to train on
            if stepCounter >= learnStart:
                print("learning on mini-batch")
                deepQ.learnOnMiniBatch(minibatch_size)
                if stepCounter % saveModelFreq == 0:
                    deepQ.saveModel()

            observation = newObservation

            if done:
                last100Scores[last100ScoresIndex] = t
                last100ScoresIndex += 1
                if last100ScoresIndex >= 100:
                    last100Filled = True
                    last100ScoresIndex = 0
                if not last100Filled:
                    print "Episode ",epoch," finished after {} timesteps".format(t+1)
                else :
                    print "Episode ",epoch," finished after {} timesteps".format(t+1)," last 100 average: ",(sum(last100Scores)/len(last100Scores))
                break

            stepCounter += 1
        # for timesteps
        explorationRate *= 0.995
        explorationRate = max (0.05, explorationRate)
    # for epoch

    env.close()
# __main__