#!/usr/bin/env python
import gym
import gym_gazebo
import numpy as np
import random
import time

###### DOES NOT ACTUALLY WORK, WAS ONLY FOR TESTING PURPOSES OF THE ENVIRONMENT #######


if __name__ == '__main__':


    print("initializing environment")
    env = gym.make('GazeboSmartBotPincherKinect-v0')    # id registered in gym-gazebo/gym_gazebo/__init__.py
    
    print("resetting environment")
    observation = env.reset()

    env.render()
    time.sleep(5)

    # boundaries_xAxis = [0.04, 0.3]      # box position possiblities: (0.06, 0.22)
    # boundaries_yAxis = [-0.25, 0.25]    # box position possiblities: (-0.2, 0.2)
    # boundaries_zAxis = [0, 0.05]        # box position possiblities: (0.05, 0.1)
    # boundaries_roll = [0, 2*np.pi]
    # boundaries_pitch = [0, 2*np.pi]
    # boundaries_yaw = [0, 2*np.pi]

    yaw = np.arange(0.0, 2*np.pi, 0.2)

    for i in range(len(yaw)):
        action = np.array([0.0, -0.2, 0.05, 0.0, np.pi/2, yaw[i]]) # [0.0, 0.2, 0.05, 0.0, 1.6, 0.0]

    # while True:
    #     input_prompt = 'Enter the position.\n-> '
    #     action = np.array(input(input_prompt))

        observation, reward, done, info = env.step(action)

    env.close()
            
# __main__