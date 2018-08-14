#!/usr/bin/env python
import gym
import gym_gazebo
import time
import numpy
import random
import time

import qlearn
import liveplot

import os


###### DOES NOT ACTUALLY WORK, WAS ONLY FOR TESTING PURPOSES OF THE ENVIRONMENT #######

def render():
    render_skip = 0 #Skip first X episodes.
    render_interval = 50 #Show render Every Y episodes.
    render_episodes = 10 #Show Z episodes every rendering.

    if (episode%render_interval == 0) and (episode != 0) and (episode > render_skip):
        env.render()
    elif ((episode-render_episodes)%render_interval == 0) and (episode != 0) and (episode > render_skip) and (render_episodes < episode):
        env.render(close=True)
# render

if __name__ == '__main__':

    moveRealArm = False

    print("initializing environment")
    if(moveRealArm):
        env = gym.make('GazeboSmartBotPincherKinectREAL_ARM-v0')    # id registered in gym-gazebo/gym_gazebo/__init__.py
    else:
        env = gym.make('GazeboSmartBotPincherKinect-v0')            # id registered in gym-gazebo/gym_gazebo/__init__.py

    # print(env.action_space) # Box(6,)
    # print(env.observation_space) # Box(220, 300)
    # print(env.observation_space.shape) # (220, 300)
    # print(env.observation_space.shape[0]) # 220

    last_time_steps = numpy.ndarray(0)

    qlearn = qlearn.QLearn(actions=range(2), alpha=0.2, gamma=0.8, epsilon=0.9)

    initial_epsilon = qlearn.epsilon

    epsilon_discount = 0.9986

    start_time = time.time()
    total_episodes = 1 # 10000
    highest_reward = 0

    for episode in range(total_episodes):
        done = False

        cumulated_reward = 0

        print("resetting environment")
        observation = env.reset()

        if qlearn.epsilon > 0.05:
            qlearn.epsilon *= epsilon_discount

        if(not moveRealArm):
            # render() #defined above, not env.render()
            env.render()

        state = ''.join(map(str, observation))

        # time.sleep(5)

        for timestep in range(1):

            # Pick an action based on the current state
            action = qlearn.chooseAction(state)

            # Execute the action and get feedback
            print("calling env.step... %i", timestep)
            observation, reward, done, info = env.step(action)
            cumulated_reward += reward

            if highest_reward < cumulated_reward:
                highest_reward = cumulated_reward

            nextState = ''.join(map(str, observation))

            qlearn.learn(state, action, reward, nextState)

            env.flush(force=True)

            if not(done):
                state = nextState
            else:
                last_time_steps = numpy.append(last_time_steps, [int(timestep + 1)])
                break
        # for timestep

        if episode%100==0:
            plotter.plot(env)

        m, s = divmod(int(time.time() - start_time), 60)
        h, m = divmod(m, 60)
        print ("EP: "+str(episode+1)+" - [alpha: "+str(round(qlearn.alpha,2))+" - gamma: "+str(round(qlearn.gamma,2))+" - epsilon: "+str(round(qlearn.epsilon,2))+"] - Reward: "+str(cumulated_reward)+"     Time: %d:%02d:%02d" % (h, m, s))
    # for episode

    #Github table content
    print ("\n|"+str(total_episodes)+"|"+str(qlearn.alpha)+"|"+str(qlearn.gamma)+"|"+str(initial_epsilon)+"*"+str(epsilon_discount)+"|"+str(highest_reward)+"| PICTURE |")

    # l = last_time_steps.tolist()
    # l.sort()

    # #print("Parameters: a="+str)
    # print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    # print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.close()
# __main__