import gym
import gym_gazebo

env = gym.make("GazeboSmartBotPincherKinect-v0")
print("random agent started")
for i_episode in range(2000):
    state = env.reset()
    for t in range(500):
        # env.render()
        action = env.action_space.sample()
        print("step {}.{}".format(i_episode, t))
        state, reward, done, info = env.step(action)

print("random agent finished")
env.close()