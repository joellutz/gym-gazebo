'''
Q-learning approach for different RL problems
as part of the basic series on reinforcement learning @
https://github.com/vmayoral/basic_reinforcement_learning

Inspired by https://gym.openai.com/evaluations/eval_kWknKOkPQ7izrixdhriurA

        @author: Victor Mayoral Vilches <victor@erlerobotics.com>
'''
import gym
import numpy
import random
import pandas

class QLearn:
    def __init__(self, actions, epsilon, alpha, gamma):
        self.q = {}
        self.epsilon = epsilon  # exploration constant
        self.alpha = alpha      # discount constant
        self.gamma = gamma      # discount factor
        self.actions = actions

    def getQ(self, state, action):
        return self.q.get((state, action), 0.0)

    def learnQ(self, state, action, reward, value):
        '''
        Q-learning:
            Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
        '''
        oldv = self.q.get((state, action), None)
        if oldv is None:
            self.q[(state, action)] = reward
        else:
            self.q[(state, action)] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state, return_q=False):
        q = [self.getQ(state, a) for a in self.actions] # q = [0.0, 0.0]
        maxQ = max(q) # maxQ = 0.0

        if random.random() < self.epsilon:
            minQ = min(q); mag = max(abs(minQ), abs(maxQ))
            # add random values to all the actions, recalculate maxQ
            q = [q[i] + random.random() * mag - .5 * mag for i in range(len(self.actions))]
            maxQ = max(q)

        count = q.count(maxQ) # count = 2
        # In case there're several state-action max values
        # we select a random one among them
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == maxQ] # best = [0, 1]
            i = random.choice(best) # i = 1
        else:
            i = q.index(maxQ)

        action = self.actions[i] # action = 1
        if return_q: # if they want it, give it!
            return action, q
        return action

    def learn(self, state1, action1, reward, state2):
        # get maximum action value for the current state (state2) over all possible actions
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        # update the previous state & action pair (state1 & action) according to the Q-Learn algorithm
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]

if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # DEPRECATED as of 12/23/2016
    # env.monitor.start('/tmp/cartpole-experiment-1', force=True)
    #    # video_callable=lambda count: count % 10 == 0)
    
    env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
        # video_callable=lambda count: count % 10 == 0)

    print(env.action_space) # Discrete(2)
    print(env.observation_space) # Box(4,)
    print(env.observation_space.shape) # (4,)
    print(env.observation_space.shape[0]) # 4

    goal_average_steps = 195
    max_number_of_steps = 200
    last_time_steps = numpy.ndarray(0)
    n_bins = 8
    n_bins_angle = 10

    number_of_features = env.observation_space.shape[0] # 4
    last_time_steps = numpy.ndarray(0)

    # Number of states is huge so in order to simplify the situation
    # we discretize the space to: 10 ** number_of_features
    # (if all n_bins for all features are 10 --> here the space is descretized into 8*10*8*10 = 6'400 elements)
    print(pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True))
    # ([(-2.405, -1.8], (1.8, 2.4]]
    # Categories (8, interval[float64]): [(-2.405, -1.8] < (-1.8, -1.2] < (-1.2, -0.6] < (-0.6, 0.0] <
    #                                     (0.0, 0.6] < (0.6, 1.2] < (1.2, 1.8] < (1.8, 2.4]], array([-2.4048, -1.8   , -1.2   , -0.6   ,  0.    ,  0.6   ,  1.2   ,
    #         1.8   ,  2.4   ]))
    print(pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1])
    # [-2.4048 -1.8    -1.2    -0.6     0.      0.6     1.2     1.8     2.4   ]
    print(pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1])
    # [-1.8 -1.2 -0.6  0.   0.6  1.2  1.8]
    cart_position_bins = pandas.cut([-2.4, 2.4], bins=n_bins, retbins=True)[1][1:-1]    # [-1.8, -1.2, -0.6,  0. ,  0.6,  1.2,  1.8]
    pole_angle_bins = pandas.cut([-2, 2], bins=n_bins_angle, retbins=True)[1][1:-1]     # [-1.6, -1.2, -0.8, -0.4,  0. ,  0.4,  0.8,  1.2,  1.6]
    cart_velocity_bins = pandas.cut([-1, 1], bins=n_bins, retbins=True)[1][1:-1]        # [-0.75, -0.5 , -0.25,  0.  ,  0.25,  0.5 ,  0.75]
    angle_rate_bins = pandas.cut([-3.5, 3.5], bins=n_bins_angle, retbins=True)[1][1:-1] # [-2.8, -2.1, -1.4, -0.7,  0. ,  0.7,  1.4,  2.1,  2.8]

    # The Q-learn algorithm
    qlearn = QLearn(actions=range(env.action_space.n),
                    alpha=0.5, gamma=0.90, epsilon=0.1)

    print(env.action_space.n) # 2
    
    for i_episode in xrange(3000):
        observation = env.reset() # array([ 0.03531468, -0.00567917, -0.00695458,  0.00468278])

        cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
        # state building: for each feature, decide which bin it belongs to and concatenate this number for all features
        state = build_state([to_bin(cart_position, cart_position_bins),
                         to_bin(pole_angle, pole_angle_bins),
                         to_bin(cart_velocity, cart_velocity_bins),
                         to_bin(angle_rate_of_change, angle_rate_bins)])
        
        print(to_bin(cart_position, cart_position_bins)) # 4
        print(state) # 4435

        for t in xrange(max_number_of_steps):
            # env.render()

            # Pick an action based on the current state
            action = qlearn.chooseAction(state) # action = 1
            # Execute the action and get feedback
            observation, reward, done, info = env.step(action)

            # Digitize the observation to get a state
            cart_position, pole_angle, cart_velocity, angle_rate_of_change = observation
            nextState = build_state([to_bin(cart_position, cart_position_bins),
                             to_bin(pole_angle, pole_angle_bins),
                             to_bin(cart_velocity, cart_velocity_bins),
                             to_bin(angle_rate_of_change, angle_rate_bins)])

            # nextState = 4534

            # # If out of bounds
            # if (cart_position > 2.4 or cart_position < -2.4):
            #     reward = -200
            #     qlearn.learn(state, action, reward, nextState)
            #     print("Out of bounds, reseting")
            #     break

            if not(done):
                qlearn.learn(state, action, reward, nextState)
                state = nextState
            else:
                # Q-learn stuff
                reward = -200
                qlearn.learn(state, action, reward, nextState)
                last_time_steps = numpy.append(last_time_steps, [int(t + 1)])
                break

    l = last_time_steps.tolist()
    l.sort()
    print("Overall score: {:0.2f}".format(last_time_steps.mean()))
    print("Best 100 score: {:0.2f}".format(reduce(lambda x, y: x + y, l[-100:]) / len(l[-100:])))

    env.monitor.close()
    # gym.upload('/tmp/cartpole-experiment-1', algorithm_id='vmayoral simple Q-learning', api_key='your-key')