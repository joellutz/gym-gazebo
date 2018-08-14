""" 
Implementation of DDPG - Deep Deterministic Policy Gradient

Algorithm and hyperparameter details can be found here: 
    http://arxiv.org/pdf/1509.02971v2.pdf

The algorithm is tested on the Pendulum-v0 OpenAI gym task 
and developed with tflearn + Tensorflow

Author: Patrick Emami
"""
import tensorflow as tf
import numpy as np
import gym
import gym_gazebo
from gym import wrappers
import tflearn
import argparse
import pprint as pp

import time
import tensorflow.contrib as tc

from replay_buffer import ReplayBuffer

def normalize(x, stats):
    if stats is None:
        return x
    return (x - stats.mean) / stats.std

# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    """
    Input to the network is the state, output is the action
    under a deterministic policy.

    The output layer activation is a tanh to keep the action
    between -action_bound and action_bound
    """

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size, use_layer_norm=False, normalize_observations=False):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size
        self.use_layer_norm = use_layer_norm
        self.normalize_observations = normalize_observations

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        # TODO: Observation normalization.
        # from baselines.common.mpi_running_mean_std import RunningMeanStd
        # if self.normalize_observations:
        #     with tf.variable_scope("obs_rms"):
        #         self.obs_rms = RunningMeanStd(shape=observation_shape)
        # else:
        #     self.obs_rms = None
        # normalized_obs0 = tf.clip_by_value(normalize(self.obs0, self.obs_rms), self.observation_range[0], self.observation_range[1])
        # normalized_obs1 = tf.clip_by_value(normalize(self.obs1, self.obs_rms), self.observation_range[0], self.observation_range[1])
        
        # # in store_transition(obs0, action, reward, obs1, terminal):
        # if self.normalize_observations:
        #     self.obs_rms.update(np.array([obs0]))

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)
    # __init__

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 64) # 400
        if self.use_layer_norm:
            net = tc.layers.layer_norm(net, center=True, scale=True)
        else:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64) # 300
        if self.use_layer_norm:
            net = tc.layers.layer_norm(net, center=True, scale=True)
        else:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim, activation="tanh", weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out
    # create_actor_network

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })
    # train

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })
    # predict

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })
    # predict_target

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    # update_target_network

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    # get_num_trainable_vars
# class ActorNetwork


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars, use_layer_norm=False, normalize_observations=False, critic_l2_reg=0.0):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.use_layer_norm = use_layer_norm
        self.normalize_observations = normalize_observations
        self.critic_l2_reg = critic_l2_reg

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        # TODO: normalize_observations?

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        # Adding L2 regularization
        # from: https://github.com/pemami4911/deep-rl/issues/9#issuecomment-289229836
        self.L2 = np.sum([tf.nn.l2_loss(v) for v in self.network_params if "W" in v.name])
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out) + self.critic_l2_reg * self.L2

        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.out, self.action)
    # __init__

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 64) #, regularizer="L2", weight_decay=self.critic_l2_reg) # 400
        if self.use_layer_norm:
            net = tc.layers.layer_norm(net, center=True, scale=True)
        else:
            net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64) #, regularizer="L2", weight_decay=self.critic_l2_reg) # 300
        t2 = tflearn.fully_connected(action, 64) #, regularizer="L2", weight_decay=self.critic_l2_reg) # 300

        net = tflearn.activation(tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation="relu")

        net = tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b
        if self.use_layer_norm:
            net = tc.layers.layer_norm(net, center=True, scale=True)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)

        return inputs, action, out
    # create_critic_network

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })
    # train

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })
    # predict

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })
    # predict_target

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })
    # action_gradients

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)
    # update_target_network
# class CriticNetwork

def scale_range(x, x_min, x_max, y_min, y_max):
    # y = a*x + b
    # a = deltaY/deltaX
    # b = y_min - a*x_min (or y_max - a*x_max)
    y = (y_max - y_min) / (x_max - x_min) * x + (y_min*x_max - y_max*x_min) / (x_max - x_min)
    return y


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return "OrnsteinUhlenbeckActionNoise(mu={}, sigma={})".format(self.mu, self.sigma)
# class OrnsteinUhlenbeckActionNoise

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
# build_summaries

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic, actor_noise, reload):

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    # from https://github.com/openai/baselines/issues/162#issuecomment-397356482
    savingModelPath = "/home/joel/Documents/saved_models_OpenAI_gym/"
    if reload == True:
        print("Restoring from saved model")
        saver.restore(sess, tf.train.latest_checkpoint(savingModelPath))
    else:
        print("Starting from scratch!")
        sess.run(tf.global_variables_initializer())

    # Set up summary Ops
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args["summary_dir"], sess.graph)

    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args["buffer_size"]), int(args["random_seed"]))

    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    tflearn.is_training(True)

    for episode in range(int(args["max_episodes"])):

        start_time_episode = time.time()
        state = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for timestep in range(int(args["max_episode_len"])):

            start_time_timestep = time.time()

            if args["render_env"]:
                env.render()

            # Added exploration noise
            action = actor.predict(np.reshape(state, (1, actor.s_dim))) + actor_noise()
            action = np.clip(action, -actor.action_bound, actor.action_bound)
            np.set_printoptions(precision=3)
            print(action)
            # action = e.g. array([[ 0.18538651, -0.15227656, -0.53626154, -0.89609986]])
            # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
            target = np.insert(action, 3, [0.0, 0.0])
            target = scale_range(target, -1, 1, env.action_space.low, env.action_space.high)
            # e.g. target = array([ 0.19410025, -0.03806914,  0.01159346,  3.14159274,  3.14159274, 0.32641194])
            # we keep the roll & pitch angle fixed
            target[3] = 0.0
            target[4] = np.pi/2

            nextState, reward, terminal, info = env.step(target)

            replay_buffer.add(np.reshape(state, (actor.s_dim,)), np.reshape(action, (actor.a_dim,)), reward,
                              terminal, np.reshape(nextState, (actor.s_dim,)))

            # Keep adding experience to the memory until there are at least minibatch size samples
            if replay_buffer.size() > int(args["minibatch_size"]):
                s_batch, a_batch, r_batch, t_batch, s2_batch = replay_buffer.sample_batch(int(args["minibatch_size"]))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args["minibatch_size"])):
                    if t_batch[k]: # terminal state
                        y_i.append(r_batch[k])
                    else: # non-terminal state
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                print("fitting the critic & actor")
                start_time_fit = time.time()
                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args["minibatch_size"]), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()
                print("runtime fitting: {}s".format(time.time() - start_time_fit))
            # if

            state = nextState
            ep_reward += reward

            if terminal:

                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: ep_reward,
                    summary_vars[1]: ep_ave_max_q / float(timestep)
                })

                writer.add_summary(summary_str, episode)
                writer.flush()

                print("| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}".format(int(ep_reward), episode, (ep_ave_max_q / float(timestep))))
                break
            # if
            print("runtime timestep: {}s".format(time.time() - start_time_timestep))
        # for timesteps
        
        # Saving the trained model
        if(saver is not None):
            print("saving the trained model")
            start_time_save = time.time()
            saver.save(sess, savingModelPath + "ddpg_model_2", global_step=episode)
            print("runtime saving: {}s".format(time.time() - start_time_save))
        
        print("runtime episode: {}s".format(time.time() - start_time_episode))
    # for episodes
# train

def main(args):

    with tf.Session() as sess:

        env = gym.make(args["env"])
        np.random.seed(int(args["random_seed"]))
        tf.set_random_seed(int(args["random_seed"]))
        env.seed(int(args["random_seed"]))

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0] - 2 # 4 action-dimensions (we keep roll & pitch angle of the robot arm fixed)
        action_bound = 1 # env.action_space.high
        # Ensure action bound is symmetric
        # assert (env.action_space.high == -env.action_space.low).all()

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(args["actor_lr"]), float(args["tau"]),
                             int(args["minibatch_size"]), use_layer_norm=bool(args["use_layer_norm"]))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args["critic_lr"]), float(args["tau"]),
                               float(args["gamma"]),
                               actor.get_num_trainable_vars(), use_layer_norm=bool(args["use_layer_norm"]), critic_l2_reg=float(args["critic_l2_reg"]))
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim), sigma=1.5)

        # TODO: adaptive parameter noise of actor?
        # from baselines.ddpg.noise import *
        # param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))

        if args["use_gym_monitor"]:
            if not args["render_env"]:
                env = wrappers.Monitor(
                    env, args["monitor_dir"], video_callable=False, force=True)
            else:
                env = wrappers.Monitor(env, args["monitor_dir"], force=True)

        train(sess, env, args, actor, critic, actor_noise, bool(args["reload"]))

        if args["use_gym_monitor"]:
            env.monitor.close()
    # with sess
# main

if __name__ == "__main__":

    print("********************************************* Starting RL algorithm *********************************************")

    parser = argparse.ArgumentParser(description="provide arguments for DDPG agent")

    # agent parameters
    parser.add_argument("--actor-lr", help="actor network learning rate", default=0.0001)
    parser.add_argument("--critic-lr", help="critic network learning rate", default=0.001)
    parser.add_argument("--gamma", help="discount factor for critic updates", default=0.99)
    parser.add_argument("--tau", help="soft target update parameter", default=0.001)
    parser.add_argument("--buffer-size", help="max size of the replay buffer", default=1000000)
    parser.add_argument("--minibatch-size", help="size of minibatch for minibatch-SGD", default=64)
    
    parser.add_argument("--normalize-observations", default=True)
    parser.add_argument("--reload", default=False)
    parser.add_argument("--use-layer-norm", default=True)
    parser.add_argument("--critic-l2-reg", default=0.01)
    parser.add_argument("--noise-type", default="adaptive_param_0.2")

    # run parameters
    parser.add_argument("--env", help="choose the gym env- tested on {Pendulum-v0}", default="GazeboSmartBotPincherKinect-v0")
    parser.add_argument("--random-seed", help="random seed for repeatability", default=1234)
    parser.add_argument("--max-episodes", help="max num of episodes to do while training", default=50000)
    parser.add_argument("--max-episode-len", help="max length of 1 episode", default=1000)
    parser.add_argument("--render-env", help="render the gym env", action="store_true")
    parser.add_argument("--use-gym-monitor", help="record gym results", action="store_true")
    parser.add_argument("--monitor-dir", help="directory for storing gym results", default="./results/gym_ddpg")
    parser.add_argument("--summary-dir", help="directory for storing tensorboard info", default="./results/tf_ddpg")

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)
    
    args = vars(parser.parse_args())
    
    pp.pprint(args)

    main(args)
# if __main__