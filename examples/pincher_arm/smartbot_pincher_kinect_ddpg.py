#!/usr/bin/env python
# based on: https://github.com/openai/baselines/blob/master/baselines/ddpg/main.py
# algorithm code has also been adapted to work with my environment
import argparse
import time
import os
import logging
from baselines import logger, bench
from baselines.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)
import baselines.ddpg.training as training
from baselines.ddpg.models import Actor, Critic
from baselines.ddpg.memory import Memory
from baselines.ddpg.noise import *

import gym
import gym_gazebo
import tensorflow as tf
from mpi4py import MPI

import subprocess

def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    print("********************************************* Starting RL algorithm *********************************************")
    # Configure logger for the process with rank 0 (main-process?)
    # MPI = Message Passing Interface, for parallel computing; rank = process identifier within a group of processes
    rank = MPI.COMM_WORLD.Get_rank()
    if rank != 0:
        # Disable logging for rank != 0 to avoid noise.
        logger.set_level(logger.DISABLED)

    # Create envs.
    env = gym.make(env_id)
    env = bench.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(), str(rank)))

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
        eval_env = bench.Monitor(eval_env, os.path.join(logger.get_dir(), 'gym_eval'))
        env = bench.Monitor(env, None)
    else:
        eval_env = None

    # Parse noise_type
    action_noise = None
    param_noise = None
    nb_actions = env.action_space.shape[-1] - 2 # 4 action-dimensions (we keep roll & pitch angle of the robot arm fixed)
    for current_noise_type in noise_type.split(','):
        current_noise_type = current_noise_type.strip()
        if current_noise_type == 'none':
            pass
        elif 'adaptive-param' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            param_noise = AdaptiveParamNoiseSpec(initial_stddev=float(stddev), desired_action_stddev=float(stddev))
        elif 'normal' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = NormalActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        elif 'ou' in current_noise_type:
            _, stddev = current_noise_type.split('_')
            action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(stddev) * np.ones(nb_actions))
        else:
            raise RuntimeError('unknown noise type "{}"'.format(current_noise_type))

    # Configure components. (initialize memory, critic & actor objects)
    # print(env.action_space) # Box(6,)
    # print(env.observation_space) # Box(220, 300)
    # print(env.observation_space.shape) # (220, 300)
    # print(env.observation_space.shape[0]) # 220
    memory = Memory(limit=int(1e3), action_shape=(env.action_space.shape[0] -2,), observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    actor = Actor(nb_actions, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    tf.reset_default_graph()
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Train the RL algorithm
    if rank == 0:
        start_time = time.time()
    training.train(env=env, eval_env=eval_env, param_noise=param_noise,
        action_noise=action_noise, actor=actor, critic=critic, memory=memory, **kwargs)
    
    # Training is done
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))

    return True
# run


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # some rl parameters
    parser.add_argument('--env-id', type=str, help="envionment to use", default='GazeboSmartBotPincherKinect-v0')
    parser.add_argument('--gamma', type=float, help="discount factor for critic updates", default=0.99)
    parser.add_argument('--noise-type', type=str,
        help="noise type for exploration (choices are adaptive-param_xx, ou_xx, normal_xx, none)", default='adaptive-param_0.2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    boolean_flag(parser, 'restore', help="load a previously trained model", default=False)

    # training duration parameters
    parser.add_argument('--nb-epochs', help="number of epochs aka episodes", type=int, default=10)
    parser.add_argument('--nb-epoch-cycles', help="number of cycles in an epoch", type=int, default=20)
    parser.add_argument('--nb-rollout-steps', help="number of rollout steps per epoch cycle", type=int, default=50)  # per epoch cycle and MPI worker
    parser.add_argument('--num-timesteps', help="number of total timesteps (= nb_epochs * nb_epoch_cycles * nb_rollout_steps", type=int, default=None)
    
    # some neural network hyper-parameters
    boolean_flag(parser, 'layer-norm', help="use layer normalization", default=True)
    parser.add_argument('--critic-l2-reg', help="l2-regularization parameter for critic", type=float, default=1e-2)
    parser.add_argument('--batch-size', help="minibatch-size for model fitting", type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', help="actor network learning rate", type=float, default=1e-4)
    parser.add_argument('--critic-lr', help="critic network learning rate", type=float, default=1e-3)
    parser.add_argument('--nb-train-steps', help="number of model-fitting steps per epoch cycle", type=int, default=50)  # per epoch cycle and MPI worker

    # graphical parameters
    boolean_flag(parser, 'render', help="display simulation", default=False)

    # various parameters
    parser.add_argument('--clip-norm', type=float, default=None)
    boolean_flag(parser, 'popart', default=False)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--reward-scale', type=float, default=1.)

    # evaluation environment (not possible with a gym-gazebo environment)
    boolean_flag(parser, 'evaluation', default=False)
    boolean_flag(parser, 'render-eval', default=False)
    parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
    
    args = parser.parse_args()
    # we don't directly specify timesteps for this script, so make sure that if we do specify them
    # they agree with the other parameters
    if args.num_timesteps is not None:
        assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
    dict_args = vars(args)
    del dict_args['num_timesteps']
    return dict_args
# parse_args


if __name__ == '__main__':
    args = parse_args()
    # if MPI.COMM_WORLD.Get_rank() == 0:
    #     logger.configure()
    # Run actual script.
    algorithmDone = False
    # while(not algorithmDone):
    #     try:
    algorithmDome = run(**args)
        # except Exception as e:
        #     print("an exception occured during training")
        #     print(e)
        #     cmd = "killall -9 rosout roslaunch rosmaster gzserver nodelet robot_state_publisher gzclient" # aka killgazebogym
        #     result = subprocess.call(cmd, shell=True)
        #     print(result)
        #     time.sleep(10)
        #     break
    # while
# if __main___