#!/usr/bin/env python3
from baselines import logger
from baselines.acer.acer_simple import learn
from baselines.acer.policies import AcerCnnPolicy, AcerLstmPolicy
from baselines.common.cmd_util import make_atari_env, atari_arg_parser

import gym
import gym_gazebo

def train(num_timesteps, seed, policy, lrschedule):
    env = gym.make('GazeboSmartBotPincherKinect-v0')
    if policy == 'cnn':
        policy_fn = AcerCnnPolicy
    elif policy == 'lstm':
        policy_fn = AcerLstmPolicy
    else:
        print("Policy {} not implemented".format(policy))
        return
    learn(policy_fn, env, seed, total_timesteps=int(num_timesteps * 1.1), lrschedule=lrschedule)
    env.close()

def main():
    # parser = atari_arg_parser()
    # parser.add_argument('--policy', help='Policy architecture', choices=['cnn', 'lstm', 'lnlstm'], default='cnn')
    # parser.add_argument('--lrschedule', help='Learning rate schedule', choices=['constant', 'linear'], default='constant')
    # parser.add_argument('--logdir', help ='Directory for logging')
    # args = parser.parse_args()
    # logger.configure(args.logdir)

    train(num_timesteps=500, seed=123, policy="cnn", lrschedule="constant")

if __name__ == '__main__':
    main()