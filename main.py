import gym
import sys
import torch
from ppo import PPO
from network import ActorCritic
from evaluation import eval_policy
import argparse
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',dest='mode',type=str,default='train')
    parser.add_argument('--ac_model',dest='ac_model',type=str,default='')
    parser.add_argument('--env', type=str, default='')
    parser.add_argument("--continuous",action="store_true",help="Continuous or not")

    args = parser.parse_args()

    return args
def train(env,hyperparameters,actor_critic_model):
    print("Training", flush = True)

    model = PPO(policy_class=ActorCritic,env = env, **hyperparameters)

    if actor_critic_model != '':
        print(f"Loading in {actor_critic_model}...", flush=True )
        model.network.load_state_dict(torch.load(actor_critic_model))
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn()

def test(env,actor_critic_model,continuous=False):

    print(f"Testing {actor_critic_model}", flush=True)
    if actor_critic_model == '':
        print(f"Didn't specify model file. Existing." , flush=True)
        sys.exit(0)


    if continuous:
        print('continuous environment!')
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        print('Discrete environment!')
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n


    policy = ActorCritic(obs_dim,act_dim,continuous=continuous)


    policy.load_state_dict(torch.load(actor_critic_model))
    print('successfully load model')
    eval_policy(policy=policy,env=env,render=True,continuous=continuous)
def main(args):
    hyperparameters = {
        'total_batch_size': 2048,
        'max_timesteps_per_episode': 200,
        'mini_batch_size':256,
        'n_updates_per_epoch': 10,
        'total_epochs':2000,
        'state_norm':False,
        'continuous':False,
        'gamma': 0.995,
        'lamda': 0.97,
        'lr': 5e-4,
        'clip': 0.2,
        'render': True,
    }
    env = gym.make(args.env)

    if args.mode =='train':
        train(env=env,hyperparameters=hyperparameters,actor_critic_model=args.ac_model)
    else:
        test(env=env,actor_critic_model=args.ac_model,continuous=args.continuous)

if __name__ == '__main__':
    args = get_args()
    main(args)