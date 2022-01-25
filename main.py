import gym
import sys
import torch
from meta_ppo import MetaPPo, gradadd
from ppo import PPO
from network import ActorCritic
from evaluation import eval_policy
import argparse
from gym_env import HalfCheetahVelEnv
import tensorboard
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

gym.logger.set_level(40)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')
    #parser.add_argument('--ac_model', dest='ac_model', type=str, default='')
    #parser.add_argument('--env', type=str, default='')
    #parser.add_argument("--continuous", action="store_true", help="Continuous or not")

    args = parser.parse_args()

    return args


def train(env, hyperparameters, actor_critic_model):
    print("Training", flush=True)

    model = PPO(policy_class=ActorCritic, env=env, **hyperparameters)

    if actor_critic_model != '':
        print(f"Loading in {actor_critic_model}...", flush=True)
        model.meta_policy.load_state_dict(torch.load(actor_critic_model))
        print(f"Successfully loaded.", flush=True)
    else:
        print(f"Training from scratch.", flush=True)

    model.learn()


def test(env, actor_critic_model, continuous=False):
    print(f"Testing {actor_critic_model}", flush=True)
    if actor_critic_model == '':
        print(f"Didn't specify model file. Existing.", flush=True)
        sys.exit(0)

    if continuous:
        print('continuous environment!')
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
    else:
        print('Discrete environment!')
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

    policy = ActorCritic(obs_dim, act_dim, continuous=continuous)

    policy.load_state_dict(torch.load(actor_critic_model))
    print('successfully load model')
    eval_policy(policy=policy, env=env, render=True, continuous=continuous)


def main(args):
    if args.mode == 'Train':
        meta_iterations = 80
        meta_policy_lr = 3e-4
        num_tasks = 5
        env = HalfCheetahVelEnv()
        # tasks = env.sample_tasks(num_tasks)
        tasks = [{'velocity': 0.4}, {'velocity': 0.8}, {'velocity': 1.2}, {'velocity': 1.6}, {'velocity': 2.0}]
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        continuous = True if type(env.action_space) == gym.spaces.box.Box else False
        hyperparameters = {
            'N': 3,
            'K': 16,
            'max_timesteps_per_episode': 200,
            'mini_batch_size': 256,
            'state_norm': False,
            'continuous': continuous,
            'gamma': 0.995,
            'lamda': 0.97,
            'lr': 1e-3,  #inner loop lr
            'clip': 0.2,
            'render': True,
        }
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        writer = SummaryWriter('runs/S2R_N_'+str(hyperparameters['N'])+'_' + TIMESTAMP)
        meta_policy = ActorCritic(obs_dim, act_dim, continuous=continuous, layer_norm=True)
        outer_optimizer = torch.optim.Adam(meta_policy.parameters(),lr=meta_policy_lr)
        print(f"meta policy training begins!")
        for meta_iteration in range(meta_iterations):
            print(f"meta_iteration: {meta_iteration + 1} / {meta_iterations}")
            total_meta_loss = 0
            total_reward = 0
            meta_gradient_total = None
            for task_index in range(num_tasks):
                print(f"starting task {task_index + 1} / {num_tasks}")
                task = HalfCheetahVelEnv(task=tasks[task_index])
                metappo = MetaPPo(meta_policy, task, **hyperparameters)
                # post_updated_reward, grad_wrt_theta = metappo.maml_learn()
                # post_updated_reward, grad_wrt_theta = metappo.sim2real_learn()
                task_val_reward, task_loss = metappo.meta_learn()
                # if meta_gradient_total is None:
                #     meta_gradient_total = grad_wrt_theta
                # else:
                #     meta_gradient_total = gradadd(meta_gradient_total, grad_wrt_theta)
                # total_reward += post_updated_reward
                total_reward += task_val_reward
                total_meta_loss += task_loss
            total_meta_loss = total_meta_loss / num_tasks
            total_reward = total_reward / num_tasks
            print('Updating meta-policy')
            outer_optimizer.zero_grad()
            # total_meta_loss = total_meta_loss / num_tasks
            # writer.add_scalar('Meta_Loss', total_meta_loss.item(), meta_iteration)
            total_meta_loss.backward()
            outer_optimizer.step()
            # with torch.no_grad():
            #     for i, p in enumerate(meta_policy.parameters()):
            #         p.copy_(p - meta_policy_lr * meta_gradient_total[i])
            # save model
            torch.save(meta_policy.state_dict(), 'model/higher_meta_PPO_'+str(hyperparameters['N'])+'.pth')

            # logging
            print('Metaiter {} \t 3-level reward: {}'.format(meta_iteration+1, total_reward))
            writer.add_scalar('Meta_Reward', total_reward, meta_iteration)
    elif args.mode == 'test':
        test_task  = {'velocity':1.0}
        env = HalfCheetahVelEnv()
        # tasks = env.sample_tasks(num_tasks)
        tasks = [{'velocity': 0.4}, {'velocity': 0.8}, {'velocity': 1.2}, {'velocity': 1.6}, {'velocity': 2.0}]
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        continuous = True if type(env.action_space) == gym.spaces.box.Box else False
        hyperparameters = {
            'total_batch_size': 2048,
            'max_timesteps_per_episode': 200,
            'mini_batch_size': 256,
            'n_updates_per_epoch': 10,
            'total_epochs': 2000,
            'state_norm': False,
            'continuous': continuous,
            'gamma': 0.995,
            'lamda': 0.97,
            'lr': 5e-4,
            'clip': 0.2,
            'render': True,
        }
        policy = ActorCritic(obs_dim, act_dim, continuous=continuous, layer_norm=True)
        checkpoint = torch.load('model/higher_meta_PPO.pth')
        policy.load_state_dict(checkpoint)
        ppo = PPO(policy,env,**hyperparameters)
        ppo.learn()
    else:
        print(f"no mode found!")


if __name__ == '__main__':
    args = get_args()
    main(args)
