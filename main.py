import argparse

import gym
import torch
from torch.utils.tensorboard import SummaryWriter

from evaluation import eval_policy
from halfcheetah import HalfCheetahVelEnv
from meta_ppo import MetaPPo
from network import ActorCritic
from ppo import PPO
from pendulum import MetaInvertedDoublePendulum
gym.logger.set_level(40)
import numpy as np


# TODO: change the distribution of test task -- training
# TODO: test on a simple env  --traiing
# TODO: add reptile
# TODO: read the paper again
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--save_name', dest='save_name', type=str, default='maml')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str, default='')
    # parser.add_argument('--ac_model', dest='ac_model', type=str, default='')
    # parser.add_argument('--env', type=str, default='')
    # parser.add_argument("--continuous", action="store_true", help="Continuous or not")

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


def main(args):
    np.random.seed(10)
    if args.mode == 'train':
        meta_iterations = 100
        meta_policy_lr = 3e-4
        num_tasks = 10
        env = MetaInvertedDoublePendulum()
        task_flag = True
        while task_flag == True:
            tasks = env.sample_tasks(5, 15, num_tasks)  # gravity: 5~15 generate 10 task
            task_flag = False
            for each in tasks:
                if abs(each['gravity'] - 9.8) < 1.0:
                    task_flag = True
        print(tasks)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        continuous = True if type(env.action_space) == gym.spaces.box.Box else False
        print(f"obs_dim: {obs_dim}, act_dim: {act_dim}, continuous? {continuous}")
        hyperparameters = {
            'N': 3,
            'K': 16,
            'max_timesteps_per_episode': 200,
            'mini_batch_size': 256,
            'state_norm': False,
            'continuous': continuous,
            'gamma': 0.995,
            'lamda': 0.97,
            'lr': 1e-2,  # inner loop lr
            'clip': 0.2,
            'render': True,
        }
        # TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        # writer = SummaryWriter('runs/T1000/S2R_N_more_epoch' + str(hyperparameters['N']) + '_task_' + str(num_tasks))
        writer = SummaryWriter('runs/meta_train/' +str(args.save_name) +'_as_' + str(hyperparameters['N']) + '_task_' + str(num_tasks))
        meta_policy = ActorCritic(obs_dim, act_dim, continuous=continuous, layer_norm=True)
        outer_optimizer = torch.optim.Adam(meta_policy.parameters(), lr=meta_policy_lr)
        print(f"meta policy training begins!")
        for meta_iteration in range(meta_iterations):
            print(f"meta_iteration: {meta_iteration + 1} / {meta_iterations}")
            total_meta_loss = 0
            total_reward = 0
            # meta_gradient_total = None
            for task_index in range(num_tasks):
                print(f"starting task {task_index + 1} / {num_tasks}")
                task = MetaInvertedDoublePendulum(task=tasks[task_index])
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
            torch.save(meta_policy.state_dict(),
                       'model/meta_model/' + str(hyperparameters['N']) + '_task_' + str(
                           num_tasks) + '.pth')

            # logging
            print('Metaiter {} \t 3-level reward: {}'.format(meta_iteration + 1, total_reward))
            writer.add_scalar('Meta_Reward', total_reward, meta_iteration)
    elif args.mode == 'ppo_train':
        mode = args.mode
        test_task = {'gravity': 9.8}
        env = MetaInvertedDoublePendulum(task=test_task)
        #env = gym.make('InvertedDoublePendulumMuJoCoEnv-v0')
        # tasks = env.sample_tasks(num_tasks)
        # tasks = [{'velocity': 0.4}, {'velocity': 0.8}, {'velocity': 1.2}, {'velocity': 1.6}, {'velocity': 2.0}]
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        continuous = True if type(env.action_space) == gym.spaces.box.Box else False
        hyperparameters = {
            'total_batch_size': 3200,
            'max_timesteps_per_episode': 200,
            'mini_batch_size': 256,
            'total_epochs': 2000,
            'state_norm': False,
            'continuous': continuous,
            'gamma': 0.995,
            'lamda': 0.97,
            'lr': 3e-4,
            'clip': 0.2,
            'render': True,
        }
        policy = ActorCritic(obs_dim, act_dim, continuous=continuous, layer_norm=True)
        if args.checkpoint != '':
            checkpoint = torch.load(args.checkpoint)
            policy.load_state_dict(checkpoint)
            print('successfully load model!')
            mode = 'meta_train'
        else:
            print('train from new model!')
        ppo = PPO(policy, env, mode, args.save_name, **hyperparameters)
        ppo.learn()
    elif args.mode == 'test':
        test_task = {'velocity': 1.0}
        env = HalfCheetahVelEnv(test_task)
        # tasks = env.sample_tasks(num_tasks)
        # tasks = [{'velocity': 0.4}, {'velocity': 0.8}, {'velocity': 1.2}, {'velocity': 1.6}, {'velocity': 2.0}]
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        continuous = True if type(env.action_space) == gym.spaces.box.Box else False
        policy = ActorCritic(obs_dim, act_dim, continuous=continuous, layer_norm=True)
        if args.checkpoint != '':
            checkpoint = torch.load(args.checkpoint)
            policy.load_state_dict(checkpoint)
            print('successfully load model!')
        else:
            print('no model can be loaded!')
            return 0
        # if continuous:
        #     print('continuous environment!')
        #     obs_dim = env.observation_space.shape[0]
        #     act_dim = env.action_space.shape[0]
        # else:
        #     print('Discrete environment!')
        #     obs_dim = env.observation_space.shape[0]
        #     act_dim = env.action_space.n

        eval_policy(policy=policy, env=env, render=True, continuous=continuous)


if __name__ == '__main__':
    args = get_args()
    main(args)
