import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
from time import sleep
#env = gym.make('InvertedDoublePendulumMuJoCoEnv-v0')
env = gym.make('ReacherPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
state = env.reset()  # should return a state vector if everything worked
for _ in range(1000):
    act = env.action_space.sample()  # 在动作空间中随机采样
    obs, reward, done, _ = env.step(act)  # 与环境交互
    sleep(1/240)

env.close()