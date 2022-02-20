from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.half_cheetah import HalfCheetah
import numpy as np
from time import sleep
import gym
gym.logger.set_level(40)
class HalfCheetahVelEnv(WalkerBaseMuJoCoEnv):
    def __init__(self,joints_coef = None,dynamics_coef = None):
        self.joints_coef = joints_coef
        self.dynamics_coef = dynamics_coef
        self.robot = HalfCheetah()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot,joints_coef=self.joints_coef,dynamics_coef=self.dynamics_coef)
    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        potential = self.robot.calc_potential()
        #potential = -1.0 * abs(potential - self.goal_vel)
        power_cost = -0.1 * np.square(a).sum()
        state = self.robot.calc_state()

        done = False

        debugmode = 0
        if debugmode:
            print("potential=")
            print(potential)
            print("power_cost=")
            print(power_cost)

        self.rewards = [
            potential,
            power_cost
        ]
        if debugmode:
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))
        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        return state, sum(self.rewards), bool(done), {}

    # def sample_tasks(self, num_tasks):
    #     velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,))
    #     tasks = [{'velocity': velocity} for velocity in velocities]
    #     return tasks
    #
    # def reset_task(self, task):
    #     self._task = task
    #     self._goal_vel = task['velocity']
if __name__ == '__main__':
    # self.jdict["bthigh"].power_coef = joints_coef['bthigh']  # 120 back thigh
    # self.jdict["bshin"].power_coef = joints_coef['bshin']  # 90 back shin
    # self.jdict["bfoot"].power_coef = joints_coef['bfoot']  # 60 back foot
    # self.jdict["fthigh"].power_coef = joints_coef['fthigh']  # 140 front thigh
    # self.jdict["fshin"].power_coef = joints_coef['fshin']  # 60  front shin
    # self.jdict["ffoot"].power_coef = joints_coef['ffoot']  # 30 front foot
    dynamics_coef = {'lateralFriction':0.8,'spinningFriction':0.1,'rollingFriction':0.1,'restitution':0.5}
    joints_coef = {'bthigh':120,'bshin':90,'bfoot':60,'fthigh':140,'fshin':60,"ffoot":30}
    env = HalfCheetahVelEnv(joints_coef = joints_coef,dynamics_coef = dynamics_coef)
    env.render('human')  # call this before env.reset, if you want a window showing the environment
    state = env.reset()  # should return a state vector if everything worked
    print("start simulation")
    for _ in range(1000):
        act = env.action_space.sample()  # 在动作空间中随机采样
        obs, reward, done, _ = env.step(act)  # 与环境交互
        env.camera_adjust()
        sleep(1 / 100)

    env.close()