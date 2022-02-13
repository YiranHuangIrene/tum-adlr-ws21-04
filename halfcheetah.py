from pybulletgym.envs.mujoco.envs.locomotion.walker_base_env import WalkerBaseMuJoCoEnv
from pybulletgym.envs.mujoco.robots.locomotors.half_cheetah import HalfCheetah
import numpy as np


class HalfCheetahVelEnv(WalkerBaseMuJoCoEnv):
    def __init__(self, task={}, low=0.0, high=2.0):
        self.low = low
        self.high = high
        self.task = task
        self.goal_vel = self.task.get('velocity', 0.0)
        self.robot = HalfCheetah()
        WalkerBaseMuJoCoEnv.__init__(self, self.robot)

    def step(self, a):
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        potential = self.robot.calc_potential()
        potential = -1.0 * abs(potential - self.goal_vel)
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

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(self.low, self.high, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']
