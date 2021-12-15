import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class CircleDrive(gym.Env):
    def __init__(self, render: bool = False):
        self._render = render
        self.action_space = spaces.Box(
            low=np.array([-100., -100.]),
            high=np.array([100., 100.]),
        )

        self.observation_space = spaces.Box(
            low = np.array([0.,0.,0.]),
            high = np.array([10.,10.,np.pi]),
        )

        # connect engine
        self.client_id = p.connect(p.GUI if self._render else p.DIRECT)

        # radius of circle
        self.target_position = np.array([4,4],dtype=np.float32)

        # counter
        self.step_num = 0

    def __apply_action(self, action):
        left_v, right_v = action
        left_v = np.clip(left_v, -100., 100.)
        right_v = np.clip(right_v, -100., 100.)
        p.setJointMotorControlArray(
            bodyUniqueId=self.robot,
            jointIndices=[3, 2],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[left_v, right_v],
            forces=[50., 50.],
            physicsClientId=self.client_id
        )

    def render(self, mode="human"):
        pass

    def __get_observatoin(self):

        state = np.zeros(3,dtype=np.float32)
        basePos,baseOri= p.getBasePositionAndOrientation(self.robot, physicsClientId=self.client_id)
        matrix = p.getMatrixFromQuaternion(baseOri, physicsClientId=self.client_id)
        direction_vecotr = np.array([matrix[0], matrix[3], matrix[6]])
        position_vector = np.array(basePos)
        d_L2 = np.linalg.norm(direction_vecotr)
        p_L2 = np.linalg.norm(position_vector)
        if d_L2 == 0 or p_L2 == 0:
            angle = np.pi
        else:
            angle = np.arccos(np.dot(direction_vecotr, position_vector) / (d_L2 * p_L2))
        state[0:2] = basePos[:2]
        state[2] = angle
        return state

    def reset(self):
        p.resetSimulation(physicsClientId=self.client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.8)
        self.robot = p.loadURDF('./differential.urdf', basePosition=[0., 0., 0.2], physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane = p.loadURDF('plane.urdf', physicsClientId=self.client_id)
        return self.__get_observatoin()

    def step(self, action):
        self.__apply_action(action)
        p.stepSimulation(physicsClientId=self.client_id)
        self.step_num += 1
        state = self.__get_observatoin()
        reward_position = -np.linalg.norm(state[:2]-self.target_position) + 2
        reward_angle = -0.5*np.abs(state[2] - np.pi*45/180)
        reward = reward_angle + reward_position
        if reward > 1.6:
            reward = 50000000
            done = True
        else:
            done = False
        info = {}
        time.sleep(1 / 240)

        return state, reward, done, info

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np.random(seed)
        return [seed]

    def close(self):
        if self.client_id > 0:
            p.disconnect()
        self.client_id = -1

if __name__ == '__main__':
    env = CircleDrive(render=True)
    for j in range(10):
        obs = env.reset()
        for i in range(5000):
            action = np.random.uniform(-100,100,size=(2,))
            obs, reward, done, _ = env.step(action)
            if done:
                break
            print('state:{}, reward:{}'.format(obs,reward))
    env.close()
    # import gym
    # trajectorys = np.load('trajectory.npy')
    # env = gym.make('BipedalWalker-v3')
    # for j in range(5000):
    #     env.reset()
    #     trajectory = trajectorys[j]
    #     step = 0
    #     for _ in range(520):
    #         env.render()
    #         action = trajectory[step:step+4]
    #         step = step+28
    #         env.step(action)
    #         # obs,reward,done,_ = env.step(env.action_space.sample())
    #         time.sleep(1/240)
            # if done == True:
            #     break
        #env.close()