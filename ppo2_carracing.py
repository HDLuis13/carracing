import gym

import numpy as np
from gym.spaces import Box, Discrete

import cv2
# from universe.wrappers import Vectorize, Unvectorize
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import dummy_vec_env, subproc_vec_env



def create_car_racing_env():
    env = gym.make('CarRacing-v0')
    # env = Vectorize(env)
    # env = CarRacingRescale32x32(env)
    #env = NormalizedEnv(env)
    #env = CarRacingDiscreteActions(env)
    env = PreProcessObservation(env)
    # env = Unvectorize(env)
    return env


class CarRacingDiscreteActions(gym.ActionWrapper):

    def __init__(self, env=None):
        super(CarRacingDiscreteActions, self).__init__(env)
        self.action_space = Discrete(5)
        # 0 left
        # 1 right
        # 2 forward
        # 3 brake
        # 4 noop

    def _make_continuous_action(self, a):
        print ("a = ", a)
        act = np.array([0., 0., 0.])
        if a == 0: # left
            act = np.array([-1., 0., 0.])
        elif a == 1: # right
            act = np.array([1., 0., 0.])
        elif a == 2: # gas
            act = np.array([0., 1., 0.])
        elif a == 3: # brake
            act = np.array([0., 0., 1.])
        elif a == 4: # noop
            act = np.array([0., 0., 0.])
        # print ("act: ", act)
        return act

    def _action(self, action_n):
        #print(action_n)
        # return action_n
        return [self._make_continuous_action(a) for a in action_n]

class CarRacingRescale32x32(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(CarRacingRescale32x32, self).__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 32, 32])

    def _process_frame32(self, frame):
        frame = cv2.resize(frame, (32, 32))
        frame = frame.mean(2)
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)
        frame = np.reshape(frame, [1, 32, 32])
        return frame

    def _observation(self, observation_n):
        return [self._process_frame32(obs) for obs in observation_n]


class NormalizedEnv(gym.ObservationWrapper):

    def __init__(self, env=None):
        super(NormalizedEnv, self).__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def _observation(self, observation_n):
        for observation in observation_n:
            self.num_steps += 1
            self.state_mean = self.state_mean * self.alpha + \
                observation.mean() * (1 - self.alpha)
            self.state_std = self.state_std * self.alpha + \
                observation.std() * (1 - self.alpha)

        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return [(observation - unbiased_mean) / (unbiased_std + 1e-8) for observation in observation_n]


# Returns a cropped and down sampled image where the background is erased
class PreProcessObservation(gym.ObservationWrapper):

    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        # Define a new Box
        self.observation_space = Box(self.observation_space.low[0, 0, 0], self.observation_space.high[0, 0, 0],
                                     [40, 48, 1]  # Channel, Width, Height
                                     )

    def observation(self, observation):
        I = observation[0:80]  # crop
        I = 0.2989 * I[:, :, 0] + 0.5879 * I[:, :, 1] + 0.1140 * I[:, :, 2]  # Grey Image
        I = I[::2, ::2]  # down sample by factor of 2

        return I.astype(np.float32)[..., np.newaxis]


# multiprocess environment
# n_cpu = 1

env = create_car_racing_env()
env = dummy_vec_env.DummyVecEnv([lambda: env])

print(env.action_space)

# model = PPO2.load("ppo2_carracing-v3-prep")
# model.set_env(env)


model = PPO2(CnnPolicy, env, verbose=2, full_tensorboard_log=True)

model.learn(total_timesteps=1000)
#
# model.save("ppo2_carracing-v4-prep")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

