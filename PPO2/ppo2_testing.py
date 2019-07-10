from gym.envs.box2d import CarRacing2

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env import dummy_vec_env

import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage import color

env = CarRacing2(
    preprocessed=False,
    continuous=True
)


# env = dummy_vec_env.DummyVecEnv([lambda: env])
#
# model = PPO2.load('PPO2_carracing_discr_prep_1mio')
# # model.set_env(env)
#
# # print(model.get_parameters())
# print(model.get_parameter_list())

# Enjoy trained agent
obs = env.reset()
print(obs.shape)

for i in range(200):
    action = env.action_space.sample()
    action[0] = 0.
    obs, rewards, dones, info = env.step(action)
    print(i)
    # if (i+1) % 50 == 0:
    #     obs = 2 * color.rgb2gray(obs) - 1.0
    #     plt.figure(figsize=(40, 40))
    #     # obs = [obs[:, :, 0], obs[:, :, 1:2], obs[:, :, 2:3]]
    #     print(obs.shape)
    #     plt.imshow(obs, cmap="gray")
    #     plt.show()
    env.render()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()

