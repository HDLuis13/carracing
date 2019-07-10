from car_racing_2 import CarRacing2
import gym
import matplotlib.pyplot as plt
import numpy as np
env = CarRacing2()
# env = gym.make('CarRacing-v0')
obs = env.reset()
# print(env.observation_space)
# print(obs.shape)
# print(obs)

for i in range(2000):
    action = env.action_space.sample()
    action[0] = -.1
    obs, rewards, dones, info = env.step(action)
    # obs = np.transpose(obs, (3, 1, 2, 0))
    # for info in info:
    # if i%100==0:
    #     env.reset()
    # if len(info)>0:
    #     print(info)
    # if i%1000==0:
    #     print(i)
    # env.render()
    # if (i+1) % 60 == 0:
    #     plt.figure(figsize=(80, 80))
    #     # print(obs.shape)
    #     # print(obs[0].shape)
    #     plt.imshow(obs[:,:], cmap="gray")
    #     plt.show()
    #     # plt.figure(figsize=(80, 80))
    #     # plt.imshow(obs[7][:,:,0], cmap="gray")
    #     # plt.show()
