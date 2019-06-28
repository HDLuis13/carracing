import numpy as np
import gym
import matplotlib.pyplot as plt


env_old = gym.make('CarRacing-v0')
env = gym.make('CarRacing-v1')

print(env.observation_space)

env.reset()
for i in range(1000):
    obs, rewards, dones, info = env.step(env.action_space.sample())
    print(obs.shape)
    if (i%100==0):
        plt.figure(figsize=(40,40))
        plt.imshow(obs[:, :, 0], cmap='gray')
        plt.show()
    env.render()
env.close()
