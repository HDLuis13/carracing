# -*- coding: utf-8 -*-
# @Author: Shubham Chandel
# @Date:   2018-04-21 18:55:18
# @Last Modified by:   Shubham Chandel
# @Last Modified time: 2018-04-22 03:02:35

from gym.envs.box2d.car_racing import CarRacing
from PIL import Image
import scipy.misc
import numpy as np
from random import choice, random, randint
import cv2
import matplotlib.pyplot as plt
import skimage.io
from skimage.color import rgb2gray
env = CarRacing()

episodes = 50
steps = 200

def get_action():
	# return choice([[1, 0, 0], [-1, 0, 0], [0, 1, 0]])
	# return [2*random()-1.05, 1, 0]
	action = env.action_space.sample()
	action[1] = 1
	action[2] = 0
	return action

for eps in range(episodes):
	obs = env.reset()
	env.render()
	r = 0
	for t in range(steps):
		action = get_action()
		obs, reward, done, _ = env.step(action)
		# env.render()
		r += reward
		if ((t<50) or (t>150)) and (t%5 == 0):
			i = ('000' + str(t//5))[-3:]
			obs = obs[0:80, 6:86]
			obs = rgb2gray(obs)
			skimage.io.imsave('rollouts/CarRacing/car_{}_{}.png'.format(eps,i), obs)
	print("Episode [{}/{}]: CummReward {:.2f}".format(eps+1, episodes, r))


