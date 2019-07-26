from gym.envs.box2d.car_racing_3 import CarRacing3

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
import gym
import os

log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('CarRacing-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
# grayscale=True, vae_input=False, skip_frames=3, num_stack=4, continuous=True, render_on_learn=False
# env = CarRacing3(num_stack=1, skip_frames=1, grayscale=True)
#
env = DummyVecEnv([lambda: env])

# Enjoy trained agent

model = PPO2.load("./t_log/ppo2_compare_stable_500000", env=env, verbose=2)
# model.learn(total_timesteps=1000000, tb_log_name="testing")
print(model.observation_space)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
