import gym

from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines import A2C, PPO2
from stable_baselines.common.vec_env import dummy_vec_env


env = gym.make('CartPole-v0')
env = dummy_vec_env.DummyVecEnv([lambda: env])


# print(env.action_space.sample())
# print(env.observation_space)

# model = PPO2.load("ppo2_carracing-v0")
# model.set_env(env)


model = A2C(MlpPolicy, env, verbose=2, tensorboard_log='./log/')

model.learn(total_timesteps=10000)

# model.save("ppo2_carracing_cont_prep_1e6")

# Enjoy trained agent
obs = env.reset()
ep_r = 0
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    ep_r += rewards
    env.render()
env.close()

