import gym

from stable_baselines.common.policies import CnnPolicy, MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv


env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env=env, verbose=2, tensorboard_log='./t_log/')
#
# for i in range(10):
#     timesteps = 10000*(i+1)
#
#     model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_diff_{}".format(timesteps), reset_num_timesteps=False)
#     model.save("ppo2_diff_{}".format(timesteps))

# Enjoy trained agent

model.learn(total_timesteps=100000, tb_log_name='test', reset_num_timesteps=False)

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
