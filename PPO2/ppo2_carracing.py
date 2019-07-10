from gym.envs.box2d import CarRacing2

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

env = CarRacing2(
    preprocessed=False,
    continuous=True,
)
env = DummyVecEnv([lambda: env])

# model = PPO2(CnnPolicy, env=env, verbose=2, tensorboard_log='./t_log/', n_steps=2048, learning_rate=3e-4)
# timesteps = 100000
# for i in range(5):
#     timesteps_total = timesteps*(i+1)
#     model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_compare_stable_{}".format(timesteps_total), reset_num_timesteps=False)
#     model.save("./t_log/ppo2_compare_stable_{}".format(timesteps_total))
#


# Enjoy trained agent

model = PPO2.load("./t_log/ppo2_compare_stable_300000.pkl", env=env, verbose=2)
# model.learn(total_timesteps=1000000, tb_log_name="testing")


obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
