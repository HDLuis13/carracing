from gym.envs.box2d import CarRacing2

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

env = CarRacing2(
    preprocessed=False,
    continuous=False
)

env = DummyVecEnv([lambda: env])

PPO2(CnnPolicy, env=env)

model = PPO2(CnnPolicy, env=env, verbose=2, tensorboard_log='./t_log/')

for i in range(10):
    timesteps = 100000*(i+1)

    model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_cont_prep_{}".format(timesteps), reset_num_timesteps=False)
    model.save("ppo2_cont_prep_{}".format(timesteps))


#
# # Enjoy trained agent
#
# model = PPO2.load("ppo2_cont_prep_1000000", env=env)
#
# obs = env.reset()
#
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()
