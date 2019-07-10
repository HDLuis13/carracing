from custom_carracing.car_racing_2 import CarRacing2


from stable_baselines.common.policies import CnnLnLstmPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

env = CarRacing2(
    preprocessed=True,
    continuous=True
)
env = DummyVecEnv([lambda: env])

# print(env.observation_space)

# model = A2C.load('a2c lunarlander 2', env=env, tensorboard_log="./log/")
# model.learn(total_timesteps=100000, tb_log_name='third run', reset_num_timesteps=False)
# model.save('a2c lunarlander 3')

model = A2C(CnnLnLstmPolicy, env, verbose=2, tensorboard_log='./log/')

model.learn(total_timesteps=250000, tb_log_name='cnn_first_run')

model.save("a2c carracing cnnlnlstm 1")

# model = A2C.load('a2c lunarlander 3')
# model.set_env(env)


# print(env.observation_space)


# Enjoy trained agent
# obs = env.reset()

# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()
#
