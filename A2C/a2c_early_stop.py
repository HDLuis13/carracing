from custom_carracing.car_racing_2 import CarRacing2


from stable_baselines.common.policies import CnnPolicy
from stable_baselines import A2C
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

env = CarRacing2(
    preprocessed=True,
    continuous=True
)
n_cpu = 4
env = DummyVecEnv([lambda: env])


model = A2C(CnnPolicy, env, verbose=2, tensorboard_log='./test/', n_steps=50)

model.learn(total_timesteps=1000000, tb_log_name='a2c_early_stop_cnn')
model.save("a2c_early_stop_cnn")



# print(env.observation_space)

#
# # Enjoy trained agent
# obs = env.reset()
# env.render()


# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()
#
