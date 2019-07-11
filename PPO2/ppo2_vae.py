from gym.envs.box2d.car_racing_3 import CarRacing3
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv


# grayscale=True, vae_input=False, skip_frames=3, num_stack=4, continuous=True, render_on_learn=False
env = CarRacing3(vae_input=True, skip_frames=1)
env = DummyVecEnv([lambda: env])
print(env.reset().shape)
model = PPO2(MlpPolicy, env=env, verbose=2, tensorboard_log='./t_log/', n_steps=2048)
timesteps = 100000
for i in range(10):
    timesteps_total = timesteps*(i+1)
    model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_vae_v2_{}".format(timesteps_total), reset_num_timesteps=False)
    model.save("./t_log/ppo2_vae_v2_{}".format(timesteps_total))


#
# # Enjoy trained agent
# env = RacingGym(render=True)
# env = DummyVecEnv([lambda: env])
# model = PPO2.load("./t_log/ppo2_custom_350000", env=env, verbose=2, tensorboard_log='./t_log/')
# timesteps = 50000
# for i in range(3):
#     timesteps_total = timesteps*(i+1)+350000
#     model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_custom_{}".format(timesteps_total), reset_num_timesteps=False,)
#     model.save("./t_log/ppo2_custom_{}".format(timesteps_total))
#
# obs = env.reset()
#
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
# env.close()
