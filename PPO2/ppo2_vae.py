from CarRacing_custom_wrapper import RacingGym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

env = RacingGym(render=False, vae=True, skip_actions=1, )
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env=env, verbose=2, tensorboard_log='./t_log/', n_steps=1024)
timesteps = 50000
for i in range(10):
    timesteps_total = timesteps*(i+1)
    model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_vae{}".format(timesteps_total), reset_num_timesteps=False)
    model.save("./t_log/ppo2_vae_{}".format(timesteps_total))


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
