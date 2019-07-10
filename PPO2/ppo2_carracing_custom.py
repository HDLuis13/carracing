from CarRacing_custom_wrapper import RacingGym
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

env = RacingGym(render=False, skip_actions=1, num_frames=4, vae=False)
env = DummyVecEnv([lambda: env])

model = PPO2(CnnPolicy, env=env, verbose=2, tensorboard_log='./t_log/', n_steps=1024)
timesteps = 100000
for i in range(5):
    timesteps_total = timesteps*(i+1)
    model.learn(total_timesteps=timesteps,  tb_log_name="ppo2_custom_noskip_4frames_{}_steps2048".format(timesteps_total), reset_num_timesteps=False)
    model.save("./t_log/ppo2_custom_noskip_4frames_{}".format(timesteps_total))



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
