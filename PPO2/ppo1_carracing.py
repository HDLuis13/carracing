import gym
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO1

env = gym.make('CarRacing-v0')

PPO1(CnnPolicy, env=env)

model = PPO1(CnnPolicy, env=env, verbose=2, tensorboard_log='./t_log/')

for i in range(9):
    timesteps = 100000*(i+1)
    model.learn(total_timesteps=timesteps,  tb_log_name="ppo1_cont_prep_1{}".format(timesteps))
    model.save("ppo1_cont_prep_1{}".format(timesteps))



# # Enjoy trained agent
# obs = env.reset()
#
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     env.render()
# env.close()
