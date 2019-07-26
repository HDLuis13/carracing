from gym.envs.box2d import CarRacing3

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.bench import Monitor
from stable_baselines.gail import generate_expert_traj
import os
import cv2
import csv
import gym

# grayscale=True, vae_input=False, skip_frames=3, num_stack=4, continuous=True, render_on_learn=False
log_dir = "/tmp/gym/"
os.makedirs(log_dir, exist_ok=True)

env = gym.make('CarRacing-v0')
env = Monitor(env, log_dir, allow_early_resets=True)
env = DummyVecEnv([lambda: env])

# model = PPO2(CnnPolicy, env=env, verbose=2, n_steps=2048, learning_rate=3e-4)
# timesteps = 100000
# for i in range(10):
#     timesteps_total = timesteps*(i+1)
#     model.learn(total_timesteps=timesteps,reset_num_timesteps=False)
#     model.save("./t_log/ppo2_sf3_ns4_steps2048_{}".format(timesteps_total))



# Enjoy trained agent

model = PPO2.load("./t_log/ppo2_compare_stable_500000.pkl", env=env, verbose=2)
# model.learn(total_timesteps=1000000, tb_log_name="testing")


obs = env.reset()
print(obs.shape)
value_list = [['timestep', 'value', 'action']]

for update in range(1000):
    action, _states, values = model.predict(obs)
    value_list.append([update, values[0], action])
    if 1000 > update > 0 == update % 50:
        image_path = os.path.join('./images', "{}.png".format(update))
        img = cv2.cvtColor(obs[0], cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, img)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()

with open('./images/values.csv', 'w') as csvFile:
    writer = csv.writer(csvFile)
    writer.writerows(value_list)
csvFile.close()
