from gym.envs.box2d import CarRacing3

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv

# grayscale=True, vae_input=False, skip_frames=3, num_stack=4, continuous=True, render_on_learn=False
env = CarRacing3(vae_input=False, grayscale=True, skip_frames=1, num_stack=4
)
env = DummyVecEnv([lambda: env])

# Enjoy trained agent

model = PPO2.load("./t_log/ppo2_sf3_ns4_steps2048_700000", env=env, verbose=2)
# model.learn(total_timesteps=1000000, tb_log_name="testing")


obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
env.close()
