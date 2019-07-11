import gym
from gym import spaces
import numpy as np
from vae import VAE
import torch
import torchvision
import cv2
from torchvision import transforms


class RacingGym:
    def __init__(self, env_name='CarRacing-v0', skip_actions=3, num_frames=8, w=80, h=80, render=True, vae=False):
        # env_name: the name of the Open AI Gym environment. By default, CarRacing-v0
        # skip_actions: the number of frames to repeat an action for
        # num_frames: the number of frames to stack in one state
        # w: width of the state input
        # h: height of the state input.
        self.env = gym.make(env_name)
        self.num_frames = num_frames
        self.skip_actions = skip_actions
        self.w = w
        self.h = h
        self.render = render
        self.vae = vae
        if self.vae:
            self.observation_space = spaces.Box(low=-1, high=1, shape=(32,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(w, h, num_frames), dtype=np.uint8)
        self.action_space = self.env.action_space
        self.metadata = self.env.metadata

        self.state = None
        self.env_name = env_name

        device = torch.device('cpu')
        self.vae_model = VAE(image_channels=1).to(device)
        vae.load_state_dict(torch.load('./PPO2/vae_gray.torch', map_location='cpu'))

    # Preprocess the input and stack the frames.
    def preprocess(self, obs, is_start=False):
        if self.vae:
            s = obs[0:80, 6:86]
            s = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY).astype(np.float32)
            s = torch.from_numpy(s).float()
            s = torch.unsqueeze(s, dim=0)
            compose = transforms.Compose([transforms.ToPILImage(), transforms.Resize((64, 64), interpolation=Image.NEAREST)])
            s = compose(s)
            z = self.vae_model.representation(s)

            self.state = z.detach().numpy()[0]
            self.observation_space = spaces.Box(shape=self.state.shape, dtype=np.float32)
            # s_g
            # if is_start or self.state is None:
            #     self.state = np.repeat(s, self.num_frames, axis=2)
            # else:
            #     self.state = np.append(s, self.state[:, :, :, :self.num_frames - 1], axis=3)
            return self.state
        else:
            grayscale = obs.astype('float32').mean(2)
            # crop bottom bar and 8 pixels left/right for quadratic image
            s_g = grayscale[0:80, 6:86]
            # Next reshape the image to a 4D array with 1st and 4th dimensions of
            # size 1
            s = s_g.reshape(1, s_g.shape[0], s_g.shape[1], 1)
            # Now stack the frames. If this is the first frame, then repeat it
            # num_frames times.
            if is_start or self.state is None:
                self.state = np.repeat(s, self.num_frames, axis=3)
            else:
                self.state = np.append(s, self.state[:, :, :, :self.num_frames-1], axis=3)

            self.observation_space = spaces.Box(low=0, high=255, shape=self.state.shape, dtype=np.uint8)


    # Render the current frame
    def render(self):
        self.env.render()

    # Reset the environment and return the state.
    def reset(self):
        return self.preprocess(self.env.reset(), is_start=True)

    # Step the environment with the given action
    def step(self, action):
        accum_reward = 0
        prev_s = None
        for _ in range(self.skip_actions):
            s, r, term, info = self.env.step(action)
            accum_reward += r
            if self.render:
                self.env.render()
            if term:
                break
            prev_s = s

        if prev_s is not None:
            s = np.maximum.reduce([s, prev_s])
        return self.preprocess(s), accum_reward, term, info