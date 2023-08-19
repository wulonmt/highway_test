import gymnasium as gym
import torch as th
from stable_baselines3 import PPO
from torch.distributions import Categorical
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env

import argparse
from Ptime import Ptime

# ==================================
#        Main script
# ==================================

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
args = parser.parse_args()

if __name__ == "__main__":
    train = True
    if train:
        n_cpu = 8
        batch_size = 64
        env = gym.make("merge-v0")
        env.configure({"observation": {
                       "type": "GrayscaleObservation",
                       "observation_shape": (128, 64),
                       "stack_size": 4,
                       "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                       "scaling": 1.75,
                   }})
        env.reset()
        #trained_env = make_vec_env(env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        #env = gym.make("highway-fast-v0", render_mode="human")
        model = PPO("CnnPolicy",
                    env,
                    policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                    n_steps=batch_size * 16 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    target_kl=0.04,
                    ent_coef=0.05,
                    tensorboard_log="highway_ppo/")
        time_str = Ptime()
        time_str.set_time_now()
        log_name = time_str.get_time() + f"_{args.log_name}"
        # Train the agent
        model.learn(total_timesteps=int(2e4), tb_log_name=log_name)
        # Save the agent
        model.save("highway_ppo/model")

    model = PPO.load("highway_ppo/model")
    env.render_mode = "human"
    while True:
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
