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
import MyEnv

# ==================================
#        Main script
# ==================================

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
args = parser.parse_args()

if __name__ == "__main__":
    train = True
    GrayScale_env = gym.make("my-merge-v0", render_mode = "rgb_array")
    
    if train:
        n_cpu = 8
        batch_size = 64
        tensorboard_log="merge_ppo/"
        trained_env = GrayScale_env
        #trained_env = make_vec_env(env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        #env = gym.make("highway-fast-v0", render_mode="human")
        model = PPO("CnnPolicy",
                    trained_env,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    n_steps=batch_size * 16 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    target_kl=0.1,
                    ent_coef=0.01,
                    vf_coef=0.8,
                    tensorboard_log=tensorboard_log)
        time_str = Ptime()
        time_str.set_time_now()
        log_name = time_str.get_time() + f"_{args.log_name}"
        # Train the agent
        model.learn(total_timesteps=int(5e5), tb_log_name=log_name)
        print("log name: ", tensorboard_log + log_name)
        # Save the agent
        model.save(tensorboard_log + "model")

    model = PPO.load(tensorboard_log + "model")
    env = GrayScale_env
    while True:
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
