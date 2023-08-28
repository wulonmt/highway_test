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
from CustomPPO import CustomPPO

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
parser.add_argument("-s", "--save_log", help="whether save log or not", type=str, default = "True") #parser can't pass bool
parser.add_argument("-e", "--environment", help="which my- env been used", type=str, required = True)
parser.add_argument("-t", "--train", help="training or not", type=str, default = "True")
args = parser.parse_args()
ENV_LIST=["merge", "highway", "racetrack", "roundabout", "intersection",]

if __name__ == "__main__":
    assert args.environment in ENV_LIST, "Wrong my-ENV"
    GrayScale_env = gym.make(f"my-{args.environment}-v0", render_mode="rgb_array")
    
    if args.train == "True":
        n_cpu = 8
        batch_size = 64
        tensorboard_log=f"{args.environment}_ppo/" if args.save_log == "True" else None
        trained_env = GrayScale_env
        #trained_env = make_vec_env(GrayScale_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        #trained_env = make_vec_env(GrayScale_env, n_envs=n_cpu,)
        #env = gym.make("highway-fast-v0", render_mode="human")
        model = CustomPPO("CnnPolicy",
                    trained_env,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
                    n_steps=batch_size * 16 // n_cpu,
                    batch_size=batch_size,
                    n_epochs=10,
                    learning_rate=5e-4,
                    gamma=0.8,
                    verbose=1,
                    target_kl=0.2,
                    ent_coef=0.03,
                    vf_coef=0.8,
                    tensorboard_log=tensorboard_log)
        time_str = Ptime()
        time_str.set_time_now()
        log_name = time_str.get_time() + f"_{args.log_name}"
        # Train the agent
        model.learn(total_timesteps=int(5e5), tb_log_name=log_name)
        print("log name: ", tensorboard_log + log_name)
        # Save the agent
        if args.save_log:
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
