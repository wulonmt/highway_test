import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env

from Ptime import Ptime

parser = argparse.ArgumentParser()
parser.add_argument("-l", "--log_name", help="modified log name", type=str, nargs='?')
args = parser.parse_args()

TRAIN = True

if __name__ == '__main__':
    n_cpu = 8
    batch_size = 64
    env = gym.make("racetrack-v0")
        env.configure({"observation": {
                       "type": "GrayscaleObservation",
                       "observation_shape": (128, 64),
                       "stack_size": 4,
                       "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                       "scaling": 1.75,
                   }})
        env.reset()
    #env = make_vec_env("racetrack-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
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
                    tensorboard_log="racetrack_ppo/")
    # Train the model
    if TRAIN:
        time_str = Ptime()
        time_str.set_time_now()
        log_name = time_str.get_time() + f"_{args.log_name}"
        # Train the agent
        model.learn(total_timesteps=int(2e4), tb_log_name=log_name)
        model.save("racetrack_ppo/model")
        del model

    # Run the algorithm
    model = PPO.load("racetrack_ppo/model", env=env)

    env.render_mode="human"
    #env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    #env.unwrapped.set_record_video_wrapper(env)

    while True:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _ = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()
    env.close()
