import gymnasium as gym
from matplotlib import pyplot as plt
import MyEnv

#env = gym.make('custom-racetrack-v0', render_mode='rgb_array')
env = gym.make('test-racetrack-v0', render_mode='rgb_array')
env.reset()
"""
for _ in range(3):
    action = [0, 0]
    obs, reward, done, truncated, info = env.step(action)
    env.render()
"""
plt.imshow(env.render())
plt.show()

