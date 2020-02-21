#!/usr/bin/env python

import time

import gym
import gym_pacman

# Set up environment
env = gym.make('PacMan-v1')
env.reset()

for x in range(200):
    # Render the scene
    env.render()
    observation, reward, is_over, info = env.step(env.action_space.sample())
    time.sleep(0.1)
    if is_over:
        break

# Clean up and exit
time.sleep(1)
env.close()
