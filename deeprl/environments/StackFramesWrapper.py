from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from collections import deque
import numpy as np
import gym
from gym import spaces


class StackFramesWrapper(gym.Wrapper):
    def __init__(self, env, maxlen=4):
        super(StackFramesWrapper, self).__init__(env)

        self.maxlen = maxlen
        self.states = deque([], maxlen=maxlen)
        shape = env.observation_space.shape
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(shape[0], shape[1], shape[2] * maxlen))

    def reset(self):
        observation = self.env.reset()
        for _ in range(self.maxlen):
            self.states.append(observation)
        return self._get_observation()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.states.append(observation)
        return self._get_observation(), reward, done, info

    def _get_observation(self):
        return np.dstack(self.states)
