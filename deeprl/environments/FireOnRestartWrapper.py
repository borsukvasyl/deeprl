from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym


class FireOnRestartWrapper(gym.Wrapper):
    FIRE_ACTION = 1

    def __init__(self, env=None):
        super(FireOnRestartWrapper, self).__init__(env)
        if env.unwrapped.get_action_meanings()[self.FIRE_ACTION] != 'FIRE':
            raise ValueError("No 'FIRE' action in environment")

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(self.FIRE_ACTION)
        if done:
            self.env.reset()
        return obs
