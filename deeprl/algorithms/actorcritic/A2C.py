from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deeprl.algorithms import BaseAlgorithm
from deeprl.policy import StochasticPolicy
from deeprl.utils import Experience


class A2C(BaseAlgorithm):
    def __init__(self, config, env, model):
        super(A2C, self).__init__(env, model)

        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size

        self.policy = StochasticPolicy()

    def choose_action(self, state):
        pi = self.model.get_one_policy(state)
        return self.policy.choose_action(pi)

    def run_episode(self):
        s = self.env.reset()
        done = False
        r_total = 0

        experience = Experience()
        loss = None

        while not done:
            a = self.choose_action(s)
            s1, r, done, _ = self.env.step(a)

            experience.add(s, a, r, s1, done)
            s = s1

            r_total += r
            if done or experience.size > self.batch_size:
                loss = self.update_model(experience)
                experience.clear()

        logs = {"loss": loss, "r_total": r_total}
        return logs

    def update_model(self, experience):
        batch_s = np.array([i[0] for i in experience])
        batch_a = np.array([i[1] for i in experience])
        batch_r = np.array([i[2] for i in experience])
        batch_s1 = np.array([i[3] for i in experience])
        batch_done = np.array([i[4] for i in experience])

        batch_target_v = self.get_v_target(batch_r, batch_s1, batch_done)
        batch_advantages = self.get_advantages(batch_target_v, batch_s)
        loss = self.model.train(batch_s, batch_a, batch_advantages, batch_target_v)
        return loss

    def get_v_target(self, batch_rewards, batch_next_states, batch_done):
        target_value = 0
        if not batch_done[-1]:
            target_value = self.model.get_v([batch_next_states[-1]])[0]

        target_values = np.zeros_like(batch_rewards)
        for i in range(len(batch_rewards) - 1, -1, -1):
            target_value = batch_rewards[i] + self.discount_factor * target_value
            target_values[i] = target_value
        return target_values

    def get_advantages(self, batch_target_v, batch_s):
        batch_v = self.model.get_v(batch_s)
        return np.array(batch_target_v) - np.array(batch_v)


