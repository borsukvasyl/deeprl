from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np

from deeprl.algorithms import BaseAlgorithm
from deeprl.callbacks import EGreedyDecay
from deeprl.policy import EGreedyPolicy
from deeprl.utils import Experience


class BaseQLearning(BaseAlgorithm):
    __metaclass__ = ABCMeta

    def __init__(self, config, env, model):
        super(BaseQLearning, self).__init__(env, model)

        self.discount_factor = config.discount_factor
        self.batch_size = config.batch_size

        self.experience = Experience(config.experience_size)
        self.policy = EGreedyPolicy(config.e)
        self.callbacks.append(EGreedyDecay(self.policy, config.e_min, config.e_decay))

    def choose_action(self, state):
        q_value = self.model.get_one_q(state)
        return self.policy.choose_action(q_value)

    def run_episode(self):
        s = self.env.reset()
        done = False
        r_total = 0

        while not done:
            a = self.choose_action(s)
            s1, r, done, _ = self.env.step(a)

            self.experience.add(s, a, r, s1, done)
            s = s1

            r_total += r
            loss = self.update_model()

        logs = {"loss": loss, "r_total": r_total}
        return logs

    def update_model(self):
        if self.experience.size > self.batch_size:
            batch = np.array(self.experience.sample(self.batch_size))
            batch_s = np.array([i[0] for i in batch])
            batch_a = np.array([i[1] for i in batch])
            batch_r = np.array([i[2] for i in batch])
            batch_s1 = np.array([i[3] for i in batch])
            batch_done = np.array([i[4] for i in batch])

            q_target = self.get_q_target(batch_r, batch_s1, batch_done)
            loss = self.model.train(batch_s, batch_a, q_target)
            return loss

    @abstractmethod
    def get_q_target(self, batch_rewards, batch_next_states, batch_dones):
        pass
