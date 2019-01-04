from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import numpy as np

from deeprl.trainers import BaseTrainer
from deeprl.callbacks import EGreedyDecay
from deeprl.policy import EGreedyPolicy


class BaseDQNTrainer(BaseTrainer):
    __metaclass__ = ABCMeta

    def __init__(self, config, agent, env):
        super(BaseDQNTrainer, self).__init__(config, agent, env)

        self.discount_factor = config.discount_factor

        self.policy = EGreedyPolicy(config.e)
        self.callbacks.append(EGreedyDecay(self.policy, config.e_min, config.e_decay))

    def choose_action(self, state):
        q_value = self.agent.model.get_one_q(state)
        return self.policy.choose_action(q_value)

    def update_model(self, batch):
        batch_s = np.array([i[0] for i in batch])
        batch_a = np.array([i[1] for i in batch])
        batch_r = np.array([i[2] for i in batch])
        batch_s1 = np.array([i[3] for i in batch])
        batch_done = np.array([i[4] for i in batch])

        q_target = self.get_q_target(batch_r, batch_s1, batch_done)
        loss = self.agent.model.train(batch_s, batch_a, q_target)
        return loss

    @abstractmethod
    def get_q_target(self, batch_rewards, batch_next_states, batch_dones):
        pass
