from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deeprl.trainers import BaseTrainer


class A2CTrainer(BaseTrainer):
    def __init__(self, config, agent, env):
        super(A2CTrainer, self).__init__(config, agent, env)

        self.discount_factor = config.discount_factor

    def update_model(self, batch):
        batch_s = np.array([i[0] for i in batch])
        batch_a = np.array([i[1] for i in batch])
        batch_r = np.array([i[2] for i in batch])
        batch_s1 = np.array([i[3] for i in batch])
        batch_done = np.array([i[4] for i in batch])

        batch_target_v = self.get_v_target(batch_r, batch_s1, batch_done)
        batch_advantages = self.get_advantages(batch_target_v, batch_s)
        loss = self.agent.model.train(batch_s, batch_a, batch_advantages, batch_target_v)
        return loss

    def get_v_target(self, batch_rewards, batch_next_states, batch_done):
        target_value = 0
        if not batch_done[-1]:
            target_value = self.agent.model.get_v([batch_next_states[-1]])[0]

        target_values = np.zeros_like(batch_rewards)
        for i in range(len(batch_rewards) - 1, -1, -1):
            target_value = batch_rewards[i] + self.discount_factor * target_value
            target_values[i] = target_value
        return target_values

    def get_advantages(self, batch_target_v, batch_s):
        batch_v = self.agent.model.get_v(batch_s)
        return np.array(batch_target_v) - np.array(batch_v)
