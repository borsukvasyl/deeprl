from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deeprl.trainers.qlearning import BaseDQNTrainer


class DQNTrainer(BaseDQNTrainer):
    def __init__(self, config, agent, env):
        super(DQNTrainer, self).__init__(config, agent, env)

    def get_q_target(self, batch_rewards, batch_next_states, batch_dones):
        q_next_state = self.agent.model.get_q(batch_next_states)

        q_target = np.zeros((len(batch_rewards),))
        for i in range(len(batch_rewards)):
            target = batch_rewards[i]
            if not batch_dones[i]:
                target += self.discount_factor * np.max(q_next_state[i])
            q_target[i] = target
        return q_target
