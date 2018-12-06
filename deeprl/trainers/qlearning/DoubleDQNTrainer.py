from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deeprl.trainers.qlearning import BaseDQNTrainer
from deeprl.callbacks import ModelSync


class DoubleDQNTrainer(BaseDQNTrainer):
    def __init__(self, config, agent, env):
        super(DoubleDQNTrainer, self).__init__(config, agent, env)

        self.target_model = self.agent.model.copy("target")
        self.callbacks.append(ModelSync(self.target_model, self.agent.model))

    def get_q_target(self, batch_rewards, batch_next_states, batch_dones):
        q_target_next_state = self.target_model.get_q(batch_next_states)
        q_next_state = self.agent.model.get_q(batch_next_states)

        q_target = np.zeros((len(batch_rewards),))
        for i in range(len(batch_rewards)):
            target = batch_rewards[i]
            if not batch_dones[i]:
                action = np.argmax(q_next_state[i])
                target += self.discount_factor * q_target_next_state[i][action]
            q_target[i] = target
        return q_target
