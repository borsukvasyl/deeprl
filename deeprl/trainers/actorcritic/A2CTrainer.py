from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deeprl.trainers import BaseTrainer


class A2CTrainer(BaseTrainer):
    def __init__(self, config, agent, env):
        super(A2CTrainer, self).__init__(config, agent, env)

        self.discount_factor = config.discount_factor

    def run_episode(self):
        s = self.env.reset()
        done = False
        r_total = 0
        loss = None

        while not done:
            a, v = self.choose_action(s)
            s1, r, done, _ = self.env.step(a)

            self.experience.add(s, a, r, s1, done, v)
            s = s1

            r_total += r
            batch = self.experience.sample(done)
            if batch is not None:
                loss = self.update_model(batch)

        logs = {"loss": loss, "r_total": r_total}
        return logs

    def update_model(self, batch):
        batch_s = np.array([i[0] for i in batch])
        batch_a = np.array([i[1] for i in batch])
        batch_r = np.array([i[2] for i in batch])
        batch_s1 = np.array([i[3] for i in batch])
        batch_done = np.array([i[4] for i in batch])
        batch_v = np.array([i[5] for i in batch])

        batch_target_v = self.get_v_target(batch_r, batch_s1, batch_done)
        batch_advantages = self.get_advantages(batch_target_v, batch_v)
        loss = self.agent.model.train(batch_s, batch_a, batch_advantages, batch_target_v)
        return loss

    def get_v_target(self, batch_rewards, batch_next_states, batch_done):
        target_value = 0
        if not batch_done[-1]:
            target_value = self.agent.model.get_v([batch_next_states[-1]])[0]

        target_values = np.zeros((batch_rewards.shape[0],))
        for i in range(batch_rewards.shape[0] - 1, -1, -1):
            target_value = batch_rewards[i] + self.discount_factor * target_value
            target_values[i] = target_value
        return target_values

    def get_advantages(self, batch_target_v, batch_v):
        return np.array(batch_target_v) - np.array(batch_v)

    def choose_action(self, state):
        pi, v = self.agent.model.get_policy_and_value([state])
        return self.agent.policy.choose_action(pi[0]), v[0][0]
