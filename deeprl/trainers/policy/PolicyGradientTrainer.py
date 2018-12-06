from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from deeprl.trainers import BaseTrainer


class PolicyGradientTrainer(BaseTrainer):
    def __init__(self, config, agent, env):
        super(PolicyGradientTrainer, self).__init__(agent, env)

        self.discount_factor = config.discount_factor

    def choose_action(self, state):
        return self.agent.choose_action(state)

    def run_episode(self):
        experience = []

        s = self.env.reset()
        done = False
        r_total = 0
        loss = None

        while not done:
            a = self.choose_action(s)
            s1, r, done, _ = self.env.step(a)

            experience.append((s, a, r, s1, done))
            s = s1

            r_total += r
            loss = self.update_model(experience)

        logs = {"loss": loss, "r_total": r_total}
        return logs

    def update_model(self, experience):
        batch = experience
        batch_s = np.array([i[0] for i in batch])
        batch_a = np.array([i[1] for i in batch])
        batch_r = np.array([i[2] for i in batch])

        discounted_episode_rewards = self.discount_episode_rewards(batch_r)
        loss = self.agent.model.train(batch_s, batch_a, discounted_episode_rewards)
        return loss

    def discount_episode_rewards(self, episode_rewards):
        discounted_episode_rewards = np.zeros_like(episode_rewards)
        cumulative = 0
        for i in range(len(episode_rewards) - 1, -1, -1):
            cumulative = cumulative * self.discount_factor + episode_rewards[i]
            discounted_episode_rewards[i] = cumulative
        return discounted_episode_rewards
