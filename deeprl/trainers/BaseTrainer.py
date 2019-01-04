from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod

from deeprl.callbacks import CallbackList, ConsoleLogger
from deeprl.trainers.experience import Experience


class BaseTrainer(object):
    __metaclass__ = ABCMeta

    def __init__(self, config, agent, env):
        self.config = config
        self.agent = agent
        self.env = env

        self.callbacks = CallbackList([
            ConsoleLogger()
        ])

        self.experience = Experience(config)

    def train(self, num_episodes, start_episode=0):
        self.callbacks.on_train_begin(start_episode=start_episode)

        for episode in range(start_episode, num_episodes):
            self.callbacks.on_episode_begin(episode)
            logs = self.run_episode()
            self.callbacks.on_episode_end(episode, logs)

        self.callbacks.on_train_end()

    def run_episode(self):
        s = self.env.reset()
        done = False
        r_total = 0
        loss = None

        while not done:
            a = self.choose_action(s)
            s1, r, done, _ = self.env.step(a)

            self.experience.add(s, a, r, s1, done)
            s = s1

            r_total += r
            batch = self.experience.sample(done)
            if batch is not None:
                loss = self.update_model(batch)

        logs = {"loss": loss, "r_total": r_total}
        return logs

    def choose_action(self, state):
        return self.agent.choose_action(state)

    @abstractmethod
    def update_model(self, batch):
        pass
