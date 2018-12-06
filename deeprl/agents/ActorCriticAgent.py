from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.agents import BaseAgent
from deeprl.policy import StochasticPolicy


class ActorCriticAgent(BaseAgent):
    def __init__(self, model):
        super(ActorCriticAgent, self).__init__(model, StochasticPolicy())

    def choose_action(self, state):
        pi = self.model.get_one_policy(state)
        return self.policy.choose_action(pi)
