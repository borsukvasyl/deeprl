from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod


class Callback():
    __metaclass__ = ABCMeta

    @abstractmethod
    def on_episode(self):
        pass
