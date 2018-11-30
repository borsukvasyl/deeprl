from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
from tensorflow.python.ops.variable_scope import variable_scope

from deeprl.utils import copy_graph


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, session, **kwargs):
        self.name = name
        self.session = session

        self.kwargs = kwargs

        with variable_scope(self.name):
            self.build(**kwargs)

    @abstractmethod
    def build(self, **kwargs):
        pass

    def sync(self, source_model):
        copy_op = copy_graph(source_model, self)
        self.session.run(copy_op)

    def copy(self, name):
        return self.__class__(name, self.session, **self.kwargs)
