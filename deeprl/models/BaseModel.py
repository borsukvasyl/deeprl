from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta, abstractmethod
import tensorflow as tf

from deeprl.utils import copy_graph


class BaseModel(object):
    __metaclass__ = ABCMeta

    def __init__(self, name, session, **kwargs):
        """
        Initializes BaseModel instance
        :param name: str, model name
        :param session: tensorflow.Session, session instance
        :param kwargs: args for build method
        """
        self.name = name
        self.session = session

        self.kwargs = kwargs

        with tf.variable_scope(self.name):
            self.build(**kwargs)

    @abstractmethod
    def build(self, **kwargs):
        """
        Builds model
        :param kwargs: args
        :return: None
        """
        pass

    def sync(self, source_model):
        """
        Synchronizes weights with source model
        :param source_model: BaseModel, model from which to copy weights
        :return: None
        """
        copy_op = copy_graph(source_model, self)
        self.session.run(copy_op)

    def copy(self, name):
        """
        Returns copy of this model with another name
        :param name: str, name of new model
        :return: BaseModel
        """
        return self.__class__(name, self.session, **self.kwargs)
