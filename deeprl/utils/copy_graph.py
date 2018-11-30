from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def copy_graph(from_model, to_model):
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_model.name)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_model.name)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder
