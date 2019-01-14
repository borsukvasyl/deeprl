from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from deeprl.environments.AtariWrapper import AtariWrapper
from deeprl.environments.StackFramesWrapper import StackFramesWrapper
from deeprl.environments.FireOnRestartWrapper import FireOnRestartWrapper


def make_wrapper(env, stack_frames=False):
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireOnRestartWrapper(env)
    env = AtariWrapper(env)
    if stack_frames:
        env = StackFramesWrapper(env)
    return env
