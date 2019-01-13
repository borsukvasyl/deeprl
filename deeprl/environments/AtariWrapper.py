from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
import cv2


class AtariWrapper(gym.ObservationWrapper):
    """
    For atari games with RGB observations
    """
    def observation(self, observation):
        grayscale = observation[:, :, 0] * 0.299 + observation[:, :, 1] * 0.587 + observation[:, :, 2] * 0.114
        cropped_frame = grayscale[8:-12, 4:-12]
        normalized_frame = cropped_frame / 255.0
        resized_frame = cv2.resize(normalized_frame, (84, 84))
        resized_frame = np.reshape(resized_frame, (84, 84, 1))
        return resized_frame
