#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2021 Apple Inc. All Rights Reserved.
#

""" Wraps Distracting Control Suite in a Gym-like wrapper."""
import numpy as np

import distractor_dmc2gym as dmc  # noqa: E402
from gym.wrappers import StepAPICompatibility

from collections import deque
from PIL import Image
import argparse
import yaml
import matplotlib.pyplot as plt
plt.ion()


class EnvironmentContainerDMC(object):
    """
    Wrapper around DCS.
    """
    def __init__(self, config, train=True, seed=None):
        self.domain = config['domain']

        self.get_other_obs = False

        # The standard task and action_repeat for each domain.
        task_info = {
            'ball_in_cup': ('catch', 4),
            'cartpole': ('swingup', 8),
            'cheetah': ('run', 4),
            'finger': ('spin', 2),
            'reacher': ('easy', 4),
            'walker': ('walk', 2),
        }
        self.task, self.action_repeat = task_info[self.domain]

        self.difficulty = config['difficulty']
        if self.difficulty in ['none', 'None']:
            self.difficulty = None

        self.dynamic = config['dynamic']
        self.background_dataset_path = config.get('background_dataset_path', './background-datasets')
        occlusion_location = config.get('occlusion_location', 'background')
        distractor = config.get('distractor', 'dots')

        self.env = dmc.make(
            domain_name=self.domain,
            task_name=self.task,
            frame_skip=self.action_repeat,
            from_pixels=True,
            height=config['image_height'],
            width=config['image_width'],
            distraction_source=distractor,
            distraction_location=occlusion_location,
            visualize_reward=False,
            channels_first=True,
            train_or_val='train' if train else 'val',
            background_dataset_path=self.background_dataset_path,
        )
        self.env = StepAPICompatibility(self.env, output_truncation_bool=False)
        action_spec = self.env.action_spec()
        self.action_dims = len(action_spec.minimum)
        self.action_low = action_spec.minimum
        self.action_high = action_spec.maximum
        self.num_frames_to_stack = config.get('num_frames_to_stack', 1)
        if self.num_frames_to_stack > 1:
            self.frame_queue = deque([], maxlen=self.num_frames_to_stack)
        self.config = config
        self.image_height, self.image_width = self.config['image_height'], self.config['image_width']
        self.num_channels = 3 * self.num_frames_to_stack
        self.other_dims = 0

    def get_action_dims(self):
        return self.action_dims

    def get_action_repeat(self):
        return self.action_repeat

    def get_action_limits(self):
        return self.action_low, self.action_high

    def get_obs_chw(self):
        return self.num_channels, self.image_height, self.image_width

    def get_obs_other_dims(self):
        return self.other_dims

    def reset(self):
        obs, info = self.env.reset()
        if self.num_frames_to_stack > 1:
            self.frame_queue.clear()

        obs = self._stack_images(obs)
        obs_dict = {'image': obs}
        return obs_dict

    def step(self, action):
        action = np.float32(action)
        obs, reward, done, info = self.env.step(action)

        obs = self._stack_images(obs)
        done = False
        info = {}
        obs_dict = {'image': obs}
        return obs_dict, reward, done, info

    def _stack_images(self, obs):
        if self.num_frames_to_stack > 1:
            if len(self.frame_queue) == 0:  # Just after reset.
                for _ in range(self.num_frames_to_stack):
                    self.frame_queue.append(obs)
            else:
                self.frame_queue.append(obs)
            obs = np.concatenate(list(self.frame_queue), axis=0)
        return obs


def argument_parser(argument):
    """ Argument parser """
    parser = argparse.ArgumentParser(description='Binder Network.')
    parser.add_argument('-c', '--config', default='', type=str, help='Training config')
    args = parser.parse_args(argument)
    return args


def test():
    args = argument_parser(None)
    try:
        with open(args.config) as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error opening specified config yaml at: {}. "
              "Please check filepath and try again.".format(args.config))

    config = config['parameters']
    seed = config['seed']
    np.random.seed(seed)
    env = EnvironmentContainerDMC(config['env'], train=True, seed=config['seed'])
    plt.figure(1)
    action_low, action_high = env.get_action_limits()
    action_dims = env.get_action_dims()
    for _ in range(1):
        env.reset()
        for _ in range(1):
            action = np.random.uniform(action_low, action_high, action_dims)
            obs_dict, reward, done, info = env.step(action)
            obs_dcs = obs_dict['image'].transpose((1, 2, 0))
            obs_dmc = obs_dict['image_clean'].transpose((1, 2, 0))
            plt.clf()
            obs = np.concatenate([obs_dcs, obs_dmc], axis=1)
            plt.imshow(obs)
            plt.pause(0.001)
            filename = 'sample_0.png'
            plt.savefig(filename)


if __name__ == '__main__':
    test()
