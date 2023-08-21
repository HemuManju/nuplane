from __future__ import print_function

import gym
from .core import XPlaneCore


class XPlaneEnv(gym.Env):
    """
    This is a XPlane environment, responsible of handling all the XPlane related steps of the training.
    """

    def __init__(self, config):
        """Initializes the environment"""
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        self.core = XPlaneCore(self.config['XPlane_server'])
        self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data)

        return observation

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""

        control = self.experiment.compute_action(action)
        sensor_data = self.core.tick(control)

        observation, info = self.experiment.get_observation(sensor_data)
        done = self.experiment.get_done_status(observation, self.core)
        reward = self.experiment.compute_reward(observation, self.core)

        return observation, reward, done, info
