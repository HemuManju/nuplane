from __future__ import print_function

import gym
from .core import XPlaneCore


class NUPlaneEnv(gym.Env):
    """
    This is a NUPlane environment, responsible of handling all the XPlane related steps of the training.
    """

    def __init__(self, config, debug=True):
        """Initializes the environment"""

        self.config = config

        # Check if experiment config is present
        if not self.config["experiment"]:
            raise Exception("The config should have experiment configuration")

        # Setup the core
        self.core = XPlaneCore(self.config, debug=debug)

        # Setup the experiment
        try:
            experiment_config = self.config["experiment"]["experiment_config"]
        except KeyError:
            experiment_config = None
        self.experiment = self.config["experiment"]["type"](
            experiment_config, self.core
        )
        if not self.experiment:
            raise Exception(
                "The experiment type cannot be empty. Please provide an experiment class"
            )
        else:
            self.core.setup_experiment(experiment_config=experiment_config)

        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

    def reset(self, *args, **kwargs):
        """Reset the simulation

        Returns
        -------
        [type]
            [description]
        """
        sensor_data = None
        obs, reward, done, info = self.experiment.reset(sensor_data, *args, **kwargs)
        self.core.reset()
        return obs, reward, done, info

    def step(self, action):
        """Computes one tick of the environment in order to return the new observation,
        as well as the rewards"""
        # TODO: implement synchronmous or asynchronous simulation here
        self.experiment.apply_actions(action, self.core)

        sensor_data = None
        observation, info = self.experiment.get_observation(sensor_data, self.core)
        done = self.experiment.get_done_status(observation, self.core)
        reward = self.experiment.compute_reward(observation, self.core)

        return observation, reward, done, info

    def close(self):
        self.core.close_simulation()
