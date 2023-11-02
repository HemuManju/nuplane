import yaml
import osmnx as ox

from nuplane.base_experiment import BaseExperiment

from experiments.taxiing.navigation.path_planner import PathPlanner


from ..agents import Hero


class SimpleExperiment(BaseExperiment):
    def __init__(self, config, core):
        super().__init__(config, core)
        self._initial_setup()

    def _initial_setup(self):
        # Setup hero
        self.hero = Hero(self.core.client, self.config['hero_config'])
        if self.hero.map is None:
            self.hero.map = self.core.map

    def reset(self, sensor_data=None):
        """Called at the beginning and each time the simulation is reset"""

        self.hero.apply_action([0, 0, 0, 0, 0, 0, 1.5, 1])

        # Get observation
        obs, info = self.get_observation(sensor_data, self.core)
        reward = 0
        done = False

        return obs, reward, done, info

    def get_action_space(self):
        """Returns the action space"""
        pass

    def get_observation_space(self):
        """Returns the observation space"""
        pass

    def get_actions(self):
        """Returns the actions"""
        pass

    def apply_actions(self, actions, core):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero

        :param action: value outputted by the policy
        """
        self.hero.apply_action(actions)

    def get_observation(self, sensor_data, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty

        """
        hero_obs = self.hero.get_observation()

        return hero_obs, {}

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return False

    def compute_reward(self, observation, core):
        """Computes the reward"""
        return 0
