from nuplane.base_experiment import BaseExperiment

from experiments.agents import AutolandActor

from gym.spaces import Box
import itertools
from nuplane.utils.transform import ft_to_m, nm_to_m


class AutoLandingExperiment(BaseExperiment):
    def __init__(self, config, core):
        super().__init__(config, core)

        self.actor = AutolandActor(core.client, config)

    def reset(self):
        """Called at the beginning and each time the simulation is reset"""
        self.actor.reset()

    def get_action_space(self):
        """
        Returns the action space
            elev - elevator flaps [-1, 1]
            aileron - aileron flaps [-1, 1]
            rudder - rudder steering [-1, 1]
            throttle - throttle amount [0, 1]
        """
        return Box(low=[-1, -1, -1,  0],
                   high=[1,  1,  1,  1])

    def get_observation_space(self):
        """Returns the observation space"""
        # these bounds seem reasonable, but are just guesses
        vel_bounds = [-100, 100]
        ang_vel_bounds = [-360, 360]
        ang_bounds = [0, 360]
        # using experiment config to bound (i.e., based on starting position plus 10% fudge)
        x_bounds = [-1000, 1.1*nm_to_m(self.config["experiment_config"]["hero_config"]["init_x"])]
        # assuming crosstrack error won't be more than 200m
        y_bounds = [-200, 200]
        h_bounds = [0, 1.1*ft_to_m(self.config["experiment_config"]["hero_config"]["init_h"])]

        low, high = [], []
        for bounds in itertools.chain([vel_bounds]*3, [ang_vel_bounds]*3, 
                                      [ang_bounds]*3, [x_bounds, y_bounds, h_bounds]):
            low.append(bounds[0])
            high.append(bounds[1])
        return Box(low=low, high=high)

    def get_actions(self):
        """Returns the actions"""
        pass

    def apply_actions(self, actions, core):
        """Given the action, returns a carla.VehicleControl() which will be applied to the hero

        :param action: value outputted by the policy
        """
        self.actor.apply_action(actions)

    def get_observation(self, sensor_data, core):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        return self.actor.est_statevec(), {}

    def get_done_status(self, observation, core):
        """Returns whether or not the experiment has to end"""
        return self._est_if_landed()

    def _est_if_landed(self):
        x, y, h = self.actor.get_pos_state()
        if x > -10 or y > 10:
            # not past runway or crosstrack error too high
            return False
        if h > self.actor._start_elev + 0.2:
            # if not close to ground, then not done
            return False

        # past runway crossing and within 5m of the start elevation
        # some flexibility for sloping runways
        return True

    def compute_reward(self, observation, core):
        """Computes the reward"""
        return 1. if self._est_if_landed() else 0.
