import numpy as np

from nuplane.controller import BaseController
from .basic_controllers import BangBang


class SinusoidController(BaseController):
    def __init__(self, steering_params, speed_params, dt):
        super(SinusoidController, self).__init__(
            dt,
            [
                np.array([c1, c2])
                for c1, c2 in zip(
                    steering_params["input_constraints"],
                    speed_params["input_constraints"],
                )
            ],
        )
        self.speed_controller = BangBang(
            speed_params["low_u"],
            speed_params["high_u"],
            speed_params["nominal_u"],
            speed_params["low_speed"],
            speed_params["high_speed"],
            speed_params["input_constraints"],
        )
        self.steering_params = steering_params
        self.cte_bias = None
        self.turn_gain = None
        self.he_limit = None
        self.rudder_bias = self.steering_params["bias"]
        self.parkbrake = 0
        self.speed_break = 0

    def reset(self):
        self.speed_controller.reset()
        self.cte_bias = np.random.uniform(
            low=self.steering_params["cte_bias_range"][0],
            high=self.steering_params["cte_bias_range"][1],
        )
        self.he_limit = np.random.uniform(
            low=self.steering_params["he_limit_range"][0],
            high=self.steering_params["he_limit_range"][1],
        )
        self.turn_gain = (
            self.steering_params["turn_min"]
            + self.steering_params["turn_gain"] * self.he_limit
        )

    def get_control(self, state):
        cte, he, speed = state['psi'], state['theta'], state['groundspeed']
        throttle = self.speed_controller.get_input(speed)

        rudder = self.rudder_bias
        if he < self.he_limit and cte < self.cte_bias:
            rudder -= self.turn_gain
        elif he > -self.he_limit and cte > self.cte_bias:
            rudder += self.turn_gain

        control = [
            0,
            0,
            rudder,
            throttle,
            0,
            0,
            self.speed_break,
            self.parkbrake,
        ]

        return control
