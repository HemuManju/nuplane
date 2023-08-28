from collections import deque

import numpy as np


wrap_pi = lambda x: np.mod(x + np.pi, 2 * np.pi) - np.pi
angle_diff = lambda x, y: wrap_pi(x - y)


class PIDLongitudinalController:
    """
    PIDLongitudinalController implements longitudinal control using a PID.
    """

    def __init__(self, K_P=1.0, K_I=0.05, K_D=0.05, dt=0.1):
        """
        Constructor method.

            :param hero: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_speed, current_speed, debug=False):
        """
        Execute one step of longitudinal control to reach a given target speed.

            :param target_speed: target speed in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """

        if debug:
            print('Current speed = {}'.format(current_speed))

        return self._pid_control(target_speed, current_speed)

    def _pid_control(self, target_speed, current_speed):
        """
        Estimate the throttle/brake of the hero based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the hero in Km/h
            :return: throttle/brake control
        """

        error = target_speed - current_speed
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return (self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PIDLateralController:
    """
    PIDLateralController implements lateral control using a PID.
    """

    def __init__(self, offset=0, K_P=0.75, K_I=0.01, K_D=0.0, dt=0.1):
        """
        Constructor method.

            :param hero: actor to apply to local planner logic onto
            :param offset: distance to the center line. If might cause issues if the value
                is large enough to make the hero invade other lanes.
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._offset = offset
        self._e_buffer = deque(maxlen=10)

    def run_step(self, heading_error):
        """
        Execute one step of lateral control to steer
        the hero towards a certain waypoin.

            :param waypoint: target waypoint
            :return: steering control in the range [-1, 1] where:
            -1 maximum steering to left
            +1 maximum steering to right
        """
        return self._pid_control(heading_error)

    def _pid_control(self, heading_error):
        """
        Estimate the steering angle of the hero based on the PID equations

            :param waypoint: target waypoint
            :param hero_transform: current transform of the hero
            :return: steering control in the range [-1, 1]
        """
        self._e_buffer.append(heading_error)
        if len(self._e_buffer) >= 2:
            _de = (self._e_buffer[-1] - self._e_buffer[-2]) / self._dt
            _ie = sum(self._e_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0
        return (self._k_p * heading_error) + (self._k_d * _de) + (self._k_i * _ie)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt


class PathController:
    # Throttle command ranges from -1 to 1
    # but too low of a throttle can stall the engine
    MIN_THROTTLE = 0.0

    def __init__(self, agent, dt=0.1, *args, **kwargs):
        self.agent = agent
        self.controller_state = "tracking"
        self.dt = dt
        self.max_speed = 3.0

        self.parkbrake = 0
        self.speed_break = 0
        self.gear = 1
        self.flaps = 0
        self.steering = 0
        self.past_steering = 0
        self.throttle = 0
        self.speed_error_integrator = 0
        self.lateral_control = PIDLateralController(dt=dt)
        self.longitudina_control = PIDLongitudinalController(dt=dt)

    def get_control(self, distance_next_pos, heading):
        # Get the control
        state = self.agent.get_observation()

        # print(state['psi'] - state['true_psi'])

        # Calculate steering
        heading_error = angle_diff(
            heading * np.pi / 180, state['true_psi'] * np.pi / 180
        )
        self.steering = self.lateral_control.run_step(heading_error)

        if self.steering > self.past_steering + 0.05:
            self.steering = self.past_steering + 0.05
        elif self.steering < self.past_steering - 0.05:
            self.steering = self.past_steering - 0.05

        # Calculate the throttle
        target_speed = min(self.max_speed, distance_next_pos * self.dt * 5)

        self.throttle = self.longitudina_control.run_step(
            target_speed, state['groundspeed']
        )

        if state['groundspeed'] > 3.0:
            self.speed_break = 1.0
            self.throttle = 0
        else:
            self.speed_break = -0.5

        self.steering = np.clip(self.steering, -0.5, 0.5)
        self.throttle = np.clip(self.throttle, self.MIN_THROTTLE, 1.0)
        control = [
            0,
            self.steering,
            0,
            self.throttle,
            self.gear,
            self.flaps,
            self.speed_break,
            self.parkbrake,
        ]

        # Update past steering
        self.past_steering = self.steering

        return control
