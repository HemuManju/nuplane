import math
import numpy as np

from experiments.auto_landing.pid import PID

from nuplane.controller import BaseController

# TCH = Threshold Crossing Height
# Default is the one for Grant Co Intl Airport (KMWH) Runway 04
# 50 ft TCH -> meters
GRANT_RWY4_TCH = 50 * 0.3048

class GlideSlopeController(BaseController):
    # Input Constraints:
    # X-Plane elevator, aileron, rudder, throttle
    # first three are [-1, 1] and throttle is [0, 1]
    def __init__(self, dt=0.1, input_constraints=[[-1, -1, -1, 0], [1]*4]):
        super().__init__(dt, input_constraints)
        self._dt          = dt

        if dt > 0.5:
            raise Warning("Running at a much slower dt than controller was designed for")

        # PI controllers
        # lateral
        self._psi_pid   = PID(dt, kp=1., ki=0.1, kd=0.)
        self._y_pid     = PID(dt, kp=0.5, ki=0.01, kd=0.)
        self._phi_pid   = PID(dt, kp=1., ki=0.1, kd=0.)

        self._u_pid     = PID(dt, kp=50., ki=5., kd=0.)
        self._theta_pid = PID(dt, kp=0.24, ki=0.024, kd=0.)

        self._pids = [self._psi_pid, self._y_pid, self._phi_pid,
                      self._u_pid, self._theta_pid]

    @property
    def runway_elevation(self):
        '''
        Returns the target height above the runway threshold
        '''
        return self._runway_elev

    def reset(self):
        for pid in self._pids:
            pid.reset()

    def get_control(self, statevec, estop=False):
        '''
        INPUTS
            Statevector based on https://arc.aiaa.org/doi/10.2514/6.2021-0998
            statevec with components
                u      - longitudinal velocity (m/s)
                v      - lateral velocity (m/s)
                w      - vertical velocity (m/s)
                p      - roll velocity (deg/s)
                q      - pitch velocity (deg/s)
                r      - yaw velocity (deg/s)
                phi    - roll angle (deg)
                theta  - pitch angle (deg)
                psi    - yaw angle (deg)
                x      - horizontal distance (m)
                y      - lateral deviation (m)
                h      - aircraft altitude (m)
        OUTPUTS
            throttle
            elevator
            rudder
            aileron
        '''

        if estop:
            raise Warning("GlideSlopeController does not implement estop=True")

        u, v, w, \
        p, q, r, \
        phi, theta, psi, \
        x, y, h = statevec

        # lateral control
        err_y = 0.0 - y
        err_psi = 0.0 - psi
        err_phi = 0.0 - phi

        delta_r = self._psi_pid(err_psi) + self._y_pid(err_y)
        delta_a = self._phi_pid(err_phi)
        rudder = max(-27, min(delta_r, 27))/27
        aileron = max(-20, min(delta_a, 20))/20

        # longitudinal control
        err_u = self._des_u - u

        fu = self._u_pid(err_u) + 5000
        throttle = min(fu, 10000)/10000

        h_c = self._h_thresh + x*self._tan_gamma
        err_h = h_c - h
        theta_c = self._theta_pid(err_h)
        elev = (theta_c - theta)*5 - 0.05*q

        if elev > 0:
            elevator = min(elev, 30)/30
        else:
            elevator = max(elev, -15)/15

        return elevator, aileron, rudder, throttle

    @property
    def runway_threshold_height(self):
        return self._h_thresh

    def set_reference(self, state_reference, input_reference=None):
        """
        Sets the state reference for the glideslope controller.
        The assumed format is [gamma, tch, runway_elev, des_u]
        where
            gamma: the glideslope angle in degrees: positive is pointing down so should be > 0
            tch: the runway crossing threshold in meters
            runway_elev: the runway elevation (above mean sea level) in meters
            des_u: the desired forward body-frame velocity

        If you want to use a default value, pass np.nan for that element
        """
        if input_reference is not None:
            raise Warning(f"GlideSlopeController does not accept input reference but got: {input_reference}")
        DEFAULT_VALS = np.array([3, GRANT_RWY4_TCH, 361, 50])
        state_reference = np.asarray(state_reference)
        nan_idx = np.isnan(state_reference)
        state_reference[nan_idx] = DEFAULT_VALS[nan_idx]

        gamma, tch, runway_elev, des_u = state_reference
        self._gamma       = gamma    # glide slope angle
        self._h_thresh    = tch + runway_elev # height of runway threshold (m) is the TCH + the elevation of the runway (m)
        self._runway_elev = runway_elev
        self._des_u       = des_u # desired longitudinal velocity (m/s)

        self._tan_gamma = math.tan(math.radians(self._gamma))
