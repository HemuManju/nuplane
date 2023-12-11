import math
import numpy as np
import time

from nuplane.actors import BaseActor
from nuplane.utils.transform import *


class AIAircraft(BaseActor):
    def __init__(self, client, config, id=1) -> None:
        super().__init__(client, config)

        self.id = id
        mode = [2] * 20
        mode[0] = 0
        self.client.sendDREF('sim/operation/override/override_plane_ai_autopilot', mode)
        time.sleep(0.1)

    def apply_action(self, *args, **kwargs):
        return None

    def get_observation(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        initRef = f"sim/multiplayer/position/plane{self.id}_"
        drefs = []
        refs = [
            'the',
            'psi',
            'phi',
            'psi',
            'v_x',
            'v_y',
            'v_z',
            'wing_sweep',
            'x',
            'y',
        ]
        for ref in refs:
            drefs += [initRef + ref]
        values = [0] * len(refs)
        self.client.sendDREFs(drefs, values)
        time.sleep(0.1)

    def set_position(self, position, elevation):
        mode = [1] * 20
        self.client.sendDREF(f'sim/operation/override/override_planepath', mode)
        self.client.sendPOSI(
            [position['lat'], position['lon'], 0, 0, 0, position['heading']], self.id
        )
        time.sleep(0.1)
        self.client.sendDREF(f"sim/multiplayer/position/plane{self.id}_y", elevation)
        return None

    def apply_action(self, action):
        elev, aileron, rudder, throttle, brake = (
            action[0],
            action[1],
            action[2],
            action[3],
            action[4],
        )
        mode = [1] * 20
        mode[0] = 1
        self.client.sendDREF('sim/multiplayer/controls/gear_request', mode)
        if brake:
            mode = [1] * 20
        else:
            mode = [0] * 20
        self.client.sendDREF(f"sim/multiplayer/controls/parking_brake", mode)
        self.client.sendCTRL([elev, aileron, rudder, throttle], self.id)
        time.sleep(0.1)


class Hero(BaseActor):
    def __init__(self, client, config) -> None:
        super().__init__(client, config)

    def get_observation(self, *args, **kwargs):
        initRef = "sim/flightmodel/position/"
        data_refs = [
            'latitude',
            'longitude',
            'true_psi',
            'psi',
            'groundspeed',
        ]
        data = self.client.getDREFs([initRef + item for item in data_refs])
        observation = {}
        for i, item in enumerate(data):
            observation[data_refs[i]] = item[0]

        return observation

    def reset(self, heading=None, *args, **kwargs):
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = [
            'theta',
            'phi',
            'psi',
            'local_vx',
            'local_vy',
            'local_vz',
            'local_ax',
            'local_ay',
            'local_az',
            'Prad',
            'Qrad',
            'Rrad',
            'q',
            'groundspeed',
            'indicated_airspeed',
            'indicated_airspeed2',
            'true_airspeed',
            'M',
            'N',
            'L',
            'P',
            'Q',
            'R',
            'P_dot',
            'Q_dot',
            'R_dot',
            'Prad',
            'Qrad',
            'Rrad',
        ]
        for ref in refs:
            drefs += [initRef + ref]
        values = [0] * len(refs)
        self.client.sendDREFs(drefs, values)

        # Get the position using node number
        node_num = self.config['spawn_location']
        position = self.map.get_node_info(node_num)
        position['heading'] = heading
        self._set_position(position)

        # Make throttle zero
        self.client.sendDREF("sim/flightmodel/controls/parkbrake", 1)
        self.client.sendCTRL([0, 0, 0, 0, 1, 1, 0])

        time.sleep(0.1)

    def get_altitude(self):
        alt = self.client.getDREF("sim/flightmodel/position/local_y")[0]
        return alt

    def _set_position(self, position):
        mode = [1] * 20
        self.client.sendDREF('sim/operation/override/override_planepath', mode)
        self.client.sendPOSI(
            [position['y'], position['x'], 0, 0, 0, position['heading']], 0
        )
        time.sleep(0.1)
        curr_agly = self.client.getDREF("sim/flightmodel/position/y_agl")[0]
        curr_localy = self.client.getDREF("sim/flightmodel/position/local_y")[0]
        self.client.sendDREF(
            "sim/flightmodel/position/local_y", curr_localy - curr_agly
        )
        return None

    def apply_action(self, action):
        elev, aileron, rudder, throttle, gear, flaps, speed_break, parkbrake = (
            action[0],
            action[1],
            action[2],
            action[3],
            action[4],
            action[5],
            action[6],
            action[7],
        )
        mode = [1] * 20
        # Deploy the gear in case
        self.client.sendDREF('sim/multiplayer/controls/gear_request', mode)

        # Release the break
        self.client.sendDREF("sim/flightmodel/controls/parkbrake", parkbrake)

        # Apply control
        self.client.sendCTRL(
            [elev, aileron, rudder, throttle, gear, flaps, speed_break]
        )


class AutolandActor(BaseActor):
    def __init__(self, client, config=None) -> None:
        super().__init__(client, config)

        # Assume plane starts aligned at beginning of runway
        start_heading = self.client.getDREF("sim/flightmodel/position/psi")[0]
        self._heading_offset = start_heading - self.client.getDREF("sim/flightmodel/position/mag_psi")[0]
        try:
            self._glideslope_heading = config["hero_config"]["glideslope_heading_deg"]
        except KeyError:
            raise RuntimeError(f"Missing heading for autolanding. Place in config['hero_config']['glideslope_heading_deg']\n"
                               "Find the heading in the AirNav approach card online.")

        # Get OpenGL coordinates so we can make this point the origin
        # of the transformed coordinates
        opengl_x = self.client.getDREF("sim/flightmodel/position/local_x")[0]
        opengl_z = self.client.getDREF("sim/flightmodel/position/local_z")[0]
        self._t = np.array((opengl_x, opengl_z)).reshape((2, 1))

        # TODO: determine this automatically
        #       relates to home heading and local coordinate frame at given airport
        self._rotrad = -0.6224851011617226
        self._R = np.array([[ np.cos(self._rotrad), -np.sin(self._rotrad) ],
                            [ np.sin(self._rotrad),  np.cos(self._rotrad)]])

        # determine elevation offset by getting difference between local y (the axis for elevation)
        # and current elevation
        # then use that to shift the coordinate to align with the desired elevation
        self._start_elev = self.client.getDREF("sim/flightmodel/position/elevation")[0]
        curr_localy = self.client.getDREF("sim/flightmodel/position/local_y")[0]
        self.height_offset = self._start_elev - curr_localy

    def __del__(self):
        # the height that's on the ground at the runway
        # in the autolanding frame
        self._set_orient_pos(0, 0, 0, 0, 0, self._start_elev)

    def reset(self, *args, **kwargs):
        """Reset the actor"""

        hero_config = self.config["hero_config"]

        init_u =     hero_config.get("init_u", 50.)
        init_v =     hero_config.get("init_v", 0.)
        init_w =     hero_config.get("init_w", 0.)
        init_p =     hero_config.get("init_p", 0.)
        init_q =     hero_config.get("init_q", 0.)
        init_r =     hero_config.get("init_r", 0.)
        init_phi =   hero_config.get("init_phi", 0.)
        init_theta = hero_config.get("init_theta", 0.)
        init_psi =   hero_config.get("init_psi", 0.)
        init_x_nm =  hero_config.get("init_x", 7.)
        init_y =     hero_config.get("init_y", 0.)
        init_h_ft =  hero_config.get("init_h", 3300.)

        # conversions
        init_x = nm_to_m(init_x_nm)
        init_h = ft_to_m(init_h_ft)

        if init_h < self._start_elev:
            raise Warning(f"Note: Initial height is below start elevation of {self._start_elev}m")

        self.client.pauseSim(True)

        # Zero out control inputs
        self.client.sendCTRL([0,0,0,0])

        # Set parking brake
        self.client.sendDREF("sim/flightmodel/controls/parkbrake", 0)

        # Zero out moments and forces
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = ['theta','phi', 'local_vx','local_vy',
                'local_vz','local_ax','local_ay','local_az',
                'Prad','Qrad','Rrad','q','groundspeed',
                'indicated_airspeed','indicated_airspeed2',
                'true_airspeed','M','N','L','P','Q','R','P_dot',
                'Q_dot','R_dot','Prad','Qrad','Rrad']
        for ref in refs:
            drefs += [initRef+ref]
        values = [0]*len(refs)
        self.client.sendDREFs(drefs,values)

        # Set position and orientation
        # Set known good start values
        # Note: setting position with lat/lon gets you within 0.3m. Setting local_x, local_z is more accurate)
        self._set_orient_pos(init_phi, init_theta, init_psi, init_x, init_y, init_h)
        self._set_orientrate_vel(init_u, init_v, init_w, init_p, init_q, init_r)

        # Fix the plane if you "crashed" or broke something
        self.client.sendDREFs(["sim/operation/fix_all_systems"], [1])

        # Set fuel mixture for engine
        self.client.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)

        # Reset fuel levels
        self.client.sendDREFs(["sim/flightmodel/weight/m_fuel1","sim/flightmodel/weight/m_fuel2"],[232,232])

        # Give time to settle
        time.sleep(1.)

    def get_observation(self, *args, **kwargs):
        """Get the observation from the actor.

        Raises
        ------
        NotImplementedError
        """
        raise NotImplementedError

    def apply_action(self, action):
        """
        Navigate the plane with a normalized control input directly to XPlane

        Args:
            elev - elevator flaps [-1, 1]
            aileron - aileron flaps [-1, 1]
            rudder - rudder steering [-1, 1]
            throttle - throttle amount [0, 1]
        """
        elev, aileron, rudder, throttle = action
        self.client.sendCTRL([elev, aileron, rudder, throttle])

    def pause(self, yes=True):
        """
        Pause or unpause the simulation

        Args:
            yes: whether to pause or unpause the sim [default: True (i.e., pause)]
        """
        self.client.pauseSim(yes)

    def _set_orient_pos(self, phi, theta, psi, x, y, h):
        '''
        Set the orientation and position of the plane.
        Args:
            phi    - roll angle (deg)
            theta  - pitch angle (deg)
            psi    - yaw angle (deg)
            x      - horizontal distance (m)
            y      - lateral deviation (m)
            h      - aircraft altitude (m)
        '''
        self._send_xy(x, y)
        self.client.sendDREF("sim/flightmodel/position/local_y", h - self.height_offset)

        self.client.sendDREF('sim/flightmodel/position/phi', phi)
        self.client.sendDREF('sim/flightmodel/position/theta', theta)
        self.client.sendDREF('sim/flightmodel/position/psi', self._home_to_opengl_heading(psi))

    def _set_orientrate_vel(self, u, v, w, p, q, r):
        # 2d rotation and flip
        uv = np.array([u, v]).reshape((2, 1))
        hr = math.radians(self._home_to_opengl_heading(0))
        R = np.array([[np.cos(hr), -np.sin(hr)],[np.sin(hr), np.cos(hr)]])
        rot_uv = R@uv
        # flip direction of longitudinal velocity to put in OpenGL coordinates
        self.client.sendDREF('sim/flightmodel/position/local_vz', -rot_uv[0])
        self.client.sendDREF('sim/flightmodel/position/local_vx', rot_uv[1])
        self.client.sendDREF('sim/flightmodel/position/local_vy', w)
        self.client.sendDREF('sim/flightmodel/position/P', p)
        self.client.sendDREF('sim/flightmodel/position/Q', q)
        self.client.sendDREF('sim/flightmodel/position/R', r)

    def _send_xy(self, x, y):
        """
        Sets the x and y variables in the autolanding frame.

        Args:
            x: the distance from the runway in meters
            y: the crosstrack error in meters
        """
        local_x, local_z = self._xy_to_opengl_xz(x, y)
        self.client.sendDREF("sim/flightmodel/position/local_x", local_x)
        self.client.sendDREF("sim/flightmodel/position/local_z", local_z)

    def _xy_to_opengl_xz(self, x, y):
        """
        Converts autoland statevec's x, y elements to local x, z coordinates.
        Note: in local frame, y is elevation (up) so we care about x and **z** for this rotation
        """
        # rotation to align to runway
        # flip a sign because of x, y orientation
        # in autoland frame, x is pointing frame the runway to the starting point
        # and y is pointing to the right from the plane's point of view
        F = np.array([[-1.,  0.],
                    [ 0., 1.]])
        r = (self._R@F)@np.array([[x], [y]]).reshape((2, 1))
        local_x, local_z = r + self._t
        return local_x.flatten(), local_z.flatten()

    def _opengl_xz_to_xy(self, local_x, local_z):
        R = self._R
        F = np.array([[-1.,  0.],
                    [ 0., 1.]])
        RF = R@F
        l = np.array([[local_x], [local_z]]).reshape((2, 1))
        r = l - self._t
        x, y = np.linalg.inv(RF)@r
        return x, y

    def _actual_to_opengl_heading(self, psi):
        """
        Convert actual heading to local frame heading
        """
        return psi + self._heading_offset

    def _home_to_actual_heading(self, psi):
        """
        Convert home heading (0 deg is aligned with runway)
        to actual heading
        """
        return self._glideslope_heading - psi

    def _home_to_opengl_heading(self, psi):
        """
        Convert home heading (0 deg is aligned with runway)
        to local frame (OpenGL) heading
        """

        return self._actual_to_opengl_heading(self._home_to_actual_heading(psi))