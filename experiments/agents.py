import math
import numpy as np
import time

from nuplane.actors import BaseActor
from nuplane.utils.transform import nm_to_m, ft_to_m
from nuplane.utils.image import receive_image_over_tcp


class AIAircraft(BaseActor):
    def __init__(self, client, config, id=1) -> None:
        super().__init__(client, config)

        self.id = id
        mode = [2] * 20
        mode[0] = 0
        self.client.sendDREF("sim/operation/override/override_plane_ai_autopilot", mode)
        time.sleep(0.1)

    def get_observation(self, *args, **kwargs):
        return None

    def reset(self, *args, **kwargs):
        initRef = f"sim/multiplayer/position/plane{self.id}_"
        drefs = []
        refs = [
            "the",
            "psi",
            "phi",
            "psi",
            "v_x",
            "v_y",
            "v_z",
            "wing_sweep",
            "x",
            "y",
        ]
        for ref in refs:
            drefs += [initRef + ref]
        values = [0] * len(refs)
        self.client.sendDREFs(drefs, values)
        time.sleep(0.1)

    def set_position(self, position, elevation):
        mode = [1] * 20
        self.client.sendDREF("sim/operation/override/override_planepath", mode)
        self.client.sendPOSI(
            [position["lat"], position["lon"], 0, 0, 0, position["heading"]], self.id
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
        self.client.sendDREF("sim/multiplayer/controls/gear_request", mode)
        if brake:
            mode = [1] * 20
        else:
            mode = [0] * 20
        self.client.sendDREF("sim/multiplayer/controls/parking_brake", mode)
        self.client.sendCTRL([elev, aileron, rudder, throttle], self.id)
        time.sleep(0.1)


class Hero(BaseActor):
    def __init__(self, client, config) -> None:
        super().__init__(client, config)

    def get_observation(self, *args, **kwargs):
        initRef = "sim/flightmodel/position/"
        data_refs = [
            "latitude",
            "longitude",
            "true_psi",
            "psi",
            "groundspeed",
        ]
        data = self.client.getDREFs([initRef + item for item in data_refs])
        observation = {}
        for i, item in enumerate(data):
            observation[data_refs[i]] = item[0]

        # Get image data
        camera_data = self.client.getDREF("nuplane/camera_data")
        width, height = camera_data[0], camera_data[1]

        tcp_port = self.client.getDREF("nuplane/camera_tcp_port")[0]
        image = receive_image_over_tcp(width, height, 3, tcp_port)

        observation["image"] = image

        return observation

    def reset(self, heading=None, *args, **kwargs):
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = [
            "theta",
            "phi",
            "psi",
            "local_vx",
            "local_vy",
            "local_vz",
            "local_ax",
            "local_ay",
            "local_az",
            "Prad",
            "Qrad",
            "Rrad",
            "q",
            "groundspeed",
            "indicated_airspeed",
            "indicated_airspeed2",
            "true_airspeed",
            "M",
            "N",
            "L",
            "P",
            "Q",
            "R",
            "P_dot",
            "Q_dot",
            "R_dot",
            "Prad",
            "Qrad",
            "Rrad",
        ]
        for ref in refs:
            drefs += [initRef + ref]
        values = [0] * len(refs)
        self.client.sendDREFs(drefs, values)

        # Get the position using node number
        node_num = self.config["spawn_location"]
        position = self.map.get_node_info(node_num)
        position["heading"] = heading
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
        self.client.sendDREF("sim/operation/override/override_planepath", mode)
        self.client.sendPOSI(
            [position["y"], position["x"], 0, 0, 0, position["heading"]], 0
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
        self.client.sendDREF("sim/multiplayer/controls/gear_request", mode)

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
        self._glideslope_heading = self.client.getDREF("sim/flightmodel/position/psi")[
            0
        ]
        true_psi = self.client.getDREF("sim/flightmodel/position/true_psi")[0]
        print(
            f"Glideslope heading inferred to be: {self._glideslope_heading} ({true_psi} true north heading)"
        )

        # Get OpenGL coordinates so we can make this point the origin
        # of the transformed coordinates
        opengl_x = self.client.getDREF("sim/flightmodel/position/local_x")[0]
        opengl_z = self.client.getDREF("sim/flightmodel/position/local_z")[0]
        # x = east, z = south, y = up
        rotrad = math.radians(-self._glideslope_heading)
        # rotation is clockwise from north
        # plus some axis flipping to align xs
        self._R = np.array(
            [[np.cos(rotrad), -np.sin(rotrad)], [np.sin(rotrad), np.cos(rotrad)]]
        )
        self._t = np.array((opengl_z, opengl_x)).reshape((2, 1))

        # determine elevation offset by getting difference between local y (the axis for elevation)
        # and current elevation
        # then use that to shift the coordinate to align with the desired elevation
        self._start_elev = self.client.getDREF("sim/flightmodel/position/elevation")[0]
        curr_localy = self.client.getDREF("sim/flightmodel/position/local_y")[0]
        self.height_offset = self._start_elev - curr_localy

    def place_at_start_position(self):
        """
        Sets the plane back at the initial position (where the plane was when this class was initialized)
        and pauses the simulation.
        """
        # the height that's on the ground at the runway
        # in the autolanding frame
        self._set_orient_pos(0, 0, 0, 0, 0, self._start_elev)
        self.pause(True)

    @property
    def runway_elev(self):
        """
        Returns the estimated runway elevation.
        This is based on the spawn location of the plane (chosen by X-Plane).
        """
        return self._start_elev

    def reset(self, *args, **kwargs):
        """Reset the actor"""

        hero_config = self.config["hero_config"]

        init_u = hero_config.get("init_u", 50.0)
        init_v = hero_config.get("init_v", 0.0)
        init_w = hero_config.get("init_w", 0.0)
        init_p = hero_config.get("init_p", 0.0)
        init_q = hero_config.get("init_q", 0.0)
        init_r = hero_config.get("init_r", 0.0)
        init_phi = hero_config.get("init_phi", 0.0)
        init_theta = hero_config.get("init_theta", 0.0)
        init_psi = hero_config.get("init_psi", 0.0)
        init_x_nm = hero_config.get("init_x", 7.0)
        init_y = hero_config.get("init_y", 0.0)
        init_h_ft = hero_config.get("init_h", 3300.0)

        # conversions
        self._init_x = nm_to_m(init_x_nm)
        self._init_h = ft_to_m(init_h_ft)

        if self._init_h < self._start_elev:
            raise Warning(
                f"Note: Initial height is below start elevation of {self._start_elev}m"
            )

        self.client.pauseSim(True)

        # Zero out control inputs
        self.client.sendCTRL([0, 0, 0, 0])

        # Set parking brake
        self.client.sendDREF("sim/flightmodel/controls/parkbrake", 0)

        # Zero out moments and forces
        initRef = "sim/flightmodel/position/"
        drefs = []
        refs = [
            "theta",
            "phi",
            "local_vx",
            "local_vy",
            "local_vz",
            "local_ax",
            "local_ay",
            "local_az",
            "Prad",
            "Qrad",
            "Rrad",
            "q",
            "groundspeed",
            "indicated_airspeed",
            "indicated_airspeed2",
            "true_airspeed",
            "M",
            "N",
            "L",
            "P",
            "Q",
            "R",
            "P_dot",
            "Q_dot",
            "R_dot",
            "Prad",
            "Qrad",
            "Rrad",
        ]
        for ref in refs:
            drefs += [initRef + ref]
        values = [0] * len(refs)
        self.client.sendDREFs(drefs, values)

        # Set position and orientation
        # Set known good start values
        # Note: setting position with lat/lon gets you within 0.3m. Setting local_x, local_z is more accurate)
        self._set_orient_pos(
            init_phi, init_theta, init_psi, self._init_x, init_y, self._init_h
        )
        self._set_orientrate_vel(init_u, init_v, init_w, init_p, init_q, init_r)

        # Fix the plane if you "crashed" or broke something
        self.client.sendDREFs(["sim/operation/fix_all_systems"], [1])

        # Set fuel mixture for engine
        self.client.sendDREF("sim/flightmodel/engine/ENGN_mixt", 0.61)

        # Reset fuel levels
        self.client.sendDREFs(
            ["sim/flightmodel/weight/m_fuel1", "sim/flightmodel/weight/m_fuel2"],
            [232, 232],
        )

        # Give time to settle
        time.sleep(1.0)

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
        self.pause(False)
        elev, aileron, rudder, throttle = action
        self.client.sendCTRL([elev, aileron, rudder, throttle])

    def pause(self, yes=True):
        """
        Pause or unpause the simulation

        Args:
            yes: whether to pause or unpause the sim [default: True (i.e., pause)]
        """
        self.client.pauseSim(yes)

    ###########################################################################
    # State estimation
    # by default these are passthroughs -- can be overloaded in a subclass
    ###########################################################################
    def est_statevec(self):
        vel = self.est_vel_state()
        ovel = self.est_orient_vel_state()
        o = self.est_orient_state()
        pos = self.est_pos_state()

        return np.stack((vel, ovel, o, pos)).flatten()

    def est_vel_state(self):
        return self.get_vel_state()

    def est_orient_vel_state(self):
        return self.get_orient_vel_state()

    def est_orient_state(self):
        return self.get_orient_state()

    def est_pos_state(self):
        return self.get_pos_state()

    ###########################################################################
    # True state getters
    ###########################################################################
    def get_statevec(self):
        """
        Returns the state vector used in the autoland scenario
        Based on https://arc.aiaa.org/doi/10.2514/6.2021-0998
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
        """

        vel = self.get_vel_state()
        ovel = self.get_orient_vel_state()
        o = self.get_orient_state()
        pos = self.get_pos_state()

        return np.stack((vel, ovel, o, pos)).flatten()

    def get_vel_state(self):
        return self._body_frame_velocity()

    def get_orient_vel_state(self):
        P = self.client.getDREF("sim/flightmodel/position/P")[0]
        Q = self.client.getDREF("sim/flightmodel/position/Q")[0]
        R = self.client.getDREF("sim/flightmodel/position/R")[0]
        return np.array([P, Q, R])

    def get_orient_state(self):
        phi = self.client.getDREF("sim/flightmodel/position/phi")[0]
        theta = self.client.getDREF("sim/flightmodel/position/theta")[0]
        psi = self._get_home_heading()
        return np.array([phi, theta, psi])

    def get_pos_state(self):
        x, y = self._get_home_xy()
        h = self.client.getDREF("sim/flightmodel/position/elevation")[0]
        return np.array([x, y, h])

    def _set_orient_pos(self, phi, theta, psi, x, y, h):
        """
        Set the orientation and position of the plane.
        Args:
            phi    - roll angle (deg)
            theta  - pitch angle (deg)
            psi    - yaw angle (deg)
            x      - horizontal distance (m)
            y      - lateral deviation (m)
            h      - aircraft altitude (m)
        """
        self._send_xy(x, y)
        self.client.sendDREF("sim/flightmodel/position/local_y", h - self.height_offset)

        self.client.sendDREF("sim/flightmodel/position/phi", phi)
        self.client.sendDREF("sim/flightmodel/position/theta", theta)
        self.client.sendDREF(
            "sim/flightmodel/position/psi", self._home_to_opengl_heading(psi)
        )

    def _set_orientrate_vel(self, u, v, w, p, q, r):
        # 2d rotation and flip
        uv = np.array([u, v]).reshape((2, 1))
        hr = math.radians(self._home_to_opengl_heading(0))
        R = np.array([[np.cos(hr), -np.sin(hr)], [np.sin(hr), np.cos(hr)]])
        rot_uv = R @ uv
        # flip direction of longitudinal velocity to put in OpenGL coordinates
        self.client.sendDREF("sim/flightmodel/position/local_vz", -rot_uv[0])
        self.client.sendDREF("sim/flightmodel/position/local_vx", rot_uv[1])
        self.client.sendDREF("sim/flightmodel/position/local_vy", w)
        self.client.sendDREF("sim/flightmodel/position/P", p)
        self.client.sendDREF("sim/flightmodel/position/Q", q)
        self.client.sendDREF("sim/flightmodel/position/R", r)

    def _send_xy(self, x, y):
        """
        Sets the x and y variables in the autolanding frame.

        Args:
            x: the distance from the runway in meters
            y: the crosstrack error in meters
        """
        local_z, local_x = self._xy_to_opengl_zx(x, y)
        self.client.sendDREF("sim/flightmodel/position/local_x", local_x)
        self.client.sendDREF("sim/flightmodel/position/local_z", local_z)

    def _xy_to_opengl_zx(self, x, y):
        """
        Converts autoland statevec's x, y elements to local x, z coordinates.
        Note: in local frame, y is elevation (up) so we care about x and **z** for this rotation
        """
        xy = np.array([[x], [y]]).reshape((2, 1))
        r = self._R @ xy
        local_z, local_x = r + self._t
        return local_z.flatten(), local_x.flatten()

    def _opengl_zx_to_xy(self, local_z, local_x):
        l = np.array([[local_z], [local_x]]).reshape((2, 1))
        r = l - self._t
        xy = np.linalg.inv(self._R) @ r
        xy = xy.flatten()
        return xy[0], xy[1]

    def _home_to_opengl_heading(self, psi):
        """
        Convert home heading (0 deg is aligned with runway)
        to actual heading
        """
        return self._glideslope_heading + psi

    ###########################################################################
    # Helper functions
    ###########################################################################
    def _body_frame_velocity(self):
        cos = math.cos
        sin = math.sin

        psi = self.client.getDREF("sim/flightmodel/position/psi")[0]
        theta = self.client.getDREF("sim/flightmodel/position/theta")[0]
        phi = self.client.getDREF("sim/flightmodel/position/phi")[0]

        h = math.radians(psi)
        Rh = np.array([[cos(h), sin(h), 0], [-sin(h), cos(h), 0], [0, 0, 1]])
        el = math.radians(theta)
        Re = np.array([[cos(el), 0, -sin(el)], [0, 1, 0], [sin(el), 0, cos(el)]])
        roll = math.radians(phi)
        Rr = np.array(
            [[1, 0, 0], [0, cos(roll), sin(roll)], [0, -sin(roll), cos(roll)]]
        )
        R = np.matmul(Rr, np.matmul(Re, Rh))

        vx = self.client.getDREF("sim/flightmodel/position/local_vx")[0]
        vy = self.client.getDREF("sim/flightmodel/position/local_vy")[0]
        vz = self.client.getDREF("sim/flightmodel/position/local_vz")[0]
        # local frame is East-Up-South and we convert to North-East-Down
        vel_vec = np.array([-vz, vx, -vy]).T

        return np.matmul(R, vel_vec)

    def _get_home_heading(self):
        """
        Get the value of the aircraft's heading in degrees from the runway
        """
        true_heading = self.client.getDREF("sim/flightmodel/position/psi")[0]
        return true_heading - self._glideslope_heading

    def _get_local_heading(self):
        """
        Get the value of the aircraft's heading in degrees from the Z axis
        """
        return self.client.getDREF("sim/flightmodel/position/psi")[0]

    def _get_home_xy(self):
        """
        Get the aircraft's current x and y position and heading in the
        home frame. The x-value represents crosstrack error,the y-value represents
        downtrack position, and theta is the heading error.
        """

        local_x = self.client.getDREF("sim/flightmodel/position/local_x")[0]
        local_z = self.client.getDREF("sim/flightmodel/position/local_z")[0]

        return self._opengl_zx_to_xy(local_z, local_x)
