import math
import time

from nuplane.actors import BaseActor
from nuplane.utils.transform import *


class AIAircraft(BaseActor):
    def __init__(self, client, config, id=1) -> None:
        super().__init__(client, config)
        self.client = client
        self.config = config

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

        self.client = client
        self.config = config

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
