import os
import random
import signal
import subprocess
import time
import psutil

from .xpc3 import XPlaneConnect

from .sensors.sensor_interface import SensorInterface
from .sensors.factory import SensorFactory

from .maps import Map

from .utils.transform import join_dicts

BASE_CORE_CONFIG = {
    "host": "localhost",  # Client host
    "timeout": 10.0,  # Timeout of the client
    "timestep": 0.05,  # Time step of the simulation
    "retries_on_error": 20,  # Number of tries to connect to the client
    "resolution_x": 600,  # Width of the server spectator camera
    "resolution_y": 600,  # Height of the server spectator camera
    "show_display": False,  # Whether or not the server will be displayed
}


def is_used(port):
    """Checks whether or not a port is used"""
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    """Kill all PIDs that start with XPlane"""
    processes = [p for p in psutil.process_iter() if 'Main Thread' in p.name()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class XPlaneCore:
    """
    Class responsible of handling all the different XPlane functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """

    def __init__(self, config={}, debug=True):
        """Initialize the server and client"""
        self.client = None
        self.world = None
        self.map = None
        self.hero = None
        self.config = join_dicts(BASE_CORE_CONFIG, config)
        self.sensor_interface = SensorInterface()

        if not debug:
            self.init_server()

        # Connect to client
        self.connect_client()

        # Setup the map
        self.map = Map(self.client, config)

    def init_server(self):
        """Start a server on a random port"""
        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))
        xplane_bin = os.path.expanduser(
            f"{self.config['xplane_server']['xplane_path']}/X-Plane-x86_64"
        )
        if not os.path.exists(xplane_bin):
            xplane_bin_orig = xplane_bin
            xplane_bin = xplane_bin.replace('X-Plane-11', "X-Plane 11")
            if not os.path.exists(xplane_bin):
                raise ValueError(
                    f"Could not find X-Plane binary at: \n{xplane_bin_orig} or \n{xplane_bin}"
                )

        server_command = [
            xplane_bin,
            f"--windowed={self.config['xplane_server']['resolution_x']}x{self.config['xplane_server']['resolution_y']}",
            "--no_sound",
        ]

        self.process = subprocess.Popen(
            server_command,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """Connect to the client"""

        for i in range(self.config["retries_on_error"]):
            try:
                self.client = XPlaneConnect()

                # Create a timer
                self.t_start = self.client.getDREF("sim/time/local_time_sec")[0]

                return self.client
            except Exception as e:
                print(
                    f"Waiting for server to be ready: {e}, attempt {i+1} of {self.config['retries_on_error']}"
                )
                time.sleep(3)

        raise Exception(
            "Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the XPlane configuration"
        )

    def setup_experiment(self, experiment_config, hero=None):
        """Initialize the hero and sensors"""
        # Set weather and time of the day
        self.client.sendDREF(
            'sim/weather/cloud_type[0]', experiment_config['cloud_type']
        )
        self.client.sendDREF(
            'sim/time/zulu_time_sec', experiment_config['time_of_the_day'] * 60 * 60
        )

    def reset(self, *args, **kwargs):
        self.client.pauseSim(True)
        self.client.sendDREFs(["sim/operation/fix_all_systems"], [1])
        self.client.pauseSim(False)

    def apply_action(self, control):
        """Applies the control calcualted at the experiment to the hero"""

        if control is not None:
            self.hero.apply_action(control)

        return None

    def tick(self):
        """Performs one tick of the simulation, moving all actors, and getting the sensor data"""

        # Return the sensor data
        return self.get_sensor_data()

    def get_sensor_data(self):
        """Returns the data sent by the different sensors at this tick"""
        sensor_data = self.sensor_interface.get_data()

        return sensor_data

    def close_simulation(self):
        raise NotImplementedError
