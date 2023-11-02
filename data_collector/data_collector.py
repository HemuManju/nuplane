import random
import os
import yaml
from tqdm import tqdm
import itertools

import json

import multiprocessing
import pandas as pd


from nuplane.core import kill_all_servers
from nuplane.env import NUPlaneEnv

from .experiments.simple_experiment import SimpleExperiment
from .controllers.xplane_controllers import SinusoidController

from .helpers import create_directory, inspect_config


def save_configuration(config, client, write_path):
    # Get file configuration
    client_config = inspect_config(client)

    # Add the configuration
    for key in config:
        if key not in ['experiment']:
            client_config[key + '_config'] = config[key]

    # Save the configuration
    file_name = config['experiment']['town']
    save_path = write_path + file_name + '_configuration.json'
    with open(save_path, 'w') as fp:
        json.dump(client_config, fp, indent=4)
    fp.close()
    return None


def run_trajectory(
    nuplane_env,
    controller,
    estimator,
    episode_params,
    monitor=None,
    data_recorder=None,
    debug=False,
):
    observation, reward, done, info = nuplane_env.reset(episode_params)
    controller.reset()

    if data_recorder is not None:
        data_recorder.reset()

    while not done:
        if data_recorder is not None:
            data_recorder.record(nuplane_env, observation)

        if monitor is not None:
            anomaly_score, estop = monitor.monitor(observation)
        else:
            estop = False

        control = controller.get_control(observation)

        observation, reward, done, info = nuplane_env.step(control)

    return observation, done


class DataCollector:
    def __init__(self, config, write_path, navigation_type=None, writer=None):
        self.cfg = config
        self.write_path = write_path
        # Setup the server
        self.nuplane_env = NUPlaneEnv(config, debug=True)
        self.controller = SinusoidController(
            config['controller']['steering'], config['controller']['speed'], 1.0
        )
        return None

    def write_loop(self, file_name=None):
        run_trajectory(
            self.nuplane_env, self.controller, estimator=None, episode_params=None
        )

        return None

    def collect(self):
        """
        Main loop of the simulation. It handles updating all the HUD information,
        ticking the agent.
        """
        try:
            # Iterate over weather and behavior
            combinations = list(
                itertools.product(
                    self.cfg['experiment']['weather'], self.cfg['vehicle']['behavior']
                )
            )
            for weather, behavior in tqdm(combinations):
                self.server.set_weather(weather)

                # Setup the agent behavior
                self.agent_manager.setup_agent(behavior)

                # Get the new file name
                file_name = '_'.join(
                    [self.cfg['experiment']['town'], weather, behavior]
                )

                # Run the simulation
                self.write_loop(file_name)

            # Finally close the writer
            self.writer.close()

        except KeyboardInterrupt:
            self.writer.close()
            kill_all_servers()
            print('-' * 16 + 'Data collection interrupted' + '-' * 16)

        finally:
            print('-' * 16 + 'Finished data collection' + '-' * 16)
            kill_all_servers()
