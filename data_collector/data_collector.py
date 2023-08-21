import random
import os
from tqdm import tqdm
import itertools

import json

import multiprocessing
import pandas as pd


from nuplane.core import kill_all_servers

from .pre_process import PreProcessData
from .data_writer import WebDatasetWriter

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


class AgentManager:
    def __init__(self, config, server):
        self.cfg = config
        self.server = server
        self.world = None
        self.steps = 0
        self.behavior = 'normal'

        return None

    def setup_agent(self, behavior=None):
        raise NotImplementedError

    def collect_data(self, pre_process=None):
        control = self.agent.run_step()

        # Get different kinds of data
        vehicle_data = self.agent.get_vehicle_data(control)
        traffic_data = self.agent.get_traffic_data()
        waypoint_data = self.agent.get_waypoint_data()
        collision_data = self.agent.get_vehicle_collision_data()
        sensor_data = self.server.step(control)

        if pre_process is not None:
            data = pre_process.process(
                sensor_data,
                waypoint_data,
                vehicle_data=vehicle_data,
                traffic_data=traffic_data,
                collision_data=collision_data,
            )
        else:
            data = {
                **sensor_data,
                **waypoint_data,
                **traffic_data,
                **vehicle_data,
                **collision_data,
            }
        return data


class DataCollector:
    def __init__(self, config, write_path, navigation_type=None, writer=None):
        self.cfg = config
        if navigation_type is None:
            navigation_type = 'straight'
        self.write_path = write_path + '/' + navigation_type + '/'

        if navigation_type == 'navigation':
            self.cfg['collector']['steps'] = 300000

        # Setup the server
        self.server = None

        # Setup agent, writer and preprocessor
        self.agent_manager = AgentManager(config=self.cfg, server=self.server)
        self.pre_process = PreProcessData(config=self.cfg)

        # Writer
        if writer is None:
            self.writer = WebDatasetWriter(config=self.cfg)
        else:
            self.writer = writer

        # Create a directory and save the configuration
        create_directory(self.write_path)

        # Save the configuration
        client = self.agent_manager.server.get_client()
        save_configuration(self.cfg, client, self.write_path)

        return None

    def write_loop(self, file_name):
        # Create the tar file
        self.writer.create_tar_file(file_name, self.write_path)

        steps = self.cfg['collector']['steps']
        for i in range(steps):
            # Collect the data from agent
            data = self.agent_manager.collect_data(self.pre_process)

            # Write data at regular intervals
            if i % self.cfg['data_writer']['data_write_freq'] == 0:
                self.writer.write(data, i)

            # Reset if collision has happened
            if data['collision'] or self.agent_manager.agent.done():
                self.agent_manager.setup_agent()
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


class ParallelDataCollector:
    def __init__(self, config, write_path, number_collectors=1):
        self.cfg = config
        self.write_path = write_path
        self.number_collector = number_collectors
        return None

    def single_instance_collector(self, weather, behavior, navigation_type, semaphore):
        # Set the weather and agent behavior
        data_collector = DataCollector(self.cfg, self.write_path, navigation_type)
        data_collector.server.set_weather(weather)
        data_collector.agent_manager.setup_agent(behavior)

        # Get the new file name
        file_name = '_'.join([self.cfg['experiment']['town'], weather, behavior])

        # Run the simulation
        data_collector.write_loop(file_name)
        data_collector.writer.close()
        data_collector.server.close()
        print('-' * 16 + 'Process Done' + '-' * 16)

        semaphore.release()
        return None

    def collect(self):
        try:
            concurrency = self.number_collector
            semaphore = multiprocessing.Semaphore(concurrency)
            all_processes = []
            for weather, behavior, navigation_type in itertools.product(
                self.cfg['experiment']['weather'],
                self.cfg['vehicle']['behavior'],
                self.cfg['experiment']['navigation_type'],
            ):
                semaphore.acquire()
                p = multiprocessing.Process(
                    target=self.single_instance_collector,
                    args=(weather, behavior, navigation_type, semaphore),
                )
                all_processes.append(p)
                p.start()

            for p in all_processes:
                p.join()

        except KeyboardInterrupt:
            self.writer.close()
            kill_all_servers()
            print('-' * 16 + 'Data collection interrupted' + '-' * 16)

        finally:
            print('-' * 16 + 'Finished data collection' + '-' * 16)
            kill_all_servers()
