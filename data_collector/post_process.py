import numpy as np
import imageio as iio

from utils import get_nonexistant_path


class PostProcess:
    def __init__(self, config):
        self.cfg = config
        self.n_collision = 0

    def preprocess_sensor_data(self, sensor_data):
        # Check if there is a collision
        if 'collision' in sensor_data.keys():
            self.n_collision += 1

    def process(self):
        """Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        raise NotImplementedError


class Replay:
    def __init__(self, config):
        self.cfg = config
        self.n_collision = 0

    def _get_unique_name(self, write_path):
        fname_path = write_path + 'video.mp4'
        save_path = get_nonexistant_path(fname_path=fname_path)
        return save_path

    def create_movie(self, samples, write_path):
        save_path = self._get_unique_name(write_path)
        writer = iio.get_writer(save_path, format='FFMPEG', mode='I')

        for sample in samples:
            array = sample['jpeg'].cpu().detach().numpy()
            array = np.swapaxes(array, 0, 2) * 255  # Scaling co-efficient

            # Write the array
            writer.append_data(np.flipud(array).astype(np.uint8))
        writer.close()
