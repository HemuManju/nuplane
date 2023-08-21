import os
import json
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy


import numpy as np

from PIL import Image as im
import matplotlib.pyplot as plt
import webdataset as wds

from .helpers import (
    get_nonexistant_shard_path,
    get_nonexistant_path,
)


def default(obj):
    if type(obj).__module__ == np.__name__:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj.item()
    raise TypeError('Unknown type:', type(obj))


class WebDatasetWriter:
    def __init__(self, config) -> None:
        self.cfg = config
        self.sink = None
        jsonpickle_numpy.register_handlers()

    def _is_jsonable(self, x):
        try:
            json.dumps(x, default=default)
            return True
        except (TypeError, OverflowError):
            return False

    def _get_serializable_data(self, data):
        keys_to_delete = []
        for key, value in data.items():
            if not self._is_jsonable(value):
                keys_to_delete.append(key)
        for key in keys_to_delete:
            del data[key]
        return data

    def create_tar_file(self, file_name, write_path):
        # Check if file already exists, increment if so
        if self.cfg['data_writer']['shard_write']:
            path_to_file = write_path + file_name + '_%06d.tar'
            shard_start_index = get_nonexistant_shard_path(path_to_file)
        else:
            path_to_file = write_path + file_name + '.tar'

        # Create a folder
        write_path = get_nonexistant_path(path_to_file)

        # Create a tar file
        if self.cfg['data_writer']['shard_write']:
            max_count = self.cfg['data_writer']['shard_maxcount']
            self.sink = wds.ShardWriter(write_path, maxcount=max_count, compress=True)
        else:
            self.sink = wds.TarWriter(write_path, compress=True)

    def sample(self, data, index):
        image_data = im.fromarray(data['rgb'])
        del data['rgb']  # No longer needed

        # Semseg data
        data['semseg'] = data['semseg'][:, :, 0].flatten().tolist()
        encoded_data = jsonpickle.encode(data)

        return {
            "__key__": "sample%06d" % index,
            'jpeg': image_data,
            'json': encoded_data,
        }

    def write(self, data, index):
        if self.sink is None:
            raise FileNotFoundError(
                'Please call create_tar_file() method before calling the write method'
            )
        self.sink.write(self.sample(data, index))

    def close(self):
        self.sink.close()


class AutoEncoderDatasetWriter(WebDatasetWriter):
    def __init__(self, config) -> None:
        super().__init__(config)

    def sample(self, data, index):
        image_front = im.fromarray(data['rgb_front'])
        del data['rgb_front']  # No longer needed

        image_back = im.fromarray(data['rgb_back'])
        del data['rgb_back']  # No longer needed

        image_right = im.fromarray(data['rgb_right'])
        del data['rgb_right']  # No longer needed

        image_left = im.fromarray(data['rgb_left'])
        del data['rgb_left']  # No longer needed

        # Other data
        encoded_data = jsonpickle.encode(data)

        return {
            "__key__": "sample%06d" % index,
            'front.jpeg': image_front,
            'back.jpeg': image_back,
            'right.jpeg': image_right,
            'left.jpeg': image_left,
            'json': encoded_data,
        }
