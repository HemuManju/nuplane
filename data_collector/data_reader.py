import numpy as np
import imageio as iio

from pathlib import Path

import pandas as pd

import webdataset as wds
import torch

from itertools import islice

from .helpers import get_nonexistant_path


class Replay:
    def __init__(self, config):
        self.cfg = config
        self.n_collision = 0

    def _get_unique_name(self, file_name, write_path):
        fname_path = Path(write_path, file_name).with_suffix('.mp4')
        save_path = get_nonexistant_path(fname_path=fname_path)
        return save_path

    def _create_movie(self, samples, file_name, write_path):
        save_path = self._get_unique_name(file_name, write_path)
        writer = iio.get_writer(save_path, format='FFMPEG', mode='I', codec='mpeg4')

        for i, sample in enumerate(samples):
            if type(sample['jpeg']) is list:
                array = torch.stack(sample['jpeg'], dim=0).cpu().detach().numpy()
            else:
                array = sample['jpeg'].cpu().detach().numpy()
            array = np.swapaxes(array, 0, 2) * 255  # Scaling co-efficient

            # Write the array
            writer.append_data(np.flipud(array).astype(np.uint8))
        writer.close()


class Summary:
    def __init__(self, config):
        self.cfg = config

    def summarize(self, samples):
        data = []
        for sample in samples:
            data.append(sample['json'])

        df = pd.DataFrame(data)
        print(df.groupby('modified_direction').count())
        print(df.describe())


class WebDatasetReader:
    def __init__(self, config, file_path) -> None:
        self.file_path = file_path
        self.cfg = config
        self.replay = Replay(config)
        self.sink = None

    def _concatenate_samples(self, samples):
        combined_data = {
            k: [d.get(k) for d in samples if k in d] for k in set().union(*samples)
        }
        return combined_data

    def _generate_seqs(self, src, nsamples=3):
        it = iter(src)
        result = tuple(islice(it, nsamples))
        if len(result) == nsamples:
            yield self._concatenate_samples(result)
        for elem in it:
            result = result[1:] + (elem,)
            yield self._concatenate_samples(result)

    def create_movie(self, file_name=None, write_path=None):
        # Get the samples
        samples = self.get_dataset()

        if write_path is None:
            write_path = Path(self.file_path).parent

        if file_name is None:
            file_name = Path(self.file_path).stem

        self.replay._create_movie(samples, file_name, write_path)

    def get_dataset(self, concat_n_samples=None):
        if concat_n_samples is None:
            dataset = wds.WebDataset(self.file_path).decode("torchrgb")
        else:
            dataset = (
                wds.WebDataset(self.file_path)
                .decode("torchrgb")
                .then(self._generate_seqs, concat_n_samples)
            )
        return dataset

    def get_dataloader(self, num_workers, batch_size, concat_n_samples=None):
        # Get the dataset
        dataset = self.get_dataset(concat_n_samples=concat_n_samples)
        dataloader = torch.utils.data.DataLoader(
            dataset.batched(batch_size), num_workers=num_workers, batch_size=None
        )
        return dataloader
