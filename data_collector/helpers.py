import os

from pathlib import Path
import natsort

import collections.abc

import cv2
import numpy as np


def inspect_config(client):
    raise NotImplementedError


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: image
    """
    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.float32)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def get_nonexistant_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.exists(fname_path):
        return fname_path
    filename, file_extension = os.path.splitext(fname_path)
    i = 1
    new_fname = "{}_{}{}".format(filename, i, file_extension)
    while os.path.exists(new_fname):
        i += 1
        new_fname = "{}_{}{}".format(filename, i, file_extension)
    return new_fname


def get_nonexistant_shard_path(fname_path):
    """
    Get the path to a filename which does not exist by incrementing path.

    Examples
    --------
    >>> get_nonexistant_path('/etc/issue')
    '/etc/issue-1'
    >>> get_nonexistant_path('whatever/1337bla.py')
    'whatever/1337bla.py'
    """
    if not os.path.isfile(fname_path % 0):
        return fname_path
    start_index = 1
    while os.path.exists(fname_path % start_index):
        start_index += 1
    return start_index


def create_directory(write_path):
    if not os.path.exists(write_path):
        # Create a new directory because it does not exist
        os.makedirs(write_path)
        print("Created new data directory!")


def find_tar_files(read_path, pattern):
    files = [str(f) for f in Path(read_path).glob('*.tar') if f.match(pattern + '*')]
    return natsort.natsorted(files)


def rotateToHome(x, y):
    """Rotate to the home coordinate frame.

    Home coordinate frame starts at (0,0) at the start of the runway
    and ends at (0, 2982 at the end of the runway). Thus, the x-value
    in the home coordinate frame corresponds to crosstrack error and
    the y-value corresponds to downtrack position.

    Args:
        x: x-value in local coordinate frame
        y: y-value in local coordinate frame
    """
    rotx = 0.583055934597441 * x + 0.8124320138514389 * y
    roty = -0.8124320138514389 * x + 0.583055934597441 * y
    return rotx, roty


def localToHome(x, y):
    """Get the home coordinates of the aircraft from the local coordinates.

    Args:
        x: x-value in the local coordinate frame
        y: y-value in the local coordinate frame
    """

    # Translate to make start x and y the origin
    startX, startY = getStartXY()
    transx = startX - x
    transy = startY - y

    # Rotate to align runway with y axis
    rotx, roty = rotateToHome(transx, transy)
    return rotx, roty
