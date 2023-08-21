import collections.abc
import os
import glob
import sys

import math

import numpy as np
from scipy.special import comb


# Geology constants
R = 6371000  # Radius of third rock from the sun, in metres
FT = 12 * 0.0254  # 1 FOOT = 12 INCHES
NAUTICAL_MILE = 1.852  # Nautical mile in meters 6076.118ft=1nm

try:
    sys.path.append(
        glob.glob(
            '../XPlane/dist/XPlane-*%d.%d-%s.egg'
            % (
                sys.version_info.major,
                sys.version_info.minor,
                'win-amd64' if os.name == 'nt' else 'linux-x86_64',
            )
        )[0]
    )
except IndexError:
    pass

from datetime import datetime
import re
import socket


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


def get_bearing(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(
        dLon
    )
    brng = np.rad2deg(math.atan2(y, x))
    if brng < 0:
        brng += 360
    return brng


def linear_refine_implicit(x, n):
    """Given a 2D ndarray (npt, m) of npt coordinates in m dimension,
    insert 2**(n-1) additional points on each trajectory segment
    Returns an (npt*2**(n-1), m) ndarray

    Parameters
    ----------
    x : array
        A 2D input array
    n : int
        Number of intermediate points to insert between two consecutive points in x

    Returns
    -------
    array
        An array with interploated points

    Raises
    ------
    NotImplementedError
        The functions is not implemented for 3D or higher dimensions
    ValueError
        Number of intermediate points should be greated than zero
    """
    if n > 1:
        m = 0.5 * (x[:-1] + x[1:])
        if x.ndim == 2:
            msize = (x.shape[0] + m.shape[0], x.shape[1])
        else:
            raise NotImplementedError

        x_new = np.empty(msize, dtype=x.dtype)
        x_new[0::2] = x
        x_new[1::2] = m
        return linear_refine_implicit(x_new, n - 1)
    elif n == 1:
        return x
    else:
        raise ValueError


def haversine_distance(p1, p2):  # in radians.
    try:
        lat1, lat2 = math.radians(p1['lat']), math.radians(p2['lat'])
        long1, long2 = math.radians(p1['lon']), math.radians(p2['lon'])
    except KeyError:
        lat1, lat2 = math.radians(p1['x']), math.radians(p2['x'])
        long1, long2 = math.radians(p1['y']), math.radians(p2['y'])

    dlat, dlong = lat2 - lat1, long2 - long1
    a = math.pow(math.sin(dlat / 2), 2) + math.cos(lat1) * math.cos(lat2) * math.pow(
        math.sin(dlong / 2), 2
    )
    return 2 * R * math.asin(math.sqrt(a))


def vec_haversine_distance(lat1, long1, lat2, long2):  # in radians.
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [long1, lat1, long2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def rotate_to_local(x, y):
    """Rotate to the local coordinate frame.

    Args:
        x: x-value in home coordinate frame
        y: y-value in home coordinate frame
    """
    rotx = 0.583055934597441 * x + -0.8124320138514389 * y
    roty = 0.8124320138514389 * x + 0.583055934597441 * y
    return rotx, roty


def rotate_to_home(x, y):
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


def home_to_local(x, y, start_x, start_y):
    # Rotate back
    rotx, roty = rotate_to_local(x, y)

    transx = start_x - rotx
    transy = start_y - roty

    return transx, transy


def local_to_home(x, y, start_x, start_y):
    """Get the home coordinates of the aircraft from the local coordinates.

    Args:
        x: x-value in the local coordinate frame
        y: y-value in the local coordinate frame
    """

    transx = start_x - x
    transy = start_y - y

    # Rotate to align runway with y axis
    rotx, roty = rotate_to_home(transx, transy)
    return rotx, roty


def bernstein(i, n, t):
    """
    The i-th Bernstein polynomial of degree n
    """
    return comb(n, i) * (t ** (n - i)) * (1 - t) ** i


def _weighted_bezier_curve(points, weights, n_points=50):
    node_x = np.array([n[0] for n in points])
    node_y = np.array([n[1] for n in points])
    weights = np.array(weights)

    t = np.linspace(0.0, 1.0, n_points)
    weighted_bernstein = np.array(
        [bernstein(i, len(points) - 1, t) * weights[i] for i in range(0, len(points))]
    )

    sum_weighted_bernstein = np.sum(weighted_bernstein, axis=0)

    p_x = np.divide(np.dot(node_x, weighted_bernstein), sum_weighted_bernstein)
    p_y = np.divide(np.dot(node_y, weighted_bernstein), sum_weighted_bernstein)
    return p_x, p_y


def bezier_curve(points, weights=None, n_points=200):
    """
    Given a set of control points, return the
    bezier curve defined by the control points.

    points should be a list of lists, or list of tuples
    such as [ [1,1],
              [2,3],
              [4,5], ..[Xn, Yn] ]
     nTimes is the number of time steps, defaults to 1000

     See http://processingjs.nihongoresources.com/bezierinfo/
    """

    if weights is None:
        node_x = np.array([n[0] for n in points])
        node_y = np.array([n[1] for n in points])
        t = np.linspace(0.0, 1.0, n_points)
        numerator = np.array(
            [bernstein(i, len(points) - 1, t) for i in range(0, len(points))]
        )
        p_x = np.dot(node_x, numerator)
        p_y = np.dot(node_y, numerator)
        return p_x, p_y
    else:
        if n_points is None:
            return _weighted_bezier_curve(points, weights)
        else:
            return _weighted_bezier_curve(points, weights, n_points=n_points)
