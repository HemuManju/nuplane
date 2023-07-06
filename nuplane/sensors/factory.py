#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from .sensor import (
    CameraRGB,
    CameraDepth,
    CameraSemanticSegmentation,
    CameraDVS,
    Lidar,
    SemanticLidar,
    Radar,
    Gnss,
    Imu,
    LaneInvasion,
    Collision,
    Obstacle,
)


class SensorFactory(object):
    """
    Class to simplify the creation of the different XPlane sensors
    """

    @staticmethod
    def spawn(name, attributes, interface, parent):
        attributes = attributes.copy()
        type_ = attributes.get("type", "")

        # Dictionary of sensors. Add a new key corresponding to a new sensor
        sensors = {
            "sensor.camera.rgb": CameraRGB,
            "sensor.camera.depth": CameraDepth,
            "sensor.camera.semantic_segmentation": CameraSemanticSegmentation,
            "sensor.camera.dvs": CameraDVS,
            "sensor.lidar.ray_cast": Lidar,
            "sensor.lidar.ray_cast_semantic": SemanticLidar,
            "sensor.other.radar": Radar,
            "sensor.other.gnss": Gnss,
            "sensor.other.imu": Imu,
            "sensor.other.lane_invasion": LaneInvasion,
            "sensor.other.collision": Collision,
            "sensor.other.obstacle": Obstacle,
        }

        if type_ in sensors.keys():
            return sensors[type_](name, attributes, interface, parent)
        else:
            raise RuntimeError("Sensor of type {} not supported".format(type_))
