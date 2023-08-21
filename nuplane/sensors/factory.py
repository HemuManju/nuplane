from .sensor import (
    CameraRGB,
    CameraDepth,
    CameraSemanticSegmentation,
    Lidar,
    SemanticLidar,
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
            "sensor.lidar.ray_cast": Lidar,
            "sensor.lidar.ray_cast_semantic": SemanticLidar,
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
