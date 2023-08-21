"""
Here are defined all the XPlane sensors
"""

import copy
import math
import numpy as np

# ==================================================================================================
# -- BaseSensor -----------------------------------------------------------------------------------
# ==================================================================================================


class BaseSensor(object):
    def __init__(self, name, attributes, interface, parent):
        self.name = name
        self.attributes = attributes
        self.interface = interface
        self.parent = parent

        self.interface.register(self.name, self)

    def is_event_sensor(self):
        return False

    def parse(self):
        raise NotImplementedError

    def update_sensor(self, data, frame):
        if not self.is_event_sensor():
            self.interface._data_buffers.put((self.name, frame, self.parse(data)))
        else:
            self.interface._event_data_buffers.put((self.name, frame, self.parse(data)))

    def callback(self, data):
        self.update_sensor(data, data.frame)

    def destroy(self):
        raise NotImplementedError


class XPlaneSensor(BaseSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def destroy(self):
        if self.sensor is not None:
            self.sensor.destroy()
            self.sensor = None


class PseudoSensor(BaseSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def callback(self, data, frame):
        self.update_sensor(data, frame)


# ==================================================================================================
# -- Cameras -----------------------------------------------------------------------------------
# ==================================================================================================
class BaseCamera(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def parse(self, sensor_data):
        """Parses the Image into an numpy array"""
        # sensor_data: [fov, height, width, raw_data]
        array = np.frombuffer(sensor_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (sensor_data.height, sensor_data.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        return array


class CameraRGB(BaseCamera):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class CameraDepth(BaseCamera):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class CameraSemanticSegmentation(BaseCamera):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


# ==================================================================================================
# -- LIDAR -----------------------------------------------------------------------------------
# ==================================================================================================
class Lidar(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class SemanticLidar(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


# ==================================================================================================
# -- Others -----------------------------------------------------------------------------------
# ==================================================================================================


class Gnss(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class Imu(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class LaneInvasion(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True


class Collision(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)


class Obstacle(XPlaneSensor):
    def __init__(self, name, attributes, interface, parent):
        super().__init__(name, attributes, interface, parent)

    def is_event_sensor(self):
        return True
