class PreProcessData:
    def __init__(self, config):
        self.cfg = config

    def process(self):
        """Function to do all the pre processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        raise NotImplementedError
