import os
from .DroneCamera import DroneCamera


class BebopROS:
    """
    Class responsible for interacting with the Bebop2 drone, capturing video,
    and handling communication with ROS (Robot Operating System).
    """
    MAIN_DIR = os.path.dirname(__file__)

    def __init__(self):
        """
        Initializes BebopROS with a predefined file path for saving images and
        drone type.
        """
        self.file_path = os.path.join(self.MAIN_DIR, 'images')
        self.drone_type = 'bebop2'
        self.camera = DroneCamera(self.file_path)
