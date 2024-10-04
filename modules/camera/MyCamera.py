import cv2
from typing import Union
from ..bebop_autonomous.BebopROS import BebopROS


class MyCamera():
    """
    MyCamera class for initializing camera or drone-based video
    capture configuration.

    Parameters:
    - source: Union[int, str, BebopROS] - Camera source or drone instance
    - fps: int - Frames per second for the capture
    - dist: float - Distance parameter (if relevant)
    - length: int - Length parameter (if relevant)
    """
    def __init__(
        self, source: Union[int, str, BebopROS], fps: int = 5,
        dist: float = 0.025, length: int = 15
    ) -> None:
        if hasattr(source, 'drone_type'):
            source.VideoCapture()
            self.cap = source.camera  # ROS-based camera
        else:
            self.cap = cv2.VideoCapture(source)  # OpenCV-based camera

        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open camera.")

        self.fps = fps
        self.dist = dist
        self.length = length

    def __del__(self):
        """ Release the camera when object is destroyed. """
        if hasattr(self.cap, 'release'):
            self.cap.release()
