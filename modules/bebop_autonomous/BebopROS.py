import os
import numpy as np
from typing import Tuple
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
        

    @staticmethod
    def _centralize_operator(frame: np.ndarray, bounding_box: Tuple[
        int, int, int, int], drone_pitch: float = 0.0, drone_yaw: float = 0.0
    ) -> Tuple[float, float]:
        """
        Adjust the camera's orientation to center the operator in the frame,
        compensating for yaw and pitch.

        :param frame: The captured frame.
        :param bounding_box: The bounding box of the operator as (x, y, width,
        height).
        :param drone_pitch: The pitch value for compensation.
        :param drone_yaw: The yaw value for compensation.
        :return: Tuple[float, float]: The horizontal and vertical distance to
        the frame's center.
        """
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)

        box_x, box_y, _, _ = bounding_box
        dist_to_center_h = (
            box_x - frame_center[0]) / frame_center[0] - drone_yaw
        dist_to_center_v = (
            box_y - frame_center[1]) / frame_center[1] - drone_pitch

        return dist_to_center_h, dist_to_center_v

    @staticmethod
    def ajust_camera(frame: np.ndarray, boxes: np.ndarray, Gi: Tuple[
        int, int], Ge: Tuple[int, int]
    ) -> Tuple[float, float]:
        """
        Adjust the camera orientation based on the operator's position in the
        frame.

        :param frame: The captured frame.
        :param boxes: Bounding boxes for detected objects.
        :param Gi: Internal gain for pitch and yaw adjustment.
        :param Ge: External gain for pitch and yaw adjustment.
        :return: Tuple[float, float]: Pitch and yaw adjustments.
        """
        dist_center_h, dist_center_v = BebopROS._centralize_operator(frame, boxes)
        sc_pitch = np.tanh(-dist_center_v * Gi[0]) * Ge[0] if abs(
            dist_center_v) > 0.25 else 0
        sc_yaw = np.tanh(dist_center_h * Gi[1]) * Ge[1] if abs(
            dist_center_h) > 0.25 else 0
        print(sc_pitch, sc_yaw)
