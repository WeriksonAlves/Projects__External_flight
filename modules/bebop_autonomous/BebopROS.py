from .DroneCamera import DroneCamera
from .DroneControl import DroneControl
from functools import wraps
from geometry_msgs.msg import Twist
from typing import List, Optional, Tuple

import os
import numpy as np
import rospy


def ensure_directory_exists(method):
    """Decorator to ensure the image directory exists before proceeding."""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not os.path.exists(self.file_path):
            os.makedirs(self.file_path)
            rospy.loginfo(f"Created directory: {self.file_path}")
        return method(self, *args, **kwargs)
    return wrapper


class BebopROS:
    """
    Class responsible for interacting with the Bebop2 drone, capturing video,
    and handling communication with ROS (Robot Operating System).
    """

    def __init__(self, drone_type="Bebop2", ip_address=None):
        """
        Initializes BebopROS with a predefined file path for saving images and
        drone type.
        """
        self.main_dir = os.path.dirname(__file__)
        self.file_path = os.path.join(self.main_dir, 'images')
        self.drone_type = drone_type
        self.simulation = True
        self.camera = DroneCamera(self.file_path, 10)
        self.control = DroneControl()

    """Section 1: Bebop camera-related methods"""

    @ensure_directory_exists
    def VideoCapture(self) -> DroneCamera:
        """
        Initializes the drone camera, sets up publishers and subscribers,
        and marks the camera as opened if successful.

        :return: DroneCamera: The initialized camera object.
        """
        try:
            self._init_camera_topics()
            self.camera.success_flags["isOpened"] = True
            rospy.loginfo("Camera successfully initialized and opened.")
        except Exception as e:
            self.camera.success_flags["isOpened"] = False
            rospy.logerr(f"Error during VideoCapture initialization: {e}")
        return self.camera

    def ajust_camera(self, frame: np.ndarray, bounding_box: np.ndarray,
                     Gi: Tuple[int, int] = (0.5, 0.5), Ge: Tuple[int, int] = (
                         50, 50)) -> None:
        """
        Adjust the camera orientation based on the operator's position in the
        frame.

        :param frame: The captured frame.
        :param boxes: Bounding boxes for detected objects.
        :param Gi: Internal gain for pitch and yaw adjustment.
        :param Ge: External gain for pitch and yaw adjustment.
        """
        dist_center_h, dist_center_v = self._centralize_operator(
            frame, bounding_box)
        sc_pitch = np.tanh(-dist_center_v * Gi[0]) * Ge[0] if abs(
            dist_center_v) > 0.25 else 0
        sc_yaw = np.tanh(dist_center_h * Gi[1]) * Ge[1] if abs(
            dist_center_h) > 0.25 else 0
        print(f"pitch: {sc_pitch}, yaw: {sc_yaw}")
        self.camera.move_camera(tilt=sc_pitch, pan=sc_yaw)

    def _init_camera_topics(self) -> None:
        """
        Sets up the necessary publishers and subscribers for drone
        communication.
        """
        publishers = ['camera_control', 'snapshot', 'set_exposure']
        subscribers = ['compressed', 'compressed_description',
                       'compressed_update']
        self.camera.init_publishers(publishers)
        self.camera.init_subscribers(subscribers)
        rospy.loginfo(f"Initialized publishers: {publishers}")
        rospy.loginfo(f"Initialized subscribers: {subscribers}")
        self.camera_initialized = True

    def _centralize_operator(self, frame: np.ndarray, bounding_box: Tuple[
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

    """Section 2: Bebop action-related methods"""

    def send_command_uav(self, command: str, simulation: bool = True) -> None:
        """
        Sends a command to the UAV.
        """
        if command == 'L':
            rospy.loginfo("Sending land command to Bebop.")
            if not simulation:
                self.land()
        elif command == 'I':
            rospy.loginfo("Sending inspection command to Bebop.")
            if not simulation:
                self.control.inspection()
        if command == 'F':
            rospy.loginfo("Sending follow me command to Bebop.")
            if not simulation:
                self.control.follow_me()
        elif command == 'P':
            rospy.loginfo("Sending photography command to Bebop.")
            if not simulation:
                self.control.take_picture()
        elif command == 'T':
            rospy.loginfo("Sending takeoff command to Bebop.")
            if not simulation:
                self.takeoff()

    
    """Section 3: Bebop control-related methods"""

    def land(self) -> None:
        """
        Sends a landing command to the drone.
        """
        self.control.land()

    def takeoff(self) -> None:
        """
        Sends a takeoff command to the drone.
        """
        self.control.takeoff()
    
