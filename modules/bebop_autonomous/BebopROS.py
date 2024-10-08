import os
import rospy
from functools import wraps
from .DroneCamera import DroneCamera


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
    def __init__(self):
        """
        Initializes BebopROS with a predefined file path for saving images and
        drone type.
        """
        self.file_path = os.path.join(os.path.dirname(__file__), 'images')
        self.drone_type = 'bebop2'
        self.camera = None

    @ensure_directory_exists
    def VideoCapture(self) -> bool:
        """
        Initializes the drone camera, sets up publishers and subscribers,
        and marks the camera as opened if successful.

        Returns:
            bool: True if the camera was successfully opened, False otherwise.
        """
        try:
            # Initialize the DroneCamera with the specified file path
            self.camera = DroneCamera(self.file_path)
            self._initialize_camera_communication()
            self.camera.success_flags["isOpened"] = True
            rospy.loginfo("Camera successfully initialized and opened.")
        except Exception as e:
            rospy.logerr(f"Error during VideoCapture initialization: {e}")
            if self.camera:
                self.camera.success_flags["isOpened"] = False
        return self.camera.success_flags.get("isOpened", False)

    def _initialize_camera_communication(self) -> None:
        """
        Sets up the necessary publishers and subscribers for drone
        communication.
        """
        publishers = ['camera_control', 'snapshot', 'set_exposure']
        subscribers = ['compressed']
        self.camera.init_publishers(publishers)
        self.camera.init_subscribers(subscribers)
        rospy.loginfo(f"Initialized publishers: {publishers}")
        rospy.loginfo(f"Initialized subscribers: {subscribers}")

    def close_camera(self) -> None:
        """
        Safely closes the camera and cleans up resources.
        """
        if self.camera:
            self.camera.close()
            rospy.loginfo("Camera closed successfully.")
