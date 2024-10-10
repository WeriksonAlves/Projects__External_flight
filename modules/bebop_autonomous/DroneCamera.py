from bebop_msgs.msg import Ardrone3CameraStateOrientation
from cv_bridge import CvBridge
from dynamic_reconfigure.msg import ConfigDescription, Config
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Empty, Float32
from typing import List, Optional, Tuple

import cv2
import numpy as np
import os
import rospy


class DroneCamera():
    """
    DroneCamera handles camera operations, including capturing images,
    managing camera orientation, and controlling exposure settings via ROS
    topics.
    """

    def __init__(self,
                 file_path: str = os.path.dirname(__file__)):
        """
        Initialize the DroneCamera class with publishers, subscribers, and
        image handling.

        Attributes:
            file_path: Path to save captured images.
            image_data: Dictionary to store image data.
            success_flags: Dictionary to track successful image captures.
            current_tilt: Current vertical camera orientation.
            current_pan: Current horizontal camera orientation.
            param_listener: Listener for dynamic reconfiguration.
            bridge: Bridge for converting ROS messages to OpenCV images.
            pubs: Dictionary of ROS publishers.
            subs: Dictionary of ROS subscribers.
        """
        self.file_path = file_path

        self.image_data = {"image": None, "compressed": None,
                           "depth": None,
                           "theora": None}
        self.success_flags = {"image": False, "compressed": False,
                              "depth": False,
                              "theora": False, "isOpened": False}
        self.current_tilt = 0.0
        self.current_pan = 0.0
        self.param_listener = ParameterListener(self)
        self.bridge = CvBridge()

        self.pubs = {}
        self.subs = {}

    """Section 1: Initializing the camera topics."""

    def init_publishers(self, topics: List[str]) -> None:
        """Initialize publishers for the given ROS topics."""
        for topic in topics:
            if topic == 'camera_control':
                self.pubs['camera_control'] = rospy.Publisher(
                    '/bebop/camera_control', Twist, queue_size=10)
            elif topic == 'snapshot':
                self.pubs['snapshot'] = rospy.Publisher('/bebop/snapshot',
                                                        Empty, queue_size=10)
            elif topic == 'set_exposure':
                self.pubs['set_exposure'] = rospy.Publisher(
                    '/bebop/set_exposure', Float32, queue_size=10)

    def init_subscribers(self, topics: List[str]) -> None:
        """Initialize subscribers for the given ROS topics."""
        topic_map = {
            'image': ("/bebop/image_raw", Image, self._process_raw_image),
            'compressed': ("/bebop/image_raw/compressed", CompressedImage,
                           self._process_compressed_image),
            'depth': ("/bebop/image_raw/compressedDepth", CompressedImage,
                      self._process_compressed_depth_image),
            'theora': ("/bebop/image_raw/theora", CompressedImage,
                       self._process_theora_image)
        }
        for topic in topics:
            if topic in topic_map:
                topic_name, msg_type, callback = topic_map[topic]
                self.subs[topic] = rospy.Subscriber(
                    topic_name,
                    msg_type,
                    callback
                )
        # Separate camera orientation subscriber
        self.subs['camera_orientation'] = rospy.Subscriber(
            "/bebop/states/ardrone3/CameraState/Orientation",
            Ardrone3CameraStateOrientation,
            self._process_camera_orientation
        )
        # Enable parameter listener for dynamic reconfiguration
        self.param_listener.init_subscribers(topics)

    def _process_raw_image(self, data: Image) -> None:
        """Process and save raw image data."""
        self.__save_and_load_image(data, "image_raw.png", "image",
                                   use_cv_bridge=True)

    def _process_compressed_image(self, data: CompressedImage) -> None:
        """Process and save compressed image data."""
        self.__save_and_load_image(data, "compressed.png",
                                   "compressed")

    def _process_compressed_depth_image(self, data: CompressedImage) -> None:
        """Process and save compressed depth image data."""
        self.__save_and_load_image(data, "depth.png",
                                   "depth")

    def _process_theora_image(self, data: CompressedImage) -> None:
        """Process and save Theora-encoded image data."""
        self.__save_and_load_image(data, "theora.png", "theora")

    def _process_camera_orientation(self, data: Ardrone3CameraStateOrientation
                                    ) -> None:
        """Process camera orientation changes."""
        self.current_tilt = data.tilt
        self.current_pan = data.pan

    def __save_and_load_image(self, data: CompressedImage, filename: str,
                              img_type: str, use_cv_bridge=False) -> None:
        """Save the image data to a file and load it."""
        try:
            # Handle image decoding based on whether CvBridge is used
            image = self.bridge.imgmsg_to_cv2(
                data, "bgr8") if use_cv_bridge else cv2.imdecode(
                    np.frombuffer(data.data, np.uint8), cv2.IMREAD_COLOR)

            # Save and load the image, update success flags
            img_path = os.path.join(self.file_path, filename)
            self.success_flags[img_type] = self.__save_image(image, img_path)
            self.image_data[img_type] = self.__load_image(img_path)
        except (cv2.error, ValueError) as e:
            rospy.logerr(f"Failed to process {img_type} image: {e}")

    def __save_image(self, image: np.ndarray, filename: str) -> bool:
        """Save an image to disk."""
        if image is not None:
            return cv2.imwrite(filename, image)
        return False

    def __load_image(self, filename: str) -> Optional[np.ndarray]:
        """Load an image from a file."""
        return cv2.imread(filename)

    """Section 2: Camera control methods"""

    def move_camera(self, tilt=0.0, pan=0.0, drone_pitch=0.0, drone_yaw=0.0
                    ) -> None:
        """
        Control the camera orientation, compensating for drone pitch and yaw.

        :param tilt: Vertical camera movement.
        :param pan: Horizontal camera movement.
        :param drone_pitch: Compensation for drone pitch.
        :param drone_yaw: Compensation for drone yaw.
        """
        camera_control_msg = Twist()
        camera_control_msg.angular.y = tilt - drone_pitch
        camera_control_msg.angular.z = pan - drone_yaw
        self.pubs['camera_control'].publish(camera_control_msg)

    def take_snapshot(self) -> None:
        """Command the drone to take a snapshot."""
        self.pubs['snapshot'].publish(Empty())

    def set_exposure(self, exposure_value: float) -> None:
        """
        Set the camera's exposure value.

        :param exposure_value: A float value to adjust exposure (-3.0 to 3.0).
        """
        exposure_msg = Float32(data=exposure_value)
        self.pubs['set_exposure'].publish(exposure_msg)

    """Section 3: Implementing OpenCV-like methods."""

    def isOpened(self) -> bool:
        """
        Check if the ROS image topics are publishing or if the camera is
        active.

        :return: bool: True if the camera is operational, False otherwise.
        """
        return self.success_flags["isOpened"]

    def read(self, subscriber: str = 'compressed'
             ) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Reads the latest image data from the camera.

        :param subscriber: The subscriber to read from.
        :return: Tuple[bool, Optional[np.ndarray]]: A tuple containing a
            boolean indicating success and the image data.
        """
        if self.success_flags[subscriber]:
            return True, self.image_data[subscriber]
        return False, None

    def release(self) -> None:
        """
        Simulate the behavior of OpenCV's release() method, resetting the
        camera's internal state.
        """
        self.success_flags = {key: False for key in self.success_flags}
        self.image_data = {key: None for key in self.image_data}
        rospy.loginfo("Camera resources released.")


class ParameterListener:
    """
    Listens to parameter updates for dynamic reconfiguration of camera
    parameters.

    Attributes:
        drone_camera: The DroneCamera instance to update.
        subs: Dictionary of ROS subscribers.
    """

    def __init__(self, drone_camera: DroneCamera) -> None:
        """
        Initializes the listener and sets up ROS subscribers for parameter
        updates.
        """
        self.drone_camera = drone_camera
        self.subs = {}

    def init_subscribers(self, topics: List[str]) -> None:
        """Initialize subscribers for the given ROS topics."""
        topic_map = {
            'compressed_description': (
                "/bebop/image_raw/compressed/parameter_descriptions",
                ConfigDescription, self._callback_param_desc),
            'compressed_update': (
                "/bebop/image_raw/compressed/parameter_updates",
                Config, self._callback_param_update)
        }
        for topic in topics:
            if topic in topic_map:
                topic_name, msg_type, callback = topic_map[topic]
                self.subs[topic] = rospy.Subscriber(topic_name,
                                                    msg_type,
                                                    callback)

    def _callback_param_desc(self, data: ConfigDescription) -> None:
        """
        Callback for parameter descriptions. Logs the parameter groups and
        their respective details. Optimized by reducing redundant logs.
        """
        for group in data.groups:
            rospy.loginfo(f"Parameter group: {group.name}")
            for param in group.parameters:
                rospy.logdebug(
                    f"  Parameter: {param.name}, Type: {param.type}"
                )

    def _callback_param_update(self, data: Config) -> None:
        """
        Callback for parameter updates. Processes updated parameter values and
        updates camera settings.
        """
        self.__update_parameters(data.doubles)

    def __update_parameters(self, parameters: list) -> None:
        """
        Updates camera settings based on received parameter updates.
        """
        for param in parameters:
            if param.name == "compression_quality":
                self.__update_compression_quality(param.value)

    def __update_compression_quality(self, value: float) -> None:
        """
        Updates the compression quality of the drone camera.
        """
        rospy.loginfo(f"Updating compression quality to {value}")
        self.drone_camera.compression_quality = value
