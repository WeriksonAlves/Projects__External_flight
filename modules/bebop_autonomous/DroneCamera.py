import rospy
from bebop_msgs.msg import Ardrone3CameraStateOrientation
from cv_bridge import CvBridge
from dynamic_reconfigure.msg import ConfigDescription, Config
from functools import wraps
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Empty, Float32
from typing import Callable, List, Optional, Tuple

import cv2
import os
import numpy as np


# Reusing the log_decorator for logging entry/exit points of functions
def log_decorator(func: Callable) -> Callable:
    """Decorator for logging function entry and exit."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        rospy.loginfo(f"Entering {func.__name__}")
        result = func(*args, **kwargs)
        rospy.loginfo(f"Exiting {func.__name__}")
        return result
    return wrapper


class DroneCamera:
    """
    DroneCamera handles camera operations, including capturing images,
    managing camera orientation, and controlling exposure settings via ROS
    topics.
    """

    def __init__(self, file_path: str):
        """
        Initialize the DroneCamera class with publishers, subscribers, and
        image handling.

        :param file_path: Path to save captured images.
        """
        self.file_path = file_path
        self.image_data = {"image": None, "image_compressed": None,
                           "image_compressed_depth": None,
                           "image_theora": None}
        self.success_flags = {"image": False, "image_compressed": False,
                              "image_compressed_depth": False,
                              "image_theora": False, "isOpened": False}
        self.current_tilt = 0.0
        self.current_pan = 0.0
        self.param_listener = ParameterListener(self)
        self.bridge = CvBridge()

        self.pubs = {}
        self.subs = {}

    def init_publishers(self, topics: List[str]):
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

    def init_subscribers(self, topics: List[str]):
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

    # @log_decorator
    def _process_raw_image(self, data: Image) -> None:
        """Process and save raw image data."""
        self.__save_and_load_image(data, "image_raw.png", "image",
                                   use_cv_bridge=True)

    # @log_decorator
    def _process_compressed_image(self, data: CompressedImage) -> None:
        """Process and save compressed image data."""
        self.__save_and_load_image(data, "image_compressed.png",
                                   "image_compressed")

    # @log_decorator
    def _process_compressed_depth_image(self, data: CompressedImage) -> None:
        """Process and save compressed depth image data."""
        self.__save_and_load_image(data, "image_compressed_depth.png",
                                   "image_compressed_depth")

    # @log_decorator
    def _process_theora_image(self, data: CompressedImage) -> None:
        """Process and save Theora-encoded image data."""
        self.__save_and_load_image(data, "image_theora.png", "image_theora")

    @log_decorator
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
            self.success_flags[img_type] = self._save_image(image, img_path)
            self.image_data[img_type] = self._load_image(img_path)
        except (cv2.error, ValueError) as e:
            rospy.logerr(f"Failed to process {img_type} image: {e}")

    # @log_decorator
    def _process_camera_orientation(self, data: Ardrone3CameraStateOrientation
                                    ) -> None:
        """Process camera orientation changes."""
        self.current_tilt = data.tilt
        self.current_pan = data.pan

    @log_decorator
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

    @log_decorator
    def take_snapshot(self) -> None:
        """Command the drone to take a snapshot."""
        self.snapshot_pub.publish(Empty())

    def set_exposure(self, exposure_value: float) -> None:
        """
        Set the camera's exposure value.

        :param exposure_value: A float value to adjust exposure (-3.0 to 3.0).
        """
        try:
            exposure_msg = Float32(data=exposure_value)
            self.set_exposure_pub.publish(exposure_msg)
            rospy.loginfo(f"Exposure set to {exposure_value}")
        except Exception as e:
            rospy.logerr(f"Failed to set exposure: {e}")

    def _save_image(self, image: np.ndarray, filename: str) -> bool:
        """Save an image to disk."""
        if image is not None:
            cv2.imwrite(filename, image)
            return True
        return False

    def _load_image(self, filename: str) -> Optional[np.ndarray]:
        """Load an image from a file."""
        return cv2.imread(filename)

    def start_camera_stream(self) -> None:
        """Start the camera stream and keep the node active."""
        rospy.spin()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Simulate the behavior of OpenCV's read() method, returning the latest available image.

        :return: (bool, np.ndarray): A tuple where the first element is a boolean indicating if an image was successfully read, 
                                      and the second element is the image array (if available).
        """
        for img_type, img_data in self.image_data.items():
            if img_data is not None:
                return True, img_data
        return False, None

    def isOpened(self) -> bool:
        """
        Check if the ROS image topics are publishing or if the camera is
        active.
        :return: bool: True if the camera is operational, False otherwise.
        """
        return self.success_flags["isOpened"]

    def release(self) -> None:
        """
        Simulate the behavior of OpenCV's release() method, which would
        release the camera.
        Here, it resets the internal state indicating the camera is no longer
        active.
        """
        rospy.loginfo("Releasing camera resources")
        self.success_flags = {key: False for key in self.success_flags}
        self.image_data = {key: None for key in self.image_data}

    def centralize_operator(self, frame: np.ndarray,
                            bounding_box: Tuple[int, int, int, int],
                            drone_pitch: float = 0.0, drone_yaw: float = 0.0
                            ) -> Tuple[float, float]:
        """
        Adjusts the camera's orientation to center the detected person in the
        frame, compensating for yaw and pitch.

        :param frame: The captured frame.
        :param bounding_box: The bounding box of the person as (x, y, width,
        height).
        :param drone_pitch: The pitch value to compensate for.
        :param drone_yaw: The yaw value to compensate for.
        :return: Tuple containing the horizontal and vertical distance to the
        center of the frame.
        """
        frame_height, frame_width = frame.shape[:2]
        frame_center = (frame_width // 2, frame_height // 2)

        box_x, box_y, _, _ = bounding_box
        dist_to_center_h = (box_x - frame_center[0]
                            ) / frame_center[0] - drone_yaw
        dist_to_center_v = (box_y - frame_center[1]
                            ) / frame_center[1] - drone_pitch

        return dist_to_center_h, dist_to_center_v

    def _ajust_camera(self, frame: np.ndarray, boxes: np.ndarray,
                      Gi: tuple[int, int], Ge: tuple[int, int]
                      ) -> Tuple[float, float]:
        """
        Adjusts the camera orientation based on the operator's position in the
        frame. Returns the pitch and yaw adjustments required.
        """
        dist_center_h, dist_center_v = self.centralize_operator(frame, boxes)
        sc_pitch = np.tanh(-dist_center_v * Gi[0]) * Ge[0] if np.abs(
            dist_center_v) > 0.25 else 0
        sc_yaw = np.tanh(dist_center_h * Gi[1]) * Ge[1] if np.abs(
            dist_center_h) > 0.25 else 0
        return sc_pitch, sc_yaw


class ParameterListener:
    """
    Listens to parameter updates for dynamic reconfiguration of camera
    parameters.
    """

    def __init__(self, drone_camera: DroneCamera) -> None:
        """
        Initializes the listener and sets up ROS subscribers for parameter
        updates.
        """
        self.drone_camera = drone_camera
        rospy.Subscriber(
            "/bebop/image_raw/compressed/parameter_descriptions",
            ConfigDescription,
            self._callback_param_desc
        )
        rospy.Subscriber(
            "/bebop/image_raw/compressed/parameter_updates",
            Config,
            self._callback_param_update
        )

    @log_decorator
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

    @log_decorator
    def _callback_param_update(self, data: Config) -> None:
        """
        Callback for parameter updates. Processes updated parameter values and
        updates camera settings.
        """
        self._update_parameters(data.doubles)

    def _update_parameters(self, parameters: list) -> None:
        """
        Updates camera settings based on received parameter updates.
        """
        for param in parameters:
            if param.name == "compression_quality":
                self._update_compression_quality(param.value)

    @log_decorator
    def _update_compression_quality(self, value: float) -> None:
        """
        Updates the compression quality of the drone camera.
        """
        rospy.loginfo(f"Updating compression quality to {value}")
        self.drone_camera.compression_quality = value
