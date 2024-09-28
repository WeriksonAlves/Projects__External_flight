#!/usr/bin/env python
import rospy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32
from bebop_msgs.msg import Ardrone3CameraStateOrientation
from dynamic_reconfigure.msg import Config, ConfigDescription
from typing import List, Optional, Tuple


class DroneCamera:
    """
    DroneCamera class handles camera operations, including capturing images, managing camera orientation,
    and controlling exposure settings.

    ROS Topics:
        /bebop/image_raw
        /bebop/image_raw/compressed
        /bebop/camera_control
        /bebop/states/ardrone3/CameraState/Orientation
        /bebop/set_exposure
        /bebop/snapshot
    """

    def __init__(self, file_path: str):
        """
        Initialize the DroneCamera class and set up necessary publishers and subscribers.

        :param file_path: The path to save images.
        """

        self.file_path = file_path
        self.image_data = {
            "image": None,
            "image_compressed": None,
            "image_compressed_depth": None,
            "image_theora": None,
        }

        self.success_flags = {
            "image": False,
            "image_compressed": False,
            "image_compressed_depth": False,
            "image_theora": False,
            "isOpened": False
        }

        self.current_tilt: float = 0.0
        self.current_pan: float = 0.0

        self.param_listener = ParameterListener(self)
        self.bridge = CvBridge()
        
    def initialize_publishers(self, topics: List[str] = ['camera_control', 'snapshot', 'set_exposure']):
        """
        Initialize publishers for camera control, snapshot, and exposure settings.
        
        :param topics: A list of topics to initialize publishers for.
        """
        if 'camera_control' in topics: self.camera_control_pub = rospy.Publisher('/bebop/camera_control', Twist, queue_size=10)
        if 'snapshot' in topics: self.snapshot_pub = rospy.Publisher('/bebop/snapshot', Empty, queue_size=10)
        if 'set_exposure' in topics: self.set_exposure_pub = rospy.Publisher('/bebop/set_exposure', Float32, queue_size=10)

    def initialize_subscribers(self, topics: List[str] = ['image', 'compressed', 'depth', 'theora']):
        """
        Initialize subscribers for image and camera orientation topics.
        
        :param topics: A list of topics to initialize subscriber for
        """
        if 'image' in topics: rospy.Subscriber("/bebop/image_raw", Image, self._process_raw_image)
        if 'compressed' in topics: rospy.Subscriber("/bebop/image_raw/compressed", CompressedImage, self._process_compressed_image)
        if 'depth' in topics: rospy.Subscriber("/bebop/image_raw/compressedDepth", CompressedImage, self._process_compressed_depth_image)
        if 'theora' in topics: rospy.Subscriber("/bebop/image_raw/theora", CompressedImage, self._process_theora_image)

        rospy.Subscriber("/bebop/states/ardrone3/CameraState/Orientation", Ardrone3CameraStateOrientation, self._process_camera_orientation)

    def _process_raw_image(self, data: Image) -> None:
        """Process and save raw image data."""
        self._save_and_load_image(data, "image_raw.png", "image", use_cv_bridge=True)

    def _process_compressed_image(self, data: CompressedImage) -> None:
        """Process and save compressed image data."""
        self._save_and_load_image(data, "image_compressed.png", "image_compressed")

    def _process_compressed_depth_image(self, data: CompressedImage) -> None:
        """Process and save compressed depth image data."""
        self._save_and_load_image(data, "image_compressed_depth.png", "image_compressed_depth")

    def _process_theora_image(self, data: CompressedImage) -> None:
        """Process and save Theora encoded image data."""
        self._save_and_load_image(data, "image_theora.png", "image_theora")

    def _save_and_load_image(self, data: CompressedImage, filename: str, img_type: str, use_cv_bridge=False) -> None:
        """Save the image data to a file and load it."""
        try:
            if use_cv_bridge:
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            else:
                np_arr = np.frombuffer(data.data, np.uint8)
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            self.success_flags[img_type] = self._save_image(image, os.path.join(self.file_path, filename))
            self.image_data[img_type] = self._load_image(os.path.join(self.file_path, filename))
        
        except Exception as e:
            rospy.logerr(f"Failed to process {img_type} image: {e}")

    def _process_camera_orientation(self, data: Ardrone3CameraStateOrientation) -> None:
        """Update the camera orientation state."""
        self.current_tilt = data.tilt
        self.current_pan = data.pan

    def move_camera(self, tilt=0.0, pan=0.0, drone_pitch=0.0, drone_yaw=0.0) -> None:
        """
        Control the camera orientation, compensating for drone pitch and yaw.

        :param tilt: Vertical movement of the camera.
        :param pan: Horizontal movement of the camera.
        :param drone_pitch: Compensate for drone pitch.
        :param drone_yaw: Compensate for drone yaw.
        """
        camera_control_msg = Twist()
        camera_control_msg.angular.y = tilt - drone_pitch
        camera_control_msg.angular.z = pan - drone_yaw
        self.camera_control_pub.publish(camera_control_msg)
        rospy.loginfo(f"Moving camera - Tilt: {tilt - drone_pitch}, Pan: {pan - drone_yaw}")

    def take_snapshot(self) -> None:
        """Command the drone to take a snapshot."""
        self.snapshot_pub.publish(Empty())
        rospy.loginfo("Snapshot command sent")

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
        Check if the ROS image topics are publishing or if the camera is active.
        :return: bool: True if the camera is operational, False otherwise.
        """
        return self.success_flags["isOpened"]
    
    def release(self) -> None:
        """
        Simulate the behavior of OpenCV's release() method, which would release the camera.
        Here, it resets the internal state indicating the camera is no longer active.
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
    Listens to parameter updates for dynamic reconfiguration of camera parameters.
    """

    def __init__(self, drone_camera: DroneCamera) -> None:
        self.drone_camera = drone_camera
        rospy.Subscriber("/bebop/image_raw/compressed/parameter_descriptions", ConfigDescription, self._callback_param_desc)
        rospy.Subscriber("/bebop/image_raw/compressed/parameter_updates", Config, self._callback_param_update)

    def _callback_param_desc(self, data: ConfigDescription) -> None:
        """Log the parameter description data."""
        for group in data.groups:
            rospy.loginfo(f"Parameter group: {group.name}")
            for param in group.parameters:
                rospy.loginfo(f"  Parameter: {param.name}, Type: {param.type}")

    def _callback_param_update(self, data: Config) -> None:
        """Process parameter updates and adjust internal parameters."""
        for param in data.doubles:
            rospy.loginfo(f"Update - Parameter: {param.name}, Value: {param.value}")
            if param.name == "compression_quality":
                rospy.loginfo(f"Changing compression quality to {param.value}")
                self.drone_camera.compression_quality = param.value
