#!/usr/bin/env python
import rospy
import cv2
import os
import numpy as np
# from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32
from bebop_msgs.msg import Ardrone3CameraStateOrientation  # Import the correct message type for camera orientation
from dynamic_reconfigure.msg import Config, ConfigDescription
from typing import Tuple, Optional


class DroneCamera:
    """
    DroneCamera
        Purpose: 
            This class handles camera operations, including capturing raw images, managing camera orientation, and controlling exposure settings.

        Topics:
            /bebop/image_raw
            /bebop/image_raw/compressed
            /bebop/camera_control
            /bebop/states/ardrone3/CameraState/Orientation
            /bebop/set_exposure
            /bebop/snapshot

        Responsibilities:
            Capture and process images for gesture recognition.
            Interface with any image compression, depth sensing, or stream encoding.
            Handle camera orientation and exposure settings.
    """

    def __init__(self, file_path: str):
        """
        Initializes the DroneCamera class, setting up subscribers and camera controls.
        """
        rospy.init_node('bebop_drone_camera', anonymous=True)

        # CvBridge object for ROS to OpenCV image conversion
        # self.bridge = CvBridge()
        
        # Image storage
        self.image: Optional[np.ndarray] = None
        self.image_compressed: Optional[np.ndarray] = None
        self.image_compressed_depth: Optional[np.ndarray] = None
        self.image_theora: Optional[np.ndarray] = None
        self.file_path: str = file_path
        self.success_image: Optional[bool] = False
        self.success_compressed_image: Optional[bool] = False
        self.success_compressed_depth_image: Optional[bool] = False
        self.success_theora_image: Optional[bool] = False

        # Camera control publishers
        self.camera_control_pub = rospy.Publisher('/bebop/camera_control', Twist, queue_size=10)
        self.snapshot_pub = rospy.Publisher('/bebop/snapshot', Empty, queue_size=10)
        self.set_exposure_pub = rospy.Publisher('/bebop/set_exposure', Float32, queue_size=10)

        # Initialize camera orientation state variables
        self.current_tilt: float = 0.0
        self.current_pan: float = 0.0

        # Subscribing to image topics
        # rospy.Subscriber("/bebop/image_raw", Image, self._process_raw_image)
        rospy.Subscriber("/bebop/image_raw/compressed", CompressedImage, self._process_compressed_image)
        rospy.Subscriber("/bebop/image_raw/compressedDepth", CompressedImage, self._process_compressed_depth_image)
        rospy.Subscriber("/bebop/image_raw/theora", CompressedImage, self._process_theora_image)

        # Subscribing to camera orientation state
        rospy.Subscriber("/bebop/states/ardrone3/CameraState/Orientation", Ardrone3CameraStateOrientation, self._process_camera_orientation)

        # Initialize parameter listener
        self.param_listener = ParameterListener(self)

    # def _process_raw_image(self, data: Image) -> None:
    #     """
    #     Processes raw image data and saves it to disk.
    #     """
    #     try:
    #         image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #         self.success_image, self.image = self._save_and_load_image(
    #             image, os.path.join(self.file_path, "image_raw.png"))
    #     except CvBridgeError as e:
    #         rospy.logerr(f"Failed to convert raw image: {e}")

    def _process_compressed_image(self, data: CompressedImage) -> None:
        """
        Processes compressed image data and saves it to disk.
        """
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            compressed_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.success_compressed_image, self.image_compressed = self._save_and_load_image(
                compressed_image, os.path.join(self.file_path, "image_compressed.png"))
        except Exception as e:
            rospy.logerr(f"Failed to decode compressed image: {e}")

    def _process_compressed_depth_image(self, data: CompressedImage) -> None:
        """
        Processes compressed depth image data and saves it to disk.
        """
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            compressed_depth_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.success_compressed_depth_image, self.image_compressed_depth = self._save_and_load_image(
                compressed_depth_image, os.path.join(self.file_path, "image_compressed_depth.png"))
        except Exception as e:
            rospy.logerr(f"Failed to decode compressed depth image: {e}")

    def _process_theora_image(self, data: CompressedImage) -> None:
        """
        Processes Theora encoded image data and saves it to disk.
        """
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            theora_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.success_theora_image, self.image_theora = self._save_and_load_image(
                theora_image, os.path.join(self.file_path, "image_theora.png"))
        except Exception as e:
            rospy.logerr(f"Failed to decode Theora image: {e}")

    def _process_camera_orientation(self, data: Ardrone3CameraStateOrientation) -> None:
        """
        Processes camera orientation data received from /bebop/states/ardrone3/CameraState/Orientation.
        """
        self.current_tilt = data.tilt
        self.current_pan = data.pan
        rospy.loginfo(f"Camera Orientation - Tilt: {self.current_tilt}, Pan: {self.current_pan}")

    def move_camera(self, tilt=0.0, pan=0.0):
        """
        Controls the camera orientation.
        :param tilt: The tilt value to set for the camera (vertical movement).
        :param pan: The pan value to set for the camera (horizontal movement).
        """
        camera_control_msg = Twist()
        camera_control_msg.angular.y = tilt  # Tilt the camera
        camera_control_msg.angular.z = pan   # Pan the camera
        self.camera_control_pub.publish(camera_control_msg)
        rospy.loginfo(f"Moving camera - Tilt: {tilt}, Pan: {pan}")

    def take_snapshot(self):
        """
        Commands the drone to take a snapshot.
        """
        self.snapshot_pub.publish(Empty())
        rospy.loginfo("Snapshot command sent")

    def set_exposure(self, exposure_value):
        """
        Sets the camera's exposure to the specified value.

        :param exposure_value: Exposure value to be set [-3.0, +3.0].
        """
        try:
            exposure_msg = Float32()
            exposure_msg.data = exposure_value
            self.set_exposure_pub.publish(exposure_msg)
            rospy.loginfo(f"Exposure set to {exposure_value}")
        except Exception as e:
            rospy.logerr(f"Failed to set exposure: {e}")

    def _save_image(self, image: np.ndarray, filename: str) -> bool:
        """
        Saves an image to a specified file.
        """
        if image is not None:
            cv2.imwrite(filename, image)
            return True
        return False

    def _load_image(self, filename: str) -> Optional[np.ndarray]:
        """
        Loads an image from a file.
        """
        return cv2.imread(filename)

    def _save_and_load_image(self, image: np.ndarray, filename: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Saves an image to disk and loads it back.
        """
        if self._save_image(image, filename):
            loaded_image = self._load_image(filename)
            return True, loaded_image
        return False, None

    def start_camera_stream(self):
        """
        Starts the camera stream and processes incoming images.
        """
        rospy.spin()  # Keep the node running to receive images


class ParameterListener:
    def __init__(self, drone_camera: DroneCamera) -> None:
        """
        Initializes the ParameterListener, which listens to parameter description and updates for the drone's camera system.
        """
        self.drone_camera: DroneCamera = drone_camera

        rospy.Subscriber("/bebop/image_raw/compressed/parameter_descriptions", ConfigDescription, self._callback_param_desc)
        rospy.Subscriber("/bebop/image_raw/compressed/parameter_updates", Config, self._callback_param_update)

    def _callback_param_desc(self, data: ConfigDescription) -> None:
        """
        Callback for processing parameter descriptions.
        """
        for group in data.groups:
            rospy.loginfo(f"Parameter group: {group.name}")
            for param in group.parameters:
                rospy.loginfo(f"  Parameter: {param.name}, Type: {param.type}, Level: {param.level}")

    def _callback_param_update(self, data: Config) -> None:
        """
        Callback to process parameter updates.
        """
        for param in data.doubles:
            rospy.loginfo(f"Update - Parameter: {param.name}, Value: {param.value}")
            # Adjust image processing parameters based on the parameter updates
            if param.name == "compression_quality":
                rospy.loginfo(f"Changing the compression quality to {param.value}")
                # Example: Change an internal compression quality setting
                self.drone_camera.compression_quality = param.value

