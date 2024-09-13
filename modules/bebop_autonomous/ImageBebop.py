import cv2
import os
import rospy

import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from dynamic_reconfigure.msg import Config, ConfigDescription
from sensor_msgs.msg import Image, CompressedImage
from typing import Tuple, Optional


class ImageListener:
    """
    ImageProcessor is responsible for handling raw, compressed, and Theora encoded images in a ROS environment.
    It provides methods for converting ROS image messages to OpenCV images, saving, and loading images.

    :param file_path: Path to the directory where images will be saved.

    Attributes:
    - bridge: CvBridge object for ROS to OpenCV image conversion.
    - image: Raw image as a numpy array.
    - image_compressed: Compressed image as a numpy array.
    - image_compressed_depth: Compressed depth image as a numpy array.
    - image_theora: Theora encoded image as a numpy array.
    - file_path: Path to the directory where images will be saved.
    - success_image: Flag indicating if the raw image was successfully processed.
    - success_compressed_image: Flag indicating if the compressed image was successfully processed.
    - success_compressed_depth_image: Flag indicating if the compressed depth image was successfully processed.
    - success_theora_image: Flag indicating if the Theora encoded image was successfully processed.
    """

    def __init__(self, file_path: str) -> None:
        """
        Initializes the ImageProcessor class, setting up the CvBridge for ROS to OpenCV image conversion.
        """
        self.bridge = CvBridge()
        self.image: Optional[np.ndarray] = None
        self.image_compressed: Optional[np.ndarray] = None
        self.image_compressed_depth: Optional[np.ndarray] = None
        self.image_theora: Optional[np.ndarray] = None
        self.file_path: str = file_path
        self.success_image: Optional[bool] = False
        self.success_compressed_image: Optional[bool] = False
        self.success_compressed_depth_image: Optional[bool] = False
        self.success_theora_image: Optional[bool] = False

    def subscribe_to_image_raw(self) -> None:
        """
        Subscribes to the raw image topic.
        """
        rospy.Subscriber("/bebop/image_raw", Image, self._process_raw_image)

    def subscribe_to_image_raw_compressed(self) -> None:
        """
        Subscribes to the compressed image topic.
        """
        rospy.Subscriber("/bebop/image_raw/compressed", CompressedImage, self._process_compressed_image)

    def subscribe_to_image_raw_compressed_Depth(self) -> None:
        """
        Subscribes to the compressed depth image topic.
        """
        rospy.Subscriber("/bebop/image_raw/compressedDepth", CompressedImage, self._process_compressed_depth_image)

    def subscribe_to_image_raw_theora(self) -> None:
        """
        Subscribes to the Theora encoded image topic.
        """
        rospy.Subscriber("/bebop/image_raw/theora", CompressedImage, self._process_theora_image)

    def _process_raw_image(self, data: Image) -> None:
        """
        Processes raw image data and saves it to disk.

        :param data: ROS Image message containing the raw image.
        """
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.success_image, self.image = self._save_and_load_image(
                image, os.path.join(self.file_path, "image_raw.png"))
        except CvBridgeError as e:
            rospy.logerr(f"Failed to convert raw image: {e}")

    def _process_compressed_image(self, data: CompressedImage) -> None:
        """
        Processes compressed image data and saves it to disk.

        :param data: ROS CompressedImage message containing the compressed image.
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

        :param data: ROS CompressedImage message containing the compressed depth image.
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

        :param data: ROS CompressedImage message containing the Theora encoded image.
        """
        try:
            np_arr = np.frombuffer(data.data, np.uint8)
            theora_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.success_theora_image, self.image_theora = self._save_and_load_image(
                theora_image, os.path.join(self.file_path, "image_theora.png"))
        except Exception as e:
            rospy.logerr(f"Failed to decode Theora image: {e}")

    def _save_image(self, image: np.ndarray, filename: str) -> bool:
        """
        Saves an image to a specified file.

        :param image: Image to be saved.
        :param filename: Path to the file where the image will be saved.
        :return: True if the image is saved successfully, False otherwise.
        """
        if image is not None:
            cv2.imwrite(filename, image)
            return True
        return False

    def _load_image(self, filename: str) -> Optional[np.ndarray]:
        """
        Loads an image from a file.

        :param filename: Path to the image file.
        :return: Loaded image as a numpy array or None if loading fails.
        """
        return cv2.imread(filename)

    def _save_and_load_image(self, image: np.ndarray, filename: str) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Saves an image to disk and then loads it back.

        :param image: Image to be saved.
        :param filename: Path to the file where the image will be saved.
        :return: Tuple containing a success flag and the loaded image.
        """
        if self._save_image(image, filename):
            loaded_image = self._load_image(filename)
            return True, loaded_image
        return False, None


class ParameterListener:
    def __init__(self, image_processor: ImageListener) -> None:
        self.image_processor: ImageListener = image_processor
        # Repetir o mesmo para outros tópicos de depth e theora

    def subscribe_to_parameter_descriptions(self) -> None:
        """
        Subscribes to the parameter descriptions topic.
        """
        rospy.Subscriber("/bebop/image_raw/compressed/parameter_descriptions", ConfigDescription, self._callback_param_desc)

    def subscribe_to_parameter_updates(self) -> None:
        """
        Subscribes to the parameter updates topic.
        """
        rospy.Subscriber("/bebop/image_raw/compressed/parameter_updates", Config, self._callback_param_update)

    def _callback_param_desc(self, data: ConfigDescription) -> None:
        """
        Callback for processing parameter descriptions.
        """
        for group in data.groups:
            print(f"Parameter group: {group.name}")
            for param in group.parameters:
                print(f"  Parameter: {param.name}, Type: {param.type}, Level: {param.level}")

    def _callback_param_update(self, data: Config) -> None:
        """
        Callback to process parameter updates.
        """
        for param in data.doubles:
            rospy.loginfo(f"Update - Parameter: {param.name}, Value: {param.value}")
            # Ajuste baseado no parâmetro, por exemplo:
            if param.name == "compression_quality":
                rospy.loginfo(f"Changing the compression quality to {param.value}")
                # Pode usar isso para influenciar o processamento da imagem
                self.image_processor.compression_quality = param.value

