"""
Class-related topics:
ok /bebop/image_raw
ok /bebop/image_raw/compressed
    /bebop/image_raw/compressed/parameter_descriptions
    /bebop/image_raw/compressed/parameter_updates
ok /bebop/image_raw/compressedDepth
    /bebop/image_raw/compressedDepth/parameter_descriptions
    /bebop/image_raw/compressedDepth/parameter_updates
ok /bebop/image_raw/theora
    /bebop/image_raw/theora/parameter_descriptions
    /bebop/image_raw/theora/parameter_updates
"""


import cv2
import rospy

import numpy as np

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CompressedImage
from typing import Tuple

class ImageRawTools(object):
    """
    ImageRawTools is a class that provides tools for handling raw, compressed, and Theora encoded images in a ROS environment. It includes methods for converting ROS image messages to OpenCV images, saving images to files, and loading images from files.
        sucess_read (bool): Indicates whether the image was successfully read.
    Methods:
        __init__() -> None:
        
        
        __save_image(image: np.ndarray, filename: str) -> bool:
        __load_image(filename: str) -> np.ndarray:
        __read(image: np.ndarray, filename: str) -> Tuple[bool, np.ndarray]:
            :param image: The image to save and read.
            :param filename: The filename to save and read the image to.
    """

    def __init__(self) -> None:
        """
        Initializes the ImageRawTools class.

        Attributes:
            bridge (CvBridge): An instance of CvBridge for converting ROS images to OpenCV images.
            image (None): Placeholder for storing the raw image.
            image_compressed (None): Placeholder for storing the compressed image.
            image_compressed_depth (None): Placeholder for storing the compressed depth image.
            image_theora (None): Placeholder for storing the Theora encoded image.
        """
        self.bridge = CvBridge()
        self.image_sucess = False
        self.image = None
        self.image_compressed_sucess = False
        self.image_compressed = None
        self.image_compressed_depth_sucess = False
        self.image_compressed_depth = None
        self.image_theora_sucess = False
        self.image_theora = None

    def listening_image_raw(self) -> None:
        """
        Subscribes to the raw image topic.
        """
        rospy.Subscriber("/bebop/image_raw", Image, self._callback_image_raw)

    def listening_image_raw_compressed(self) -> None:
        """
        Subscribes to the compressed image topic.
        """
        rospy.Subscriber("/bebop/image_raw/compressed", CompressedImage, self._callback_image_raw_compressed)

    def listening_image_raw_compressed_Depth(self) -> None:
        """
        Subscribes to the compressed depth image topic.
        """
        rospy.Subscriber("/bebop/image_raw/compressedDepth", CompressedImage, self._callback_image_raw_compressed_depth)

    def listening_image_raw_theora(self) -> None:
        """
        Subscribes to the Theora encoded image topic.
        """
        rospy.Subscriber("/bebop/image_raw/theora", CompressedImage, self._callback_image_raw_theora)

    def _callback_image_raw(self, data: Image) -> None:
        """
        Callback function for the raw image topic.
        
        :param data: The raw image message.
        """

        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.image_sucess, self.image = self.__read(self.image, "/home/ubuntu/Imagens/test/image_raw.png")
        except CvBridgeError as e:
            self.sucess_read = False
            print(f"Error converting image: {e}")

    def _callback_image_raw_compressed(self, data: CompressedImage) -> None:
        """
        Callback function for the compressed image topic.
        
        :param data: The compressed image message.
        """

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            self.image_compressed = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"/home/ubuntu/Imagens/test/image_raw_compressed.png"
            self.image_compressed_sucess, self.image_compressed = self.__read(self.image_compressed, filename)
        except Exception as e:
            print(f"Error decoding image: {e}")

    def _callback_image_raw_compressed_depth(self, data: CompressedImage) -> None:
        """
        Callback function for the compressed depth image topic.
        
        :param data: The compressed depth image message.
        """

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            self.image_compressed_depth = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"/home/ubuntu/Imagens/test/image_raw_compressed_depth.png"
            self.image_compressed_depth_sucess, self.image_compressed_depth = self.__read(self.image_compressed_depth, filename)
        except Exception as e:
            print(f"Error decoding image: {e}")

    def _callback_image_raw_theora(self, data):
        """
        Callback function for the Theora encoded image topic.
        
        :param data: The Theora encoded image message.
        """
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            self.image_theora = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"/home/ubuntu/Imagens/test/image_raw_theora.png"
            self.image_theora_sucess, self.image_theora = self.__read(self.image_theora, filename)
        except Exception as e:
            print(f"Error decoding image: {e}")

    def __save_image(self, image: np.ndarray, filename: str) -> bool:
        """
        Saves an image to a file.
        
        :param image: The image to save.
        :param filename: The filename to save the image to.
        :return: True if the image was saved successfully, False otherwise.
        """

        if image is not None:
            cv2.imwrite(filename, image)
            return True
        else:
            return False

    def __load_image(self, filename: str) -> np.ndarray:
        """
        Loads an image from a file.
        
        :param filename: The filename of the image to load.
        :return: The loaded image as a numpy array.
        """

        return cv2.imread(filename)

    def __read(self, image:np.ndarray, filename: str) -> Tuple[bool, np.ndarray]:
        """
        Reads an image from a file.
        
        :param filename: The filename of the image to read.
        :return: A tuple containing a boolean indicating if the image was read successfully and the loaded image as a numpy array.
        """

        if self.__save_image(image, filename):
            return True, self.__load_image(filename)
        else:
            return False, None

# ...............................................................................................
