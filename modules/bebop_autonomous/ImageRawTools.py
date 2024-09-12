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
        _image_raw_callback(data: Image) -> None:
        _image_raw_compressed_callback(data: CompressedImage) -> None:
        _image_raw_compressed_depth_callback(data: CompressedImage) -> None:
        _image_raw_theora_callback(data) -> None:
        __save_image(image: np.ndarray, filename: str) -> bool:
        __load_image(filename: str) -> np.ndarray:
        read(image: np.ndarray, filename: str) -> Tuple[bool, np.ndarray]:
            :param image: The image to save and read.
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
        self.image = None
        self.image_compressed = None
        self.image_compressed_depth = None
        self.image_theora = None
        self.sucess_read = False

    def _image_raw_callback(self, data: Image) -> None:
        """
        Callback function for the raw image topic.
        
        :param data: The raw image message.
        """

        try:
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.sucess_read, self.image = self.read(self.image, "/home/ubuntu/Imagens/test/image_raw.png")
        except CvBridgeError as e:
            self.sucess_read = False
            print(f"Error converting image: {e}")

    def _image_raw_compressed_callback(self, data: CompressedImage) -> None:
        """
        Callback function for the compressed image topic.
        
        :param data: The compressed image message.
        """

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            self.image_compressed = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"/home/ubuntu/Imagens/test/image_raw_compressed.png"
            self.sucess_read, self.image = self.read(self.image_compressed, filename)
        except Exception as e:
            print(f"Error decoding image: {e}")

    def _image_raw_compressed_depth_callback(self, data: CompressedImage) -> None:
        """
        Callback function for the compressed depth image topic.
        
        :param data: The compressed depth image message.
        """

        try:
            np_arr = np.fromstring(data.data, np.uint8)
            self.image_compressed_depth = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"/home/ubuntu/Imagens/test/image_raw_compressed_depth.png"
            self.sucess_read, self.image = self.read(self.image_compressed_depth, filename)
        except Exception as e:
            print(f"Error decoding image: {e}")

    def _image_raw_theora_callback(self, data):
        """
        Callback function for the Theora encoded image topic.
        
        :param data: The Theora encoded image message.
        """
        try:
            np_arr = np.fromstring(data.data, np.uint8)
            self.image_theora = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            filename = f"/home/ubuntu/Imagens/test/image_raw_theora.png"
            #self.sucess_read, self.image = self.read(self.image_theora, filename)
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

    def read(self, image:np.ndarray, filename: str) -> Tuple[bool, np.ndarray]:
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
