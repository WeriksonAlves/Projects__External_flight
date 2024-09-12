import cv2
import os
import numpy as np
from pyparrot.Bebop import Bebop
from pyparrot.DroneVision import DroneVision
from ..auxiliary.Model import Model
from typing import Union, Optional

class InitializeConfig:
    _instance = None

    def __new__(cls, *args, **kwargs) -> 'InitializeConfig':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, source: Union[int, str], fps: int = 5, dist: float = 0.025, length: int = 15) -> None:
        """
        Initializes the configuration object. Supports either a local camera or Bebop drone stream as the video source.
        """
        self.cap = Camera(source, {'attempts': 100, "buffer_size": 10})
        
        self.fps = fps
        self.dist = dist
        self.length = length

    


class Camera:
    def __init__(self, source:Union[int, str], param: Optional[dict] = None) -> None:
        """
        Initializes the camera object with the specified parameters.

        :param source: The source of the camera (RealSense, ESP32cam or Bebop drone).
        :param param: The parameters for the camera.
        """
        self.source = source
        self.param = param
        self.camera_initialize()

    def camera_initialize(self):
        """
        Initializes the camera based on the source.
        """
        if self.source == "bebop":
            self.bebop = Bebop()
            self._define_parameters()
            self._camera_prepare()
            return self.user_vision
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                exit()
            return self.cap

    def _define_parameters(self):
        """
        Defines the parameters for the Bebop drone camera.
        """
        self.attempts = self.param.get('attempts', 100)
        self.buffer_size = self.param.get('buffer_size', 10)

    def _camera_prepare(self):
        """
        Prepares the camera for capturing frames of bebop.
        """
        self.success = self.bebop.connect(self.attempts)
        if not self.success:
            print("Error: Could not connect to Bebop drone.")
            exit()

        self.bebop.set_video_stream_mode("low_latency")
        self.bebop.set_video_framerate("24_FPS")
        self.bebop.set_video_resolutions("rec720_stream720")
        self.bebop.start_video_stream()

        bebop_vision = DroneVision(
            self.bebop, Model.BEBOP, buffer_size=self.buffer_size
        )
        self.user_vision = UserVision(bebop_vision)
        self.image = bebop_vision.set_user_callback_function(
            self.user_vision.show_image, user_callback_args=(True,)
        )

        self.success = bebop_vision.open_video()
        if not self.success:
            print("Error: Could not open video stream.")
            exit()

    def read(self):
        """
        Captures a frame from the camera.
        """
        if self.source == "bebop":
            return self.success, self.image#_ self.bebop.get_frame()
        else:
            return self.cap.read()


class UserVision:
    def __init__(
        self,
        vision: DroneVision
    ) -> None:
        """
        Initialize the UserVision class with a vision object.

        :param vision: The DroneVision object responsible for image capture.
        """
        self.image_index = 1
        self.vision = vision

    def show(self, display_image):
        """
        Displays the image captured by the vision system.
        """
        image = self.vision.get_latest_valid_picture()
        if display_image:
            cv2.imshow("Captured Image", image)
            cv2.waitKey(100)
        return image

    def save_image(
        self,
        args: tuple = (False, None)
    ) -> None:
        """
        Saves the latest valid picture captured by the vision system.

        :param args: Flag indicating whether to save the image or not and the path where the images should be saved.
        """
        image = self.vision.get_latest_valid_picture()
        cv2.imshow("Captured Image", image)
        cv2.waitKey(100)

        if image is not None:
            if args[0]:
                if not os.path.exists(args[1]):
                    os.makedirs(args[1])

                filename = os.path.join(
                    args[1], f"image_{self.image_index:04d}.png"
                )
                cv2.imwrite(filename, image)
            self.image_index += 1


class ModeFactory:
    @staticmethod
    def create_mode(mode_type, **kwargs):
        """
        The function `create_mode` dynamically creates instances of different mode classes based on the
        specified `mode_type`.
        """
        if mode_type == 'dataset':
            return ModeDataset(**kwargs)
        elif mode_type == 'validate':
            return ModeValidate(**kwargs)
        elif mode_type == 'real_time':
            return ModeRealTime(**kwargs)
        else:
            raise ValueError("Invalid mode type")

class ModeDataset:
    def __init__(self, database: dict[str, list], file_name_build: str, max_num_gest: int = 50, 
                    dist: float = 0.025, length: int = 15) -> None:
        """
        This function initializes an object with specified parameters including a database, file name,
        maximum number of gestures, distance, and length.
        """
        self.mode = 'D'
        self.database = database
        self.file_name_build = file_name_build
        self.max_num_gest = max_num_gest
        self.dist = dist
        self.length = length

class ModeValidate:
    def __init__(self, files_name: list[str], database: dict[str, list], name_val: str,                    
                    proportion: float = 0.7, n_class: int = 5, n_sample_class: int = 10) -> None:
        """
        This function initializes various attributes including file names, database, proportion, and
        calculates a value based on input parameters.
        """
        self.mode = 'V'
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = int(np.round(np.sqrt(int(len(self.files_name) * self.proportion * n_class * n_sample_class))))
        self.file_name_val = self.rename(n_class, n_sample_class, name_val)

    def rename(self, n_class, n_sample_class, name_val):
        """
        The `rename` function generates a file name based on input parameters such as class, sample
        size, proportion, and a custom name value.
        """
        c = n_class
        s = int(len(self.files_name) * (1 - self.proportion) * n_class * n_sample_class)
        ma_p = int(10 * self.proportion)
        me_p = int(10 * (1 - self.proportion))
        return f"Results\C{c}_S{s}_p{ma_p}{me_p}_k{self.k}_{name_val}"

class ModeRealTime:
    def __init__(self, files_name: list[str], database: dict[str, list], proportion: float = 0.7,            
                    n_class: int = 5, n_sample_class: int = 10) -> None:
        """
        This function initializes an object with specified parameters for files, database, proportion,
        number of classes, and number of samples per class.
        """
        self.mode = 'RT'
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = int(np.round(np.sqrt(int(len(self.files_name) * self.proportion * n_class * n_sample_class))))
