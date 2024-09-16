#!/usr/bin/env python
"""
...............................................................................................
Description
    Operation mode:
        Build:     Creates a new database and saves it in json format
        Recognize: Load the database, create the classifier and classify the actions

    Operation stage:
        0 - Processes the image and analyzes the operator's hand
        1 - Processes the image and analyzes the operator's body
        2 - Reduces the dimensionality of the data
        3 - Updates and save the database
        4 - Performs classification from kMeans
...............................................................................................
""" 
from modules import *
import os
import rospy
from typing import Union
import cv2
import time
import numpy as np

class InitializeConfig:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, source: Union[int, str, BebopROS], fps: int = 5, dist: float = 0.025, length: int = 15) -> None:
        if hasattr(source, 'drone_type'):
            source.VideoCapture()
            if source.camera.isOpened():
                self.cap = source.camera  # ROS-based camera
            else:
                raise RuntimeError("Error: Could not open DroneCamera.")
        else:
            self.cap = cv2.VideoCapture(source)  # For other OpenCV-based cameras
            if not self.cap.isOpened():
                print("Error: Could not open camera.")
                exit()
        self.fps = fps
        self.dist = dist
        self.length = length


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

#%%


# Initialize the Gesture Recognition System
database = {'F': [], 'I': [], 'L': [], 'P': [], 'T': []}
file_name_build = f"datasets/DataBase_(5-10)_16.json"
files_name= ['datasets/DataBase_(5-10)_G.json',
            'datasets/DataBase_(5-10)_H.json',
            'datasets/DataBase_(5-10)_L.json',
            'datasets/DataBase_(5-10)_M.json',
            'datasets/DataBase_(5-10)_T.json',
            'datasets/DataBase_(5-10)_1.json',
            'datasets/DataBase_(5-10)_2.json',
            'datasets/DataBase_(5-10)_3.json',
            'datasets/DataBase_(5-10)_4.json',
            'datasets/DataBase_(5-10)_5.json',
            'datasets/DataBase_(5-10)_6.json',
            'datasets/DataBase_(5-10)_7.json',
            'datasets/DataBase_(5-10)_8.json',
            'datasets/DataBase_(5-10)_9.json',
            'datasets/DataBase_(5-10)_10.json'
            ]
name_val=f"val99"

dataset_mode = ModeFactory.create_mode('dataset', database=database, file_name_build=file_name_build)
validate_mode = ModeFactory.create_mode('validate', files_name=files_name, database=database, name_val=name_val)
real_time_mode = ModeFactory.create_mode('real_time', files_name=files_name, database=database)
B = BebopROS()
track = MyYolo('yolov8n-pose.pt')
# config=InitializeConfig('http://192.168.209.199:81/stream') #ESP32cam
# config=InitializeConfig(4,10) #realsense
config = InitializeConfig(B,10) #bebop














def my_imshow():
    time.sleep(2)
    start_time = time.time()
    tic = time.time()
    sc_pitch = 0
    sc_yaw = 0
    while not rospy.is_shutdown() and time.time() - start_time < 120:  
        if time.time() - tic > 1/30:
            tic = time.time()
            past = config.cap.image_data["image_compressed"]

            if config.cap.success_flags["image_compressed"]:# and not (past == config.cap.image_data["image_compressed"]).all():
                try:
                    # Show the image in real time
                    cv2.imshow('frame', config.cap.image_data["image_compressed"])
                    cv2.waitKey(1)
                    # Detect people in the frame
                    results_people, results_identifies = track.detect_people_in_frame(config.cap.image_data["image_compressed"])

                    # Identify operator
                    boxes, track_ids = track.identify_operator(results_people)

                    # Crop operator in frame
                    cropped_image, _ = track.crop_operator_from_frame(boxes, track_ids, results_identifies, config.cap.image_data["image_compressed"])
                    
                    # Centralize person in frame, compensating for yaw
                    dist_center_h, dist_center_v = track.centralize_person_in_frame(config.cap.image_data["image_compressed"], boxes[0])
                    
                    # Signal control
                    gain = [45, 45]
                    
                    if np.abs(dist_center_v) > 0.25:
                        sc_pitch = np.tanh(-dist_center_v*0.75) * gain[0]
                    
                    if np.abs(dist_center_h) > 0.25:
                        sc_yaw = np.tanh(dist_center_h*0.75) * gain[1]
                    
                    print(f"Distance: ({dist_center_h:8.4f}, {dist_center_v:8.4f})   Signal Control: ({sc_pitch:8.4f}, {sc_yaw:8.4f})", end='')

                    config.cap.move_camera(sc_pitch, sc_yaw)
                    
                    # Show the image in real time
                    cv2.imshow('cropped_image', cropped_image)
                    cv2.waitKey(1)
                except Exception as e:
                    rospy.logerr(f"Error processing image: {e}")
                    cv2.imshow('frame', config.cap.image_data["image_compressed"])
                    cv2.waitKey(1)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


try:
    rospy.loginfo("Drone camera system initialized")
    my_imshow()

except rospy.ROSInterruptException:
    rospy.logerr("ROS node interrupted.")
