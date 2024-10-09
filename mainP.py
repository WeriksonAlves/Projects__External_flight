#!/usr/bin/env python
import os
import rospy
from modules import (
    ModeFactory,
    BebopROS,
    GestureRecognitionSystem,
    MyCamera,
    MyYolo,
    MyHandsMediaPipe,
    MyPoseMediaPipe,
    KNN
)
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp

# Constants
DATABASE_FILE = "datasets/DataBase_(5-10)_16.json"
DATABASE_FILES = [
    'datasets/DataBase_(5-10)_G.json',
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
NAME_VAL = "val99"


def initialize_modes(mode: int):
    """Initialize operation modes for gesture recognition."""
    database_empty = {'F': [], 'I': [], 'L': [], 'P': [], 'T': []}

    if mode == 1:
        operation_mode = ModeFactory.create_mode(
            mode_type='dataset',
            database=database_empty,
            file_name_build=DATABASE_FILE
        )
    elif mode == 2:
        operation_mode = ModeFactory.create_mode(
            mode_type='validate',
            files_name=DATABASE_FILES,
            database=database_empty,
            name_val=NAME_VAL
        )
    elif mode == 3:
        operation_mode = ModeFactory.create_mode(
            mode_type='real_time',
            files_name=DATABASE_FILES,
            database=database_empty
        )
    else:
        raise ValueError("Invalid mode")
    return operation_mode


def create_gesture_recognition_system(camera, operation_mode, bebop):
    """Create the Gesture Recognition System."""
    return GestureRecognitionSystem(
        base_dir=os.path.dirname(__file__),  # Get the current folder
        configs=MyCamera(camera, 15),  # Initialize the configuration
        operation_mode=operation_mode,
        tracker_model=MyYolo('yolov8n-pose.pt'),
        hand_extractor_model=MyHandsMediaPipe(
            mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75
            )
        ),
        body_extractor_model=MyPoseMediaPipe(
            mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                smooth_segmentation=True,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75
            )
        ),
        classifier=KNN(
            KNeighborsClassifier(
                n_neighbors=operation_mode.k,
                algorithm='auto',
                weights='uniform'
            )
        ) if hasattr(operation_mode, 'k') else None,
        bebop=bebop
    )


def initialize_camera(camera):
    """Initialize the camera to be used."""
    if camera == 'realsense':
        return 4
    elif camera == 'espcam':
        return "http://192.168.209.199:81/stream"
    elif camera == 'bebop':
        return BebopROS()


def main():
    """Main function to run the Gesture Recognition System."""
    rospy.init_node('External_Flight', anonymous=True)

    # Initialize the Bebop2 drone
    bebop = BebopROS()

    # Initialize Gesture Recognition System
    operation_mode = initialize_modes(3)

    # Create and run the gesture recognition system
    gesture_system = create_gesture_recognition_system(bebop.camera,
                                                       operation_mode,
                                                       bebop)
    try:
        gesture_system.run()
    finally:
        gesture_system.stop()


if __name__ == "__main__":
    main()
