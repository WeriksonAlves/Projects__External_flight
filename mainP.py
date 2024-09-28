#!/usr/bin/env python
import os
import rospy
from std_msgs.msg import Int32
from modules import (
    ModeFactory,
    InitializeConfig,
    ServoPositionSystem,
    BebopROS,
    GestureRecognitionSystem,
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


def initialize_servo_system(num_servos):
    """Initialize the Servo Position System."""
    if num_servos != 0:
        dir_rot = 1  # Direction of rotation
        pub_hor_rot = rospy.Publisher(
            '/EspSystem/horizontal', Int32, queue_size=10
        )
        pub_ver_rot = rospy.Publisher(
            '/EspSystem/vertical', Int32, queue_size=10
        )
        return ServoPositionSystem(
            num_servos, pub_hor_rot, pub_ver_rot, dir_rot
        )
    else:
        return None


def create_gesture_recognition_system(camera, mode, sps):
    """Create the Gesture Recognition System."""
    return GestureRecognitionSystem(
        config=InitializeConfig(camera, 15),
        operation=mode,
        current_folder=os.path.dirname(__file__),
        tracking_model=MyYolo('yolov8n-pose.pt'),
        feature_hand=MyHandsMediaPipe(
            mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.75,
                min_tracking_confidence=0.75
            )
        ),
        feature_pose=MyPoseMediaPipe(
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
                n_neighbors=mode.k,
                algorithm='auto',
                weights='uniform'
            )
        ) if hasattr(mode, 'k') else None,
        sps=sps
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
    rospy.init_node('RecognitionSystem', anonymous=True)

    # Initialize Gesture Recognition System
    operation_mode = initialize_modes(1)

    # Initialize the Servo Position System
    num_servos = 0  # Adjust the number of servos if necessary
    sps = initialize_servo_system(num_servos)

    # Initialize the camera to be used
    camera = initialize_camera('realsense')

    # Create and run the gesture recognition system
    gesture_system = create_gesture_recognition_system(camera, operation_mode,
                                                       sps)
    # try:
    gesture_system.run()
    # finally:
    #     gesture_system.stop()


if __name__ == "__main__":
    main()
