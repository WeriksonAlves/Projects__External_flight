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

# rospy.init_node('RecognitionSystem', anonymous=True)

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

mode = real_time_mode

# Initialize the Servo Position System
num_servos = 0 # Number of servos in the system
if num_servos != 0:
    dir_rot = 1 #direction of rotation
    pub_hor_rot = rospy.Publisher('/EspSystem/hor_rot', Int32, queue_size=10)
    pub_ver_rot = rospy.Publisher('/EspSystem/ver_rot', Int32, queue_size=10)
else:
    pub_hor_rot = None
    pub_ver_rot = None
    dir_rot = 0
    
    
SPS = ServoPositionSystem(num_servos, pub_hor_rot, pub_ver_rot, dir_rot)
B = BebopROS()
grs = GestureRecognitionSystem(
        # config=InitializeConfig('http://192.168.209.199:81/stream'),
        config=InitializeConfig(4,10),
        # config = InitializeConfig(B,10),
        operation=mode,
        file_handler=FileHandler(),
        current_folder=os.path.dirname(__file__),
        data_processor=DataProcessor(), 
        time_functions=TimeFunctions(), 
        gesture_analyzer=GestureAnalyzer(),
        tracking_processor=MyYolo('yolov8n-pose.pt'), 
        feature=MyMediaPipe(
            mp.solutions.hands.Hands(
                static_image_mode=False, 
                max_num_hands=1, 
                model_complexity=1, 
                min_detection_confidence=0.75, 
                min_tracking_confidence=0.75
                ),
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
            ),
        sps=SPS
        )

try:
    grs.run()
finally:
    grs.stop()
