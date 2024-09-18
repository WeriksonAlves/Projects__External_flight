import cv2
import os
import numpy as np
import threading
from typing import Union
from ..auxiliary.DrawGraphics import DrawGraphics
from ..auxiliary.FileHandler import FileHandler
from ..auxiliary.TimeFunctions import TimeFunctions
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..interfaces.ExtractorInterface import ExtractorInterface
from ..interfaces.TrackerInterface import TrackerInterface
from ..gesture.DataProcessor import DataProcessor
from ..gesture.FeatureExtractor import FeatureExtractor
from ..gesture.GestureAnalyzer import GestureAnalyzer
from ..system.ServoPositionSystem import ServoPositionSystem
from ..system.SystemSettings import InitializeConfig
from ..system.SystemSettings import ModeDataset
from ..system.SystemSettings import ModeValidate
from ..system.SystemSettings import ModeRealTime


class GestureRecognitionSystem:
    def __init__(self, config: InitializeConfig, operation: Union[ModeDataset, ModeValidate, ModeRealTime], 
                file_handler: FileHandler, current_folder: str, data_processor: DataProcessor, 
                time_functions: TimeFunctions, gesture_analyzer: GestureAnalyzer, tracking_processor: TrackerInterface, 
                feature: ExtractorInterface, classifier: ClassifierInterface = None, sps: ServoPositionSystem = None) -> None:
        
        self._initialize_camera(config)
        self._initialize_operation(operation)
        
        self.file_handler = file_handler
        self.current_folder = current_folder
        self.data_processor = data_processor
        self.time_functions = time_functions
        self.gesture_analyzer = gesture_analyzer
        self.tracking_processor = tracking_processor
        self.feature = feature
        self.classifier = classifier
        self.sps = sps
        
        self._initialize_simulation_variables()
        self._initialize_storage_variables()
        self._initialize_threads()

    def _initialize_camera(self, config: InitializeConfig) -> None:
        """
        The function `_initialize_camera` initializes camera settings based on the provided
        configuration.
        """
        self.cap = config.cap
        self.fps = config.fps
        self.dist = config.dist
        self.length = config.length

    def _initialize_operation(self, operation: Union[ModeDataset, ModeValidate, ModeRealTime]) -> None:
        """
        The function `_initialize_operation` initializes attributes based on the mode specified in the
        input operation.
        """
        self.mode = operation.mode
        if self.mode == 'D':
            self.database = operation.database
            self.file_name_build = operation.file_name_build
            self.max_num_gest = operation.max_num_gest
            self.dist = operation.dist
            self.length = operation.length
        elif self.mode == 'V':
            self.database = operation.database
            self.proportion = operation.proportion
            self.files_name = operation.files_name
            self.file_name_val = operation.file_name_val
        elif self.mode == 'RT':
            self.database = operation.database
            self.proportion = operation.proportion
            self.files_name = operation.files_name
        else:
            raise ValueError("Invalid mode")

    def _initialize_simulation_variables(self) -> None:
        """
        The function `_initialize_simulation_variables` initializes various simulation variables to
        default values.
        """
        self.stage = 0
        self.num_gest = 0
        self.dist_virtual_point = 1
        self.sc_pitch: float = 0
        self.sc_yaw: float = 0
        self.hands_results = None
        self.pose_results = None
        self.time_gesture = None
        self.time_action = None
        self.y_val = None
        self.frame_captured = None
        self.center_person = False
        self.loop = False
        self.y_predict = []
        self.time_classifier = []

    def _initialize_storage_variables(self) -> None:
        """
        The function `_initialize_storage_variables` initializes storage variables using data processed
        by `data_processor`.
        """
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)

    def _initialize_threads(self) -> None:
        # For threading
        self.frame_lock = threading.Lock()
        
        # Thread for reading images
        self.image_thread = threading.Thread(target=self._read_image_thread)
        self.image_thread.daemon = True
        self.image_thread.start()

    def run(self) -> None:
        """
        Run the gesture recognition system based on the specified mode.

        - If the mode is 'D' (Batch), initialize the database and set the loop flag to True.
        - If the mode is 'RT' (Real-Time), load and fit the classifier, and set the loop flag to True.
        - If the mode is 'V' (Validation), validate the classifier and set the loop flag to False.
        - If the mode is invalid, print a message and set the loop flag to False.

        During the loop:
        - Measure the time for each frame.
        - Check for user input to quit the loop (pressing 'q').
        - If the mode is 'D', break the loop if the maximum number of gestures is reached.
        - Process each stage of the gesture recognition system.

        After the loop, release the capture and close all OpenCV windows.

        Returns:
            None
        """
        if self.mode == 'D':
            self._initialize_database()
            self.loop = True
        elif self.mode == 'RT':
            self._load_and_fit_classifier()
            self.loop = True
            self.servo_enabled = True
        elif self.mode == 'V':
            self._validate_classifier()
            self.loop = False
            self.servo_enabled = False
        else:
            print(f"Operation mode invalid!")
            self.loop = False
            self.servo_enabled = False
            
        t_frame = self.time_functions.tic()
        while self.loop:
            if self.time_functions.toc(t_frame) > (1 / self.fps):
                t_frame = self.time_functions.tic()
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop()
                
                if self.mode == "B":
                    if self.num_gest == self.max_num_gest:
                        self.stop()
                
                self._process_stage()

    def _initialize_database(self) -> None:
        """
        This method initializes the target names and ground truth labels (y_val) by calling the
        initialize_database method of the file_handler object.
        """
        self.target_names, self.y_val = self.file_handler.initialize_database(self.database)

    def _load_and_fit_classifier(self) -> None:
        """
        This method loads the training data, fits the classifier with the training data, and performs
        model training.
        """
        x_train, y_train, _, _ = self.file_handler.load_database(self.current_folder, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)

    def _validate_classifier(self) -> None:
        """
        This method validates the classifier with validation data and saves the validation results.
        """
        x_train, y_train, x_val, self.y_val = self.file_handler.load_database(self.current_folder, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)
        self.y_predict, self.time_classifier = self.classifier.validate(x_val)
        self.target_names, _ = self.file_handler.initialize_database(self.database)
        self.file_handler.save_results(self.y_val, self.y_predict, self.time_classifier, self.target_names, os.path.join(self.current_folder, self.file_name_val))

    def _process_stage(self) -> None:
        """
        The `_process_stage` function handles different stages and modes of processing in the system.        
        Returns:
        - If conditions are met, the function may return `None` or continue execution without returning anything.
        """
        if self.stage in [0, 1] and self.mode in ['D', 'RT']:
            success, frame = self._read_image()
            if not success:
                return
            if not self._image_processing(frame):
                return
            self._extract_features()
        elif self.stage == 2 and self.mode in ['D', 'RT']:
            self.process_reduction()
            if self.mode == 'D':
                self.stage = 3
            elif self.mode == 'RT':
                self.stage = 4
        elif self.stage == 3 and self.mode == 'D':
            if self._update_database():
                self.loop = False
            self.stage = 0
        elif self.stage == 4 and self.mode == 'RT':
            self._classify_gestures()
            self.stage = 0

    def _read_image_thread(self) -> None:
        """
        Reads frames from the video capture device and stores the captured frame in the instance variable `frame_captured`.

        This method runs in a separate thread and continuously reads frames from the video capture device. If the frame size is not 640x480,
        it resizes the frame to the desired size. The captured frame is stored in the `frame_captured` instance variable, which can be accessed
        by other methods.

        Returns:
            None
        """
        while True:
            success, frame = self.cap.read()
            if success:
                # Verify if the size image is 640x480, otherwise, resize it
                if frame.shape[0] != 640 or frame.shape[1] != 480:
                    frame = cv2.resize(frame, (640, 480))
                with self.frame_lock:
                    self.frame_captured = frame

    def _read_image(self) -> tuple[bool, np.ndarray]:
            """
            Reads and returns the captured frame from the video stream.

            Returns:
                A tuple containing a boolean value indicating whether the frame was successfully read,
                and the captured frame as a numpy array.
            """
            with self.frame_lock:
                if self.frame_captured is None:
                    return False, None
                frame = self.frame_captured.copy()  # Create a copy of the frame for thread-safe processing
            return True, frame

    def _image_processing(self, frame: np.ndarray) -> bool:
        """
        Process the input frame for gesture recognition.

        Args:
            frame (np.ndarray): The input frame to be processed.

        Returns:
            bool: True if the processing is successful, False otherwise.
        """
        try:
            results_people, results_identifies = self.tracking_processor.detect_people_in_frame(frame)
            boxes, track_ids = self.tracking_processor.identify_operator(results_people)
            cropped_image, _ = self.tracking_processor.crop_operator_from_frame(boxes, track_ids, results_identifies, frame)
            dist_center_h, dist_center_v = self.tracking_processor.centralize_person_in_frame(frame, boxes[0])

            # Signal control
            gain = [45, 45]
            if np.abs(dist_center_v) > 0.25: self.sc_pitch = np.tanh(-dist_center_v*0.75) * gain[0]
            if np.abs(dist_center_h) > 0.25: self.sc_yaw = np.tanh(dist_center_h*0.75) * gain[1]
            
            # print(f"Distance: ({dist_center_h:8.4f}, {dist_center_v:8.4f})   Signal Control: ({sc_pitch:8.4f}, {sc_yaw:8.4f})", end='')
            # self.cap.move_camera(self.sc_pitch, self.sc_yaw)
            
            # Finds the operator's hand(s) and body
            self.hands_results, self.pose_results = self.feature.find_features(cropped_image)
            frame_results = self.feature.draw_features(cropped_image, self.hands_results, self.pose_results)

            # Shows the skeleton formed on the body, and indicates which gesture is being 
            # performed at the moment.
            if self.mode == 'D':
                cv2.putText(frame_results, f"S{self.stage} N{self.num_gest+1}: {self.y_val[self.num_gest]} D{self.dist_virtual_point:.3f}" , (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            elif self.mode == 'RT':
                cv2.putText(frame_results, f"S{self.stage} D{self.dist_virtual_point:.3f}" , (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow('RealSense Camera', frame_results)
            return True
        except Exception as e:
            print(f"E1 - Error during operator detection, tracking or feature extraction: {e}")
            with self.frame_lock:
                frame = self.frame_captured
            cv2.imshow('RealSense Camera', cv2.flip(frame, 1))
            self.hand_history = np.concatenate((self.hand_history, np.array([self.hand_history[-1]])), axis=0)
            self.wrists_history = np.concatenate((self.wrists_history, np.array([self.wrists_history[-1]])), axis=0)
            return False

    def _extract_features(self) -> None:
        """
        The function `_extract_features` processes hand and pose data to track specific joints and
        trigger gestures based on proximity criteria.
        """
        if self.stage == 0:
            try:
                # Tracks the fingertips of the left hand and centers them in relation to the center
                # of the hand.
                hand_ref = np.tile(self.gesture_analyzer.calculate_ref_pose(self.hands_results.multi_hand_landmarks[0], self.sample['joints_trigger_reference']), len(self.sample['joints_trigger']))
                hand_pose = [FeatureExtractor.calculate_joint_xy(self.hands_results.multi_hand_landmarks[0], marker) for marker in self.sample['joints_trigger']]
                hand_center = np.array([np.array(hand_pose).flatten() - hand_ref])
                self.hand_history = np.concatenate((self.hand_history, hand_center), axis=0)
            except:
                # If this is not possible, repeat the last line of the history.
                self.hand_history = np.concatenate((self.hand_history, np.array([self.hand_history[-1]])), axis=0)
            
            # Check that the fingertips are close together, if they are less than "dist" the 
            # trigger starts and the gesture begins.
            _, self.hand_history, self.dist_virtual_point = self.gesture_analyzer.check_trigger_enabled(self.hand_history, self.sample['par_trigger_length'], self.sample['par_trigger_dist'])
            if self.dist_virtual_point < self.sample['par_trigger_dist']:
                self.stage = 1
                self.dist_virtual_point = 1
                self.time_gesture = self.time_functions.tic()
                self.time_action = self.time_functions.tic()
        elif self.stage == 1:
            try:
                # Tracks the operator's wrists throughout the action
                track_ref = np.tile(self.gesture_analyzer.calculate_ref_pose(self.pose_results.pose_landmarks, self.sample['joints_tracked_reference'], 3), len(self.sample['joints_tracked']))
                track_pose = [FeatureExtractor.calculate_joint_xyz(self.pose_results.pose_landmarks, marker) for marker in self.sample['joints_tracked']]
                track_center = np.array([np.array(track_pose).flatten() - track_ref])
                self.wrists_history = np.concatenate((self.wrists_history, track_center), axis=0)
            except:
                # If this is not possible, repeat the last line of the history.
                self.wrists_history = np.concatenate((self.wrists_history, np.array([self.wrists_history[-1]])), axis=0)
            
            # Evaluates whether the execution time of a gesture has been completed
            if self.time_functions.toc(self.time_action) > 4:
                self.stage = 2
                self.sample['time_gest'] = self.time_functions.toc(self.time_gesture)
                self.t_classifier = self.time_functions.tic()

    def process_reduction(self) -> None:
        """
        The function `process_reduction` removes the zero's line from a matrix, applies filters, and
        reduces the matrix to a 6x6 matrix based on certain conditions.
        """
        # Remove the zero's line
        self.wrists_history = self.wrists_history[1:]
        
        # Test the use of filters before applying pca
        
        # Reduces to a 6x6 matrix 
        self.sample['data_reduce_dim'] = np.dot(self.wrists_history.T, self.wrists_history)

    def _update_database(self) -> bool:
        """
        This function updates a database with sample data and saves it in JSON format.
        
        Note: Exclusive function for database construction operating mode.
        """
        # Updates the sample
        self.sample['data_pose_track'] = self.wrists_history
        self.sample['answer_predict'] = self.y_val[self.num_gest]
        
        # Updates the database
        self.database[str(self.y_val[self.num_gest])].append(self.sample)
        
        # Save the database in JSON format
        self.file_handler.save_database(self.sample, self.database, os.path.join(self.current_folder, self.file_name_build))
        
        # Resets sample data variables to default values
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)
        
        # Indicates the next gesture and returns to the image processing step
        self.num_gest += 1
        if self.num_gest == self.max_num_gest: 
            return True
        else: 
            return False

    def _classify_gestures(self) -> None:
        """
        This function classifies gestures based on the stage and mode, updating predictions and
        resetting sample data variables accordingly.
        
        Note: Exclusive function for Real-Time operating mode
        """
        # Classifies the action performed
        self.y_predict.append(self.classifier.predict(self.sample['data_reduce_dim']))
        self.time_classifier.append(self.time_functions.toc(self.t_classifier))
        print(f"\nThe gesture performed belongs to class {self.y_predict[-1]} and took {self.time_classifier[-1]:.3}ms to be classified.\n")
        
        # Resets sample data variables to default values
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)

    def stop(self) -> None:
        """
        Stops the gesture recognition system.

        """
        self.loop = False
        self.cap.release()
        cv2.destroyAllWindows()