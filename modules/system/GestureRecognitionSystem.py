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
from ..system.SystemSettings import (
    InitializeConfig,
    ModeDataset,
    ModeValidate,
    ModeRealTime
)


class GestureRecognitionSystem:
    """
    Gesture recognition system for real-time, validation, and dataset
    collection modes.

    Attributes:
        file_handler (FileHandler): Handles file operations.
        current_folder (str): Directory to store and access gesture data.
        data_processor (DataProcessor): Processes input data for gesture
        recognition.
        time_functions (TimeFunctions): Time-related utilities for FPS and
        performance tracking.
        gesture_analyzer (GestureAnalyzer): Analyzes detected gestures.
        tracking_processor (TrackerInterface): Handles tracking of detected
        objects and gestures.
        feature (ExtractorInterface): Extracts gesture-related features from
        input.
        classifier (ClassifierInterface): Classifier for recognizing gestures
        (used in validation and real-time modes).
        sps (ServoPositionSystem): Controls the servo motor position system
        (optional).
    """

    def __init__(
        self,
        config: InitializeConfig,
        operation: Union[ModeDataset, ModeValidate, ModeRealTime],
        file_handler: FileHandler,
        current_folder: str,
        data_processor: DataProcessor,
        time_functions: TimeFunctions,
        gesture_analyzer: GestureAnalyzer,
        tracking_processor: TrackerInterface,
        feature: ExtractorInterface,
        classifier: ClassifierInterface = None,
        sps: ServoPositionSystem = None
    ) -> None:
        """
        Initialize the gesture recognition system based on the configuration,
        operation mode, and various processors and analyzers.

        :param config: Configuration settings for the camera.
        :param operation: Operation mode for the system.
        :param file_handler: Handles file operations.
        :param current_folder: Directory to store and access gesture data.
        :param data_processor: Processes input data for gesture recognition.
        :param time_functions: Time-related utilities for FPS and performance
        tracking.
        :param gesture_analyzer: Analyzes detected gestures.
        :param tracking_processor: Handles tracking of detected objects and
        gestures.
        :param feature: Extracts gesture-related features from input.
        :param classifier: Classifier for recognizing gestures (used in
        validation and real-time modes).
        :param sps: Controls the servo motor position system (optional).
        """
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
        """Initializes camera settings based on the provided configuration."""
        self.cap = config.cap
        self.fps = config.fps
        self.dist = config.dist
        self.length = config.length

    def _initialize_operation(
        self, operation: Union[ModeDataset, ModeValidate, ModeRealTime]
    ) -> None:
        """Initializes operation mode and specific parameters for each mode."""
        self.mode = operation.mode
        if self.mode == 'D':
            self._initialize_dataset_mode(operation)
        elif self.mode == 'V':
            self._initialize_validation_mode(operation)
        elif self.mode == 'RT':
            self._initialize_real_time_mode(operation)
        else:
            raise ValueError("Invalid mode")

    def _initialize_dataset_mode(self, operation: ModeDataset) -> None:
        """Initializes dataset collection mode."""
        self.database = operation.database
        self.file_name_build = operation.file_name_build
        self.max_num_gest = operation.max_num_gest
        self.dist = operation.dist
        self.length = operation.length

    def _initialize_validation_mode(self, operation: ModeValidate) -> None:
        """Initializes validation mode."""
        self.database = operation.database
        self.proportion = operation.proportion
        self.files_name = operation.files_name
        self.file_name_val = operation.file_name_val

    def _initialize_real_time_mode(self, operation: ModeRealTime) -> None:
        """Initializes real-time gesture recognition mode."""
        self.database = operation.database
        self.proportion = operation.proportion
        self.files_name = operation.files_name

    def _initialize_simulation_variables(self) -> None:
        """Initializes simulation-related variables to default values."""
        self.stage = 0
        self.num_gest = 0
        self.dist_virtual_point = 1.0
        self.sc_pitch = 0.0
        self.sc_yaw = 0.0
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
        """Initializes variables for storing hand and pose data."""
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(
            dist=self.dist, length=self.length
        )

    def _initialize_threads(self) -> None:
        """Initializes threads for reading images."""
        self.frame_lock = threading.Lock()

        # Start thread for reading images
        self.image_thread = threading.Thread(
            target=self._read_image_thread, daemon=True
        )
        self.image_thread.start()

    def run(self) -> None:
        """
        Main execution loop for gesture recognition system based on the
        specified mode. Handles dataset collection, real-time gesture
        recognition, and validation.
        """
        self._setup_mode()

        t_frame = self.time_functions.tic()
        while self.loop:
            if self.time_functions.toc(t_frame) > (1 / self.fps):
                print(f"FPS: {int(1/self.time_functions.toc(t_frame))}")
                t_frame = self.time_functions.tic()
                self._process_frame(t_frame)
                t_frame = self.time_functions.tic()

    def _setup_mode(self) -> None:
        """Set up the system based on the mode of operation."""
        if self.mode == 'D':
            self._initialize_database()
            self.loop = True
        elif self.mode == 'RT':
            self._load_and_fit_classifier()
            self.loop = True
        elif self.mode == 'V':
            self._validate_classifier()
            self.loop = False
        else:
            print(f"Invalid operation mode: {self.mode}")
            self.loop = False

    def _process_frame(self, t_frame) -> None:
        """Process each frame during the system's run loop."""
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()

        if self.mode == 'D' and self.num_gest == self.max_num_gest:
            self.stop()

        self._process_stage()

    def _initialize_database(self) -> None:
        """Initialize the gesture database."""
        self.target_names, self.y_val = self.file_handler.initialize_database(
            self.database
        )

    def _load_and_fit_classifier(self) -> None:
        """Load training data and fit the classifier."""
        x_train, y_train, _, _ = self.file_handler.load_database(
            self.current_folder, self.files_name, self.proportion
        )
        self.classifier.fit(x_train, y_train)

    def _validate_classifier(self) -> None:
        """Validate the classifier with the validation dataset."""
        x_train, y_train, x_val, self.y_val = self.file_handler.load_database(
            self.current_folder, self.files_name, self.proportion
        )
        self.classifier.fit(x_train, y_train)
        self.y_predict, self.time_classifier = self.classifier.validate(x_val)
        self.target_names, _ = self.file_handler.initialize_database(
            self.database
        )
        self.file_handler.save_results(
            self.y_val,
            self.y_predict,
            self.time_classifier,
            self.target_names,
            os.path.join(self.current_folder, self.file_name_val)
        )

    def _process_stage(self) -> None:  # Voltar AQUI
        """Processes each stage in the gesture recognition pipeline."""
        if self.stage in [0, 1] and self.mode in ['D', 'RT']:
            success, frame = self._read_image()
            if success:
                self._image_processing(frame)
                self._extract_features()
        elif self.stage == 2 and self.mode in ['D', 'RT']:
            self.process_reduction()
            self.stage = 3 if self.mode == 'D' else 4
        elif self.stage == 3 and self.mode == 'D':
            self._update_database()
            self.stage = 0
        elif self.stage == 4 and self.mode == 'RT':
            self._classify_gestures()
            self.stage = 0

    def _read_image_thread(self) -> None:  # VOLTAR AQUI
        """Thread for continuously reading images from the camera."""
        while True:
            success, frame = self.cap.read()
            if success:
                # Verify if the size image is 640x480, otherwise, resize it
                if frame.shape[0] != 640 or frame.shape[1] != 480:
                    frame = cv2.resize(frame, (640, 480))
                with self.frame_lock:
                    self.frame_captured = frame

    def _read_image(self) -> tuple[bool, np.ndarray]:
        """Reads the next image frame from the captured stream."""
        with self.frame_lock:
            return self.frame_captured is not None, self.frame_captured

    def stop(self) -> None:
        """Stops the gesture recognition system and releases resources."""
        self.loop = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

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
            boxes, track_ids = self.tracking_processor.extract_boxes_and_ids(results_people)
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
