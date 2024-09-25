import cv2
import os
import numpy as np
import threading
from typing import Union, Tuple
from ..auxiliary.FileHandler import FileHandler
from ..auxiliary.TimeFunctions import TimeFunctions
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..interfaces.ExtractorInterface import ExtractorInterface
from ..interfaces.TrackerInterface import TrackerInterface
from ..gesture.DataProcessor import DataProcessor
from ..system.ServoPositionSystem import ServoPositionSystem
from ..system.SystemSettings import (
    InitializeConfig,
    ModeDataset,
    ModeValidate,
    ModeRealTime
)


class GestureRecognitionSystem2:
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

    @TimeFunctions.timer
    def __init__(
        self,
        config: InitializeConfig,
        operation: Union[ModeDataset, ModeValidate, ModeRealTime],
        file_handler: FileHandler,
        current_folder: str,
        data_processor: DataProcessor,
        time_functions: TimeFunctions,
        tracking_processor: TrackerInterface,
        feature_hand: ExtractorInterface,
        feature_pose: ExtractorInterface,
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
        self.tracker = tracking_processor
        self.feature_hand = feature_hand
        self.feature_pose = feature_pose
        self.classifier = classifier
        self.sps = sps

        self._initialize_simulation_variables()
        self._initialize_storage_variables()
        self._initialize_threads()

    @TimeFunctions.timer
    def _initialize_camera(self, config: InitializeConfig) -> None:
        """Initializes camera settings based on the provided configuration."""
        self.cap = config.cap
        self.fps = config.fps
        self.dist = config.dist
        self.length = config.length

    @TimeFunctions.timer
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

    @TimeFunctions.timer
    def _initialize_simulation_variables(self) -> None:
        """Initializes simulation-related variables to default values."""
        self.stage = 0
        self.num_gest = 0
        self.dist_virtual_point = 1.0
        self.sc_pitch = 0.0
        self.sc_yaw = 0.0
        self.hand_results = None
        self.pose_results = None
        self.time_gesture = None
        self.time_action = None
        self.y_val = None
        self.frame_captured = None
        self.center_person = False
        self.loop = False
        self.y_predict = []
        self.time_classifier = []

    @TimeFunctions.timer
    def _initialize_storage_variables(self) -> None:
        """Initializes variables for storing hand and pose data."""
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(
            dist=self.dist, length=self.length
        )

    @TimeFunctions.timer
    def _initialize_threads(self) -> None:
        """Initializes threads for reading images."""
        self.frame_lock = threading.Lock()

        # Start thread for reading images
        self.image_thread = threading.Thread(
            target=self._read_image_thread, daemon=True
        )
        self.image_thread.start()

    # @TimeFunctions.timer
    def _initialize_dataset_mode(self, operation: ModeDataset) -> None:
        """Initializes dataset collection mode."""
        self.database = operation.database
        self.file_name_build = operation.file_name_build
        self.max_num_gest = operation.max_num_gest
        self.dist = operation.dist
        self.length = operation.length

    # @TimeFunctions.timer
    def _initialize_validation_mode(self, operation: ModeValidate) -> None:
        """Initializes validation mode."""
        self.database = operation.database
        self.proportion = operation.proportion
        self.files_name = operation.files_name
        self.file_name_val = operation.file_name_val

    # @TimeFunctions.timer
    def _initialize_real_time_mode(self, operation: ModeRealTime) -> None:
        """Initializes real-time gesture recognition mode."""
        self.database = operation.database
        self.proportion = operation.proportion
        self.files_name = operation.files_name

    # @TimeFunctions.timer
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
                t_frame = self.time_functions.tic()
                self._process_frame(t_frame)
                t_frame = self.time_functions.tic()

    # @TimeFunctions.timer
    def stop(self) -> None:
        """Stops the gesture recognition system and releases resources."""
        self.loop = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    # @TimeFunctions.timer
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

    # @TimeFunctions.timer
    def _initialize_database(self) -> None:
        """Initialize the gesture database."""
        self.target_names, self.y_val = self.file_handler.initialize_database(
            self.database
        )

    # @TimeFunctions.timer
    def _load_and_fit_classifier(self) -> None:
        """Load training data and fit the classifier."""
        x_train, y_train, _, _ = self.file_handler.load_database(
            self.current_folder, self.files_name, self.proportion
        )
        self.classifier.fit(x_train, y_train)

    # @TimeFunctions.timer
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

    @TimeFunctions.timer
    def _process_frame(self, t_frame) -> None:
        """Process each frame during the system's run loop."""
        if cv2.waitKey(1) & 0xFF == ord("q"):
            self.stop()

        if self.mode == 'D' and self.num_gest == self.max_num_gest:
            self.stop()

        self._process_stage()

    # @TimeFunctions.timer
    def _process_stage(self) -> None:  # Voltar AQUI
        """Processes each stage in the gesture recognition pipeline."""
        if self.stage in [0, 1] and self.mode in ['D', 'RT']:
            success, frame = self._read_image()
            if success:
                cropped_image = self._tracking_processor(frame)
                self._extraction_processor(cropped_image)
        elif self.stage == 2 and self.mode in ['D', 'RT']:
            self.process_reduction()
            self.stage = 3 if self.mode == 'D' else 4
        elif self.stage == 3 and self.mode == 'D':
            self._update_database()
            self.stage = 0
        elif self.stage == 4 and self.mode == 'RT':
            self._classify_gestures()
            self.stage = 0

    # @TimeFunctions.timer
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

    # @TimeFunctions.timer
    def _read_image(self) -> tuple[bool, np.ndarray]:
        """Reads the next image frame from the captured stream."""
        with self.frame_lock:
            return self.frame_captured is not None, self.frame_captured

    @TimeFunctions.timer
    def _tracking_processor(self, frame: np.ndarray
                            ) -> tuple[np.ndarray, float, float]:
        """Processes the input frame for operator detection and tracking."""
        try:
            results_people, annotated_frame = self.tracker.detect_people(frame)
            boxes, track_ids = self.tracker.identify_operator(results_people)
            cropped_image = self.tracker.crop_operator(boxes, track_ids,
                                                       annotated_frame, frame)
            return cropped_image
        except Exception as e:
            print(f"Error during operator detection and tracking extraction: {e}")
            return frame

    def _ajust_camera(self, frame: np.ndarray, boxes: np.ndarray,
                      Gi: Tuple[int, int], Ge: Tuple[int, int]
                      ) -> Tuple[float, float]:
        dist_center_h, dist_center_v = self.tracker.centralize_operator(frame,
                                                                        boxes)
        if np.abs(dist_center_v) > 0.25:
            sc_pitch = np.tanh(-dist_center_v * Gi[0]) * Ge[0]
        else:
            sc_pitch = 0
        if np.abs(dist_center_h) > 0.25:
            sc_yaw = np.tanh(dist_center_h * Gi[1]) * Ge[1] 
        else:
            sc_yaw = 0
        return sc_pitch, sc_yaw

    @TimeFunctions.timer
    def _extraction_processor(self, cropped_image: np.ndarray) -> None:
        """
        Extracts hand and pose features, tracking specific joints and
        triggering gestures based on criteria.
        """
        try:
            if self.stage == 0:
                self._track_hand_gesture(cropped_image)
            elif self.stage == 1:
                self._track_wrist_movement(cropped_image)

            # Check if the gesture duration has been exceeded
            if self.stage == 1 and self.time_functions.toc(self.time_action) > 4:
                self.stage = 2
                self.sample['time_gest'] = self.time_functions.toc(
                    self.time_gesture
                    )
                self.t_classifier = self.time_functions.tic()
            return True
        except Exception as e:
            print(
                "E1 - Error during operator detection, tracking, or feature" +
                f"extraction: {e}"
            )
            self._handle_processing_error(cropped_image)
            return False

    # @TimeFunctions.timer
    def _handle_processing_error(self, frame: np.ndarray) -> None:
        """Handles errors during image processing by flipping the frame and updating history."""
        with self.frame_lock: frame = self.frame_captured
        cv2.imshow('RealSense Camera', cv2.flip(frame, 1))

        # Repeat the last history line if there's an error
        self.hand_history = np.concatenate((self.hand_history, [self.hand_history[-1]]), axis=0)
        self.wrists_history = np.concatenate((self.wrists_history, [self.wrists_history[-1]]), axis=0)

    # @TimeFunctions.timer
    def _track_hand_gesture(self, cropped_image: np.ndarray) -> None:
        """Tracks the fingertips and checks if the hand gesture is enabled based on proximity criteria."""
        try:
            # Find and draw feature hands
            self.hand_results = self.feature_hand.find_features(cropped_image)
            frame_results = self.feature_hand.draw_features(cropped_image, self.hand_results)

            # Annotate the frame with current gesture stage and distance
            self._annotation_image(frame_results)

            # Calculate reference position and hand pose
            hand_ref = self.feature_hand.calculate_reference_pose(self.hand_results, self.sample['joints_trigger_reference'], self.sample['joints_trigger'])
            hand_pose = self.feature_hand.calculate_pose(self.hand_results, self.sample['joints_trigger'])
            hand_center = np.array([hand_pose.flatten() - hand_ref])
            self.hand_history = np.concatenate((self.hand_history, hand_center), axis=0)
        except:
            # If tracking fails, repeat the last hand history entry
            self.hand_history = np.concatenate((self.hand_history, [self.hand_history[-1]]), axis=0)

        # Check trigger conditions for gesture activation
        _, self.hand_history, self.dist_virtual_point = self._check_enabled_trigger(self.hand_history, self.sample['par_trigger_length'], self.sample['par_trigger_dist'])
        if self.dist_virtual_point < self.sample['par_trigger_dist']:
            self.stage = 1
            self.dist_virtual_point = 1
            self.time_gesture = self.time_functions.tic()
            self.time_action = self.time_functions.tic()

    # @TimeFunctions.timer
    def _annotation_image(self, frame: np.ndarray) -> None:
        """Annotates the frame with the information more relevant."""
        # Annotate the frame with current gesture stage and distance
        annotation = f"S{self.stage} D{self.dist_virtual_point:.3f}"
        if self.mode == 'D':
            annotation += f" N{self.num_gest+1}: {self.y_val[self.num_gest]}"
        cv2.putText(
            frame,
            annotation,
            (25, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        cv2.imshow('RealSense Camera', frame)

    # @TimeFunctions.timer
    def _check_enabled_trigger(self, storage_trigger: np.ndarray, length: int = 30, dist: float = 0.03) -> tuple[bool, np.ndarray, float]:
        """
        Checks if a trigger is enabled based on the input array, length, and
        distance criteria.

        :param storage_trigger: Array containing trigger data points.
        :param length: Minimum number of elements in the `storage_trigger`
        array. Defaults to 30.
        :param dist: Threshold distance value. Defaults to 0.03.
        :return: Boolean indicating whether the trigger is enabled, a subset
        of `storage_trigger`, and the calculated distance of the virtual point.
        """
        if len(storage_trigger) < length:
            return False, storage_trigger, 1

        # Use only the last `length` data points
        storage_trigger = storage_trigger[-length:]
        dimension = np.shape(storage_trigger)
        media_coordinates_fingers = np.mean(
            storage_trigger, axis=0
        ).reshape(int(dimension[1] / 2), 2)
        std_fingers_xy = np.std(media_coordinates_fingers, axis=0)

        # Calculate the distance of the virtual point
        dist_virtual_point = np.sqrt(
            std_fingers_xy[0] ** 2 + std_fingers_xy[1] ** 2
        )

        if dist_virtual_point < dist:
            return True, storage_trigger[-1:], dist_virtual_point
        return False, storage_trigger[-length:], dist_virtual_point

    # @TimeFunctions.timer
    def _track_wrist_movement(self, cropped_image: np.ndarray) -> None:
        """
        Tracks wrist movements during the action phase of a gesture.
        """
        try:
            # Find and draw feature pose
            self.pose_results = self.feature_pose.find_features(cropped_image)
            frame_results = self.feature_pose.draw_features(cropped_image, self.pose_results)

            # Annotate the frame with current gesture stage and distance
            self._annotation_image(frame_results)

            # Calculate reference pose for wrist tracking
            body_ref = self.feature_pose.calculate_reference_pose(self.pose_results, self.sample['joints_tracked_reference'], self.sample['joints_tracked'], 3)
            body_pose = self.feature_pose.calculate_pose(self.pose_results, self.sample['joints_tracked'])
            body_center = np.array([body_pose.flatten() - body_ref])
            self.wrists_history = np.concatenate((self.wrists_history, body_center), axis=0)
        except:
            # If tracking fails, repeat the last wrist history entry
            self.wrists_history = np.concatenate((self.wrists_history, [self.wrists_history[-1]]), axis=0)

    # @TimeFunctions.timer
    def process_reduction(self) -> None:
        """
        The function `process_reduction` removes the zero's line from a matrix,
        applies filters, and reduces the matrix to a 6x6 matrix based on
        certain conditions.
        """
        # Remove the zero's line
        self.wrists_history = self.wrists_history[1:]

        # Test the use of filters before applying pca

        # Reduces to a 6x6 matrix
        self.sample['data_reduce_dim'] = np.dot(
            self.wrists_history.T, self.wrists_history
        )

    # @TimeFunctions.timer
    def _update_database(self) -> bool:
        """
        Updates a database with sample data and saves it in JSON format.

        Note: Exclusive function for database construction mode.

        Returns:
            bool: True if all gestures are processed, False otherwise.
        """
        # Update sample data with the current pose track and predicted answer
        self.sample['data_pose_track'] = self.wrists_history
        self.sample['answer_predict'] = self.y_val[self.num_gest]

        # Update database and save it in JSON format
        self._append_to_database()
        self._save_database_to_file()

        # Reset sample data and prepare for the next gesture
        self._reset_sample_data()

        # Move to the next gesture or finish processing
        self.num_gest += 1
        return self.num_gest == self.max_num_gest

    # @TimeFunctions.timer
    def _append_to_database(self) -> None:
        """
        Appends the current sample data to the database.
        """
        gesture_class = str(self.y_val[self.num_gest])
        self.database[gesture_class].append(self.sample)

    # @TimeFunctions.timer
    def _save_database_to_file(self) -> None:
        """
        Saves the database in JSON format to the specified file.
        """
        file_path = os.path.join(self.current_folder, self.file_name_build)
        self.file_handler.save_database(self.sample, self.database, file_path)

    # @TimeFunctions.timer
    def _reset_sample_data(self) -> None:
        """
        Resets sample data and history for the next gesture.
        """
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)

    # @TimeFunctions.timer
    def _classify_gestures(self) -> None:
        """
        Classifies gestures based on the current stage and mode, updates
        predictions, and resets sample data variables for real-time
        gesture classification.

        Note: Exclusive function for Real-Time operating mode.
        """
        # Predict the current gesture and record classification time
        self._predict_gesture()

        # Reset sample data for the next classification
        self._reset_sample_data()

    # @TimeFunctions.timer
    def _predict_gesture(self) -> None:
        """
        Predicts the gesture class and logs the classification time.
        """
        predicted_class = self.classifier.predict(
            self.sample['data_reduce_dim']
        )
        self.y_predict.append(predicted_class)

        classification_time = self.time_functions.toc(self.t_classifier)
        self.time_classifier.append(classification_time)

        print(
            f"\nThe gesture performed belongs to class {predicted_class} " +
            f"and took {classification_time:.3f}ms to be classified.\n"
        )
