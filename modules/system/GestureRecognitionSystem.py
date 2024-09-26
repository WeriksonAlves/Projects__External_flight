import cv2
import os
import numpy as np
import threading
from typing import Union, Tuple
from ..auxiliary.MyDataHandler import MyDataHandler
from ..auxiliary.MyTimer import MyTimer
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..interfaces.ExtractorInterface import ExtractorInterface
from ..interfaces.TrackerInterface import TrackerInterface
from ..servo.ServoPositionSystem import ServoPositionSystem
from ..system.SystemSettings import (InitializeConfig, ModeDataset,
                                     ModeValidate, ModeRealTime)


class GestureRecognitionSystem:
    """"
    Gesture recognition system for real-time, validation, and dataset
    collection modes.

    Attributes:
        file_handler (FileHandler): Handles file operations.
        current_folder (str): Directory to store and access gesture data.
        data_processor (DataProcessor): Processes input data for gesture
        recognition.
        time_functions (TimeFunctions): Time-related utilities for FPS and
        performance tracking.
        tracking_model (TrackerInterface): Tracks detected objects and
        gestures.
        feature_hand (ExtractorInterface): Extracts gesture-related features
        from hand input.
        feature_pose (ExtractorInterface): Extracts gesture-related features
        from pose input.
        classifier (ClassifierInterface, optional): Classifier for recognizing
        gestures (used in validation and real-time modes).
        sps (ServoPositionSystem, optional): Controls the servo motor position
        system.
    """

    log_output = True
    use_cv2 = True

    def __init__(
        self,
        config: InitializeConfig,
        operation: Union[ModeDataset, ModeValidate, ModeRealTime],
        current_folder: str,
        tracking_model: TrackerInterface,
        feature_hand: ExtractorInterface,
        feature_pose: ExtractorInterface,
        classifier: ClassifierInterface = None,
        sps: ServoPositionSystem = None
    ) -> None:
        """
        Initializes the gesture recognition system based on the configuration,
        operation mode, and various processors.
        """
        self.current_folder = current_folder
        self.tracker = tracking_model
        self.feature_hand = feature_hand
        self.feature_pose = feature_pose
        self.classifier = classifier
        self.sps = sps

        # Camera and operation initialization
        self.__initialize_camera(config)
        self.__initialize_operation(operation)
        self.__initialize_variables()
        self.__start_image_thread()

    @property
    def fps(self):
        """Retrieve frames per second from the camera."""
        return self.cap.get(cv2.CAP_PROP_FPS) if self.cap else None

    @MyTimer.timing_decorator(use_cv2)
    def __initialize_camera(self, config: InitializeConfig) -> None:
        """Initializes camera settings based on the provided configuration."""
        self.cap = config.cap
        # self.fps = config.fps
        self.dist = config.dist
        self.length = config.length

    @MyTimer.timing_decorator(use_cv2)
    def __initialize_operation(self, operation: Union[ModeDataset,
                               ModeValidate, ModeRealTime]) -> None:
        """Initializes operation mode and parameters for each mode."""
        self.mode = operation.mode
        if self.mode == 'D':
            self.__initialize_dataset_mode(operation)
        elif self.mode == 'V':
            self.__initialize_validation_mode(operation)
        elif self.mode == 'RT':
            self.__initialize_real_time_mode(operation)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def __initialize_dataset_mode(self, operation: ModeDataset) -> None:
        """Initializes dataset collection mode."""
        self.database = operation.database
        self.file_name_build = operation.file_name_build
        self.max_num_gest = operation.max_num_gest
        self.dist = operation.dist
        self.length = operation.length

    def __initialize_validation_mode(self, operation: ModeValidate) -> None:
        """Initializes validation mode."""
        self.database = operation.database
        self.proportion = operation.proportion
        self.files_name = operation.files_name
        self.file_name_val = operation.file_name_val

    def __initialize_real_time_mode(self, operation: ModeRealTime) -> None:
        """Initializes real-time gesture recognition mode."""
        self.database = operation.database
        self.proportion = operation.proportion
        self.files_name = operation.files_name

    @MyTimer.timing_decorator(use_cv2)
    def __initialize_variables(self) -> None:
        """Initializes core and storage variables."""
        self.stage = 0
        self.num_gest = 0
        self.dist_virtual_point = 1.0
        self.sc_pitch = 0.0
        self.sc_yaw = 0.0
        self.hand_results = None
        self.pose_results = None
        self.y_val = None
        self.frame_captured = None
        self.center_person = False
        self.loop = False
        self.y_predict = []
        self.time_classifier = []

        # Initializing hand and pose storage
        self.__initialize_storage_variables()

    def __initialize_storage_variables(self) -> None:
        """Initializes storage variables for hand and pose data."""
        self.hand_history, self.wrists_history, self.sample = MyDataHandler.initialize_data(
            dist=self.dist, length=self.length
        )

    @MyTimer.timing_decorator(use_cv2)
    def __start_image_thread(self) -> None:
        """Starts a thread for reading images."""
        self.frame_lock = threading.Lock()
        self.image_thread = threading.Thread(target=self._read_image_thread,
                                             daemon=True)
        self.image_thread.start()

    def _read_image_thread(self) -> None:  # Voltar aqui
        """Thread function to read and capture images."""
        # while self.cap.isOpened():
        #     success, frame = self.cap.read()
        #     if success:
        #         # Resize if frame isn't 640x480
        #         if frame.shape[:2] != (640, 480):
        #             frame = cv2.resize(frame, (640, 480))
        #         with self.frame_lock:
        #             self.frame_captured = frame
        while True:
            success, frame = self.cap.read()
            if success:
                # Verify if the size image is 640x480, otherwise, resize it
                if frame.shape[0] != 640 or frame.shape[1] != 480:
                    frame = cv2.resize(frame, (640, 480))
                with self.frame_lock:
                    self.frame_captured = frame

    @MyTimer.timing_decorator(use_cv2)
    def run(self) -> None:
        """
        Main execution loop for the gesture recognition system.
        Based on the mode, it processes frames for gesture detection
        and recognition in real-time.
        """
        self._setup_mode()
        t_frame = MyTimer.get_current_time(self.use_cv2)

        while self.loop:
            if MyTimer.elapsed_time(t_frame, self.use_cv2) > (1 / self.fps):
                t_frame = MyTimer.get_current_time(self.use_cv2)

                # Stop the system if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.stop()

                # In dataset mode, stop after max gestures are collected
                if self.mode == 'D' and self.num_gest >= self.max_num_gest:
                    self.stop()

                # Process the current stage (for gesture detection/recognition)
                self.__process_stage()

    def stop(self) -> None:
        """
        Stops the gesture recognition system and releases resources.
        Ensures that the camera and all OpenCV windows are properly closed.
        """
        self.loop = False
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

    @MyTimer.timing_decorator(use_cv2)
    def _setup_mode(self) -> None:
        """
        Set up the system based on the mode of operation:
        - Dataset ('D'): Initializes the gesture database for collection.
        - Real-Time ('RT'): Loads and fits the classifier for real-time
        recognition.
        - Validation ('V'): Validates the classifier using a validation
        dataset.
        """
        mode_actions = {
            'D': self.__initialize_database,
            'RT': self.__load_and_fit_classifier,
            'V': self.__validate_classifier
        }

        if self.mode in mode_actions:
            mode_actions[self.mode]()
            self.loop = True if self.mode != 'V' else False
        else:
            print(f"Invalid operation mode: {self.mode}")
            self.loop = False

    def __initialize_database(self) -> None:
        """
        Initializes the gesture database, loading target names and validation
        labels from the file handler.
        """
        self.target_names, self.y_val = MyDataHandler.initialize_database(
            self.database)

    def __load_and_fit_classifier(self) -> None:
        """
        Loads training data and fits the classifier for real-time gesture
        recognition.
        The classifier is trained on data loaded from the file handler.
        """
        x_train, y_train, _, _ = MyDataHandler.load_database(
            self.current_folder, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)

    def __validate_classifier(self) -> None:
        """
        Validates the classifier using a validation dataset.
        Trains the classifier and then validates it, saving the results using
        the file handler.
        """
        x_train, y_train, x_val, self.y_val = MyDataHandler.load_database(
            self.current_folder, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)
        self.y_predict, self.time_classifier = self.classifier.validate(x_val)

        # Save results
        self.target_names, _ = MyDataHandler.initialize_database(
            self.database)
        MyDataHandler.save_results(self.y_val, self.y_predict,
                                 self.time_classifier, self.target_names,
                                 os.path.join(self.current_folder,
                                              self.file_name_val))

    @MyTimer.timing_decorator(use_cv2=True, log_output=log_output)
    def __process_stage(self) -> None:
        """
        Handles different stages of gesture recognition depending on the
        current mode.
        Processes image frames, tracks objects, extracts features, and
        classifies gestures.
        """
        if self.stage in [0, 1] and self.mode in ['D', 'RT']:
            success, frame = self._read_image()
            if success:
                cropped_image = self._tracking_processor(frame)
                self._extraction_processor(cropped_image)
        elif self.stage == 2 and self.mode in ['D', 'RT']:
            self._process_reduction_stage()
            self.stage = 3 if self.mode == 'D' else 4
        elif self.stage == 3 and self.mode == 'D':
            self._update_database()
            self.stage = 0
        elif self.stage == 4 and self.mode == 'RT':
            self._classify_gestures()
            self.stage = 0

    def _read_image(self) -> tuple[bool, np.ndarray]:
        """
        Reads the next image frame from the captured stream.
        Locks the frame to ensure safe access across threads.
        """
        with self.frame_lock:
            return self.frame_captured is not None, self.frame_captured

    @MyTimer.timing_decorator(use_cv2)
    def _tracking_processor(self, frame: np.ndarray) -> np.ndarray:
        """
        Processes the input frame for operator detection and tracking. Returns
        the cropped image of the operator.
        """
        try:
            results_people, annotated_frame = self.tracker.detect_people(frame)
            boxes, track_ids = self.tracker.identify_operator(results_people)
            return self.tracker.crop_operator(boxes, track_ids,
                                              annotated_frame, frame)
        except Exception as e:
            print(f"Error during operator detection and tracking: {e}")
            return frame

    def _ajust_camera(self, frame: np.ndarray, boxes: np.ndarray,
                      Gi: tuple[int, int], Ge: tuple[int, int]
                      ) -> Tuple[float, float]:
        """
        Adjusts the camera orientation based on the operator's position in the
        frame. Returns the pitch and yaw adjustments required.
        """
        dist_center_h, dist_center_v = self.tracker.centralize_operator(frame,
                                                                        boxes)
        sc_pitch = np.tanh(-dist_center_v * Gi[0]) * Ge[0] if np.abs(
            dist_center_v) > 0.25 else 0
        sc_yaw = np.tanh(dist_center_h * Gi[1]) * Ge[1] if np.abs(
            dist_center_h) > 0.25 else 0
        return sc_pitch, sc_yaw

    @MyTimer.timing_decorator(use_cv2)
    def _extraction_processor(self, cropped_image: np.ndarray) -> bool:
        """
        Extracts features from the cropped image, such as hand and wrist
        movements. Updates the gesture stage based on criteria.
        """
        try:
            if self.stage == 0:
                self.__track_hand_gesture(cropped_image)
            elif self.stage == 1:
                self.__track_wrist_movement(cropped_image)

            # Check if gesture duration has exceeded
            if self.stage == 1 and MyTimer.elapsed_time(self.time_action,
                                                        self.use_cv2) > 4:
                self.stage = 2
                self.sample['time_gest'] = MyTimer.elapsed_time(
                    self.time_gesture, self.use_cv2)
                self.t_classifier = MyTimer.get_current_time(self.use_cv2)
            return True
        except Exception as e:
            print(f"Error during feature extraction: {e}")
            self.__handle_processing_error(cropped_image)
            return False

    def __handle_processing_error(self, frame: np.ndarray) -> None:
        """
        Handles errors during image processing by flipping the frame and
        updating history.
        """
        with self.frame_lock:
            frame = self.frame_captured
        cv2.imshow('RealSense Camera', cv2.flip(frame, 1))
        self.__repeat_last_history_entry()

    def __repeat_last_history_entry(self) -> None:
        """
        Repeats the last history entry for both hand and wrist tracking.
        Useful when tracking fails, preventing system crashes.
        """
        self.hand_history = np.concatenate((self.hand_history,
                                            [self.hand_history[-1]]), axis=0)
        self.wrists_history = np.concatenate((self.wrists_history,
                                              [self.wrists_history[-1]]),
                                             axis=0)

    def __track_hand_gesture(self, cropped_image: np.ndarray) -> None:
        """
        Tracks fingertips and checks for gesture activation based on proximity
        criteria.
        """
        try:
            self.hand_results = self.feature_hand.find_features(cropped_image)
            frame_results = self.feature_hand.draw_features(cropped_image,
                                                            self.hand_results)
            self.__annotation_image(frame_results)

            hand_ref = self.feature_hand.calculate_reference_pose(
                self.hand_results, self.sample['joints_trigger_reference'],
                self.sample['joints_trigger'])
            hand_pose = self.feature_hand.calculate_pose(
                self.hand_results, self.sample['joints_trigger'])
            hand_center = np.array([hand_pose.flatten() - hand_ref])
            self.hand_history = np.concatenate((self.hand_history,
                                                hand_center), axis=0)

            self.__check_gesture_trigger()
        except:
            self.__repeat_last_history_entry()

    def __check_gesture_trigger(self) -> None:
        """
        Checks if a gesture trigger is enabled based on hand history and
        proximity. If conditions are met, moves to the next gesture stage.
        """
        trigger, self.hand_history, self.dist_virtual_point = self.__check_enabled_trigger(
            self.hand_history, self.sample['par_trigger_length'],
            self.sample['par_trigger_dist'])
        if trigger:
            self.stage = 1
            self.dist_virtual_point = 1
            self.time_gesture = MyTimer.get_current_time(self.use_cv2)
            self.time_action = MyTimer.get_current_time(self.use_cv2)

    def __annotation_image(self, frame: np.ndarray) -> None:
        """
        Annotates the frame with relevant information such as current gesture
        stage and distance.
        """
        annotation = f"S{self.stage} D{self.dist_virtual_point:.3f}"
        if self.mode == 'D':
            annotation += f" N{self.num_gest + 1}: {self.y_val[self.num_gest]}"
        cv2.putText(frame, annotation, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('RealSense Camera', frame)

    def __check_enabled_trigger(self, storage_trigger: np.ndarray,
                                length: int = 30, dist: float = 0.03
                                ) -> tuple[bool, np.ndarray, float]:
        """
        Checks if a trigger is enabled based on the input array and distance
        criteria. Returns a boolean indicating if the trigger is enabled,
        updated storage, and the calculated distance.
        """
        if len(storage_trigger) < length:
            return False, storage_trigger, 1

        storage_trigger = storage_trigger[-length:]
        media_coordinates_fingers = np.mean(storage_trigger, axis=0
                                            ).reshape(-1, 2)
        std_fingers_xy = np.std(media_coordinates_fingers, axis=0)

        dist_virtual_point = np.sqrt(
            std_fingers_xy[0] ** 2 + std_fingers_xy[1] ** 2)
        return dist_virtual_point < dist, storage_trigger, dist_virtual_point

    def __track_wrist_movement(self, cropped_image: np.ndarray) -> None:
        """
        Tracks wrist movements and calculates the body pose based on the
        tracked joints.
        """
        try:
            self.pose_results = self.feature_pose.find_features(cropped_image)
            frame_results = self.feature_pose.draw_features(cropped_image,
                                                            self.pose_results)
            self.__annotation_image(frame_results)

            body_ref = self.feature_pose.calculate_reference_pose(
                self.pose_results,
                self.sample['joints_tracked_reference'],
                self.sample['joints_tracked'],
                3
            )
            body_pose = self.feature_pose.calculate_pose(
                self.pose_results, self.sample['joints_tracked'])
            body_center = np.array([body_pose.flatten() - body_ref])
            self.wrists_history = np.concatenate((self.wrists_history,
                                                  body_center), axis=0)
        except:
            self.__repeat_last_history_entry()

    @MyTimer.timing_decorator(use_cv2)
    def _process_reduction_stage(self) -> None:
        """
        Reduces the dimensionality of the `wrists_history` matrix.
        Removes the zero-line from the matrix, applies necessary filters,
        and reduces it to a 6x6 matrix via a dot product.
        """
        # Efficient zero-line removal
        if len(self.wrists_history) > 1:
            self.wrists_history = self.wrists_history[1:]

        # Reduce dimensionality using matrix multiplication
        self.sample['data_reduce_dim'] = np.dot(self.wrists_history.T,
                                                self.wrists_history)

    @MyTimer.timing_decorator(use_cv2)
    def _update_database(self) -> bool:
        """
        Updates the database with the current gesture data, saving it in JSON
        format. Resets sample data after each update.
        """
        # Update sample with the latest tracking and prediction data
        self.sample['data_pose_track'] = self.wrists_history
        self.sample['answer_predict'] = self.y_val[self.num_gest]

        # Append sample to the database and save it
        self.__append_to_database()
        self.__save_database_to_file()

        # Reset sample data and move to the next gesture
        self.__reset_sample_data()
        self.num_gest += 1

        # Return True if all gestures are processed
        return self.num_gest == self.max_num_gest

    def __append_to_database(self) -> None:
        """
        Appends the current sample data to the database under the correct
        gesture class.
        """
        gesture_class = str(self.y_val[self.num_gest])
        self.database[gesture_class].append(self.sample)

    def __save_database_to_file(self) -> None:
        """
        Saves the database in JSON format to the specified file.
        """
        file_path = os.path.join(self.current_folder, self.file_name_build)
        MyDataHandler.save_database(self.sample, self.database, file_path)

    def __reset_sample_data(self) -> None:
        """
        Resets sample data, including history, for the next gesture.
        """
        self.hand_history, self.wrists_history, self.sample = MyDataHandler.initialize_data(self.dist, self.length)

    @MyTimer.timing_decorator(use_cv2)
    def _classify_gestures(self) -> None:
        """
        Classifies gestures in real-time mode, updates predictions, and resets
        data for the next classification.
        """
        # Predict gesture and log classification time
        self.__predict_gesture()

        # Reset sample data after classification
        self.__reset_sample_data()

    def __predict_gesture(self) -> None:
        """
        Predicts the gesture class based on reduced-dimensionality data.
        Logs the time taken for classification.
        """
        # Predict gesture class using the reduced matrix
        predicted_class = self.classifier.predict(self.sample['data_reduce_dim'
                                                              ])
        self.y_predict.append(predicted_class)

        # Log the time taken for classification
        classification_time = MyTimer.elapsed_time(self.t_classifier,
                                                   self.use_cv2)
        self.time_classifier.append(classification_time)

        print(
            f"\nThe gesture performed belongs to class {predicted_class} " +
            f"and took {classification_time:.3f}ms to be classified.\n"
        )
