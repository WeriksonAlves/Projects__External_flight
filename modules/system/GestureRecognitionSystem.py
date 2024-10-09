import cv2
import os
import numpy as np
import threading
import rospy
from typing import Optional, Union, Tuple
from ..auxiliary.MyDataHandler import MyDataHandler
from ..auxiliary.MyTimer import MyTimer
from ..bebop_autonomous.BebopROS import BebopROS as B
from ..camera.MyCamera import MyCamera
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..interfaces.ExtractorInterface import ExtractorInterface
from ..interfaces.TrackerInterface import TrackerInterface
from ..system.SystemSettings import (ModeDataset, ModeValidate, ModeRealTime)


class GestureRecognitionSystem:
    """
    Gesture recognition system supporting real-time, validation, and
    dataset collection modes.
    """

    def __init__(
        self,
        base_dir: str,
        configs: MyCamera,
        operation_mode: Union[ModeDataset, ModeValidate, ModeRealTime],
        tracker_model: TrackerInterface,
        hand_extractor_model: ExtractorInterface,
        body_extractor_model: ExtractorInterface,
        classifier: Optional[ClassifierInterface] = None,
        bebop: Optional[B] = None
    ) -> None:
        """
        Initialize the GestureRecognitionSystem.

        :param base_dir: Directory to store and access gesture data.
        :param configs: Configuration settings for the camera.
        :param operation_mode: Operation mode for the system.
        :param tracking_model: Object tracking model for gesture recognition.
        :param feature_hand: Feature extractor for hand gestures.
        :param feature_pose: Feature extractor for pose gestures.
        :param classifier: Classifier for gesture recognition (optional).
        :param sps: Servo Position System for controlling the camera
            (optional).
        """
        self.base_dir = base_dir
        self.configs = configs
        self.operation_mode = operation_mode
        self.tracker = tracker_model
        self.hand_extractor = hand_extractor_model
        self.body_extractor = body_extractor_model
        self.classifier = classifier
        self.B = bebop

        # Initialize system components
        self.__initialize_system()

    @MyTimer.timing_decorator()
    def __initialize_system(self) -> None:
        """Initializes the gesture recognition system."""
        if not (self.__initialize_operation(
        ) and self.__initialize_variables() and self.__start_image_thread()):
            rospy.logerr("System initialization failed.")
            self.__terminate_system()

    def __initialize_operation(self) -> bool:
        """Initializes operation mode and parameters for each mode."""
        try:
            self.mode = self.operation_mode.task
            mode_initializers = {
                'D': self.__initialize_dataset_mode,
                'V': self.__initialize_validation_mode,
                'RT': self.__initialize_real_time_mode
            }

            initializer = mode_initializers.get(self.mode)
            if initializer:
                initializer()
                rospy.loginfo(f"Initialized mode: {self.mode}")
                return True

            raise ValueError(f"Invalid mode: {self.mode}")
        except Exception as e:
            rospy.logerr(f"Error initializing operation mode: {e}")
            return False

    def __initialize_dataset_mode(self) -> None:
        """Initializes dataset collection mode."""
        self.database = self.operation_mode.database
        self.file_name_build = self.operation_mode.file_name_build
        self.max_num_gest = self.operation_mode.max_num_gest
        rospy.logdebug("Dataset mode initialized.")

    def __initialize_validation_mode(self) -> None:
        """Initializes validation mode."""
        self.database = self.operation_mode.database
        self.proportion = self.operation_mode.proportion
        self.files_name = self.operation_mode.files_name
        self.file_name_val = self.operation_mode.file_name_val
        rospy.logdebug("Validation mode initialized.")

    def __initialize_real_time_mode(self) -> None:
        """Initializes real-time gesture recognition mode."""
        self.database = self.operation_mode.database
        self.proportion = self.operation_mode.proportion
        self.files_name = self.operation_mode.files_name
        rospy.logdebug("Real-time mode initialized.")

    def __initialize_variables(self) -> bool:
        """
        Initializes core and storage variables.

        :return: True if initialization is successful, False otherwise.
        """
        try:
            self.cap = self.configs.cap
            self.fps = self.configs.fps
            self.dist = self.configs.dist
            self.length = self.configs.length
            self.stage = 0
            self.num_gest = 0
            self.dist_point = 1.0
            self.hand_results = None
            self.body_results = None
            self.y_val = None
            self.frame_captured = None
            self.loop = False
            self.predictions = []
            self.time_classifier = []

            # Initialize hand and pose storage
            self.__initialize_storage_variables()
            rospy.loginfo("System variables initialized.")
            return True
        except Exception as e:
            rospy.logerr(f"Error initializing variables: {e}")
            return False

    def __initialize_storage_variables(self) -> None:
        """
        Initializes storage variables for hand and pose data.
        """
        self.hand_history, self.body_history, self.sample = MyDataHandler.initialize_data(
            dist=self.dist,
            length=self.length
        )
        rospy.logdebug("Storage variables initialized.")

    def __start_image_thread(self) -> bool:
        """
        Starts a thread for reading images.

        :return: True if the thread starts successfully, False otherwise.
        """
        try:
            self.frame_lock = threading.Lock()
            self.stop_event = threading.Event()
            self.image_thread = threading.Thread(
                target=self.__read_image_thread,
                daemon=True
            )
            self.image_thread.start()
            rospy.loginfo("Image reading thread started.")
            return True
        except Exception as e:
            rospy.logerr(f"Error starting image thread: {e}")
            return False

    def __read_image_thread(self) -> None:
        """
        Thread function to read and capture images.
        Continuously reads frames from the camera and updates `frame_captured`.
        """
        rospy.logdebug("Image thread running.")
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if success:
                with self.frame_lock:
                    self.frame_captured = cv2.resize(
                        frame, (640, 480)) if frame.shape[:2] != (
                            480, 640) else frame
            else:
                rospy.logwarn("Failed to read frame from camera.")
                break
        rospy.logdebug("Image thread terminating.")

    def __terminate_system(self) -> None:
        """Gracefully terminate system."""
        if self.image_thread and self.image_thread.is_alive():
            self.stop_event.set()
            self.image_thread.join()
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        if self.sps:
            self.sps.terminate()
        rospy.loginfo("System terminated successfully.")

    def run(self) -> None:
        """Main execution loop for the gesture recognition system."""
        self._setup_mode()
        t_frame = MyTimer.get_current_time()
        rospy.loginfo("Starting main execution loop.")
        while self.loop:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                rospy.loginfo("Exit signal received (q pressed).")
                self.stop()

            if self.mode == 'D' and self.num_gest >= self.max_num_gest:
                rospy.loginfo("Maximum number of gestures collected.")
                self.stop()

            current_time = MyTimer.get_current_time()
            if MyTimer.elapsed_time(t_frame) > (1 / self.fps):
                t_frame = current_time
                self._process_stage()

    def stop(self) -> None:
        """Stops the gesture recognition system and releases resources."""
        if not self.loop:
            rospy.logdebug("Stop called, but system is already stopping.")
            return

        rospy.loginfo("Stopping gesture recognition system.")
        self.loop = False
        if hasattr(self, 'stop_event'):
            self.stop_event.set()

        if hasattr(self, 'image_thread') and self.image_thread.is_alive():
            self.image_thread.join(timeout=2)
            rospy.logdebug("Image thread joined.")

        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            rospy.logdebug("Camera released.")

        cv2.destroyAllWindows()
        rospy.loginfo("Gesture recognition system stopped.")

    def _setup_mode(self) -> None:
        """Setup mode based on the operation."""
        mode_actions = {
            'D': self.__initialize_database,
            'RT': self.__load_and_fit_classifier,
            'V': self.__validate_classifier
        }
        action = mode_actions.get(self.operation_mode.task)
        if action:
            action()

        self.loop = True if self.mode != 'V' else False

        rospy.logdebug(f"Setting up system for mode: "
                       f"{self.operation_mode.task}")

    def __initialize_database(self) -> None:
        """Initialize gesture database."""
        self.target_names, self.y_val = MyDataHandler.initialize_database(
            self.database)

    def __load_and_fit_classifier(self) -> None:
        """Load and fit classifier for real-time recognition."""
        x_train, y_train, _, _ = MyDataHandler.load_database(
            self.base_dir, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)

    def __validate_classifier(self) -> None:
        """Validate classifier using dataset."""
        x_train, y_train, x_val, self.y_val = MyDataHandler.load_database(
            self.base_dir, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)
        self.predictions, self.time_classifier = self.classifier.validate(
            x_val)
        self.target_names, _ = MyDataHandler.initialize_database(self.database)
        MyDataHandler.save_results(self.y_val.tolist(), self.predictions,
                                   self.time_classifier, self.target_names,
                                   os.path.join(self.base_dir,
                                                self.file_name_val))

    # @MyTimer.timing_decorator()
    def _process_stage(self) -> None:
        """
        Handles different stages of gesture recognition based on the mode.
        Processes image frames, tracks objects, extracts features, and
        classifies gestures.
        """
        if self.stage in [0, 1] and self.mode in ['D', 'RT']:
            if self.process_frame():
                return

        if self.stage == 2 and self.mode in ['D', 'RT']:
            self.process_reduction_stage()
            self.stage = 3 if self.mode == 'D' else 4

        if self.stage == 3 and self.mode == 'D':
            self.update_database()
            self.stage = 0

        if self.stage == 4 and self.mode == 'RT':
            self.classify_gestures()
            self.stage = 0

    def process_frame(self) -> bool:
        """
        Reads the frame, processes tracking, and extracts features.
        Returns True if the frame was successfully processed.
        """
        success, frame = self.read_image()
        if success:
            cropped_image = self.tracking_processor(frame)
            if cropped_image is not None:
                return self.extraction_processor(cropped_image)
            else:
                self._annotate_image(frame)
        else:
            print("Failed to read frame from camera.")
        return False

    def read_image(self) -> Tuple[bool, np.ndarray]:
        """
        Reads the next image frame from the captured stream.
        Safeguards access to the frame by locking across threads.
        """
        with self.frame_lock:
            return self.frame_captured is not None, self.frame_captured

    def tracking_processor(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        """
        Detects and tracks the operator from the given frame.
        Returns the cropped image of the operator.
        """
        try:
            results_people, annotated_frame = self.tracker.detect_people(frame)
            bounding_box, track_id = self.tracker.identify_operator(
                results_people)
            success, cropped_image = self.tracker.crop_operator(
                bounding_box, track_id, annotated_frame, frame)
            if self.B is not None:
                B.ajust_camera(frame, bounding_box, (0.5, 0.5), (0.5, 0.5))
            return cropped_image if success else None
        except Exception as e:
            rospy.logerr(f"Error during operator detection and tracking: {e}")
            return None

    def extraction_processor(self, cropped_image: np.ndarray) -> bool:
        """
        Extracts hand and wrist features and updates gesture stage.
        Returns True if processing was successful.
        """
        try:
            if self.stage == 0:
                self.track_hand_gesture(cropped_image)
            elif self.stage == 1:
                self.track_wrist_movement(cropped_image)

            if self.stage == 1 and MyTimer.elapsed_time(self.time_action) > 4:
                self._transition_to_reduction_stage()
            return True
        except Exception as e:
            rospy.logerr(f"Error during feature extraction: {e}")
            self._handle_processing_error(cropped_image)
            return False

    def _transition_to_reduction_stage(self) -> None:
        """
        Transitions to the reduction stage after sufficient time in stage 1.
        """
        self.stage = 2
        self.sample['time_gest'] = MyTimer.elapsed_time(self.time_gesture)
        self.t_classifier = MyTimer.get_current_time()

    def _handle_processing_error(self, frame: np.ndarray) -> None:
        """
        Handles errors during image processing by showing the frame
        and repeating the last valid history entry.
        """
        with self.frame_lock:
            frame = self.frame_captured
        cv2.imshow('RealSense Camera', cv2.flip(frame, 1))
        self._repeat_last_history_entry()

    def _repeat_last_history_entry(self) -> None:
        """
        Repeats the last valid entries in hand and wrist history.
        Useful for preventing system crashes during tracking failures.
        """
        self.hand_history = np.concatenate((self.hand_history,
                                            [self.hand_history[-1]]), axis=0)
        self.body_history = np.concatenate((self.body_history,
                                            [self.body_history[-1]]), axis=0)

    def track_hand_gesture(self, cropped_image: np.ndarray) -> None:
        """
        Tracks hand gestures using feature extraction.
        Checks for gesture activation based on proximity.
        """
        try:
            self.hand_results = self.hand_extractor.find_features(
                cropped_image
            )
            frame_results = self.hand_extractor.draw_features(
                cropped_image, self.hand_results
            )
            self._annotate_image(frame_results)

            hand_ref = self.hand_extractor.calculate_reference_pose(
                self.hand_results,
                self.sample['joints_trigger_reference'],
                self.sample['joints_trigger']
            )
            hand_pose = self.hand_extractor.calculate_pose(
                self.hand_results,
                self.sample['joints_trigger']
            )
            hand_center = np.array([hand_pose.flatten() - hand_ref])
            self.hand_history = np.concatenate(
                (self.hand_history, hand_center),
                axis=0
            )

            self.check_gesture_trigger()
        except Exception:
            self._repeat_last_history_entry()

    def check_gesture_trigger(self) -> None:
        """
        Checks if the gesture trigger is activated based on hand history.
        If activated, moves to the next stage.
        """
        trigger, self.hand_history, self.dist_point = self._is_trigger_enabled(
            self.hand_history,
            self.sample['par_trigger_length'],
            self.sample['par_trigger_dist']
        )
        if trigger:
            self.stage = 1
            self.dist_point = 1
            self.time_gesture = MyTimer.get_current_time()
            self.time_action = MyTimer.get_current_time()

    def _is_trigger_enabled(self, storage: np.ndarray, length: int,
                            dist: float) -> Tuple[bool, np.ndarray, float]:
        """
        Determines whether a gesture trigger is enabled based on storage and
        distance.
        Returns a tuple with the trigger status, updated storage, and the
        calculated distance.
        """
        if len(storage) < length:
            return False, storage, 1

        storage = storage[-length:]
        mean_coords = np.mean(storage, axis=0).reshape(-1, 2)
        std_dev = np.std(mean_coords, axis=0)

        dist_point = np.sqrt(std_dev[0] ** 2 + std_dev[1] ** 2)
        return dist_point < dist, storage, dist_point

    def track_wrist_movement(self, cropped_image: np.ndarray) -> None:
        """
        Tracks wrist movements using pose extraction.
        Updates the wrist history with the new data.
        """
        try:
            self.body_results = self.body_extractor.find_features(
                cropped_image
            )
            frame_results = self.body_extractor.draw_features(
                cropped_image, self.body_results
            )
            self._annotate_image(frame_results)

            body_ref = self.body_extractor.calculate_reference_pose(
                self.body_results,
                self.sample['joints_tracked_reference'],
                self.sample['joints_tracked'],
                3
            )
            body_pose = self.body_extractor.calculate_pose(
                self.body_results,
                self.sample['joints_tracked']
            )
            body_center = np.array([body_pose.flatten() - body_ref])
            self.body_history = np.concatenate(
                (self.body_history, body_center),
                axis=0
            )
        except Exception:
            self._repeat_last_history_entry()

    def _annotate_image(self, frame: np.ndarray) -> None:
        """
        Annotates the given frame with relevant information such as the
        current gesture stage and distance.
        """
        annotation = f"S{self.stage} D{self.dist_point:.3f}"
        if self.mode == 'D':
            annotation += f" N{self.num_gest + 1}: {self.y_val[self.num_gest]}"
        cv2.putText(frame, annotation, (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow('RealSense Camera', frame)

    def process_reduction_stage(self) -> None:
        """
        Reduces the dimensionality of the wrist history matrix.
        Applies necessary filters and performs dimensionality reduction.
        """
        self.body_history = self.body_history[1:]  # Remove zero line
        self.sample['data_reduce_dim'] = np.dot(self.body_history.T,
                                                self.body_history)

    def update_database(self) -> None:
        """
        Updates the database with the current gesture data and resets sample
        data.
        """
        self.sample['data_pose_track'] = self.body_history
        self.sample['answer_predict'] = self.y_val[self.num_gest]

        self.__append_to_database()
        self.__save_database_to_file()
        self.__initialize_storage_variables()
        self.num_gest += 1

    def __append_to_database(self) -> None:
        """
        Appends the current gesture sample to the database.
        """
        gesture_class = str(self.y_val[self.num_gest])
        self.database[gesture_class].append(self.sample)

    def __save_database_to_file(self) -> None:
        """
        Saves the current database to a file.
        """
        file_path = os.path.join(self.base_dir, self.file_name_build)
        MyDataHandler.save_database(self.sample, self.database, file_path)

    def classify_gestures(self) -> None:
        """
        Classifies gestures in real-time mode and resets sample data for the
        next classification.
        """
        self.__predict_gesture()
        self.__initialize_storage_variables()

    def __predict_gesture(self) -> None:
        """
        Predicts the gesture class based on reduced data and logs the
        classification time.
        """
        predicted_class = self.classifier.predict(
            self.sample['data_reduce_dim']
        )
        self.predictions.append(predicted_class)

        classification_time = MyTimer.elapsed_time(self.t_classifier)
        self.time_classifier.append(classification_time)

        print(
            f"\nThe gesture performed belongs to class {predicted_class} " +
            f"and took {classification_time:.3f}ms to be classified.\n"
        )
        rospy.loginfo(f"\nThe gesture belongs to class {predicted_class} and "
                      f"took {classification_time:.3f}ms to classify.\n")
