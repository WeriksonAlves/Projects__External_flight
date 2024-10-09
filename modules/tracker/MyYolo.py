import cv2
import numpy as np
from ..interfaces import TrackerInterface
from collections import defaultdict
from ultralytics import YOLO
from ultralytics.engine.results import Results
from typing import Tuple, List


def ensure_valid_frame(func):
    """
    Decorator to ensure a valid frame is passed before proceeding.
    """

    def wrapper(self, frame: np.ndarray, *args, **kwargs):
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Invalid frame provided")
        return func(self, frame, *args, **kwargs)

    return wrapper


class MyYolo(TrackerInterface):
    """
    YOLO processor class for detecting and tracking people in a video frame.
    Utilizes a pre-trained YOLO model to identify and track people within a
    given frame.
    """

    def __init__(self, yolo_model_path: str) -> None:
        """
        Initializes the MyYolo class by loading the specified YOLO model.

        :param yolo_model_path: Path to the YOLO model file.
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.track_history = defaultdict(list)

    @ensure_valid_frame
    def detect_people(self, frame: np.ndarray, persist: bool = True,
                      verbose: bool = False
                      ) -> Tuple[List[Results], np.ndarray]:
        """
        Detects people in the given video frame using YOLO model.

        :param frame: The captured video frame as a numpy array.
        :param persist: If True, the tracking data will be saved.
        :param verbose: If True, outputs detailed information about the
        detection process.
        :return: Tuple containing the detection results and the annotated
        frame.
        """
        detection_results = self.yolo_model.track(frame, persist=persist,
                                                  verbose=verbose)
        annotated_frame = detection_results[0].plot()
        return detection_results, annotated_frame

    def identify_operator(self, detection_results: List[Results]
                          ) -> Tuple[np.ndarray, List[int]]:
        """
        Extracts bounding boxes and tracking IDs for people detected in the
        frame.

        :param detection_results: List of detection results from the YOLO
            model.
        :return: Tuple containing the bounding box and tracking ID for the
            detected person.
        """
        detection_result = detection_results[0].boxes
        if detection_result:
            bounding_boxes = detection_result.xywh.cpu().numpy()
            track_ids = detection_result.id.cpu().numpy().astype(int).tolist()

            for box, track_id in zip(bounding_boxes, track_ids):
                x, y, w, h = map(int, box)
                return np.array([x, y, w, h]), track_id
        else:
            return np.array([]), []

    def _draw_track_history(self, annotated_frame: np.ndarray, track_id: int,
                            box: np.ndarray, track_length: int) -> None:
        """
        Draws the track history on the annotated frame for a specific person.

        :param annotated_frame: The frame with drawn annotations.
        :param track_id: The tracking ID of the person.
        :param box: The bounding box for the person.
        :param track_length: The number of points to keep in the track history
        for the operator's path.
        """
        x, y, w, h = map(int, box)
        self.track_history[track_id].append((x + w // 2, y + h // 2))
        self.track_history[track_id] = self.track_history[track_id][
            -track_length:]

        points = np.array(self.track_history[track_id], np.int32).reshape(
            (-1, 1, 2))
        cv2.polylines(annotated_frame, [points], isClosed=False,
                      color=(230, 230, 230), thickness=10)

    def crop_operator(self, bounding_box: np.ndarray, track_id: int,
                      annotated_frame: np.ndarray, frame: np.ndarray,
                      track_length: int = 90) -> Tuple[bool, np.ndarray]:
        """
        Tracks and highlights the operator in the captured frame, and crops
        the region of interest (ROI) for the operator.

        :param box: The bounding box for the detected person.
        :param track_id: The tracking ID of the person.
        :param annotated_frame: The frame with drawn annotations.
        :param frame: The original frame where the operator is to be cropped.
        :param track_length: The number of points to keep in the track history
        for the operator's path.
        :return: Cropped operator region of interest or None if no person is
        detected.
        """
        if bounding_box.size == 0:
            return False, None  # Fallback if no person is detected

        x, y, w, h = map(int, bounding_box)
        self._draw_track_history(annotated_frame, track_id, bounding_box,
                                 track_length)
        person_roi = frame[max(0, y - h // 2): y + h // 2,
                           max(0, x - w // 2): x + w // 2]
        return True, cv2.flip(person_roi, 1)
