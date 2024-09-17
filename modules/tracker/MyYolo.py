from ..interfaces import TrackerInterface
from collections import defaultdict
from typing import Tuple, List
from ultralytics import YOLO

import cv2

import numpy as np


class MyYolo(TrackerInterface):
    """
    YOLO processor class for detecting and tracking people in a video frame.
    Utilizes a pre-trained YOLO model to identify and track people within a given frame.
    """
    
    def __init__(self, yolo_model_path: str) -> None:
        """
        Initializes the MyYolo class by loading the specified YOLO model.
        
        :param yolo_model_path: Path to the YOLO model file.
        """
        self.yolo_model = YOLO(yolo_model_path)
        self.track_history = defaultdict(list)

    def detect_people_in_frame(self, frame: np.ndarray, persist: bool = True, verbose: bool = False) -> Tuple[list, np.ndarray]:
        """
        Detects people in the given video frame.

        :param frame: The captured video frame as a numpy array.
        :param persist: If True, the tracking data will be saved.
        :param verbose: If True, outputs detailed information about the detection process.
        :return: Tuple containing the tracking results and the image with annotations.
        """
        results_people = self.yolo_model.track(frame, persist=persist, verbose=verbose)
        annotated_frame = results_people[0].plot()
        return results_people, annotated_frame

    def identify_operator(self, detection_results: list) -> Tuple[np.ndarray, List[int]]:
        """
        Extracts bounding boxes and tracking IDs for people detected in the frame.

        :param detection_results: List of detection results from YOLO.
        :return: Tuple containing the bounding boxes (x, y, width, height) and tracking IDs.
        """
        boxes = detection_results[0].boxes.xywh.cpu().numpy()
        track_ids = detection_results[0].boxes.id.cpu().numpy().astype(int).tolist()
        return boxes, track_ids

    def crop_operator_from_frame(self, boxes: np.ndarray, track_ids: List[int], annotated_frame: np.ndarray, frame: np.ndarray, track_length: int = 90) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Tracks and highlights the operator in the captured frame, and crops the region of interest (ROI) for the operator.

        :param boxes: Array of bounding boxes for detected people.
        :param track_ids: List of tracking IDs for detected people.
        :param annotated_frame: The frame with drawn annotations.
        :param frame: The original frame where the operator is to be cropped.
        :param track_length: The number of points to keep in track history for the operator's path.
        :return: Cropped image of the operator's region and the bounding box of the operator.
        """
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = map(int, box)  # Convert the box to integer values
            track = self.track_history[track_id]
            track.append((x + w // 2, y + h // 2))  # Track center of the bounding box
            track = track[-track_length:]  # Limit tracking history

            points = np.array(track, np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Crop operator from the frame
            person_roi = frame[max(0, y - h // 2): y + h // 2, max(0, x - w // 2): x + w // 2]
            return cv2.flip(person_roi, 1), (x, y, w, h)

        return frame, (0, 0, 0, 0)  # Fallback if no person is detected

    def centralize_person_in_frame(self, frame: np.ndarray, bounding_box: Tuple[int, int, int, int], drone_pitch: float = 0.0, drone_yaw: float = 0.0) -> Tuple[float, float]:
        """
        Adjusts the camera's orientation to center the detected person in the frame, compensating for yaw.

        :param frame: The captured frame.
        :param bounding_box: The bounding box of the person as (x, y, width, height).
        :param yaw: The yaw value to compensate for.
        :return: Tuple containing the horizontal and vertical distance to the center of the frame.
        """
        frame_height, frame_width, _ = frame.shape
        frame_center = (frame_width // 2, frame_height // 2)

        box_x, box_y, _, _ = bounding_box

        # Calculate the relative distance from the person's center to the frame center
        distance_to_center_h = (box_x - frame_center[0]) / frame_center[0] - drone_yaw
        distance_to_center_v = (box_y - frame_center[1]) / frame_center[1] - drone_pitch

        return distance_to_center_h, distance_to_center_v

