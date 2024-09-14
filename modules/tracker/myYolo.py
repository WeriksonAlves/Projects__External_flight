from ..interfaces import TrackerInterface

from collections import defaultdict
from ultralytics import YOLO

import cv2

import numpy as np


class myYolo(TrackerInterface):
    """
    YOLO processor class for tracking.
    """
    def __init__(self, file_YOLO: str) -> None:
        """
        Initialize YOLO processor.

        :param: file_YOLO (str): The YOLO model file.
        """
        self.yolo_model = YOLO(file_YOLO)
        self.track_history = defaultdict(list)
    
    def find_people(self, captured_frame: np.ndarray, persist: bool = True, verbose: bool = False) -> list:
        """
        Find people in captured frame.

        :param: captured_frame (np.ndarray): The captured frame.
        :param: persist (bool): Whether to persist the results.
        :param: verbose (bool): Whether to output verbose information.
        :return: List: A list of people found in the frame.
        """
        return self.yolo_model.track(captured_frame, persist=persist, verbose=verbose)
    
    def identify_operator(self, results_people: list) -> np.ndarray:
        """
        Identify operator.

        :param: results_people: Results of people found in the frame.
        :return: np.ndarray: Image results.
        """
        return results_people[0].plot()
    
    def track_operator(self, results_people: list, results_identifies: np.ndarray, captured_frame: np.ndarray, length: int = 90) -> np.ndarray:
        """
        Tracks the operator identified in the captured frame.

        :param: results_people (list): List of detected people results.
        :param: results_identifies (np.ndarray): Array of identification results.
        :param: captured_frame (np.ndarray): The captured frame.
        :param: length (int, optional): Length of the track history. Defaults to 90.
        :return: np.ndarray: The flipped person ROI and the coordinates of the bounding box (x, y, w, h).
        """
        boxes = results_people[0].boxes.xywh.cpu()
        track_ids = results_people[0].boxes.id.int().cpu().tolist()
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = map(int, box)
            track = self.track_history[track_id]
            track.append((x + w // 2, y + h // 2))
            track = track[-length:]
            points = np.vstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(results_identifies, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            person_roi = captured_frame[(y - h // 2):(y + h // 2), (x - w // 2):(x + w // 2)]
            break
        return cv2.flip(person_roi, 1), (x, y, w, h)
    