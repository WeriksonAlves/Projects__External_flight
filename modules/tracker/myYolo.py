from ..interfaces import TrackerInterface

from collections import defaultdict
from ultralytics import YOLO
from typing import Tuple

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
    
    def detects_people_in_frame(
            self, captured_frame: np.ndarray, persist: bool = True, verbose: bool = False
        ) -> tuple:
        """
        Detects people in the captured frame.

        :param: captured_frame (np.ndarray): The captured frame.
        :param: persist (bool): Whether to persist the results.
        :param: verbose (bool): Whether to output verbose information.
        :return: tuple: Results of people found in the frame.
        """

        results_people = self.yolo_model.track(captured_frame, persist=persist, verbose=verbose)
        results_identifies = results_people[0].plot()
        return results_people, results_identifies

    def identifies_operator(self, results_people: list) -> tuple:
        """
        Identify operator.
        """
        boxes = results_people[0].boxes.xywh.cpu()
        track_ids = results_people[0].boxes.id.int().cpu().tolist()
        return boxes, track_ids

    def crop_operator_in_frame(
            self, boxes: np.ndarray, track_ids: list, results_identifies, captured_frame: np.ndarray, length: int = 90
        ) -> np.ndarray:
        """
        Tracks the operator identified in the captured frame.

        """
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
    
    def centralize_person_in_frame(self, captured_frame: np.ndarray, bounding_box: Tuple[int, int, int, int], yaw: float) -> Tuple[float, float]:
        """
        Adjusts the camera's orientation to center the person in the frame, compensating for yaw.

        :param: captured_frame: The captured frame.
        :param: bounding_box (Center point x and y, width, height): The bounding box of the person.
        :param: yaw: The yaw value to compensate for.
        :return: Tuple[float, float]: The distance to the center horizontally and vertically.
        """
    
        frame_height, frame_width, _ = captured_frame.shape
        frame_center = (frame_width // 2, frame_height // 2)

        box_x, box_y, _, _ = bounding_box

        # Calculate distance to the center with yaw compensation
        distance_to_center_h = (box_x - frame_center[0]) / frame_center[0] - yaw  # Compensate horizontal distance with yaw
        distance_to_center_v = (box_y - frame_center[1]) / frame_center[1]

        return distance_to_center_h, distance_to_center_v

