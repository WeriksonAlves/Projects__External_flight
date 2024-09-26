import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union


class TrackerInterface(ABC):
    """
    Abstract base class for tracking processors.

    This interface defines the contract for any tracking processor that
    implements person detection, box extraction, cropping, and centralizing
    functionality in a video frame.
    """

    @abstractmethod
    def detect_people(self, frame: np.ndarray, *args: Any, **kwargs: Any
                      ) -> Tuple[List, np.ndarray]:
        """
        Abstract method to detect people in the captured frames.

        :param frame: The captured video frame as a numpy array.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Tuple containing the detection results and the annotated
            frame.
        """
        raise NotImplementedError("Subclasses must implement 'detect_people'.")

    @abstractmethod
    def identify_operator(self, detection_results: List, *args: Any,
                          **kwargs: Any) -> Tuple[np.ndarray, List[int]]:
        """
        Abstract method to extract bounding boxes and tracking IDs from
        detection results.

        :param detection_results: List of detection results from the tracking
            processor.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Tuple containing the bounding boxes and the list of tracking
            IDs.
        """
        raise NotImplementedError(
            "Subclasses must implement 'identify_operator'.")

    @abstractmethod
    def crop_operator(self, boxes: np.ndarray, track_ids: List[int],
                      frame: np.ndarray, *args: Any, **kwargs: Any
                      ) -> Union[bool, np.ndarray]:
        """
        Abstract method to crop the operator (person) from the captured frame.

        :param boxes: The bounding boxes of the detected people.
        :param track_ids: The tracking IDs of the detected people.
        :param frame: The captured video frame as a numpy array.
        :param args: Additional arguments.
        :param kwargs: Additional keyword arguments.
        :return: Cropped frame containing the operator or False if no operator
            is detected.
        """
        raise NotImplementedError("Subclasses must implement 'crop_operator'.")
