from abc import ABC, abstractmethod


class TrackerInterface(ABC):
    """
    Abstract base class for tracking processors.

    This interface defines the contract for any tracking processor that
    implements person detection, box extraction, cropping, and centralizing
    functionality in a video frame.
    """

    @abstractmethod
    def detect_people_in_frame(self, *args, **kwargs):
        """
        Abstract method to detect people in the captured frames.

        Implementations should return a tuple containing detection results and
        an annotated frame.

        :param args: Positional arguments for detection.
        :param kwargs: Keyword arguments for detection.
        :return: A tuple with detection results and the processed frame (e.g.,
        List of detections, np.ndarray of the frame).
        """
        raise NotImplementedError(
            "Subclasses must implement 'detect_people_in_frame'."
        )

    @abstractmethod
    def extract_boxes_and_ids(self, *args, **kwargs):
        """
        Abstract method to extract bounding boxes and tracking IDs from
        detection results.

        Implementations should return bounding boxes (x, y, width, height) and
        tracking IDs.

        :param args: Positional arguments for box and ID extraction.
        :param kwargs: Keyword arguments for box and ID extraction.
        :return: A tuple containing the bounding boxes and their respective
        tracking IDs (e.g., np.ndarray of boxes, List of tracking IDs).
        """
        raise NotImplementedError(
            "Subclasses must implement 'extract_boxes_and_ids'."
        )

    @abstractmethod
    def crop_operator_from_frame(self, *args, **kwargs):
        """
        Abstract method to crop the operator (person) from the captured frame.

        Implementations should track operators in frames and crop the region
        of interest (ROI) for the operator.

        :param args: Positional arguments for cropping.
        :param kwargs: Keyword arguments for cropping.
        :return: A tuple containing the cropped region and the bounding box of
        the operator (e.g., np.ndarray of the cropped frame, Tuple of bounding
        box).
        """
        raise NotImplementedError(
            "Subclasses must implement 'crop_operator_from_frame'."
        )

    @abstractmethod
    def centralize_person_in_frame(self, *args, **kwargs):
        """
        Abstract method to adjust the frame so that the person is centered.

        Implementations should adjust camera orientation (e.g., yaw, pitch) to
        center the detected person in the frame.

        :param args: Positional arguments for centralization.
        :param kwargs: Keyword arguments for centralization.
        :return: A tuple containing the distances to center the person
        horizontally and vertically (e.g., Tuple of horizontal and vertical
        distances).
        """
        raise NotImplementedError(
            "Subclasses must implement 'centralize_person_in_frame'."
        )
