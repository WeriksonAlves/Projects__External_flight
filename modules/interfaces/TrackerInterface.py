from abc import ABC, abstractmethod
from typing import Any


class TrackerInterface(ABC):
    """
    Abstract base class for tracking processors.

    This interface defines the contract for any tracking processor that
    implements person detection, box extraction, cropping, and centralizing
    functionality in a video frame.
    """

    @abstractmethod
    def detect_people(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to detect people in the captured frames.

        Implementations should return a tuple containing detection results and
        an annotated frame.
        """
        raise NotImplementedError(
            "Subclasses must implement 'detect_people'."
        )

    @abstractmethod
    def identify_operator(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to extract bounding boxes and tracking IDs from
        detection results.

        Implementations should return bounding boxes (x, y, width, height) and
        tracking IDs.
        """
        raise NotImplementedError(
            "Subclasses must implement 'identify_operator'."
        )

    @abstractmethod
    def crop_operator(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to crop the operator (person) from the captured frame.

        Implementations should track operators in frames and crop the region
        of interest (ROI) for the operator.
        """
        raise NotImplementedError(
            "Subclasses must implement 'crop_operator'."
        )

    @abstractmethod
    def centralize_operator(self, *args: Any, **kwargs: Any) -> Any:
        """
        Abstract method to adjust the frame so that the person is centered.

        Implementations should adjust camera orientation (e.g., yaw, pitch) to
        center the detected person in the frame.
        """
        raise NotImplementedError(
            "Subclasses must implement 'centralize_operator'."
        )
