from abc import ABC, abstractmethod
from typing import Any, List
import numpy as np


class ExtractorInterface(ABC):
    """
    Abstract base class for feature extraction processors.
    Defines the template methods for finding and drawing features.
    """

    @abstractmethod
    def find_features(
        self, cropped_image: np.ndarray, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Abstract method to find features in a projected window. Concrete
        subclasses must implement this method.

        :param cropped_image: The input image (projected window) to find
        features in.
        :param args: Positional arguments for feature extraction.
        :param kwargs: Keyword arguments for feature extraction.
        :return: The results containing the features found.
        """
        pass

    @abstractmethod
    def draw_features(self, cropped_image: np.ndarray, results: Any,
                      *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to draw features on a projected window.
        Concrete subclasses must implement this method.

        :param cropped_image: The cropped image to draw features on.
        :param results: The results containing the features found.
        :param args: Positional arguments for feature drawing.
        :param kwargs: Keyword arguments for feature drawing.
        :return: The modified projected window with features drawn.
        """
        pass

    @abstractmethod
    def calculate_reference_pose(self, results: Any, ref_joints: List[int],
                                 joints: List[int], dimensions: int,
                                 *args: Any, **kwargs: Any) -> np.ndarray:
        """
        Abstract method to calculate a reference pose based on the features
        found. Concrete subclasses must implement this method.

        :param results: The results containing the features found.
        :param ref_joints: The reference joints to calculate the pose.
        :param joints: The joints to calculate the pose.
        :param dimensions: The number of dimensions to calculate the pose.
        :param args: Positional arguments for pose calculation.
        :param kwargs: Keyword arguments for pose calculation.
        :return: The reference pose calculated based on the features found.
        """
        pass

    @abstractmethod
    def calculate_pose(self, results: Any, joints: List[int], *args: Any,
                       **kwargs: Any) -> np.ndarray:
        """
        Abstract method to calculate a pose based on the features found.
        Concrete subclasses must implement this method.

        :param results: The results containing the features found.
        :param joints: The joints to calculate the pose.
        :param args: Positional arguments for pose calculation.
        :param kwargs: Keyword arguments for pose calculation.
        :return: The pose calculated based on the features found.
        """
        pass
