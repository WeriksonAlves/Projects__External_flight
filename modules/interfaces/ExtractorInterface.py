from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class ExtractorInterface(ABC):
    """
    Abstract base class for feature extraction processors.
    Defines the template methods for finding and drawing features.
    """

    @abstractmethod
    def find_features(
        self, projected_window: np.ndarray, *args: Any, **kwargs: Any
    ) -> Any:
        """
        Abstract method to find features in a projected window. Concrete
        subclasses must implement this method.

        :param projected_window: The input image (projected window) to find
        features in.
        :param args: Positional arguments for feature extraction.
        :param kwargs: Keyword arguments for feature extraction.
        :return: The results containing the features found.
        """
        pass

    @abstractmethod
    def draw_features(
        self, projected_window: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Abstract method to draw features on a projected window.
        Concrete subclasses must implement this method.

        :param projected_window: The input image (projected window) to draw
        features on.
        :param args: Positional arguments for feature drawing.
        :param kwargs: Keyword arguments for feature drawing.
        :return: The modified projected window with features drawn.
        """
        pass

    def process_and_draw(
        self, projected_window: np.ndarray, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Template method that defines the skeleton of the feature extraction
        and drawing process. This method uses the `find_features` and
        `draw_features` methods.

        :param projected_window: The projected window (image) to process.
        :param args: Positional arguments for feature extraction and drawing.
        :param kwargs: Keyword arguments for feature extraction and drawing.
        :return: The modified projected window with features drawn.
        """
        features = self.find_features(projected_window, *args, **kwargs)
        return self.draw_features(projected_window, *features)
