from abc import ABC, abstractmethod
import numpy as np


class TrackerInterface(ABC):
    """
    Abstract base class for tracking processors.
    """
    @abstractmethod
    def detects_people_in_frame(self, captured_frame: np.ndarray):
        """
        Abstract method to find people in captured frames.

        :param: captured_frame (np.ndarray): The captured frame.
        """
        pass
    
    @abstractmethod
    def identifies_operator(self, results_people):
        """
        Abstract method to identify an operator.

        :param: results_people: Results of people found in the frame.
        """
        pass
    
    @abstractmethod
    def crop_operator_in_frame(self, results_people, results_identifies, captured_frame):
        """
        Abstract method to track operators in frames.

        :param: results_people: List of detected people results.
        :param: results_identifies: Array of identification results.
        :param: captured_frame: The captured frame.
        """
        pass
    
