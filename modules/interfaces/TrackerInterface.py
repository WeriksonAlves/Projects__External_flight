from abc import ABC, abstractmethod


class TrackerInterface(ABC):
    """
    Abstract base class for tracking processors.
    """
    @abstractmethod
    def find_people(self, captured_frame):
        """
        Abstract method to find people in captured frames.

        Args:
            captured_frame (np.ndarray): The captured frame.

        Returns:
            List: A list of people found in the frame.
        """
        pass
    
    @abstractmethod
    def identify_operator(self, results_people):
        """
        Abstract method to identify an operator.

        Args:
            results_people: Results of people found in the frame.
        """
        pass
    
    @abstractmethod
    def track_operator(self, results_people, results_identifies, captured_frame):
        """
        Abstract method to track operators in frames.

        Args:
            results_people: Results of people found in the frame.
            results_identifies (np.ndarray): Image results.
            captured_frame (np.ndarray): The captured frame.
        """
        pass
    
