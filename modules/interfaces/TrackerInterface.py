from abc import ABC, abstractmethod


class TrackerInterface(ABC):
    """
    Abstract base class for tracking processors.
    """
    @abstractmethod
    def detect_people_in_frame(self, *args, **kwargs):
        """
        Abstract method to find people in captured frames.
        """
        pass
    
    @abstractmethod
    def identify_operator(self, *args, **kwargs):
        """
        Abstract method to identify an operator.
        """
        pass
    
    @abstractmethod
    def crop_operator_from_frame(self, *args, **kwargs):
        """
        Abstract method to track operators in frames.
        """
        pass

    @abstractmethod
    def centralize_person_in_frame(self, *args, **kwargs):
        """
        Abstract method to centralize a person in a frame.
        """
        pass
    
