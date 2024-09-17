from abc import ABC, abstractmethod

class ExtractorInterface(ABC):
    """
    Abstract base class for feature extraction processors.
    """
    @abstractmethod
    def find_features(self, projected_window):
        """
        Abstract method to find features in a projected window.

        Args:
            projected_window (np.ndarray): The projected window.

        Returns:
            Tuple: A tuple containing the found features.
        """
        pass
    
    @abstractmethod
    def draw_features(self, projected_window, results):
        """
        Abstract method to draw features on a projected window.

        Args:
            projected_window (np.ndarray): The projected window.
            results: The results to draw.
            
        Returns:
            np.ndarray: The modified projected window.
        """
        pass