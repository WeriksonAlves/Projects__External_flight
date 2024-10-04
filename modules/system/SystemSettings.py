import numpy as np
from typing import Union, List, Dict


class SingletonMeta(type):
    """ A metaclass for implementing Singleton design pattern. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class ModeFactory:
    """
    Factory class to create different operation modes based on input mode_type.
    """
    @staticmethod
    def create_mode(
        mode_type: str, **kwargs
    ) -> Union['ModeDataset', 'ModeValidate', 'ModeRealTime']:
        """
        Factory method to create instances of different mode classes based on
        mode_type.

        :param mode_type: str: The type of mode ('dataset', 'validate',
        'real_time')
        :param kwargs: dict: Additional parameters for mode initialization
        :return: ModeDataset/ModeValidate/ModeRealTime: An instance of the
        mode class
        """
        modes = {
            'dataset': ModeDataset,
            'validate': ModeValidate,
            'real_time': ModeRealTime
        }

        try:
            return modes[mode_type](**kwargs)
        except KeyError:
            raise ValueError(f"Invalid mode type: {mode_type}")


class ModeDataset:
    """
    Mode for handling datasets for gesture recognition.

    :param database: A database of gestures
    :param file_name_build: The name of the file to build
    :param max_num_gest: Maximum number of gestures to process
    :param dist: Distance parameter for gesture processing
    :param length: Length parameter for gesture processing
    """
    def __init__(
        self, database: Dict[str, List], file_name_build: str,
        max_num_gest: int = 50, dist: float = 0.025, length: int = 15
    ) -> None:
        self.task = 'D'
        self.database = database
        self.file_name_build = file_name_build
        self.max_num_gest = max_num_gest


class ModeValidate:
    """
    Mode for validating gesture recognition models.

    :param files_name: List of file names for validation
    :param database: Database of gestures
    :param name_val: Name of the validation file
    :param proportion: Proportion of data used for validation
    :param n_class: Number of classes
    :param n_sample_class: Number of samples per class
    """
    def __init__(
        self, files_name: List[str], database: Dict[str, List], name_val: str,
        proportion: float = 0.7, n_class: int = 5, n_sample_class: int = 10
    ) -> None:
        self.task = 'V'
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = self._calculate_k(n_class, n_sample_class)
        self.file_name_val = self._rename(n_class, n_sample_class, name_val)

    def _calculate_k(self, n_class: int, n_sample_class: int) -> int:
        """ Helper method to calculate the value of k based on the data. """
        return int(
            np.round(
                np.sqrt(
                    len(self.files_name)*self.proportion*n_class*n_sample_class
                )
            )
        )

    def _rename(self, n_class: int, n_sample_class: int, name_val: str) -> str:
        """
        Generate a validation file name based on input parameters.

        Parameters:
        - n_class: int - Number of classes
        - n_sample_class: int - Number of samples per class
        - name_val: str - Name of the validation file
        """
        s = int(
            len(self.files_name)*(1 - self.proportion)*n_class*n_sample_class
        )
        ma_p = int(10 * self.proportion)
        me_p = int(10 * (1 - self.proportion))
        return f"results/C{n_class}_S{s}_p{ma_p}{me_p}_k{self.k}_{name_val}"


class ModeRealTime:
    """
    Mode for real-time gesture recognition.

    :param files_name: List of file names for real-time processing
    :param database: Database of gestures
    :param proportion: Proportion of data used
    :param n_class: Number of classes
    :param n_sample_class: Number of samples per class
    """
    def __init__(
        self, files_name: List[str], database: Dict[str, List],
        proportion: float = 0.7, n_class: int = 5, n_sample_class: int = 10
    ) -> None:
        self.task = 'RT'
        self.files_name = files_name
        self.database = database
        self.proportion = proportion
        self.k = self._calculate_k(n_class, n_sample_class)

    def _calculate_k(self, n_class: int, n_sample_class: int) -> int:
        """
        Helper method to calculate the value of k for real-time processing.
        """
        return int(
            np.round(
                np.sqrt(
                    len(self.files_name)*self.proportion*n_class*n_sample_class
                )
            )
        )
