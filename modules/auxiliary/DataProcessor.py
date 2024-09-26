import numpy as np
from typing import Tuple
from functools import wraps
from sklearn.decomposition import PCA


def validate_parameters(func):
    """
    Decorator to validate input parameters for methods that initialize data.
    Ensures positive values for required parameters.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'dist' in kwargs and kwargs['dist'] <= 0:
            raise ValueError("Distance (dist) must be a positive float.")
        for key in ['length', 'num_coordinate_trigger',
                    'num_coordinate_tracked']:
            if key in kwargs and kwargs[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")
        return func(*args, **kwargs)
    return wrapper


class DataProcessor:
    """
    Class responsible for initializing data structures and parameters for
    a pose tracking system.
    """

    @staticmethod
    @validate_parameters
    def initialize_data(dist: float = 0.03, length: int = 20,
                        num_coordinate_trigger: int = 2,
                        num_coordinate_tracked: int = 3
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Initializes data structures and parameters for a pose tracking system.

        :param dist: Distance for trigger detection.
        :param length: Number of elements in storage_trigger arrays.
        :param num_coordinate_trigger: Number of coordinates tracked for
        trigger joints.
        :param num_coordinate_tracked: Number of coordinates tracked for pose
        joints.
        :return: Tuple containing storage arrays and sample dictionary.
        """
        sample = {
            'answer_predict': '?',
            'data_pose_track': [],
            'data_reduce_dim': [],
            'joints_tracked_reference': [0],
            'joints_tracked': [15, 16],
            'joints_trigger_reference': [9],
            'joints_trigger': [4, 8, 12, 16, 20],
            'par_trigger_dist': dist,
            'par_trigger_length': length,
            'time_gest': 0.0,
            'time_classifier': 0.0
        }

        num_joints_trigger = len(sample['joints_trigger'])
        num_joints_tracked = len(sample['joints_tracked'])

        # Efficient initialization of storage arrays
        storage_trigger = np.ones(
            (1, num_joints_trigger * num_coordinate_trigger), dtype=np.float32)
        storage_tracker = np.ones(
            (1, num_joints_tracked * num_coordinate_tracked), dtype=np.float32)

        return storage_trigger, storage_tracker, sample

    @staticmethod
    def calculate_pca(data: np.ndarray, n_components: int = 3,
                      verbose: bool = False) -> Tuple[PCA, np.ndarray]:
        """
        Performs Principal Component Analysis (PCA) on the dataset.

        :param data: Dataset for PCA analysis.
        :param n_components: Number of principal components to retain.
        :param verbose: If True, prints detailed PCA results.
        :return: PCA model object and covariance matrix.
        """
        if data.size == 0:
            raise ValueError("Input data cannot be empty for PCA.")

        pca_model = PCA(n_components=n_components)
        pca_model.fit(data)

        covariance_matrix = pca_model.get_covariance()

        if verbose:
            explained_variance = pca_model.explained_variance_
            explained_variance_ratio = pca_model.explained_variance_ratio_
            print(f"Cumulative explained variance: {explained_variance}")
            print(f"Explained variance per principal component: "
                  f"{explained_variance_ratio}")

        return pca_model, covariance_matrix
