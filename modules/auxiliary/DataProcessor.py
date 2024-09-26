import numpy as np
from typing import Tuple
from functools import wraps
from sklearn.decomposition import PCA


def validate_parameters(func):
    """
    Decorator to validate the input parameters for the data initialization
    method. Ensures that distances, lengths, and coordinates are positive
    values.
    """
    @wraps(func)
    def wrapper(
        self, dist: float, length: int, num_coordinate_trigger: int = 2,
        num_coordinate_tracked: int = 3, *args, **kwargs
    ):
        if dist <= 0:
            raise ValueError("Distance (dist) must be a positive float.")
        if length <= 0 or num_coordinate_trigger <= 0 or num_coordinate_tracked <= 0:
            raise ValueError(
                "Length and coordinate counts must be positive integers."
            )
        return func(
            self, dist, length, num_coordinate_trigger,
            num_coordinate_tracked, *args, **kwargs
        )
    return wrapper


class DataProcessor:
    """
    Class responsible for initializing data structures and parameters for a
    pose tracking system.
    """

    @validate_parameters
    def initialize_data(
        self,
        dist: float = 0.03,
        length: int = 20,
        num_coordinate_trigger: int = 2,
        num_coordinate_tracked: int = 3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Initializes data structures and parameters for a pose tracking system.

        :param dist: Distance value used as a parameter for trigger detection.
        :param length: Number of elements in the `storage_trigger` arrays.
        :param num_coordinate_trigger: Number of coordinates to be tracked for
        each joint in the trigger set.
        :param num_coordinate_tracked: Number of coordinates tracked for each
        joint in the `data_pose_track` array.
        :return: A tuple containing:
            storage_trigger_left: Array for tracking left-hand trigger joint
            coordinates.
            storage_trigger_right: Array for tracking right-hand trigger joint
            coordinates.
            storage_pose_tracked: Array for storing pose data of tracked
            joints.
            sample: Dictionary of sample parameters for the pose tracking
            system.
        """

        # Sample dictionary for tracking pose and gesture classification data
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

        # Initialize arrays for storing trigger and pose tracking data
        num_triggers = len(sample['joints_trigger'])
        num_tracked_joints = len(sample['joints_tracked'])

        storage_trigger_left = np.ones(
            (1, num_triggers * num_coordinate_trigger), dtype=np.float32
        )
        storage_trigger_right = np.ones(
            (1, num_triggers * num_coordinate_trigger), dtype=np.float32
        )
        storage_pose_tracked = np.zeros(
            (1, num_tracked_joints * num_coordinate_tracked), dtype=np.float32
        )

        return storage_trigger_left, storage_trigger_right, storage_pose_tracked, sample

    @staticmethod
    def calculate_pca(
        data: np.ndarray, n_components: int = 3, verbose: bool = False
    ) -> Tuple[PCA, np.ndarray]:
        """
        Performs Principal Component Analysis (PCA) on a dataset.

        :param data: Dataset for PCA analysis.
        :param n_components: Number of principal components to retain.
        :param verbose: Whether to print the PCA output. Defaults to False.
        :return: PCA model object and covariance matrix.
        """
        pca_model = PCA(n_components=n_components)
        pca_model.fit(data)

        covariance_matrix = pca_model.get_covariance()

        if verbose:
            explained_variance = pca_model.explained_variance_
            explained_variance_ratio = pca_model.explained_variance_ratio_
            print(f"Cumulative explained variance: {explained_variance}")
            print(
                "Explained variance per principal component: " +
                f"{explained_variance_ratio}"
            )

        return pca_model, covariance_matrix
