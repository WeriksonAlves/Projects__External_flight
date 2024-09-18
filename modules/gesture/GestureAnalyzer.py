import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA
from .FeatureExtractor import FeatureExtractor
from functools import wraps


def validate_dimension(func):
    """
    Decorator to validate the dimension parameter for 2D or 3D calculations.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        dimension = kwargs.get('dimension', 2)
        if dimension not in [2, 3]:
            raise ValueError("Dimension must be either 2 or 3.")
        return func(*args, **kwargs)
    return wrapper


class GestureAnalyzer:
    """
    Class responsible for analyzing gestures.
    """

    @staticmethod
    @validate_dimension
    def calculate_ref_pose(
        data: np.ndarray, joints: np.ndarray, dimension: int = 2
    ) -> np.ndarray:
        """
        Calculates the reference pose based on input data and joint positions
        in either 2D or 3D dimensions.

        :param data: Input data containing joint positions.
        :param joints: Indices of joints in the skeleton.
        :param dimension: Dimensionality for pose calculation (2D or 3D).
        :return: Reference pose calculated based on the input data and joints.
        """
        pose_vector = [
            FeatureExtractor.calculate_joint_xyz(data, joint) if dimension == 3
            else FeatureExtractor.calculate_joint_xy(data, joint)
            for joint in joints
        ]
        reference_pose = np.mean(pose_vector, axis=0)
        return reference_pose

    @staticmethod
    def check_trigger_enabled(
        storage_trigger: np.ndarray, length: int = 30, dist: float = 0.03
    ) -> Tuple[bool, np.ndarray, float]:
        """
        Checks if a trigger is enabled based on the input array, length, and
        distance criteria.

        :param storage_trigger: Array containing trigger data points.
        :param length: Minimum number of elements in the `storage_trigger`
        array. Defaults to 30.
        :param dist: Threshold distance value. Defaults to 0.03.
        :return: Boolean indicating whether the trigger is enabled, a subset
        of `storage_trigger`, and the calculated distance of the virtual point.
        """
        if len(storage_trigger) < length:
            return False, storage_trigger, 1

        # Use only the last `length` data points
        storage_trigger = storage_trigger[-length:]
        dimension = np.shape(storage_trigger)
        media_coordinates_fingers = np.mean(
            storage_trigger, axis=0
        ).reshape(int(dimension[1] / 2), 2)
        std_fingers_xy = np.std(media_coordinates_fingers, axis=0)

        # Calculate the distance of the virtual point
        dist_virtual_point = np.sqrt(
            std_fingers_xy[0] ** 2 + std_fingers_xy[1] ** 2
        )

        if dist_virtual_point < dist:
            return True, storage_trigger[-1:], dist_virtual_point
        return False, storage_trigger[-length:], dist_virtual_point

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
