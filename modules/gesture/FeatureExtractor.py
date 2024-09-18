import numpy as np
from functools import wraps


def validate_pose_data(func):
    """
    Decorator to validate the input pose data and joint index.
    Ensures pose data has valid landmarks and joint_index is within bounds.
    """
    @wraps(func)
    def wrapper(pose_data, joint_index: int):
        if not hasattr(pose_data, 'landmark'):
            raise AttributeError(
                "Invalid pose_data: Missing 'landmark' attribute."
            )
        if not 0 <= joint_index < len(pose_data.landmark):
            raise IndexError(
                "Invalid joint_index: Must be between 0 and " +
                f"{len(pose_data.landmark) - 1}."
            )
        return func(pose_data, joint_index)
    return wrapper


class FeatureExtractor:
    """
    Class responsible for extracting 2D or 3D features from pose data.
    """

    @staticmethod
    def _get_joint_coordinates(
        pose_data, joint_index: int, dimensions: int
    ) -> np.ndarray:
        """
        Helper method to retrieve joint coordinates.

        :param pose_data: Data structure containing information about a
        person's pose.
        :param joint_index: Index of the joint to extract from the `pose_data`.
        :param dimensions: Number of dimensions to return (2 for x, y or 3 for
        x, y, z).
        :return: Array containing the joint coordinates (x, y[, z]).
        """
        joint = pose_data.landmark[joint_index]
        if dimensions == 2:
            return np.array([joint.x, joint.y])
        elif dimensions == 3:
            return np.array([joint.x, joint.y, joint.z])
        else:
            raise ValueError(
                "Invalid dimensions: Must be 2 (x, y) or 3 (x, y, z)."
            )

    @staticmethod
    @validate_pose_data
    def calculate_joint_xy(pose_data, joint_index: int) -> np.ndarray:
        """
        Extracts the x and y coordinates of a specific joint from pose data.

        :param pose_data: Data structure containing information about a
        person's pose.
        :param joint_index: Index of the joint to extract from the `pose_data`.
        :return: Array containing the x and y coordinates of the specified
        joint.
        """
        return FeatureExtractor._get_joint_coordinates(
            pose_data, joint_index, dimensions=2
        )

    @staticmethod
    @validate_pose_data
    def calculate_joint_xyz(pose_data, joint_index: int) -> np.ndarray:
        """
        Extracts the x, y, and z coordinates of a specific joint from pose
        data.

        :param pose_data: Data structure containing information about a
        person's pose.
        :param joint_index: Index of the joint to extract from the `pose_data`.
        :return: Array containing the x, y, and z coordinates of the specified
        joint.
        """
        return FeatureExtractor._get_joint_coordinates(
            pose_data, joint_index, dimensions=3
        )
