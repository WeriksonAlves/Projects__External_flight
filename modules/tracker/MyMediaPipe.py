import cv2
import mediapipe as mp
import numpy as np
from functools import wraps
from typing import Tuple, NamedTuple, List
from ..interfaces.ExtractorInterface import ExtractorInterface
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions.pose import Pose, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import (
    get_default_hand_landmarks_style,
    get_default_pose_landmarks_style
)


def ensure_rgb(func):
    """
    Decorator to ensure the input image is in RGB format.
    """
    @wraps(func)
    def wrapper(self, projected_window: np.ndarray, *args, **kwargs):
        rgb_window = cv2.cvtColor(projected_window, cv2.COLOR_BGR2RGB)
        return func(self, rgb_window, *args, **kwargs)
    return wrapper


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


class MyMediaPipe():
    """
    MediaPipe processor class for feature extraction, implementing the
    ExtractorInterface. It uses MediaPipe's Hands and Pose models to detect
    and extract features.
    """

    def __init__(
        self, hands_model: mp.solutions.hands.Hands,
        pose_model: mp.solutions.pose.Pose
    ) -> None:
        """
        Initialize MediaPipe processor with hand and pose models.

        :param hands_model: The hand detection model.
        :param pose_model: The pose detection model.
        """
        self.hands_model = hands_model
        self.pose_model = pose_model

    @ensure_rgb
    def find_features(
        self, projected_window: np.ndarray
    ) -> Tuple[mp.solutions.hands.Hands, mp.solutions.pose.Pose]:
        """
        Detect hand and pose features in the projected window.

        :param projected_window: The input image for feature detection,
        expected to be in BGR format.
        :return: A tuple containing the hand and pose detection results.
        """
        hands_results = self.hands_model.process(projected_window)
        pose_results = self.pose_model.process(projected_window)
        return hands_results, pose_results

    def draw_landmarks2(
        self, projected_window: np.ndarray, results, drawing_spec,
        connections=None
    ) -> None:
        """
        Utility function to draw landmarks on the projected window.

        :param projected_window: The image on which to draw landmarks.
        :param results: The MediaPipe results containing landmarks to draw.
        :param drawing_spec: The drawing specifications.
        :param connections: The connections between landmarks (optional).
        """
        mp.solutions.drawing_utils.draw_landmarks(
            projected_window,
            results,
            connections,
            landmark_drawing_spec=drawing_spec
        )

    def draw_hands(
        self, projected_window: np.ndarray,
        hands_results: mp.solutions.hands.Hands
    ) -> None:
        """
        Draw hand landmarks on the projected window.

        :param projected_window: The image on which to draw hand landmarks.
        :param hands_results: The hand detection results.
        """
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                self.draw_landmarks(
                    projected_window,
                    hand_landmarks,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.hands.HAND_CONNECTIONS
                )

    def draw_pose(
        self, projected_window: np.ndarray,
        pose_results: mp.solutions.pose.Pose
    ) -> None:
        """
        Draw pose landmarks on the projected window.

        :param projected_window: The image on which to draw pose landmarks.
        :param pose_results: The pose detection results.
        """
        if pose_results.pose_landmarks:
            self.draw_landmarks(
                projected_window,
                pose_results.pose_landmarks,
                mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
                mp.solutions.pose.POSE_CONNECTIONS
            )

    def draw_features(
        self, projected_window: np.ndarray,
        hands_results: mp.solutions.hands.Hands,
        pose_results: mp.solutions.pose.Pose
    ) -> np.ndarray:
        """
        Draw both hand and pose landmarks on the projected window.

        :param projected_window: The image to draw features on.
        :param hands_results: The hand detection results.
        :param pose_results: The pose detection results.
        :return: The modified projected window with landmarks drawn.
        """
        projected_window.flags.writeable = True
        self.draw_hands(projected_window, hands_results)
        self.draw_pose(projected_window, pose_results)
        return projected_window


class MyHandsMediaPipe(ExtractorInterface):
    """
    Class for hand feature extraction using MediaPipe's Hands model.
    """
    def __init__(
        self, hands_model: Hands
    ) -> None:
        self.hands_model = hands_model

    @ensure_rgb
    def find_features(
        self, projected_window: np.ndarray
    ) -> NamedTuple:
        """
        Detect hand features in the projected window.

        :param projected_window: The input image for feature detection,
        expected to be in BGR format.
        :return: The hand detection result.
        """
        return self.hands_model.process(projected_window)

    def draw_features(
        self, cropped_image: np.ndarray, hands_results: NamedTuple
    ) -> np.ndarray:
        """
        Draw hand landmarks on the projected window.

        :param cropped_image: The image to draw landmarks on.
        :param hands_results: The hand detection results.
        :return: The modified projected window with landmarks drawn.
        """
        cropped_image.flags.writeable = True
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                draw_landmarks(
                    image=cropped_image,
                    landmark_list=hand_landmarks,
                    connections=HAND_CONNECTIONS,
                    landmark_drawing_spec=get_default_hand_landmarks_style()
                )
        return cropped_image

    def calculate_reference_pose(
        self, hands_results: NamedTuple, ref_joints: List[int],
        trigger_joints: List[int], dimensions: int = 2
    ) -> np.ndarray:
        """
        Calculate the reference pose based on hand landmarks.

        :param hands_results: The hand detection results.
        :param ref_joints: Indices of joints for reference pose.
        :param trigger_joints: Indices of joints for trigger pose.
        :param dimensions: Number of dimensions to return (2 or 3).
        :return: The reference pose.
        """
        hand_ref = np.tile(
            FeatureExtractor.calculate_ref_pose(
                hands_results.multi_hand_landmarks[0], ref_joints, dimensions
            ), len(trigger_joints)
        )
        return hand_ref

    def calculate_pose(
        self, hands_results: NamedTuple, trigger_joints: List[int]
    ) -> np.ndarray:
        """
        Calculate the pose based on hand landmarks.

        :param hands_results: The hand detection results.
        :param trigger_joints: Indices of joints for trigger pose.
        :return: The calculated pose.
        """
        return np.array([
            FeatureExtractor.calculate_joint_xy(
                hands_results.multi_hand_landmarks[0], joint
            )
            for joint in trigger_joints
        ])


class MyPoseMediaPipe(ExtractorInterface):
    """
    Class for pose feature extraction using MediaPipe's Pose model.
    """
    def __init__(
        self, pose_model: Pose
    ) -> None:
        self.pose_model = pose_model

    @ensure_rgb
    def find_features(
        self, cropped_image: np.ndarray
    ) -> NamedTuple:
        """
        Detect pose features in the input image.

        :param cropped_image: The input image for feature detection, expected
        to be in BGR format.
        :return: The pose detection result.
        """
        return self.pose_model.process(cropped_image)

    def draw_features(
        self, cropped_image: np.ndarray, pose_results: NamedTuple
    ) -> np.ndarray:
        """
        Draw pose landmarks on the input image.

        :param cropped_image: The image to draw landmarks on.
        :param pose_results: The pose detection results.
        :return: The modified image with landmarks drawn.
        """
        cropped_image.flags.writeable = True
        if pose_results.pose_landmarks:
            draw_landmarks(
                image=cropped_image,
                landmark_list=pose_results.pose_landmarks,
                connections=POSE_CONNECTIONS,
                landmark_drawing_spec=get_default_pose_landmarks_style()
            )
        return cropped_image

    def calculate_reference_pose(
        self, pose_results: NamedTuple, ref_joints: List[int],
        tracked_joints: List[int], dimensions: int = 2
    ) -> np.ndarray:
        """
        Calculate the reference pose based on pose landmarks.

        :param pose_results: The pose detection results.
        :param ref_joints: Indices of joints for reference pose.
        :param tracked_joints: Indices of joints for tracking.
        :param dimensions: Number of dimensions to return (2 or 3).
        :return: The reference pose.
        """
        pose_ref = np.tile(
            FeatureExtractor.calculate_ref_pose(
                pose_results.pose_landmarks, ref_joints, dimensions
            ), len(tracked_joints)
        )
        return pose_ref

    def calculate_pose(
        self, pose_results: NamedTuple, tracked_joints: List[int]
    ) -> np.ndarray:
        """
        Calculate the pose based on pose landmarks.

        :param pose_results: The pose detection results.
        :param tracked_joints: Indices of joints for tracking.
        :return: The calculated pose.
        """
        return np.array([
            FeatureExtractor.calculate_joint_xyz(
                pose_results.pose_landmarks, joint
            )
            for joint in tracked_joints
        ])


class FeatureExtractor:
    """
    A utility class to extract 2D or 3D joint coordinates from pose data.
    """

    @staticmethod
    def _get_joint_coordinates(
        pose_data, joint_index: int, dimensions: int
    ) -> np.ndarray:
        """
        Retrieve joint coordinates (x, y, [z]).

        :param pose_data: Pose data containing landmark information.
        :param joint_index: The index of the joint to retrieve.
        :param dimensions: Number of dimensions to return (2 or 3).
        :return: An array containing the joint coordinates.
        """
        joint = pose_data.landmark[joint_index]
        if dimensions == 2:
            return np.array([joint.x, joint.y])
        if dimensions == 3:
            return np.array([joint.x, joint.y, joint.z])
        raise ValueError("Invalid dimensions: Must be 2 or 3.")

    @staticmethod
    @validate_pose_data
    def calculate_joint_xy(pose_data, joint_index: int) -> np.ndarray:
        """
        Extract the x and y coordinates of a specific joint.

        :param pose_data: Pose data containing landmark information.
        :param joint_index: Index of the joint to extract.
        :return: An array with the x, y coordinates of the joint.
        """
        return FeatureExtractor._get_joint_coordinates(
            pose_data, joint_index, dimensions=2
        )

    @staticmethod
    @validate_pose_data
    def calculate_joint_xyz(pose_data, joint_index: int) -> np.ndarray:
        """
        Extract the x, y, and z coordinates of a specific joint.

        :param pose_data: Pose data containing landmark information.
        :param joint_index: Index of the joint to extract.
        :return: An array with the x, y, z coordinates of the joint.
        """
        return FeatureExtractor._get_joint_coordinates(
            pose_data, joint_index, dimensions=3
        )

    @staticmethod
    @validate_dimension
    def calculate_ref_pose(
        data: np.ndarray, joints: List[int], dimension: int = 2
    ) -> np.ndarray:
        """
        Calculate the reference pose from joint coordinates.

        :param data: Input pose data.
        :param joints: Indices of joints to use for calculating the reference
        pose.
        :param dimension: Number of dimensions (2 or 3).
        :return: The reference pose as a numpy array.
        """
        pose_data = [
            FeatureExtractor._get_joint_coordinates(
                data, joint, dimension
            ) for joint in joints
        ]
        return np.mean(pose_data, axis=0)
