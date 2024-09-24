import cv2
import mediapipe as mp
import numpy as np
from ..interfaces.ExtractorInterface import ExtractorInterface
from functools import wraps
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions.pose import Pose, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import (
    get_default_hand_landmarks_style,
    get_default_pose_landmarks_style
)
from typing import Tuple, NamedTuple, List


def ensure_rgb(func):
    """
    Decorator to ensure the input image is in RGB format.
    """
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
    def __init__(self, hands_model: mp.solutions.hands.Hands) -> None:
        """
        Initialize with the hand detection model.
        """
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
        hands_results = self.hands_model.process(projected_window)
        return hands_results

    def draw_features(
        self,
        cropped_image: np.ndarray,
        hands_results: NamedTuple
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
                mp.solutions.drawing_utils.draw_landmarks(
                    image=cropped_image,
                    landmark_list=hand_landmarks,
                    connections=HAND_CONNECTIONS,
                    landmark_drawing_spec=get_default_hand_landmarks_style()
                )
        return cropped_image

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

    def calculate_reference_pose(
        self,
        hands_results: NamedTuple,
        ref_joints: List[int],
        trigger_joints: List[int],
        dimensions: int = 2
    ) -> np.ndarray:
        """
        Find the reference pose based on the hand landmarks.

        :param hands_results: The hand detection results.
        :param ref_joints: Indices of joints to use for reference pose.
        :param trigger_joints: Indices of joints to use for trigger pose.
        :param dimensions: Number of dimensions to return (2 for x, y or 3 for
        x, y, z).
        :return: The reference pose calculated based on the hand landmarks.
        """
        hand_ref = np.tile(
            self.calculate_ref_pose(
                hands_results.multi_hand_landmarks[0], ref_joints, dimensions
            ),
            len(trigger_joints)
        )
        return hand_ref

    def calculate_pose(
        self,
        hands_results: NamedTuple,
        trigger_joints: List[int]
    ) -> np.ndarray:
        """
        Calculate the pose based on the hand landmarks.

        :param hands_results: The hand detection results.
        :param trigger_joints: Indices of joints to use for trigger pose.
        :return: The pose calculated based on the hand landmarks.
        """
        hand_pose = np.array([
            FeatureExtractor.calculate_joint_xy(
                hands_results.multi_hand_landmarks[0], joint
            )
            for joint in trigger_joints
        ])
        return hand_pose


class MyPoseMediaPipe(ExtractorInterface):
    def __init__(self, pose_model: mp.solutions.pose.Pose) -> None:
        """
        Initialize with the pose detection model.
        """
        self.pose_model = pose_model

    @ensure_rgb
    def find_features(
        self, cropped_image: np.ndarray
    ) -> NamedTuple:
        """
        Detect pose features in the projected window.

        :param cropped_image: The input image for feature detection, expected
        to be in BGR format.
        :return: The pose detection result.
        """
        pose_results = self.pose_model.process(cropped_image)
        return pose_results

    def draw_features(
        self,
        cropped_image: np.ndarray,
        pose_results: NamedTuple
    ) -> np.ndarray:
        """
        Draw pose landmarks on the projected window.

        :param cropped_image: The image to draw landmarks on.
        :param pose_results: The pose detection results.
        :return: The modified projected window with landmarks drawn.
        """
        cropped_image.flags.writeable = True
        if pose_results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=cropped_image,
                landmark_list=pose_results.pose_landmarks,
                connections=POSE_CONNECTIONS,
                landmark_drawing_spec=get_default_pose_landmarks_style()
            )
        return cropped_image


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
