import cv2
import mediapipe as mp
import numpy as np
from ..interfaces.ExtractorInterface import ExtractorInterface
from mediapipe.python.solution_base import SolutionBase
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.hands import Hands, HAND_CONNECTIONS
from mediapipe.python.solutions.pose import Pose, POSE_CONNECTIONS
from mediapipe.python.solutions.drawing_styles import (
    get_default_hand_landmarks_style,
    get_default_pose_landmarks_style
)
from typing import Tuple, NamedTuple


def ensure_rgb(func):
    """
    Decorator to ensure the input image is in RGB format.
    """
    def wrapper(self, projected_window: np.ndarray, *args, **kwargs):
        rgb_window = cv2.cvtColor(projected_window, cv2.COLOR_BGR2RGB)
        return func(self, rgb_window, *args, **kwargs)
    return wrapper


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


class MyMediaPipe(ExtractorInterface):
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

    def draw_landmarks(
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
