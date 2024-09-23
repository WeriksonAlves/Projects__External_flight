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


class MyHandMediaPipe(ExtractorInterface):
    """
    MediaPipe processor class for feature extraction, implementing the
    ExtractorInterface. It uses MediaPipe's Hands models to detect and
    extract features.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 1,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75
    ) -> None:
        """
        Initialize MediaPipe processor with hand models.

        :param static_image_mode: Whether to treat the input images as static
        images.
        :param max_num_hands: Maximum number of hands to detect.
        :param model_complexity: The complexity of the hand detection model.
        :param min_detection_confidence: Minimum confidence value for
        detection.
        :param min_tracking_confidence: Minimum confidence value for tracking.
        """
        self.hands_model = mp.solutions.hands.Hands(
                static_image_mode=static_image_mode,
                max_num_hands=max_num_hands,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

    @ensure_rgb
    def find_features(
        self, cropped_image: np.ndarray
    ) -> mp.solutions.hands.Hands:
        """
        Detect hand features in the projected window.

        :param cropped_image: The input image for feature detection,
        expected to be in RGB format.
        :return: The hand detection results.
        """
        hands_results = self.hands_model.process(cropped_image)
        print("Aqui")
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
                draw_landmarks(
                    image=cropped_image,
                    landmark_list=hand_landmarks,
                    connections=HAND_CONNECTIONS,
                    landmark_drawing_spec=get_default_hand_landmarks_style()
                )


class MyPoseMediaPipe(ExtractorInterface):
    """
    MediaPipe processor class for feature extraction, implementing the
    ExtractorInterface. It uses MediaPipe's Pose model to detect and extract
    features.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        model_complexity: int = 1,
        smooth_landmarks=True,
        enable_segmentation=False,
        smooth_segmentation=True,
        min_detection_confidence: float = 0.75,
        min_tracking_confidence: float = 0.75
    ) -> None:
        """
        Initialize MediaPipe processor with pose models.

        :param static_image_mode: Whether to treat the input images as static
        images.
        :param model_complexity: The complexity of the pose detection model.
        :param smooth_landmarks: Whether to smooth landmark coordinates.
        :param enable_segmentation: Whether to enable segmentation.
        :param smooth_segmentation: Whether to smooth segmentation masks.
        :param min_detection_confidence: Minimum confidence value for
        detection.
        :param min_tracking_confidence: Minimum confidence value for tracking.
        """
        self.pose_model: Pose = mp.solutions.pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                smooth_landmarks=smooth_landmarks,
                enable_segmentation=enable_segmentation,
                smooth_segmentation=smooth_segmentation,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

    @ensure_rgb
    def find_features(
        self, cropped_image: np.ndarray
    ) -> NamedTuple:
        """
        Detect pose features in the projected window.

        :param cropped_image: The input image for feature detection,
        expected to be in RGB format.
        :return: The pose detection results.
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
            draw_landmarks(
                image=cropped_image,
                landmark_list=pose_results.pose_landmarks,
                connections=POSE_CONNECTIONS,
                landmark_drawing_spec=get_default_pose_landmarks_style()
            )


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
