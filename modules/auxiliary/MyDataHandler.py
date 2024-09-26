import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict
from functools import wraps
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from .MyGraphics import MyGraphics
from .MyTimer import MyTimer


def validate_parameters(func):
    """
    Decorator to validate input parameters for methods that initialize data.
    Ensures positive values for required parameters.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Validate that 'dist' is a positive float
        dist = kwargs.get('dist', None)
        if dist is not None and dist <= 0:
            raise ValueError("Distance (dist) must be a positive float.")

        # Validate that specific parameters are positive integers
        for key in ['length', 'num_coordinate_trigger',
                    'num_coordinate_tracked']:
            if key in kwargs and kwargs[key] <= 0:
                raise ValueError(f"{key} must be a positive integer.")

        return func(*args, **kwargs)
    return wrapper


class SingletonMeta(type):
    """
    A Singleton metaclass to ensure a class only has one instance.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class MyDataHandler(metaclass=SingletonMeta):
    """
    Singleton class for handling file operations, data initialization, and
    processing for gesture recognition and pose tracking.
    """

    @staticmethod
    @validate_parameters
    def initialize_data(dist: float = 0.03, length: int = 20,
                        num_coordinate_trigger: int = 2,
                        num_coordinate_tracked: int = 3
                        ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Initializes the data structures and parameters for the tracking system.

        :param dist: The distance between the trigger and tracked joints.
        :param length: The length of the trigger and tracked joint arrays.
        :param num_coordinate_trigger: The number of coordinates for trigger
            joints.
        :param num_coordinate_tracked: The number of coordinates for tracked
            joints.
        :return: Tuple of trigger history, tracked history, and sample data.
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

        # Efficient initialization of storage arrays
        trigger_history = np.ones(
            (1, len(sample['joints_trigger']) * num_coordinate_trigger),
            dtype=np.float32)
        tracked_history = np.ones(
            (1, len(sample['joints_tracked']) * num_coordinate_tracked),
            dtype=np.float32)

        return trigger_history, tracked_history, sample

    @staticmethod
    def initialize_database(database: dict, num_gest: int = 10,
                            randomize: bool = False) -> Tuple[List[str],
                                                              np.ndarray]:
        """
        Initializes the database and returns gesture classes and true labels.

        :param database: The database dictionary to initialize.
        :param num_gest: The number of gestures per class.
        :param randomize: Whether to randomize the order of the labels.
        :return: Tuple of target names and true labels.
        """
        target_names = list(database.keys()) + ['Z']
        y_true = np.array(['I', 'L', 'F', 'T', 'P'] * num_gest)

        if randomize:
            np.random.shuffle(y_true)

        return target_names, y_true

    @staticmethod
    @MyTimer.timing_decorator(use_cv2=True, log_output=False)
    def load_database(current_folder: str, file_names: List[str],
                      proportion: float) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
        """
        Load and process gesture data from the given files, splitting it into
        training and validation sets.

        :param current_folder: The current folder path.
        :param file_names: The list of file names to load.
        :param proportion: The proportion of samples to use for training.
        :return: Tuple of training and validation data and labels.
        """
        X_train, Y_train, X_val, Y_val = [], [], [], []
        time_reg = np.zeros(5)

        for file_name in file_names:
            file_path = os.path.join(current_folder, file_name)
            database = MyDataHandler._load_json(file_path)
            MyDataHandler._process_samples(database, proportion, X_train,
                                           Y_train, X_val, Y_val, time_reg)

        MyDataHandler._log_dataset_info(X_train, X_val, time_reg)

        return np.array(X_train), np.array(Y_train), np.array(X_val), np.array(
            Y_val)

    @staticmethod
    @MyTimer.timing_decorator(use_cv2=True, log_output=False)
    def save_results(y_true: List[str], y_predict: List[str],
                     time_classifier: List[float], target_names: List[str],
                     file_path: str) -> None:
        """
        Save classification results and generate confusion matrices.

        :param y_true: The true labels.
        :param y_predict: The predicted labels.
        :param time_classifier: The time taken for classification.
        :param target_names: The target class names.
        :param file_path: The file path to save the results
        """
        results = {
            "y_true": y_true,
            "y_predict": y_predict,
            "time_classifier": time_classifier
        }

        with open(file_path + ".json", 'w') as file:
            json.dump(results, file)

        MyGraphics.plot_confusion_matrix(y_true, y_predict, target_names,
                                         file_path)

        print(classification_report(y_true, y_predict,
                                    target_names=target_names, zero_division=0
                                    ))

    @staticmethod
    def save_database(sample: dict, database: dict, file_path: str) -> None:
        """
        Save the database to a JSON file with specific fields converted to
        lists.

        :param sample: The sample data to save.
        :param database: The database to save.
        :param file_path: The file path to save the database.
        """
        sample['data_pose_track'] = np.array(sample['data_pose_track']
                                             ).tolist()
        sample['data_reduce_dim'] = np.array(sample['data_reduce_dim']
                                             ).tolist()

        with open(file_path, 'w') as file:
            json.dump(database, file)

    @staticmethod
    def calculate_pca(data: np.ndarray, n_components: int = 3,
                      verbose: bool = False) -> Tuple[PCA, np.ndarray]:
        """
        Perform Principal Component Analysis (PCA) on the dataset.

        :param data: The input data to perform PCA on.
        :param n_components: The number of components to keep.
        :param verbose: Whether to print additional information.
        :return: Tuple of the PCA model and the covariance matrix.
        """
        if data.size == 0:
            raise ValueError("Input data cannot be empty for PCA.")

        pca_model = PCA(n_components=n_components)
        pca_model.fit(data)

        if verbose:
            print(f"Cumulative explained variance: "
                  f"{pca_model.explained_variance_}")
            print(f"Explained variance ratio: "
                  f"{pca_model.explained_variance_ratio_}")

        return pca_model, pca_model.get_covariance()

    @staticmethod
    def _load_json(file_path: str) -> dict:
        """
        Helper method to load JSON data from a file.
        """
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _process_samples(database: dict, proportion: float, X_train: List,
                         Y_train: List, X_val: List, Y_val: List,
                         time_reg: np.ndarray) -> None:
        """
        Helper method to split data into training and validation sets.
        """
        for g, (_, samples) in enumerate(database.items()):
            np.random.shuffle(samples)
            split_idx = int(proportion * len(samples))

            for i, sample in enumerate(samples):
                flattened_data = np.array(sample['data_reduce_dim']).flatten()
                if i < split_idx:
                    X_train.append(flattened_data)
                    Y_train.append(sample['answer_predict'])
                else:
                    X_val.append(flattened_data)
                    Y_val.append(sample['answer_predict'])

                time_reg[g % 5] += sample['time_gest']

    @staticmethod
    def _log_dataset_info(X_train: List, X_val: List, time_reg: np.ndarray
                          ) -> None:
        """
        Log information about the dataset and average collection time.
        """
        total_samples = len(X_train) + len(X_val)
        avg_times = time_reg / (total_samples / 5)
        print(f"Training => Samples: {len(X_train)}")
        print(f"Validation => Samples: {len(X_val)}")
        print(f"Average collection time per class: {avg_times}")

    @staticmethod
    def _plot_confusion_matrix(y_true: List[str], y_predict: List[str],
                               target_names: List[str], file_path: str
                               ) -> None:
        """
        Generate and save confusion matrices as images.
        """
        cm_percentage = confusion_matrix(y_true, y_predict,
                                         labels=target_names, normalize='true')
        cm_absolute = confusion_matrix(y_true, y_predict, labels=target_names)

        # Save percentage confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=cm_percentage,
                               display_labels=target_names).plot()
        plt.savefig(f"{file_path}_percentage.jpg")

        # Save absolute confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=cm_absolute,
                               display_labels=target_names).plot()
        plt.savefig(f"{file_path}_absolute.jpg")
