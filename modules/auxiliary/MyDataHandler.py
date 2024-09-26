import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from functools import wraps
from sklearn.decomposition import PCA
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
from .MyTimer import MyTimer


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


class SingletonMeta(type):
    """
    A Singleton metaclass to ensure a class only has one instance.
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class MyDataHandler(metaclass=SingletonMeta):
    """
    A singleton class to handle file operations like saving/loading databases
    and results. Class responsible for initializing data structures and
    parameters for a pose tracking system.
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
    def initialize_database(database: dict, num_gest: int = 10,
                            random: bool = False) -> tuple[list[str],
                                                           np.ndarray]:
        """
        Initialize the database, return a list of gesture classes and true
        labels.

        :param database: Dictionary representing gesture classes and their
        data.
        :param num_gest: Number of samples per gesture class.
        :param random: Whether to shuffle the labels randomly.
        :return: A tuple of gesture classes and true labels.
        """
        target_names = list(database.keys()) + ['Z']
        y_true = np.array(
            ['I'] * num_gest +
            ['L'] * num_gest +
            ['F'] * num_gest +
            ['T'] * num_gest +
            ['P'] * num_gest
        )
        if random:
            np.random.shuffle(y_true)
        return target_names, y_true

    @staticmethod
    @MyTimer.timing_decorator(use_cv2=True, log_output=False)
    def load_database(current_folder: str, file_names: list[str],
                      proportion: float) -> tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
        """
        Load data from files, split into training and validation sets, and
        calculate average collection time per class.

        :param current_folder: Folder containing the database files.
        :param file_names: List of file names to load.
        :param proportion: Proportion of data used for training.
        :return: Training and validation data (features and labels).
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
    def save_results(y_true: list[str], y_predict: list[str],
                     time_classifier: list[float], target_names: list[str],
                     file_path: str) -> None:
        """
        Save classification results and generate confusion matrices.

        :param y_true: Ground truth labels.
        :param y_predict: Predicted labels.
        :param time_classifier: Classifier time.
        :param target_names: List of class names.
        :param file_path: Path to save the results.
        """
        results = {
            "y_true": y_true,
            "y_predict": y_predict,
            "time_classifier": time_classifier
        }

        with open(file_path, 'w') as file:
            json.dump(results, file)
        target_names, _ = MyDataHandler.initialize_database(      self.database)
        MyDataHandler._plot_confusion_matrix(y_true, y_predict, target_names,
                                           file_path)

        print(classification_report(y_true, y_predict,
                                    target_names=target_names, zero_division=0)
              )

    @staticmethod
    def save_database(sample: dict, database: dict, file_path: str) -> None:
        """
        Save the database to a JSON file with specific fields converted to
        lists.

        :param sample: Data sample to be saved.
        :param database: Database dictionary.
        :param file_path: Path where the JSON file will be saved.
        """
        sample['data_pose_track'] = sample['data_pose_track'].tolist()
        sample['data_reduce_dim'] = sample['data_reduce_dim'].tolist()

        with open(file_path, 'w') as file:
            json.dump(database, file)

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















    @staticmethod
    def _load_json(file_path: str) -> dict:
        """
        Helper method to load a JSON file.

        :param file_path: Path to the JSON file.
        :return: Dictionary containing JSON data.
        """
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _process_samples(database: dict, proportion: float, X_train: list,
                         Y_train: list, X_val: list, Y_val: list,
                         time_reg: np.ndarray) -> None:
        """
        Helper method to split data into training and validation sets.

        :param database: Dictionary with gesture data.
        :param proportion: Proportion of data used for training.
        :param X_train: List to hold training data.
        :param Y_train: List to hold training labels.
        :param X_val: List to hold validation data.
        :param Y_val: List to hold validation labels.
        :param time_reg: Array to store collection times for each gesture.
        """
        for g, (_, samples) in enumerate(database.items()):
            np.random.shuffle(samples)
            split_idx = int(proportion * len(samples))
            for i, sample in enumerate(samples):
                data_flatten = np.array(sample['data_reduce_dim']).flatten()
                if i < split_idx:
                    X_train.append(data_flatten)
                    Y_train.append(sample['answer_predict'])
                else:
                    X_val.append(data_flatten)
                    Y_val.append(sample['answer_predict'])
                time_reg[g % 5] += sample['time_gest']

    @staticmethod
    def _log_dataset_info(X_train: list, X_val: list, time_reg: np.ndarray
                          ) -> None:
        """
        Log dataset info and average collection time.

        :param X_train: Training data.
        :param X_val: Validation data.
        :param time_reg: Time collection array for gestures.
        """
        total_samples = len(X_train) + len(X_val)
        avg_times = time_reg / (total_samples / 5)
        print(f"\nTraining => Samples: {len(X_train)} Classes: {len(X_train)}")
        print(f"Validation => Samples: {len(X_val)} Classes: {len(X_val)}")
        print(f"Average collection time per class: {avg_times}\n")

    
    @staticmethod
    def _plot_confusion_matrix(y_true: list[str], y_predict: list[str],
                               target_names: list[str], file_path: str
                               ) -> None:
        """
        Generate and save confusion matrices as images.

        :param y_true: Ground truth labels.
        :param y_predict: Predicted labels.
        :param target_names: List of class names.
        :param file_path: Path to save the confusion matrices.
        """
        cm_percentage = confusion_matrix(y_true, y_predict,
                                         labels=target_names, normalize='true')
        cm_absolute = confusion_matrix(y_true, y_predict, labels=target_names)

        # Plot and save percentage confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=cm_percentage,
                               display_labels=target_names).plot()
        plt.savefig(f"{file_path}_percentage.jpg")

        # Plot and save absolute confusion matrix
        ConfusionMatrixDisplay(confusion_matrix=cm_absolute,
                               display_labels=target_names).plot()
        plt.savefig(f"{file_path}_absolute.jpg")
