import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from .TimeFunctions import TimeFunctions


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


class FileHandler(metaclass=SingletonMeta):
    """
    A class for handling file operations, such as saving/loading databases and
    results. Implements the Singleton design pattern to ensure only one
    instance.
    """

    def initialize_database(
        self, database: dict, num_gest: int = 10, random: bool = False
    ) -> tuple[list[str], np.ndarray]:
        """
        Initialize the database and return a list of gesture classes and a
        NumPy array of true labels.

        :param database: Dictionary representing gesture classes and their
        data.
        :param num_gest: Number of samples per gesture class.
        :param random: Whether to shuffle the labels randomly.
        :return: A tuple containing a list of gesture classes and an array of
        true labels.
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

    def save_database(
        self, sample: dict, database: dict, file_path: str
    ) -> None:
        """
        Save the database to a JSON file after converting certain fields to
        lists.

        :param sample: Data sample to be saved in the database.
        :param database: Database dictionary.
        :param file_path: Path to save the JSON file.
        """
        sample['data_pose_track'] = sample['data_pose_track'].tolist()
        sample['data_reduce_dim'] = sample['data_reduce_dim'].tolist()

        with open(file_path, 'w') as file:
            json.dump(database, file)

    @TimeFunctions.run_timer
    def load_database(
        self, current_folder: str, file_names: list[str], proportion: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data from multiple files, split it into training and validation
        sets, and calculate average collection time per class.

        :param current_folder: Path to the folder containing database files.
        :param file_names: List of file names to load.
        :param proportion: Proportion of data to be used for training.
        :return: Arrays for training and validation data (features and labels).
        """
        X_train, Y_train, X_val, Y_val = [], [], [], []
        time_reg = np.zeros(5)

        for file_name in file_names:
            file_path = os.path.join(current_folder, file_name)
            database = self._load_json(file_path)
            self._process_samples(
                database, proportion, X_train, Y_train, X_val, Y_val, time_reg
            )

        self._log_dataset_info(X_train, X_val, time_reg)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_val = np.array(X_val)
        Y_val = np.array(Y_val)

        return X_train, Y_train, X_val, Y_val

    @staticmethod
    def _load_json(file_path: str) -> dict:
        """
        Helper method to load a JSON file.

        :param file_path: Path to the JSON file.
        :return: Dictionary containing the JSON data."""
        with open(file_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _process_samples(
        database: dict, proportion: float, X_train: list, Y_train: list,
        X_val: list, Y_val: list, time_reg: np.ndarray
    ) -> None:
        """
        Helper method to process each sample for training and validation.

        :param database: Dictionary containing gesture classes and their data.
        :param proportion: Proportion of data to be used for training.
        :param X_train: List to store training data.
        :param Y_train: List to store training labels.
        :param X_val: List to store validation data.
        :param Y_val: List to store validation labels.
        :param time_reg: Array to store collection times for each gesture
        class.
        """
        for g, (_, value) in enumerate(database.items()):
            np.random.shuffle(value)
            split_idx = int(proportion * len(value))
            for i, sample in enumerate(value):
                if i < split_idx:
                    X_train.append(
                        np.array(sample['data_reduce_dim']).flatten()
                    )
                    Y_train.append(sample['answer_predict'])
                else:
                    X_val.append(np.array(sample['data_reduce_dim']).flatten())
                    Y_val.append(sample['answer_predict'])
                time_reg[g % 5] += sample['time_gest']

    @staticmethod
    def _log_dataset_info(
        X_train: list, X_val: list, time_reg: np.ndarray
    ) -> None:
        """
        Helper method to log dataset information and average collection time.

        :param X_train: Training data.
        :param X_val: Validation data.
        :param time_reg: Array of collection times for each gesture class.
        """
        total_samples = len(X_train) + len(X_val)
        avg_times = time_reg / (total_samples / 5)
        print(f"\nTraining => Samples: {len(X_train)} Classes: {len(X_train)}")
        print(f"Validation => Samples: {len(X_val)} Classes: {len(X_val)}")
        print(f"Average collection time per class: {avg_times}\n")

    @TimeFunctions.run_timer
    def save_results(
        self, y_true: list[str], y_predict: list[str],
        time_classifier: list[float], target_names: list[str], file_path: str
    ) -> None:
        """
        Save classification results and generate confusion matrices.

        :param y_true: True labels.
        :param y_predict: Predicted labels.
        :param time_classifier: Classification time.
        :param target_names: Target class names.
        :param file_path: File path to save the results.
        """
        results = {
            "y_true": y_true,
            "y_predict": y_predict,
            "time_classifier": time_classifier,
        }

        with open(file_path, 'w') as file:
            json.dump(results, file)

        self._plot_confusion_matrix(y_true, y_predict, target_names, file_path)

        print(
            classification_report(
                y_true, y_predict, target_names=target_names, zero_division=0
            )
        )

    @staticmethod
    def _plot_confusion_matrix(
        y_true: list[str], y_predict: list[str], target_names: list[str],
        file_path: str
    ) -> None:
        """
        Generate and save confusion matrices (both percentage and absolute).

        :param y_true: True labels.
        :param y_predict: Predicted labels.
        :param target_names: Target class names.
        :param file_path: File path to save the confusion matrices.
        """
        cm_percentage = confusion_matrix(
            y_true, y_predict, labels=target_names, normalize='true'
        )
        cm_absolute = confusion_matrix(
            y_true, y_predict, labels=target_names
        )

        # Plot and save percentage confusion matrix
        disp_percentage = ConfusionMatrixDisplay(
            confusion_matrix=cm_percentage, display_labels=target_names
        )
        disp_percentage.plot()
        plt.savefig(f"{file_path}_percentage.jpg")

        # Plot and save absolute confusion matrix
        disp_absolute = ConfusionMatrixDisplay(
            confusion_matrix=cm_absolute, display_labels=target_names
        )
        disp_absolute.plot()
        plt.savefig(f"{file_path}_absolute.jpg")
