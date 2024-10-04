from sklearn.neighbors import KNeighborsClassifier
from typing import List, Tuple
import numpy as np
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..auxiliary.MyTimer import MyTimer


class KNN(ClassifierInterface):
    """
    KNN Classifier wrapping around sklearn's KNeighborsClassifier,
    with added timing and custom logic for training, predicting, and
    validating.
    """

    def __init__(self, initializer: KNeighborsClassifier):
        """
        Initialize the KNN classifier with a given KNeighborsClassifier
        instance.

        :param initializer: Instance of sklearn.neighbors.KNeighborsClassifier.
        :raises ValueError: If the initializer is not an instance of
        KNeighborsClassifier.
        """
        if not isinstance(initializer, KNeighborsClassifier):
            raise ValueError(f"Expected KNeighborsClassifier, got "
                             f"{type(initializer).__name__}")
        self.neigh = initializer

    @MyTimer.timing_decorator(use_cv2=True, log_output=False)
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Fit the KNN model using training data.

        :param x_train: Training feature data (numpy array).
        :param y_train: Corresponding target labels (numpy array).
        :raises ValueError: If inputs are not numpy arrays.
        """
        if not isinstance(x_train, np.ndarray) or not isinstance(y_train,
                                                                 np.ndarray):
            raise ValueError("Both X_train and y_train must be numpy arrays.")

        self.neigh.fit(x_train, y_train)

    def predict(self, reduced_data: np.ndarray, prob_min: float = 0.6) -> str:
        """
        Predict the class label for a given input sample based on probability.

        :param reduced_data: Input data sample (numpy array).
        :param prob_min: Minimum probability threshold for classification.
        :return: Predicted class label as a string, or 'Z' if below the
            threshold.
        :raises ValueError: If input is not a numpy array.
        """
        if not isinstance(reduced_data, np.ndarray):
            raise ValueError("reduced_data must be a numpy array")

        reduced_data = reduced_data.flatten().reshape(1, -1)
        probabilities = self.neigh.predict_proba(reduced_data)

        if np.max(probabilities) > prob_min:
            return self.neigh.predict(reduced_data)[0]
        return 'Z'

    @MyTimer.timing_decorator(use_cv2=True, log_output=False)
    def validate(self, X_val: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Validate the KNN model on a validation dataset.

        :param X_val: Validation data (numpy array).
        :return: Tuple of predicted class labels and their corresponding
            classification times.
        :raises ValueError: If X_val is not a numpy array.
        """
        if not isinstance(X_val, np.ndarray):
            raise ValueError("X_val must be a numpy array")

        predictions = []
        classification_times = []

        for sample in X_val:
            start_time = MyTimer.get_current_time()
            predictions.append(self.predict(sample))
            elapsed = MyTimer.elapsed_time(start_time)
            classification_times.append(elapsed)

        return predictions, classification_times
