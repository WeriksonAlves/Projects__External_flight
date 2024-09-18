from sklearn.neighbors import KNeighborsClassifier
from typing import List, Tuple
import numpy as np
from ..interfaces.ClassifierInterface import ClassifierInterface
from ..auxiliary.TimeFunctions import TimeFunctions


class KNN(ClassifierInterface):
    """
    KNN Classifier that wraps around the sklearn's KNeighborsClassifier,
    with added functionality for validation and custom prediction logic.
    """

    def __init__(self, initializer: KNeighborsClassifier):
        """
        Initializes the KNN classifier with a given KNeighborsClassifier
        instance.

        :param initializer: The KNeighborsClassifier instance from sklearn.
        """
        if not isinstance(initializer, KNeighborsClassifier):
            raise ValueError(
                "Initializer must be an instance of KNeighborsClassifier"
            )
        self.neigh = initializer

    @TimeFunctions.run_timer
    def fit(self, Xtrain: np.ndarray, Ytrain: np.ndarray) -> None:
        """
        Fit a KNN model using the input training data X_train and
        corresponding target labels Y_train.

        :param Xtrain: The input training data.
        :param Ytrain: The corresponding target labels.
        """
        if not isinstance(Xtrain, np.ndarray) or not isinstance(Ytrain, np.ndarray):
            raise ValueError("Invalid input types for X_train and Y_train")

        self.neigh.fit(Xtrain, Ytrain)

    def predict(self, reduced_data: np.ndarray, prob_min: float = 0.6) -> str:
        """
        Predict the class label for a given sample based on a minimum
        probability threshold.

        :param reduced_data: The input sample (numpy array).
        :param prob_min: The minimum probability threshold for prediction.
        :return: The predicted class label (str) or 'Z' if the probability is
        below the threshold.
        """
        if not isinstance(reduced_data, np.ndarray):
            raise ValueError("reduced_data must be a numpy array")

        # Flatten and convert the data into a 1D list for prediction
        reduced_data = reduced_data.flatten().tolist()

        # Perform probability prediction
        probabilities = self.neigh.predict_proba(np.array([reduced_data]))

        if max(probabilities[0]) > prob_min:
            return self.neigh.predict(np.array([reduced_data]))[0]
        return 'Z'

    @TimeFunctions.run_timer
    def validate(self, X_val: np.ndarray) -> Tuple[List[str], List[float]]:
        """
        Validate the KNN model using the input validation data X_val.

        :param X_val: The input validation data (numpy array).
        :return: A tuple containing a list of predicted class labels and the
        time taken to classify each sample.
        """
        if not isinstance(X_val, np.ndarray):
            raise ValueError("X_val must be a numpy array")

        time_func = TimeFunctions()
        predictions = []
        classification_times = []

        for sample in X_val:
            start_time = time_func.tic()
            predictions.append(self.predict(sample))
            classification_times.append(time_func.toc(start_time))

        return predictions, classification_times
