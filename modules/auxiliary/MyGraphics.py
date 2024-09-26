import matplotlib.pyplot as plt
import numpy as np
from typing import Type, List
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from functools import wraps


def validate_ndarray(func):
    """
    Decorator to validate if all args are NumPy ndarrays. Raises ValueError if
    not.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise ValueError(f"Expected np.ndarray, got {type(arg)}")
        return func(*args, **kwargs)
    return wrapper


class MyGraphics:
    """
    A utility class to plot graphs for tracked trajectories, PCA results,
    and confusion matrices.
    """

    @staticmethod
    @validate_ndarray
    def tracked_xyz(storage_pose_tracked: np.ndarray, figsize: tuple = (8, 6)
                    ) -> None:
        """
        Plot the smoothed trajectory of tracked points using their X, Y
        coordinates.

        :param storage_pose_tracked: np.ndarray of tracked points. Must have
        at least 5 columns.
        :param figsize: Size of the plot (width, height).
        :raises ValueError: If the input array has fewer than 5 columns.
        """
        if storage_pose_tracked.shape[1] < 5:
            raise ValueError("Input array must have at least 5 columns for X, "
                             "Y and additional coordinates.")

        plt.figure(figsize=figsize)
        plt.title('Smoothed Trajectory of Tracked Points')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)

        # Plot first and second trajectory
        plt.plot(storage_pose_tracked[:, 0], storage_pose_tracked[:, 1], 'ro-',
                 label='Trajectory 1 (X, Y)')
        plt.plot(storage_pose_tracked[:, 3], storage_pose_tracked[:, 4], 'bx-',
                 label='Trajectory 2 (X, Y)')

        plt.legend()
        plt.show()

    @staticmethod
    def results_pca(pca_model: Type[PCA], n_components: int = 3,
                    figsize: tuple = (8, 6)) -> None:
        """
        Visualize explained variance ratio per principal component with a bar
        and cumulative step plot.

        :param pca_model: Fitted PCA model instance.
        :param n_components: Number of components to display (default: 3).
        :param figsize: Size of the plot (width, height).
        :raises ValueError: If the PCA model is not fitted or n_components
        exceeds available components.
        """
        if not hasattr(pca_model, 'explained_variance_ratio_'):
            raise ValueError("PCA model must be fitted to visualize explained "
                             "variance.")

        if n_components > len(pca_model.explained_variance_ratio_):
            raise ValueError(f"n_components cannot exceed "
                             f"{len(pca_model.explained_variance_ratio_)}.")

        explained_variance = pca_model.explained_variance_ratio_

        plt.figure(figsize=figsize)
        plt.bar(range(1, n_components + 1), explained_variance[:n_components],
                alpha=0.5, align='center', label='Variance')
        plt.step(range(1, n_components + 1), np.cumsum(
            explained_variance[:n_components]), where='mid',
            label='Cumulative Variance')

        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Per Principal Component')
        plt.grid(True)
        plt.legend()
        plt.show()

    @staticmethod
    def plot_confusion_matrix(y_true: List[str], y_predict: List[str],
                              target_names: List[str], file_path: str,
                              figsize: tuple = (8, 6)) -> None:
        """
        Generate and save confusion matrices as images (both percentage and
        absolute).

        :param y_true: List of true labels.
        :param y_predict: List of predicted labels.
        :param target_names: List of class names.
        :param file_path: Path to save the confusion matrices.
        :param figsize: Size of the confusion matrix plot (default: (8, 6)).
        """
        cm_percentage = confusion_matrix(y_true, y_predict,
                                         labels=target_names, normalize='true')
        cm_absolute = confusion_matrix(y_true, y_predict, labels=target_names)

        # Save percentage confusion matrix
        plt.figure(figsize=figsize)
        ConfusionMatrixDisplay(confusion_matrix=cm_percentage,
                               display_labels=target_names).plot(cmap='Blues')
        plt.title('Normalized Confusion Matrix')
        plt.savefig(f"{file_path}_percentage.jpg")
        plt.close()

        # Save absolute confusion matrix
        plt.figure(figsize=figsize)
        ConfusionMatrixDisplay(confusion_matrix=cm_absolute,
                               display_labels=target_names).plot(cmap='Blues')
        plt.title('Absolute Confusion Matrix')
        plt.savefig(f"{file_path}_absolute.jpg")
        plt.close()
