import matplotlib.pyplot as plt
import numpy as np
from typing import Type
from sklearn.decomposition import PCA


def validate_ndarray(func):
    """
    Decorator to validate if the argument passed is a NumPy ndarray.
    Raises ValueError if the argument is not a NumPy array.
    """
    def wrapper(*args, **kwargs):
        for arg in args:
            if isinstance(arg, np.ndarray):
                continue
            raise ValueError(
                f"Expected argument of type np.ndarray but got {type(arg)}"
            )
        return func(*args, **kwargs)
    return wrapper


class DrawGraphics:
    """
    A utility class to plot graphs for tracked trajectories and PCA results.
    """

    @staticmethod
    @validate_ndarray
    def tracked_xyz(storage_pose_tracked: np.ndarray) -> None:
        """
        Plot the smoothed trajectory of tracked points using their X and Y
        coordinates.

        :param storage_pose_tracked: Array containing the smoothed trajectory
        of tracked points. Assumes the array has at least 5 columns, where
        columns 0, 1 represent X, Y coordinates for one trajectory, and
        columns 3, 4 represent another set of coordinates.

        :raises ValueError: If the input is not a NumPy array.
        """
        if storage_pose_tracked.shape[1] < 5:
            raise ValueError(
                "Input array must have at least 5 columns for X, Y, and " +
                "additional coordinates."
            )

        plt.figure(figsize=(8, 6))
        plt.title('Smoothed Trajectory of Tracked Points')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.grid(True)
        plt.plot(
            storage_pose_tracked[:, 0],
            storage_pose_tracked[:, 1],
            color='red',
            marker='o',
            linestyle='-',
            label='Trajectory 1 (X, Y)'
        )
        plt.plot(
            storage_pose_tracked[:, 3],
            storage_pose_tracked[:, 4],
            color='blue',
            marker='x',
            linestyle='-',
            label='Trajectory 2 (X, Y)'
        )
        plt.legend()
        plt.show()

    @staticmethod
    def results_pca(pca_model: Type[PCA], n_components: int = 3) -> None:
        """
        Visualize the explained variance ratio per principal component using a
        bar plot and cumulative step plot.

        :param pca_model: Instance of a fitted Principal Component Analysis
        (PCA) model.
        :param n_components: Number of principal components to display.
        Defaults to 3.
        :return: None
        :raises ValueError: If the PCA model is not fitted or n_components is
        greater than the number of components.
        """
        if not hasattr(pca_model, 'explained_variance_ratio_'):
            raise ValueError(
                "PCA model must be fitted before visualizing explained " +
                "variance."
            )

        if n_components > len(pca_model.explained_variance_ratio_):
            raise ValueError(
                "n_components cannot be greater than the number of available" +
                " components."
            )

        explained_variance = pca_model.explained_variance_ratio_

        plt.figure(figsize=(8, 6))
        plt.bar(
            range(1, n_components + 1),
            explained_variance[:n_components],
            alpha=0.5,
            align='center'
        )
        plt.step(
            range(1, n_components + 1),
            np.cumsum(explained_variance[:n_components]),
            where='mid'
        )
        plt.xlabel('Principal Components')
        plt.ylabel('Explained Variance Ratio')
        plt.title('Explained Variance Per Principal Component')
        plt.grid(True)
        plt.show()
