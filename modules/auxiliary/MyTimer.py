import cv2
import time
import functools
from typing import Callable, Any
import rospy


class MyTimer:
    """
    Utility class for measuring and tracking the execution time of functions.
    Provides decorators for timing and profiling code execution.
    """

    @staticmethod
    def get_current_time(use_cv2: bool = True) -> float:
        """
        Capture the current time based on the selected method.

        :param use_cv2: Whether to use OpenCV's high-resolution clock.
        :return: The current time in seconds.
        """
        return cv2.getTickCount() if use_cv2 else time.perf_counter()

    @staticmethod
    def elapsed_time(start_time: float, use_cv2: bool = True) -> float:
        """
        Calculate the elapsed time since the start time.

        :param start_time: The start time of the operation.
        :param use_cv2: Whether to use OpenCV's high-resolution clock.
        :return: The elapsed time in seconds.
        """
        if use_cv2:
            return (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        return time.perf_counter() - start_time

    @staticmethod
    def timing_decorator(use_cv2: bool = True, log_output: bool = True
                         ) -> Callable:
        """
        Decorator to measure and log the execution time of a function.

        :param use_cv2: Flag to use OpenCV for timing (default is False).
        :param log_output: Flag to log the result instead of printing.
        :return: Wrapped function with timing measurements.
        """
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                start_time = MyTimer.get_current_time(use_cv2)
                result = func(*args, **kwargs)
                elapsed = MyTimer.elapsed_time(start_time, use_cv2)

                fps = 1 / elapsed if elapsed > 0 else float('inf')
                message = (f"{func.__name__} executed in {elapsed:.5f} "
                           f"-> {fps:.0f} FPS")

                if log_output:
                    rospy.loginfo(message)
                else:
                    print(message)

                return result
            return wrapper
        return decorator
