import cv2
import time
import functools
from typing import Callable, Any


class TimeFunctions:
    """
    Utility class to measure time and track the execution duration of
    functions.
    """

    @staticmethod
    def tic() -> float:
        """
        Capture the current time.

        :return: The current time in seconds.
        """
        return time.perf_counter()

    @staticmethod
    def toc(start_time: float) -> float:
        """
        Calculate the elapsed time since a given starting time.

        :param start_time: The initial time to measure from.
        :return: The elapsed time in seconds since the start time.
        """
        return time.perf_counter() - start_time

    @staticmethod
    def run_timer(func: Callable) -> Callable:
        """
        Decorator that prints the runtime of the decorated function.

        :param func: The function to wrap and measure.
        :return: A wrapped function that will print its execution time.
        """
        @functools.wraps(func)
        def wrapper_timer(*args: Any, **kwargs: Any) -> Any:
            """
            Wrapper function that measures execution time and prints the
            result.

            :param *args: Positional arguments for the wrapped function.
            :param **kwargs: Keyword arguments for the wrapped function.
            :return: The result of the wrapped function's execution.
            """
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Finished {func.__name__}() in {run_time:.4f} seconds")
            return value

        return wrapper_timer

    @staticmethod
    def timer(func: Callable) -> Callable:
        """
        Decorator to measure the execution time of a function.

        :param func: The function to wrap and measure.
        :return: A wrapped function that will print its execution time.
        """
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = cv2.getTickCount()
            result = func(*args, **kwargs)
            end = cv2.getTickCount()
            print(
                f"{func.__name__} executed in " +
                f"{(end - start) / cv2.getTickFrequency():.6f} seconds"
                )
            return result
        return wrapper
